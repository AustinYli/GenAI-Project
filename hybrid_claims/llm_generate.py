"""Generate fixed decomposition and HyDE prompts on one evaluation partition."""

import argparse
from copy import deepcopy
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import paths
from common.data import load_queries, split_queries


OUT = Path(paths.OUT) / 'llm_decomp_vs_hyde'
CACHE = Path(paths.CACHE) / 'llm_decomp_vs_hyde'
DESIGN = {
    'model': 'Qwen/Qwen2.5-3B-Instruct', 'dtype': 'bf16', 'padding': 'left',
    'seed': 13, 'split_seed': 13, 'cap': 2, 'cuts': [10, 6],
    'whole_depth': 100, 'extra_depth': 60, 'rrf_k': 60, 'bm25_k1': 0.9, 'bm25_b': 0.4,
    'decomp': {
        'prompt': 'Split this question into the separate facts that must be looked up, '
                  'one short sub-question per line, at most 4 lines, no numbering, no other text.',
        'max_new_tokens': 128, 'batch': 40, 'num_return_sequences': 1, 'do_sample': False,
    },
    'hyde': {
        'prompt': 'Write a short news article excerpt that answers this question.',
        'max_new_tokens': 150, 'batch': 10, 'num_return_sequences': 4,
        'do_sample': True, 'temperature': 0.7, 'top_k': 0, 'top_p': 1.0,
    },
    'selection': 'Within each LLM family pick by dev complete@10, prefer the simpler variant on a tie. '
                 'Report all predeclared systems on test, regardless of the dev selection.',
    'decomp_filter': 'Remove line bullets or numbering. Keep the first four lines with at least three '
                     'tokens containing a letter after the existing BM25 stopword filter. '
                     'If all lines fail, use the rule-based claim texts with the variant search backend.',
    'hyde_dense': 'Equal-weight mean of four normalized passage embeddings and the normalized cached '
                  'question embedding, followed by L2 normalization.',
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def frozen_design():
    file = OUT / 'design.json'
    if file.exists():
        if json.loads(file.read_text(encoding='utf-8')) != DESIGN:
            raise ValueError('Saved design differs from the fixed experiment design')
    else:
        write_json(file, DESIGN)
    return sha256(file.read_bytes()).hexdigest()


def partition(queries, split):
    dev, test = split_queries(queries, seed=DESIGN['split_seed'])
    return [q for q in (dev if split == 'dev' else test) if queries[q]['question_type'] != 'null_query']


def generate(model, tokenizer, questions, settings, seed, *, processor=None):
    import torch

    texts = [f"{settings['prompt']}\n\nQuestion: {q}" for q in questions]
    if processor is None:
        prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                   add_generation_prompt=True, tokenize=False) for text in texts]
        batch = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
    else:
        messages = [[{'role': 'user', 'content': [{'type': 'text', 'text': text}]}] for text in texts]
        batch = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                             tokenize=True, return_dict=True, return_tensors='pt',
                                             processor_kwargs={'padding': True}).to('cuda')
    config = deepcopy(model.generation_config)
    config.max_new_tokens = settings['max_new_tokens']
    config.do_sample = settings['do_sample']
    config.num_return_sequences = settings['num_return_sequences']
    config.temperature = settings.get('temperature')
    config.top_k = settings.get('top_k')
    config.top_p = settings.get('top_p')
    config.repetition_penalty = 1.0
    config.pad_token_id = tokenizer.pad_token_id
    torch.manual_seed(seed)
    with torch.inference_mode():
        tokens = model.generate(**batch, generation_config=config)[:, batch.input_ids.shape[1]:].cpu()
    if processor is not None:
        raw = tokenizer.batch_decode(tokens, skip_special_tokens=False)
        if any('<think>' in text.lower() or '</think>' in text.lower() for text in raw):
            raise ValueError('A generated continuation contains a thinking tag')
    eos = config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    lengths = []
    for row in tokens.tolist():
        count = next((i + 1 for i, tok in enumerate(row) if tok in eos_ids), len(row))
        lengths.append(count)
    return tokenizer.batch_decode(tokens, skip_special_tokens=True), lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('dev', 'test'))
    args = parser.parse_args()
    signature = frozen_design()
    if args.split == 'test':
        decision_path = OUT / 'dev_decisions.json'
        decision = json.loads(decision_path.read_text(encoding='utf-8'))
        if decision['design_sha256'] != signature:
            raise ValueError('Test requires dev decisions from this frozen design')
    queries = load_queries()
    qids = sorted(partition(queries, args.split), key=lambda q: (len(queries[q]['query']), q))
    if len(qids) != {'dev': 681, 'test': 1574}[args.split]:
        raise ValueError('Unexpected benchmark partition')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.hf import dtype_kwargs

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The frozen experiment requires CUDA with bf16 support')
    tokenizer = AutoTokenizer.from_pretrained(DESIGN['model'], padding_side='left', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        DESIGN['model'], **dtype_kwargs(torch.bfloat16), attn_implementation='sdpa', local_files_only=True,
    ).to('cuda').eval()
    for form in ('decomp', 'hyde'):
        settings = DESIGN[form]
        file = CACHE / f'{args.split}_{form}.json'
        saved = {'design_sha256': signature, 'rows': {}}
        if file.exists():
            saved = json.loads(file.read_text(encoding='utf-8'))
            if saved['design_sha256'] != signature:
                raise ValueError(f'Prompt or settings changed for {file}')
        rows = saved['rows']
        pending = [q for q in qids if str(q) not in rows]
        n = settings['num_return_sequences']
        batch_size = settings['batch']
        start = 0
        print(f'{args.split} {form} with {len(rows)}/{len(qids)} cached', flush=True)
        while start < len(pending):
            batch_qids = pending[start:start + batch_size]
            tick = time.perf_counter()
            try:
                texts, lengths = generate(model, tokenizer, [queries[q]['query'] for q in batch_qids],
                                          settings, DESIGN['seed'] + batch_qids[0])
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f'Reducing {form} CUDA batch size to {batch_size}', flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            elapsed = (time.perf_counter() - tick) / len(batch_qids)
            for i, q in enumerate(batch_qids):
                rows[str(q)] = {'texts': texts[i*n:(i+1)*n], 'tokens': sum(lengths[i*n:(i+1)*n]),
                                'seconds': elapsed}
            write_json(file, saved)
            start += len(batch_qids)
            print(f'{args.split} {form} {len(rows)}/{len(qids)} ({elapsed:.3f} s/question, '
                  f'{sum(lengths)/len(batch_qids):.1f} tokens/question in batch)', flush=True)


if __name__ == '__main__':
    main()
