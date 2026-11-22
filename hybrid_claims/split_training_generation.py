"""Generate decomposition completions for the fixed few-shot and training runs."""

from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import paths
from common.data import load_queries
from hybrid_claims.llm_generate import DESIGN, write_json


OUT = Path(paths.OUT) / 'split_training'
DEMO_FILE = OUT / 'examples.json'
MODEL_NAMES = {'3b': 'Qwen/Qwen2.5-3B-Instruct', '7b': 'Qwen/Qwen2.5-7B-Instruct'}
DEMO_LINES = {
    1006: ["What did Fortune report about FTX's success being built on lies?",
           'What did Cnbc report about the FTX founder misleading an investor about governance or board plans?',
           'What did TechCrunch report about the FTX founder being charged with fraud and conspiracy?'],
    164: ['What did The Age report about worsening economic conditions, the middle class and Black Friday shopping?',
          'What did Engadget report about updating its list of Black Friday deals?'],
    654: ["What did Business Line report about Leqembi (lecanemab) being the only approved drug for Alzheimer's?",
          'What did The Guardian report about scientific developments in tackling dementia?'],
    1966: ["What did Sporting News report about a sportsbook's cash-out option to settle bets early?",
           'What did Sporting News report about a Vermont sign-up offer of a $1,000 Bonus Bet if the first bet loses?'],
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def user_turn(question):
    return {'role': 'user', 'content': f"{DESIGN['decomp']['prompt']}\n\nQuestion: {question}"}


def demo_messages():
    examples = read_json(DEMO_FILE)['examples']
    if list(map(int, examples)) != list(DEMO_LINES):
        raise ValueError('Unexpected few-shot examples or order')
    messages = []
    for demo in examples.values():
        messages.extend([user_turn(demo['question']), {'role': 'assistant', 'content': '\n'.join(demo['final'])}])
    return messages


def build_messages(prompt, question):
    if prompt not in ('zero', 'few'):
        raise ValueError(f'Unknown prompt {prompt}')
    return [*(demo_messages() if prompt == 'few' else []), user_turn(question)]


def load_model(model_key, adapter=None, base_precision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    repo = MODEL_NAMES[model_key]
    if model_key == '7b':
        from hybrid_claims.bigger_run import load_nf4

        model, tokenizer, _ = load_nf4('qwen2.5-7b-instruct')
    else:
        tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True, padding_side='left')
        args = dict(dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa')
        if base_precision == 'nf4':
            args.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True))
        model = AutoModelForCausalLM.from_pretrained(repo, **args)
        if base_precision != 'nf4':
            model = model.to('cuda')
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model.eval(), tokenizer


def shared_prompt_cache(model, tokenizer, formatted):
    """Prefill the token prefix shared by unpadded, single-question batches."""
    import torch

    all_ids = tokenizer(formatted, add_special_tokens=False)['input_ids']
    common = len(min(all_ids, key=len)) - 1
    first = all_ids[0]
    for ids in all_ids[1:]:
        common = next((i for i, (a, b) in enumerate(zip(first[:common], ids)) if a != b), common)
    if common <= 0:
        raise ValueError('The few-shot prompts have no cacheable common prefix')
    prefix = torch.tensor([first[:common]], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        return model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True,
                     logits_to_keep=1).past_key_values


def run_generation(model, prompt, qids, path, *, n=1, temperature=None, top_p=None, adapter=None, base_precision=None):
    """Cache raw completions per question and return them in generation order."""
    import torch

    path = Path(path)
    history = demo_messages() if prompt == 'few' else []
    qids = list(qids)
    queries = load_queries()
    order = sorted(qids, key=lambda q: (len(queries[q]['query']), q))
    settings = {'model': model, 'prompt': prompt, 'examples': history, 'instruction': DESIGN['decomp']['prompt'],
                'qids': order, 'n': n, 'temperature': temperature, 'top_p': top_p, 'max_new_tokens': 128,
                'adapter': str(Path(adapter).resolve()) if adapter is not None else None, 'base_precision': base_precision}
    use_shared_prefix = os.environ.get('SPLIT_TRAINING_CACHE_PREFIX') == '1' and prompt == 'few' and n == 16
    if use_shared_prefix:
        settings['prefix_cache'] = 'dynamic_single_question'
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    saved = read_json(path) if path.exists() else {'key': key, 'settings': settings, 'rows': {}}
    if saved['key'] != key or not set(saved['rows']).issubset(map(str, order)):
        raise ValueError(f'Cache {path} does not match this generation run')
    rows = saved['rows']
    pending = [q for q in order if str(q) not in rows]
    if not pending:
        return {int(q): row['texts'] for q, row in rows.items()}

    print(f'Loading {model} with {prompt} prompt for {n} completion(s) per question', flush=True)
    torch.cuda.reset_peak_memory_stats()
    llm, tokenizer = load_model(model, adapter, base_precision)
    torch.cuda.empty_cache()
    config = deepcopy(llm.generation_config)
    config.max_new_tokens = 128
    config.do_sample = temperature is not None
    config.temperature = temperature
    config.top_p = top_p
    config.top_k = 0 if config.do_sample else None
    config.repetition_penalty = 1.0
    config.pad_token_id = tokenizer.pad_token_id
    max_sequences = 16 if prompt == 'few' else 40
    nf4_budget = None
    if prompt == 'few' and base_precision == 'nf4' and n > 1:
        nf4_budget = os.environ.get('SPLIT_TRAINING_NF4_FEW_SEQUENCES')
    if nf4_budget is not None:
        max_sequences = int(nf4_budget)
        if max_sequences < 1:
            raise ValueError('SPLIT_TRAINING_NF4_FEW_SEQUENCES must be positive')
    resumed_budget = saved.get('nf4_sequence_budget')
    batch = max(1, max_sequences // n)
    if resumed_budget == nf4_budget:
        batch = min(saved.get('batch', batch), batch)
    parallel_samples = saved.get('parallel_samples', n)
    batch = min(batch, max(1, int(os.environ.get('SPLIT_TRAINING_MAX_BATCH', batch))))
    parallel_samples = min(parallel_samples, max(1, int(os.environ.get('SPLIT_TRAINING_MAX_SAMPLES', parallel_samples))))
    reserved_limit = float(os.environ.get('SPLIT_TRAINING_CUDA_CACHE_LIMIT_GIB', '0')) * 1024 ** 3
    prefix_cache = None
    if use_shared_prefix:
        if batch != 1:
            raise ValueError('Shared-prefix sampling requires one question per batch')
        transcripts = [tokenizer.apply_chat_template([*history, user_turn(queries[q]['query'])],
                                                     add_generation_prompt=True, tokenize=False) for q in order]
        prefix_cache = shared_prompt_cache(llm, tokenizer, transcripts)
        config.cache_implementation = None
    position = 0
    while position < len(pending):
        batch_qids = pending[position:position + batch]
        begin = time.perf_counter()
        formatted = [tokenizer.apply_chat_template([*history, user_turn(queries[q]['query'])],
                                                  add_generation_prompt=True, tokenize=False) for q in batch_qids]
        had_oom = False
        try:
            inputs = tokenizer(formatted, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
            torch.manual_seed(DESIGN['seed'] + batch_qids[0] + (n * 100_003 if config.do_sample else 0))
            by_question = [[] for _ in batch_qids]
            with torch.inference_mode():
                for start in range(0, n, parallel_samples):
                    count = min(parallel_samples, n - start)
                    config.num_return_sequences = count
                    cached = {}
                    if prefix_cache is not None:
                        copied = deepcopy(prefix_cache)
                        copied.batch_repeat_interleave(count)
                        cached['past_key_values'] = copied
                    tokens = llm.generate(**inputs, **cached,
                                          generation_config=config)[:, inputs.input_ids.shape[1]:].cpu()
                    cached = None
                    copied = None
                    decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                    for i, group in enumerate(by_question):
                        group.extend(decoded[i*count:(i+1)*count])
        except torch.cuda.OutOfMemoryError:
            if batch == 1 and parallel_samples == 1:
                raise
            had_oom = True
        finally:
            inputs = cached = copied = None
        if had_oom:
            if batch > 1:
                batch //= 2
            else:
                parallel_samples //= 2
            gc.collect()
            torch.cuda.empty_cache()
            print(f'Retrying with {batch} questions and {parallel_samples} samples per call after a CUDA memory error',
                  flush=True)
            continue
        elapsed = (time.perf_counter() - begin) / len(batch_qids)
        for q, group in zip(batch_qids, by_question, strict=True):
            rows[str(q)] = {'texts': group, 'seconds': elapsed}
        saved.update(batch=batch, parallel_samples=parallel_samples, prefix_cached=prefix_cache is not None,
                     peak_vram_gib=max(saved.get('peak_vram_gib', 0), torch.cuda.max_memory_reserved() / (1024 ** 3)))
        if nf4_budget is not None:
            saved['nf4_sequence_budget'] = nf4_budget
        write_json(path, saved)
        position += len(batch_qids)
        print(f'{path.stem} {len(rows)}/{len(order)} ({elapsed:.3f} s/question)', flush=True)
        if reserved_limit and torch.cuda.memory_reserved() > reserved_limit:
            torch.cuda.empty_cache()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return {int(q): row['texts'] for q, row in rows.items()}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--prompt', choices=('zero', 'few'), required=True)
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--top-p', type=float)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--base-precision', choices=('bf16', 'nf4'))
    args = parser.parse_args()
    qids = read_json(args.split_file)[args.partition]
    run_generation(args.model, args.prompt, qids, args.output, n=args.samples, temperature=args.temperature,
                   top_p=args.top_p, adapter=args.adapter, base_precision=args.base_precision)


if __name__ == '__main__':
    main()
