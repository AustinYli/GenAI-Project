"""Run the frozen MultiHop-RAG comparison with larger NF4 Qwen checkpoints."""

import argparse
from copy import deepcopy
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import paths
from common.data import load_queries
from hybrid_claims.llm_generate import DESIGN, generate, partition, write_json


OUT = Path(paths.OUT) / 'llm_decomp_bigger'
MODELS = {
    'qwen2.5-7b-instruct': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'),
    'qwen3.5-9b': ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
}
QUANT = 'bitsandbytes 4-bit NF4 with double quantization, bf16 compute, CUDA only'
HYDE_LIMIT_SECONDS = 3 * 60 * 60
VRAM_MEASURE = 'Peak PyTorch CUDA reserved memory in GiB, includes model loading and generation'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def setup(slug):
    model_name, revision = MODELS[slug]
    out = OUT / slug
    design = deepcopy(DESIGN)
    design.update(model=model_name, revision=revision, dtype=QUANT, thinking=False if slug == 'qwen3.5-9b' else None)
    path = out / 'design.json'
    if path.exists() and read_json(path) != design:
        raise ValueError('The saved model design differs from the frozen protocol')
    if not path.exists():
        write_json(path, design)
    status_file = out / 'status.json'
    status = read_json(status_file) if status_file.exists() else {'model': model_name, 'quantization': QUANT}
    return out, status_file, status, sha256(path.read_bytes()).hexdigest()


def load_nf4(slug):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with bf16 support is required')
    name, revision = MODELS[slug]
    cached = {'revision': revision, 'local_files_only': True}
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model_args = dict(**cached, dtype=torch.bfloat16, device_map={'': 0}, quantization_config=quantization,
                      attn_implementation='sdpa')
    processor = None
    if slug == 'qwen3.5-9b':
        processor = AutoProcessor.from_pretrained(name, **cached)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'left'
        model = AutoModelForMultimodalLM.from_pretrained(name, **model_args).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left', **cached)
        model = AutoModelForCausalLM.from_pretrained(name, **model_args).eval()
    if not model.is_loaded_in_4bit or any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('The checkpoint did not load as GPU-resident 4-bit NF4')
    return model, tokenizer, processor


def load_generation(path, signature, partition_ids):
    saved = read_json(path) if path.exists() else {'design_sha256': signature, 'rows': {}}
    if saved['design_sha256'] != signature or not set(map(int, saved['rows'])).issubset(partition_ids):
        raise ValueError(f'Invalid generation cache {path}')
    return saved


def generate_phase(slug, form, out, status, status_file, signature):
    import torch

    queries = load_queries()
    settings = DESIGN[form]
    n = settings['num_return_sequences']
    jobs = {}
    for split, expected in (('dev', 681), ('test', 1574)):
        qids = sorted(partition(queries, split), key=lambda q: (len(queries[q]['query']), q))
        if len(qids) != expected:
            raise ValueError(f'Split {split} has {len(qids)} questions, expected {expected}')
        file = out / 'generation' / f'{split}_{form}.json'
        jobs[split] = (qids, file, load_generation(file, signature, qids))
    if all(len(saved['rows']) == len(qids) for qids, _, saved in jobs.values()):
        print(f'{slug} {form}: all generations already saved', flush=True)
        return 0

    info = status.setdefault(form, {})
    info.update(status='loading', vram_measurement=VRAM_MEASURE, hardware=torch.cuda.get_device_name())
    if allocator := os.environ.get('PYTORCH_ALLOC_CONF'):
        info['allocator'] = allocator
    info.pop('reason', None)
    write_json(status_file, status)
    torch.cuda.reset_peak_memory_stats()
    print(f'Loading {MODELS[slug][0]} with {QUANT}', flush=True)
    try:
        model, tokenizer, processor = load_nf4(slug)
    except Exception as exc:
        info.update(status='load_failed', reason=f'{type(exc).__name__}: {exc}')
        write_json(status_file, status)
        raise
    torch.cuda.empty_cache()
    info['status'] = 'generating'
    info['loaded_vram_gib'] = torch.cuda.memory_allocated() / (1024**3)
    info['cached_questions_at_resume'] = sum(len(saved['rows']) for _, _, saved in jobs.values())
    if memory_fraction := os.environ.get('BIGGER_CUDA_GENERATION_FRACTION'):
        fraction = float(memory_fraction)
        torch.cuda.set_per_process_memory_fraction(fraction)
        info['decode_memory_fraction'] = fraction
    write_json(status_file, status)
    print(f'Loaded {MODELS[slug][0]} on {torch.cuda.get_device_name()}', flush=True)
    started = time.monotonic()
    previous = sum(row['seconds'] for _, _, saved in jobs.values() for row in saved['rows'].values())
    batch_size = info.get('batch_size', settings['batch'])
    try:
        for split, (qids, file, saved) in jobs.items():
            rows = saved['rows']
            pending = [q for q in qids if str(q) not in rows]
            start = 0
            while start < len(pending):
                if form == 'hyde' and previous + time.monotonic() - started >= HYDE_LIMIT_SECONDS:
                    info.update(status='skipped', reason='Reached the three-hour HyDE generation limit')
                    return 3
                batch = pending[start:start + batch_size]
                tick = time.perf_counter()
                try:
                    texts, lengths = generate(model, tokenizer, [queries[q]['query'] for q in batch],
                                              settings, DESIGN['seed'] + batch[0], processor=processor)
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(f'CUDA memory reduced {form} batch size to {batch_size} questions', flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                elapsed = (time.perf_counter() - tick) / len(batch)
                for i, q in enumerate(batch):
                    rows[str(q)] = {'texts': texts[i*n:(i+1)*n], 'seconds': elapsed,
                                    'tokens': sum(lengths[i*n:(i+1)*n])}
                write_json(file, saved)
                start += len(batch)
                print(f'{slug} {split} {form} {len(rows)}/{len(qids)} ({elapsed:.3f} s/question, '
                      f'{sum(lengths)/len(batch):.1f} tokens/question)', flush=True)
        info['status'] = 'generated'
    except Exception as exc:
        info.update(status='generation_failed', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        info['batch_size'] = batch_size
        info['peak_vram_gib'] = max(info.get('peak_vram_gib', 0), torch.cuda.max_memory_reserved() / (1024**3))
        if processor is not None:
            info['continuations_checked_for_thinking'] = n * sum(len(saved['rows']) for _, _, saved in jobs.values())
        write_json(status_file, status)
    return 0


def score_phase(form, out, status, status_file, signature):
    from hybrid_claims.llm_compare import metrics, score_stage

    if status.get(form, {}).get('status') not in ('generated', 'complete'):
        raise ValueError(f'{form} must finish generation before it can be scored')
    with_hyde = form == 'hyde'
    decisions_path = out / 'dev_decisions.json'
    decisions = read_json(decisions_path) if decisions_path.exists() else {'design_sha256': signature, 'selected': {}}
    if decisions['design_sha256'] != signature:
        raise ValueError('Dev choices were recorded for a different experiment')
    for split in ('dev', 'test'):
        rows = score_stage(split, signature, cache_dir=out / 'generation', include_hyde=with_hyde)
        totals = {key: metrics(rows, key) for key in rows[0]['system']}
        if split == 'dev':
            variants = ('D', 'D_bm25') if with_hyde else ('C', 'C_hybrid')
            selected = max(variants, key=lambda key: totals[key]['complete@10'])
            if decisions['selected'].get(form, selected) != selected:
                raise ValueError(f'Dev selection changed for {form}')
            decisions['selected'][form] = selected
            decisions['dev'] = totals
            write_json(decisions_path, decisions)
            print(f'Locked {form} choice using dev complete@10: {selected}', flush=True)
        write_json(out / f'{split}_per_question.json', rows)
        print(json.dumps({split: totals}, indent=2), flush=True)
    status[form]['status'] = 'complete'
    write_json(status_file, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('command', choices=('generate', 'score'))
    parser.add_argument('phase', choices=('decomp', 'hyde'))
    args = parser.parse_args()
    out, status_file, status, signature = setup(args.model)
    if args.phase == 'hyde' and status.get('decomp', {}).get('status') != 'complete':
        raise ValueError('Complete decomposition before starting HyDE')
    if args.command == 'generate':
        return generate_phase(args.model, args.phase, out, status, status_file, signature)
    score_phase(args.phase, out, status, status_file, signature)
    return 0


if __name__ == '__main__':
    sys.exit(main())
