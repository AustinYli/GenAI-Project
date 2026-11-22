"""Evaluate the frozen cross-encoder over the same lists with three ranking policies."""

import argparse
from collections import Counter
from contextlib import closing
import gc
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time

from common import paths
from common.data import load_chunks
from common.rerank import CrossEncoderReranker
from hybrid_claims.llm_generate import write_json
from hybrid_claims.reranker_step1_candidates import prepare_candidates
from hybrid_claims.run import pick


OUT = Path(paths.OUT) / 'reranker'
MODEL = 'BAAI/bge-reranker-large'
REVISION = '55611d7bca2a7133960a6d3b71e083071bbfc312'
MAX_LENGTH = 512
CAP = 2
CUT = 10


def candidate_pairs(row):
    texts, ranked = row['texts'], row['lists']
    union = list(dict.fromkeys(itertools.chain.from_iterable(ranked)))
    b_pairs = [(texts[0], chunk_id) for chunk_id in union]
    c_pairs = list(dict.fromkeys((text, chunk_id) for text, ids in zip(texts, ranked, strict=True)
                                 for chunk_id in ids))
    return union, b_pairs, c_pairs


def lengths(tokenizer, keys, get_text):
    out = {}
    keys = list(dict.fromkeys(keys))
    for start in range(0, len(keys), 1024):
        batch = keys[start:start + 1024]
        encoded = tokenizer([get_text(key) for key in batch], add_special_tokens=False, verbose=False)
        out.update((key, len(ids)) for key, ids in zip(batch, encoded['input_ids'], strict=True))
    return out


def compute_scores(db, pairs, chunks, model_path, batch_size):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('The FP16 reranker requires CUDA for this run')
    cached = {(q, c): (score, sec) for q, c, score, sec in db.execute('SELECT * FROM logits')}
    if not cached.keys() <= pairs.keys():
        raise ValueError('The logit cache contains pairs outside the frozen candidate pool')
    pending = [pair for pair in pairs if pair not in cached]
    print(f'{len(pairs):,} distinct pairs requested. {len(cached):,} already cached.', flush=True)
    run_time = 0.0
    if not pending:
        return cached, {'new_pairs': 0, 'new_scoring_wall_seconds': run_time}

    if not (model_path / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Missing frozen model snapshot {model_path}')
    reranker = CrossEncoderReranker(str(model_path), device='cuda', fp16=True, max_length=MAX_LENGTH)
    if next(reranker.model.parameters()).dtype != torch.float16:
        raise ValueError('Reranker parameters are not FP16')
    q_len = lengths(reranker.tokenizer, (q for q, _ in pending), lambda q: q)
    c_len = lengths(reranker.tokenizer, (c for _, c in pending), lambda c: chunks[c]['text'])
    special = reranker.tokenizer.num_special_tokens_to_add(pair=True)
    pending.sort(key=lambda pair: min(MAX_LENGTH, q_len[pair[0]] + c_len[pair[1]] + special))
    torch.cuda.reset_peak_memory_stats()
    started = last_report = time.perf_counter()
    position = 0
    while position < len(pending):
        block = pending[position:position + batch_size]
        start = time.perf_counter()
        inputs = [(query, chunks[chunk_id]['text']) for query, chunk_id in block]
        try:
            values = reranker.score_pairs(inputs, batch_size=len(block))
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            values = None
        if values is None:
            batch_size //= 2
            gc.collect()
            torch.cuda.empty_cache()
            print(f'Retrying with {batch_size} pairs per batch after CUDA memory exhaustion.', flush=True)
            continue
        if any(not math.isfinite(value) for value in values):
            raise ValueError('The cross-encoder returned a non-finite score')
        if torch.cuda.memory_reserved() > 7 * 1024**3:
            torch.cuda.empty_cache()
        seconds = (time.perf_counter() - start) / len(block)
        records = [(q, c, logit, seconds) for (q, c), logit in zip(block, values, strict=True)]
        with db:
            db.executemany('INSERT INTO logits VALUES (?, ?, ?, ?)', records)
        cached.update(((q, c), (logit, sec)) for q, c, logit, sec in records)
        position += len(block)
        now = time.perf_counter()
        if position == len(block) or now - last_report >= 30 or position == len(pending):
            speed = position / (now - started)
            print(f'{len(cached):,}/{len(pairs):,} scored at {speed:.1f} pairs/s. '
                  f'Estimated {(len(pending) - position) / speed / 60:.1f} minutes left.', flush=True)
            last_report = now
    run_time = time.perf_counter() - started
    return cached, {'new_pairs': position, 'new_scoring_wall_seconds': run_time,
                    'final_batch_size': batch_size, 'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
                    'attention': reranker.model.config._attn_implementation}


def select(row, chunks, scores):
    def record(lists, pairs, start):
        ids = pick(lists, CUT, chunks, CAP)
        selection_seconds = time.perf_counter() - start
        reranker_seconds = sum(scores[pair][1] for pair in pairs)
        return {'ids@10': ids, 'ids@6': ids[:6], 'pairs': len(pairs),
                'selection_seconds': selection_seconds, 'reranker_seconds': reranker_seconds,
                'seconds': row['candidate_seconds'] + selection_seconds + reranker_seconds}

    union, b_pairs, c_pairs = candidate_pairs(row)
    whole, lists = row['texts'][0], row['lists']
    start = time.perf_counter()
    a = record(lists, [], start)
    start = time.perf_counter()
    reranked_union = sorted(union, key=lambda chunk_id: -scores[(whole, chunk_id)][0])
    b = record([reranked_union], b_pairs, start)
    start = time.perf_counter()
    reranked_lists = [sorted(ids, key=lambda chunk_id: -scores[(text, chunk_id)][0])
                      for text, ids in zip(row['texts'], lists, strict=True)]
    c = record(reranked_lists, c_pairs, start)
    per_article = Counter(chunks[chunk_id]['article_id'] for chunk_id in union)
    capacity = sum(min(CAP, count) for count in per_article.values())
    if any(len(system['ids@10']) != min(CUT, capacity) for system in (a, b, c)):
        raise ValueError(f'Question {row["qid"]} did not fill the feasible context budget')
    return {'qid': row['qid'], 'question_type': row['question_type'], 'candidate_seconds': row['candidate_seconds'],
            'union_chunks': len(union), 'capacity': capacity, 'claims': len(lists) - 1,
            'systems': {'A': a, 'B': b, 'C': c}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=192)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    data = prepare_candidates()
    rows = data['rows']
    chunks = load_chunks()
    design = {
        **data['design'], 'reranker': MODEL, 'reranker_revision': REVISION, 'dtype': 'float16',
        'max_length': MAX_LENGTH, 'input': 'Raw question or claim and original chunk text including its metadata header.',
        'tie_rule': 'Stable sort preserves the source list order, or first occurrence in the union for B.',
        'optional_D': 'Skipped to avoid scoring every candidate against every query in the list set.',
        'timing': 'Seconds include shared candidate retrieval, selection, and amortized reranker time. '
                  'Mixed batches were sorted by approximate pair token length. GPU-forward and tokenizer '
                  'wall time is divided among batch pairs and credited to each system requiring that pair. '
                  'Model loading, length sorting, indexed retrieval setup, cache I/O and metrics are excluded.',
        'shared_candidate_backfill': False,
    }
    signature = sha256((OUT / 'step1_candidates.json').read_bytes() + json.dumps(design, sort_keys=True).encode())
    cache_path = OUT / f'step1_logits_{signature.hexdigest()[:16]}.sqlite3'
    model_path = Path(paths.CACHE) / 'hf' / 'hub' / 'models--BAAI--bge-reranker-large' / 'snapshots' / REVISION
    pairs = {}
    for row in rows:
        _, b_pairs, c_pairs = candidate_pairs(row)
        pairs.update(dict.fromkeys(b_pairs + c_pairs))
    with closing(sqlite3.connect(cache_path)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS logits (query TEXT NOT NULL, chunk_id INTEGER NOT NULL, '
                   'logit REAL NOT NULL, seconds REAL NOT NULL, PRIMARY KEY (query, chunk_id))')
        scores, runtime = compute_scores(db, pairs, chunks, model_path, args.batch_size)
    write_json(OUT / 'step1_rankings.json', {'design': design, 'rows': [select(row, chunks, scores) for row in rows],
                                           'runtime': {**runtime, 'unique_pairs': len(pairs),
                                                       'logit_cache': cache_path.name,
                                                       'bm25_index_seconds': data['bm25_index_seconds']}})


if __name__ == '__main__':
    main()
