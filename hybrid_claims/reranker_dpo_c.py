"""Apply System C to out-of-fold untrained and DPO LLM sub-queries."""

import argparse
from collections import Counter
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from common import paths
from common.data import load_chunks, load_json
from hybrid_claims.llm_generate import write_json
from hybrid_claims.reranker_dpo_c_candidates import prepare_candidates
from hybrid_claims.reranker_step1 import CAP, CUT, MAX_LENGTH, MODEL, OUT, REVISION, compute_scores
from hybrid_claims.run import pick


VARIANTS = ('zero_base', 'zero_dpo', 'few_base', 'few_dpo')
STEP1 = OUT / 'step1_rankings.json'
CANDIDATES = OUT / 'dpo_c_candidates.json'


def list_pairs(variant):
    return dict.fromkeys((text, chunk_id)
                         for text, ids in zip(variant['texts'], variant['lists'], strict=True)
                         for chunk_id in ids)


def step1_database():
    step1 = load_json(STEP1)
    original = step1['design']
    expected = {'reranker': MODEL, 'reranker_revision': REVISION, 'dtype': 'float16', 'max_length': MAX_LENGTH}
    if any(original[key] != value for key, value in expected.items()):
        raise ValueError('Step 1 used different cross-encoder settings')
    name = step1['runtime']['logit_cache']
    if Path(name).name != name or not name.startswith('step1_logits_'):
        raise ValueError('Unexpected Step 1 logit database path')
    return OUT / name


def seed_step1_scores(db, original_path, pairs):
    """Copy only overlapping raw query–chunk logits from the completed Step 1 run."""
    with closing(sqlite3.connect(original_path.resolve().as_uri() + '?mode=ro', uri=True)) as original:
        reused = [(text, chunk_id, logit, seconds) for text, chunk_id, logit, seconds in
                  original.execute('SELECT query, chunk_id, logit, seconds FROM logits')
                  if (text, chunk_id) in pairs]
    with db:
        db.executemany('INSERT OR IGNORE INTO logits VALUES (?, ?, ?, ?)', reused)
    return len(reused)


def choose_chunks(variant, chunks, scores):
    pairs = list_pairs(variant)
    start = perf_counter()
    lists = [sorted(ids, key=lambda chunk_id: -scores[(text, chunk_id)][0])
             for text, ids in zip(variant['texts'], variant['lists'], strict=True)]
    chosen = pick(lists, CUT, chunks, CAP)
    selection_seconds = perf_counter() - start
    by_article = Counter(chunks[chunk_id]['article_id'] for chunk_id in {chunk_id for _, chunk_id in pairs})
    capacity = sum(min(CAP, count) for count in by_article.values())
    if len(chosen) != min(CUT, capacity):
        raise ValueError('Reranking did not fill the achievable context budget')
    reranker_seconds = sum(scores[pair][1] for pair in pairs)
    candidate_seconds = variant['candidate_seconds']
    split_seconds = variant['split_seconds']
    seconds = candidate_seconds + reranker_seconds + selection_seconds
    return {'ids@10': chosen, 'ids@6': chosen[:6], 'pairs': len(pairs), 'capacity': capacity,
            'seconds': seconds, 'candidate_seconds': candidate_seconds, 'reranker_seconds': reranker_seconds,
            'selection_seconds': selection_seconds, 'split_seconds': split_seconds,
            'seconds_with_split': seconds + split_seconds, 'fallback': variant['fallback']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=192)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    data = prepare_candidates()
    design = {**data['design'], 'reranker': MODEL, 'reranker_revision': REVISION, 'dtype': 'float16',
              'max_length': MAX_LENGTH, 'article_cap': CAP, 'cuts': [10, 6], 'backfill': False,
              'tie_rule': 'Stable sort preserves original rank among equal raw logits.',
              'score_cache_rule': 'Reuse Step 1 FP16 logits for identical raw text and chunk ID.',
              'timing': 'Candidate preparation and selection wall time plus amortized batch time for requested pairs. '
                        'The combined timing also adds per-question generation time from archived CV runs.'}
    signature = sha256(CANDIDATES.read_bytes() + json.dumps(design, sort_keys=True).encode()).hexdigest()
    local_path = OUT / f'dpo_c_logits_{signature[:16]}.sqlite3'
    original_path = step1_database()
    model_path = Path(paths.CACHE) / 'hf' / 'hub' / 'models--BAAI--bge-reranker-large' / 'snapshots' / REVISION
    pairs = {}
    for row in data['rows']:
        for name in VARIANTS:
            pairs.update(list_pairs(row['systems'][name]))
    chunks = load_chunks()
    with closing(sqlite3.connect(local_path)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS logits (query TEXT NOT NULL, chunk_id INTEGER NOT NULL, '
                   'logit REAL NOT NULL, seconds REAL NOT NULL, PRIMARY KEY (query, chunk_id))')
        reused = seed_step1_scores(db, original_path, pairs)
        cached = db.execute('SELECT COUNT(*) FROM logits').fetchone()[0]
        print(f'Reusing {reused:,} Step 1 logits. {cached - reused:,} additional logits cached for this run.', flush=True)
        scores, runtime = compute_scores(db, pairs, chunks, model_path, args.batch_size)
    rows = [{'qid': row['qid'], 'question_type': row['question_type'],
             'systems': {name: choose_chunks(row['systems'][name], chunks, scores) for name in VARIANTS}}
            for row in data['rows']]
    output = OUT / 'dpo_c_rankings.json'
    write_json(output, {'design': design, 'rows': rows,
                        'runtime': {**runtime, 'unique_pairs': len(pairs), 'reused_step1_pairs': reused,
                                    'reused_current_run_pairs': cached - reused, 'logit_cache': local_path.name,
                                    'step1_logit_cache': original_path.name}})
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
