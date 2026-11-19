"""Compare three fixed ways to bundle cached 7B sub-questions into one list."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.bge import BGEEncoder, DenseIndex, _texts_hash, encode_cached
from common.data import gold_facts, load_chunks, load_queries
from common.metrics import facts_found
from hybrid_claims import run
from hybrid_claims.llm_compare import BUDGET_RULE, METRICS, ci, metrics, read_json
from hybrid_claims.llm_generate import DESIGN, partition, write_json


OUT = Path(run.paths.OUT) / 'llm_decomp_bigger' / 'bundled'
SOURCE = OUT.parent / 'qwen2.5-7b-instruct'
BASE = {'C_hybrid': 'Separate 7B searches', 'B': 'Rule split'}
VARIANTS = {'concat': 'Concat', 'mean': 'Mean', 'max': 'Max'}


def existing_sub_vectors(model, texts):
    path = SOURCE / 'generation' / 'test_sub_vectors.npy'
    metadata = read_json(path.with_suffix('.npy.json'))
    key = _texts_hash(texts, f'{model.model_name}|{model.dtype}|True|{model.query_instruction}')
    vectors = np.load(path, mmap_mode='r')
    if metadata['key'] != key or vectors.shape != (len(texts), model.model.config.hidden_size):
        raise ValueError('The existing sub-question vectors do not match the saved retrieval queries')
    return vectors


def fused_bundle_lists(qids, subquestions, bm, index, model):
    extras = {q: {name: [] for name in VARIANTS} for q in qids}
    active = [q for q in qids if subquestions[q]]
    if not active:
        return extras
    depth, rrf_k = DESIGN['extra_depth'], DESIGN['rrf_k']
    texts = list(dict.fromkeys(text for q in active for text in subquestions[q]))
    offsets = {text: i for i, text in enumerate(texts)}
    vectors = existing_sub_vectors(model, texts)
    ids = {q: [offsets[text] for text in subquestions[q]] for q in active}
    joined = ['\n'.join(subquestions[q]) for q in active]
    joined_vectors = encode_cached(model, joined, str(OUT / 'joined_vectors.npy'), is_query=True)
    concat_dense, _ = index.search(joined_vectors, depth)
    averaged = np.stack([vectors[ids[q]].mean(axis=0) for q in active])
    averaged /= np.linalg.norm(averaged, axis=1, keepdims=True)
    mean_dense, _ = index.search(averaged, depth)

    for q, text, concat_ids, mean_ids in zip(active, joined, concat_dense, mean_dense, strict=True):
        joined_bm25 = bm.search(text, depth)
        extras[q]['concat'] = run.rrf([concat_ids, joined_bm25], k=rrf_k)[:depth]
        extras[q]['mean'] = run.rrf([mean_ids, joined_bm25], k=rrf_k)[:depth]

    # Each reduction sees every corpus score. Candidate truncation happens after the maxima.
    for start in range(0, len(active), 32):
        questions = active[start:start + 32]
        groups = [ids[q] for q in questions]
        starts = np.cumsum([0, *(len(group) for group in groups[:-1])])
        scores = index.scores(vectors[[i for group in groups for i in group]])
        best_dense = np.maximum.reduceat(scores, starts, axis=0)
        for q, dense_scores in zip(questions, best_dense, strict=True):
            lex_scores = np.maximum.reduce([bm.scores(text) for text in subquestions[q]])
            top = [run.top_ids(dense_scores, depth), run.top_ids(lex_scores, depth)]
            extras[q]['max'] = run.rrf(top, k=rrf_k)[:depth]
    return extras


def verify_context(ids, k, chunks, qid):
    if len(ids) != k or len(set(ids)) != k or max(Counter(chunks[c]['article_id'] for c in ids).values()) > DESIGN['cap']:
        raise ValueError(f'Invalid {k}-chunk context for question {qid}')


def score():
    queries, chunks = load_queries(), load_chunks()
    qids = partition(queries, 'test')
    old_rows = read_json(SOURCE / 'test_per_question.json')
    decision = read_json(SOURCE / 'dev_decisions.json')
    if (len(qids) != 1574 or [row['qid'] for row in old_rows] != qids
            or decision['selected']['decomp'] != 'C_hybrid'):
        raise ValueError('The saved comparison does not match the fixed test IDs and variant')
    subquestions = {row['qid']: row['subquestions'] for row in old_rows}
    bm = run.BM25([run.chunk_doc(chunk) for chunk in chunks], DESIGN['bm25_k1'], DESIGN['bm25_b'])
    cache = Path(run.paths.CACHE)
    dense = read_json(cache / f'dense_ranking_{run.MODEL}_fp32.json')['ids']
    whole_depth, rrf_k, cap = DESIGN['whole_depth'], DESIGN['rrf_k'], DESIGN['cap']
    whole = {q: run.rrf([dense[q][:whole_depth], bm.search(queries[q]['query'], whole_depth)], k=rrf_k) for q in qids}
    model = BGEEncoder(device='cuda', dtype='fp32')
    index = DenseIndex(np.load(cache / f'emb_{run.MODEL}_fp32_chunks.npy'), device='cuda')
    extras = fused_bundle_lists(qids, subquestions, bm, index, model)
    del model
    print('Bundled rankings ready. Applying the common cap and backfill.', flush=True)

    max_k, backfill_depth = max(DESIGN['cuts']), BUDGET_RULE['backfill_depth']
    short = [q for q in qids if len(run.pick([whole[q]], max_k, chunks, cap)) < max_k]
    q_vectors = np.load(cache / f'emb_{run.MODEL}_fp32_queries.npy')[short]
    backfill_dense, _ = index.search(q_vectors, backfill_depth)
    backfill = {q: run.rrf([ids, bm.search(queries[q]['query'], backfill_depth)], k=rrf_k)
                for q, ids in zip(short, backfill_dense, strict=True)}

    rows = []
    for q, old in zip(qids, old_rows, strict=True):
        result = {name: old['system'][name] for name in BASE}
        facts = gold_facts(queries[q])
        for name in BASE:
            for k in DESIGN['cuts']:
                ids = result[name][f'ids@{k}']
                verify_context(ids, k, chunks, q)
                found = facts_found([chunks[c]['text'] for c in ids], facts, k)
                if int(all(found)) != result[name][f'complete@{k}'] or sum(found) / len(found) != result[name][f'recall@{k}']:
                    raise ValueError(f'Saved {name} coverage disagrees for question {q}')
        for name, ranking in extras[q].items():
            result[name] = {}
            for k in DESIGN['cuts']:
                ids = run.pick([whole[q], ranking], k, chunks, cap)
                filled = k - len(ids)
                if filled:
                    ids = run.pick([ids + backfill[q]], k, chunks, cap)
                verify_context(ids, k, chunks, q)
                found = facts_found([chunks[c]['text'] for c in ids], facts, k)
                result[name].update({f'ids@{k}': ids, f'backfilled@{k}': filled,
                                     f'complete@{k}': int(all(found)), f'recall@{k}': sum(found) / len(found)})
        rows.append({'qid': q, 'system': result})
    return rows


def report(rows):
    names = BASE | VARIANTS
    totals = {name: metrics(rows, name) for name in names}
    paired = {name: {ref: ci(rows, name, ref) for ref in BASE} for name in VARIANTS}
    backfill = {name: {str(k): sum(row['system'][name][f'backfilled@{k}'] > 0 for row in rows)
                       for k in DESIGN['cuts']} for name in names}
    write_json(OUT / 'results.json', {'questions': len(rows), 'metrics': totals, 'intervals': paired, 'backfill': backfill})
    lines = ['# Bundled 7B sub-question retrieval', '',
             'All rows use the same 1,574 seed-13 test questions and the original cached 7B sub-questions, including '
             'their rule-based fallback when needed. The separate-search row uses the previously dev-selected hybrid '
             'variant. All three bundled variants were fixed before scoring.', '',
             '- **Concat** joins the sub-questions with newlines and runs one BGE and one BM25 search.',
             '- **Mean** averages the normalized sub-question BGE vectors, normalizes the mean, and searches it. '
             'BM25 searches the joined text.',
             '- **Max** scores every chunk for each sub-question and takes the maximum separately for BGE and BM25, '
             'before ranking and fusion.', '',
             'Each bundle uses the top 60 dense and lexical results, fuses them with RRF k=60, and keeps the top 60 '
             'fused IDs. Selection alternates between the unchanged whole-question hybrid list and the bundle, '
             'starting with the whole question. All contexts contain exactly K unique chunks and at most two per '
             'article. When both lists run out, the same deeper whole-question fallback fills the remaining slots.', '',
             '## Test scores', '',
             '| System | Complete@10 | Complete@6 | Recall@10 | Questions backfilled at K=10 / K=6 |',
             '|---|---:|---:|---:|---:|']
    for name, label in names.items():
        values = ' | '.join(f'{totals[name][metric]:.2%}' for metric in METRICS)
        lines.append(f"| {label} | {values} | {backfill[name]['10']} / {backfill[name]['6']} |")
    lines += ['', '## Paired 95% confidence intervals', '',
              'Differences are in percentage points. Each of the 10,000 bootstrap draws samples the same questions '
              'for both systems with seed 13.', '',
              '| Bundle | Reference | Change in complete@10 | Change in complete@6 | Change in recall@10 |',
              '|---|---|---:|---:|---:|']
    for name, references in paired.items():
        for ref, intervals in references.items():
            values = ' | '.join(f'{100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]' for d, lo, hi in intervals.values())
            lines.append(f'| {names[name]} | {names[ref]} | {values} |')
    lines += ['', '[Per-question rankings](test_per_question.json) · [Numeric results](results.json)', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rows = score()
    write_json(OUT / 'test_per_question.json', rows)
    report(rows)
    print(OUT / 'REPORT.md')
