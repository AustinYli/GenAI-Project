"""Build outputs/ALL_RESULTS.md: published MultiHop-RAG and Ammann et al. (2025) results next to all of ours.

Our rows are recomputed from saved rankings: Hits@4/Hits@10/MAP@10/MRR@10 with the upstream MultiHop-RAG
evaluator (commit 3dd4d4e), plus complete@10, complete@6 (first 6 chunks of the K=10 list) and recall@10.
Archived values are used only where no rankings were saved. A blank cell means not measured / not reported.
Re-run after new experiments:  python outputs/all_results/build_all_results.py
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
TRM = OUT / 'test_retrieval_metrics'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRM))

import collect_and_report as cr  # noqa: E402
import collect_early as ce  # noqa: E402
from common.data import gold_facts, load_chunks, load_queries, split_queries  # noqa: E402
from common.metrics import calculate_metrics, facts_found  # noqa: E402

EXTRA = ('complete@10', 'complete@6', 'recall@10')
COLS = ('complete@10', 'complete@6', 'recall@10', 'Hits@4', 'Hits@10', 'MAP@10', 'MRR@10')


def coverage(texts, gold):
    c10 = c6 = r10 = 0.0
    for t, g in zip(texts, gold):
        f10, f6 = facts_found(t, g, 10), facts_found(t, g, 6)
        c10 += all(f10)
        c6 += all(f6)
        r10 += sum(f10) / len(f10)
    n = len(gold)
    return {'complete@10': c10 / n, 'complete@6': c6 / n, 'recall@10': r10 / n}


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def mean_seconds(paths):
    vals = []
    for p in paths:
        if (ROOT / p).exists():
            vals += [r['seconds'] for r in read(p)['rows'].values() if 'seconds' in r]
    return statistics.mean(vals) if vals else None


# ---------------- our test-split rows (recomputed from rankings) ----------------
queries, chunks = load_queries(), load_chunks()
test = [q for q in split_queries(queries, seed=13)[1] if queries[q]['question_type'] != 'null_query']
gold = [gold_facts(queries[q]) for q in test]


def score(rankings):
    texts = [[chunks[c]['text'] for c in ids] for ids in rankings]
    m = calculate_metrics(texts, gold)
    m.update(coverage(texts, gold))
    return m


_orig_result = ce.result


def _result(group, method, source, source_key, n, scores, rankings=None, note=None):
    row = _orig_result(group, method, source, source_key, n, scores, rankings, note)
    row.update({k: scores[k] for k in EXTRA if k in scores})
    return row


ce.result = _result
dense = ce.read_json(ce.DENSE_RANKING)['ids']
early = list(ce.replication(queries, chunks, test, score, dense))
early += list(ce.idea1_capped(queries, test, score))
early += list(ce.idea1_uncapped(queries, chunks, test, score, dense))
early += list(ce.idea2(len(test), early))


class _Eval(cr.RankingEvaluator):
    def row(self, group, method, path, key, by_qid, **kw):
        r = super().row(group, method, path, key, by_qid, **kw)
        r.update(coverage([[self.chunks[i] for i in by_qid[q]] for q in self.qids], self.gold))
        return r


_orig_read, _orig_archived = cr.read, cr.archived
cr.RankingEvaluator = _Eval
cr.read = lambda p: early if str(p).endswith('early.json') else _orig_read(p)


def _archived(group, method, path, key, value):
    row = _orig_archived(group, method, path, key, value)
    row.update({k: value[k] for k in EXTRA if k in value})
    return row


cr.archived = _archived
_, test_rows = cr.collect()
test_rows = [r for r in test_rows if 'duplicate_of' not in r]

# reranker Step 1 (rule split A/B/C) on the test split and on all 2,251 CV questions
step1 = read('outputs/reranker/step1_rankings.json')['rows']
pos = {q: i for i, q in enumerate(test)}
step1_names = {'A': 'Rule split, round-robin, no reranker (Step 1 A)',
               'B': 'Rule split + reranker vs whole question (Step 1 B, Ammann-style)',
               'C': 'Rule split + reranker per sub-query, round-robin (Step 1 C)'}
pooled_step1 = []
for cohort, rows in (('test', [r for r in step1 if r['qid'] in pos]), ('pooled', step1)):
    g = [gold_facts(queries[r['qid']]) for r in rows]
    for key, name in step1_names.items():
        texts = [[chunks[c]['text'] for c in r['systems'][key]['ids@10']] for r in rows]
        m = calculate_metrics(texts, g)
        m.update(coverage(texts, g))
        sec = statistics.mean(r['systems'][key]['seconds'] for r in rows)
        row = {'group': 'Reranker (Step 1)', 'method': name, 'n': len(rows), **m,
               'latency': sec, 'latency_note': 'measured; cached dense ranking (BGE query encoding ~0.007 s not included)'}
        (test_rows if cohort == 'test' else pooled_step1).append(row)

# DPO splitter + per-sub-query reranker (System C), test split and all 2,251 CV questions
dpo_c_path = ROOT / 'outputs/reranker/dpo_c_rankings.json'
dpo_c_names = {'zero_base': 'Qwen2.5-3B zero-shot, untrained + reranker per sub-query (C)',
               'zero_dpo': 'Qwen2.5-3B zero-shot, DPO + reranker per sub-query (C)',
               'few_base': 'Qwen2.5-3B few-shot, untrained + reranker per sub-query (C)',
               'few_dpo': 'Qwen2.5-3B few-shot, DPO + reranker per sub-query (C)'}
if dpo_c_path.exists():
    dpo_rows = read('outputs/reranker/dpo_c_rankings.json')['rows']
    for cohort, rows in (('test', [r for r in dpo_rows if r['qid'] in pos]), ('pooled', dpo_rows)):
        g = [gold_facts(queries[r['qid']]) for r in rows]
        for key, name in dpo_c_names.items():
            texts = [[chunks[c]['text'] for c in r['systems'][key]['ids@10']] for r in rows]
            m = calculate_metrics(texts, g)
            m.update(coverage(texts, g))
            sec = statistics.mean(r['systems'][key]['seconds_with_split'] for r in rows)
            row = {'group': 'DPO splitter + reranker (System C)', 'method': name, 'n': len(rows), **m,
                   'latency': sec, 'latency_note': 'measured: split + candidates + reranking + selection '
                                                   '(sub-query lists BM25-only, top 60)'}
            (test_rows if cohort == 'test' else pooled_step1).append(row)

# ---------------- latency for test rows (only where measured) ----------------
llm = read('outputs/llm_decomp_bigger/results.json')['generation']
idea3_lat = read('outputs/idea3/eval_original.json')['latency']
cap_seconds = {}
for stem, label in (('word_K10_cap2_test', 'keyword list'), ('word_K10_cap2_trimmed_test', 'trimmed sentence'),
                    ('word_K10_cap2_rewrite_Qwen__Qwen2.5-1.5B-Instruct_test', 'Qwen 1.5B rewrite')):
    cap_seconds[label] = statistics.mean(e['extra_search_seconds'] for e in read(f'outputs/idea1_cap/{stem}_per_question.json'))
scoring = {'Word overlap, tuned': 'word_K10', 'Qwen in-place, tuned': 'llm_inplace__Qwen__Qwen2.5-3B-Instruct_K10',
           'Qwen probe, tuned': 'llm_probe__Qwen__Qwen2.5-3B-Instruct_K10',
           'ModernBERT masked LM, tuned': 'mlm__answerdotai__ModernBERT-large_K10'}
step2_gen = {arm: (mean_seconds([f'outputs/split_training/step2/{arm}/test_base.json']),
                   mean_seconds([f'outputs/split_training/step2/{arm}/test_trained.json']))
             for arm in ('3b_zero', '3b_few', '7b_zero')}
few7b = mean_seconds(['outputs/split_training/step1/test_few_generations.json'])

for r in test_rows:
    m = r['method']
    if 'latency' in r:
        continue
    lat, step, note = None, None, ''
    if m in ('BGE dense', 'BGE, no cap'):
        d = idea3_lat['dense BGE']
        lat, note = (d['encode_ms_per_query'] + d['search_ms_per_query']) / 1000, 'measured: BGE query encoding + exact search'
    elif m.startswith('Four-vector pooled'):
        d = idea3_lat['pooled control']
        lat, note = (d['encode_ms_per_query'] + d['search_ms_per_query']) / 1000, 'measured: encoding + search'
    elif m.startswith('Four-vector token'):
        d = idea3_lat['token heads']
        lat, note = (d['encode_ms_per_query'] + d['search_ms_per_query']) / 1000, 'measured: encoding + search'
    elif m.startswith('Word overlap + cap 2'):
        label = m.split(', ', 1)[1]
        step = cap_seconds[label]
        note = 'timed: rewrite (Qwen 1.5B) + second searches' if 'rewrite' in label else 'timed: second searches only'
    elif m in scoring:
        step = read(f'outputs/idea1/{scoring[m]}.json').get('scoring_sec_mean')
        note = 'timed: coverage scoring only' + (' (CPU, no model)' if m.startswith('Word') else '')
    elif m.startswith('Qwen2.5-3B') or m.startswith('Qwen2.5-7B'):
        size = '3B' if m.startswith('Qwen2.5-3B') else '7B'
        dec, hyde = llm[size]['test']['decomp']['seconds'], llm[size]['test']['hyde']['seconds']
        if 'DPO' in m or 'untrained' in m:
            arm = {'3B zero': '3b_zero', '3B few': '3b_few', '7B zero': '7b_zero'}[
                size + (' few' if 'few-shot' in m else ' zero')]
            base, trained = step2_gen[arm]
            step = trained if 'DPO' in m else base
            if step is None and 'zero-shot' in m:
                step = dec  # same zero-shot prompt and model as the decomposition run
        elif 'few-shot' in m:
            step = few7b
        elif 'decomposition' in m and 'HyDE' in m:
            step = dec + hyde
        elif 'HyDE' in m:
            step = hyde
        elif 'decomposition' in m:
            step = dec
        note = 'timed: LLM sub-query / passage generation only' if step else ''
    elif m in ('Concat sub-questions + hybrid', 'Mean dense scores via mean query vector + joined BM25',
               'Max dense and BM25 scores across sub-questions + RRF'):
        step, note = llm['7B']['test']['decomp']['seconds'], 'timed: 7B sub-query generation only'
    elif m in ('Hybrid + rule-split BM25 claims + cap 2', 'BGE hybrid + rule splits + cap 2, exact K'):
        note = 'BM25 + claim split: 9.1 ms/q on CPU (dense part not timed)'
    r['latency'], r['llm_step'], r['latency_note'] = lat, step, note

# ---------------- pooled 2,251 CV rows (archived metrics; no rankings saved) ----------------
cv = read('outputs/split_training/cv/pooled_cv_metrics.json')
cv_rows = []
names = {'whole': 'whole-question hybrid (no split)', 'rule': 'rule split (no reranker)',
         'untrained': '{arm}, untrained', 'trained': '{arm}, DPO'}
for enc, label in (('bge', 'BGE'), ('e5', 'e5')):
    seen = set()
    for arm, arm_label in (('3b_zero', 'Qwen2.5-3B zero-shot'), ('3b_few', 'Qwen2.5-3B few-shot')):
        block = cv[enc]['pooled'][arm]['all']
        for key, sc in block['scores'].items():
            name = f"{label}: " + names[key].format(arm=arm_label)
            if name in seen:
                continue
            seen.add(name)
            step = None
            if key in ('untrained', 'trained'):
                f = 'test_trained.json' if key == 'trained' else 'test_base.json'
                step = mean_seconds([f'outputs/split_training/cv/fold_{k}/{arm}/{f}' for k in (1, 2)])
            cv_rows.append({'group': 'DPO 2-fold CV', 'method': name, 'n': block['n'], **sc, 'llm_step': step,
                            'latency_note': 'timed: sub-query generation only' if step else ''})
cv_rows += pooled_step1

# ---------------- published rows ----------------
T5 = {  # Tang & Yang (2024), Table 5: MRR@10, MAP@10, Hits@10, Hits@4 (no reranker | bge-reranker-large)
    'text-embedding-ada-002': ((0.4203, 0.3431, 0.6381, 0.504), (0.5477, 0.4625, 0.7059, 0.6169)),
    'text-search-ada-query-001': ((0.4203, 0.3431, 0.6399, 0.5031), (0.5483, 0.4625, 0.7064, 0.6174)),
    'llm-embedder': ((0.2558, 0.1725, 0.4499, 0.3189), (0.425, 0.3059, 0.5478, 0.4756)),
    'bge-large-en-v1.5': ((0.4298, 0.3423, 0.6718, 0.5221), (0.563, 0.4759, 0.7183, 0.6364)),
    'jina-embeddings-v2-base-en': ((0.0621, 0.031, 0.1479, 0.0802), (0.1412, 0.0772, 0.1909, 0.1639)),
    'e5-base-v2': ((0.1843, 0.1161, 0.3556, 0.2334), (0.3237, 0.2165, 0.4176, 0.3716)),
    'voyage-02': ((0.3934, 0.3143, 0.6506, 0.4619), (0.586, 0.4795, 0.7467, 0.6625)),
    'instructor-large': ((0.3458, 0.265, 0.5717, 0.4229), (0.5115, 0.4118, 0.659, 0.5775)),
}
published = []
for emb, (plain, rr) in T5.items():
    for vals, suffix in ((plain, ''), (rr, ' + bge-reranker-large')):
        published.append({'source': 'MultiHop-RAG (Tang & Yang, 2024), Table 5', 'method': emb + suffix,
                          'n': '2,255 (all non-null)', 'MRR@10': vals[0], 'MAP@10': vals[1],
                          'Hits@10': vals[2], 'Hits@4': vals[3], 'latency_note': 'not reported'})
AM = [  # Ammann et al. (2025), Table 1 (Hits@4, Hits@10, MAP@10, MRR@10) and Table 4 latency (s/query)
    ('text-ada-002 + RR (copied from Tang & Yang)', (0.616, 0.706, 0.463, 0.548), None),
    ('voyage-02 + RR (copied from Tang & Yang)', (0.663, 0.747, 0.480, 0.586), None),
    ('Naive RAG (bge-large-en-v1.5)', (0.611, 0.781, 0.217, 0.464), 0.03),
    ('+ QD', (0.655, 0.810, 0.238, 0.498), 16.7),
    ('+ RR (bge-reranker-large)', (0.687, 0.781, 0.274, 0.574), 0.88),
    ('+ QD + RR (their method)', (0.763, 0.872, 0.322, 0.635), 18.9),
]
for name, (h4, h10, mp, mrr), lat in AM:
    published.append({'source': 'Ammann et al. (2025), Tables 1 and 4', 'method': name,
                      'n': '"eval split" (size not stated)', 'Hits@4': h4, 'Hits@10': h10, 'MAP@10': mp,
                      'MRR@10': mrr, 'latency': lat,
                      'latency_note': ('measured on 250 queries, retrieval only (no answer generation), their GPU'
                                       if lat is not None else 'not reported')})

pending = [
    ('Ammann et al. reimplemented in our setup (QD + reranking vs original query)', 'not run yet'),
    ('Reranker preference fine-tuning, Bradley-Terry (Algorithm 3)', 'planned'),
    ('Idea 1 with claim-level residual queries; DPO with graded pairs (audit fixes)', 'not run yet'),
]


# ---------------- rendering ----------------
def f(v, d=4):
    return '' if v is None or v == '' else (f'{v:.{d}f}' if isinstance(v, float) else str(v))


def lat(v):
    return '' if v is None else f'{v:.3f}' if v >= 0.01 else f'{v:.4f}'


HEAD = ('| System | n | ' + ' | '.join(COLS) + ' | Latency total (s/q) | Timed part (s/q) | What was timed |\n'
        '|---|---:|' + '---:|' * len(COLS) + '---:|---:|---|')


def table(rows, name_key='method', prefix=None):
    out = [HEAD]
    for r in rows:
        name = (prefix(r) if prefix else '') + r[name_key]
        out.append(f"| {name} | {r.get('n', '')} | " + ' | '.join(f(r.get(c)) for c in COLS) +
                   f" | {lat(r.get('latency'))} | {lat(r.get('llm_step'))} | {r.get('latency_note', '')} |")
    return '\n'.join(out)


by_name = {}
for r in test_rows:
    if 'masked' not in r['group']:
        by_name.setdefault(r['method'], r)
pairs = [
    ('MultiHop-RAG: bge-large-en-v1.5', 'BGE dense'),
    ('MultiHop-RAG: bge-large-en-v1.5 + bge-reranker-large', 'BGE + BGE reranker'),
    ('Ammann: Naive RAG (bge-large-en-v1.5)', 'BGE dense'),
    ('Ammann: + RR (bge-reranker-large)', 'BGE + BGE reranker'),
    ('Ammann: + QD', 'Qwen2.5-7B decomposition + hybrid'),
    ('Ammann: + QD', step1_names['A']),
    ('Ammann: + QD + RR (their method)', step1_names['B']),
    ('Ammann: + QD + RR (their method)', step1_names['C']),
    ('Ammann: + QD + RR (their method)', dpo_c_names['few_dpo']),
]
pub_by = {('MultiHop-RAG: ' if p['source'].startswith('MultiHop') else 'Ammann: ') + p['method']: p for p in published}
h2h = []
for theirs, ours in pairs:
    t, o = pub_by[theirs], by_name[ours]
    h2h.append({**t, 'method': f'**Theirs:** {theirs}'})
    h2h.append({**o, 'method': f'Ours: {ours}', 'n': o['n']})

groups = []
for r in test_rows:
    if r['group'] not in groups:
        groups.append(r['group'])

md = ['# All retrieval results: ours vs. published (MultiHop-RAG)', '',
      'Generated by `outputs/all_results/build_all_results.py` from saved outputs. Re-run it after new experiments.', '',
      '**How to read this file**',
      '- All metrics are fractions (0-1). A blank cell means the number was not measured (ours) or not reported (theirs).',
      '- Hits@4, Hits@10, MAP@10 and MRR@10 use the upstream MultiHop-RAG evaluator (commit 3dd4d4e). complete@10 = all gold facts in the top 10; complete@6 = all gold facts in the first 6 chunks of the same list; recall@10 = share of gold facts in the top 10. The papers report no complete@k or recall@k.',
      '- **MAP@10 is not comparable across papers:** the repo evaluator gives our replicated BGE 0.19 vs. 0.34 in Tang & Yang; Ammann et al. list both conventions in one table.',
      '- Question sets differ: ours = the fixed 1,574 test questions (Sections 2, 4) or all 2,251 CV questions (Section 3); Tang & Yang = all 2,255 non-null questions; Ammann et al. = an unstated "eval split".',
      '- Latency = seconds per question for retrieval only (no answer generation), the same scope as Ammann et al. Table 4. "Latency total" is filled only when every retrieval step was timed; otherwise "Timed part" gives the one step we did time (named in the last column), so the real total is higher. Our GPU: RTX 4070 Laptop (8 GB); theirs: not the same hardware.',
      '- Not included: answer accuracy (Tang & Yang Table 6) and HotpotQA results (Ammann et al. Table 2), because we have not measured them.', '',
      '## 1. Published results (theirs)', '', table(published, prefix=lambda r: ('Tang & Yang 2024: ' if r['source'].startswith('MultiHop') else 'Ammann et al. 2025: ')), '',
      '## 2. Ours: fixed test split (1,574 questions)', '']
for gname in groups:
    md += [f'### {gname}', '', table([r for r in test_rows if r['group'] == gname]), '']
md += ['## 3. Ours: all 2,251 cross-validation questions', '',
       'DPO rows: each question scored by the fold model that never trained on it. These saved results have no rankings, so complete@6 and MAP@10 are blank.', '',
       table(cv_rows), '',
       '## 4. Head-to-head (theirs vs. ours, same embedder; their setup is not fully published, so treat as context)', '',
       table(h2h), '',
       '## 5. Pending or planned (no results yet)', '',
       '| System | Status |\n|---|---|'] + [f'| {a} | {b} |' for a, b in pending]
(OUT / 'ALL_RESULTS.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print(f'wrote outputs/ALL_RESULTS.md: {len(published)} published, {len(test_rows)} test, {len(cv_rows)} CV rows')
