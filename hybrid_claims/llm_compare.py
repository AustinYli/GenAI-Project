"""Evaluate fixed decomposition and HyDE variants inside the hybrid claims pipeline."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.bge import BGEEncoder, DenseIndex, encode_cached
from common.data import gold_facts, load_chunks, load_queries, outlet_names
from common.metrics import facts_found
from hybrid_claims import run
from hybrid_claims.llm_generate import CACHE, DESIGN, OUT, frozen_design, partition, write_json


SYSTEMS = {
    'A': 'A. Whole-question hybrid', 'B': 'B. Rule-based claims',
    'C': 'C. Decomposition + BM25', 'C_hybrid': 'C. Decomposition + hybrid',
    'D': 'D. HyDE dense', 'D_bm25': 'D. HyDE dense + BM25',
    'E': 'E. Decomposition + HyDE',
}
METRICS = ('complete@10', 'complete@6', 'recall@10')
TYPES = {'inference_query': 'Inference', 'comparison_query': 'Comparison', 'temporal_query': 'Temporal'}
BULLET = re.compile(r'^\s*(?:[-*•]\s+|(?:\d+[.)]|\(\d+\))\s*)')
BUDGET_RULE = {
    'backfill_depth': 2000,
    'when': 'Only after the original round-robin exhausts all of a system’s lists with fewer than K chunks.',
    'how': 'Append unseen candidates from a common deeper whole-question BGE+BM25 RRF list, '
           'preserving the original prefix, the exact K budget, and the two-chunk article cap.',
}
RETRIEVAL_DESIGN = {
    'systems': SYSTEMS, 'budget': BUDGET_RULE,
    'combination': 'E visits the C BM25 lists in order, then the D dense list, after the common first list.',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def valid_lines(text):
    lines = [BULLET.sub('', line).strip() for line in text.splitlines()]
    return [line for line in lines if sum(any(ch.isalpha() for ch in token) for token in run.tokenize(line)) >= 3][:4]


def dense_extras(split, qids, subquestions, passages, backfill_qids, cache_dir=CACHE):
    vectors = Path(run.paths.CACHE)
    model = BGEEncoder(device='cuda', dtype='fp32')
    index = DenseIndex(np.load(vectors / f'emb_{run.MODEL}_fp32_chunks.npy'), device='cuda')
    question_vectors = np.load(vectors / f'emb_{run.MODEL}_fp32_queries.npy')[qids]
    sub_texts = list(dict.fromkeys(text for q in qids for text in subquestions[q]))
    sub_ranks = {}
    if sub_texts:
        sub_vectors = encode_cached(model, sub_texts, str(cache_dir / f'{split}_sub_vectors.npy'), is_query=True)
        sub_ids, _ = index.search(sub_vectors, DESIGN['extra_depth'])
        sub_ranks = dict(zip(sub_texts, sub_ids, strict=True))
    hyde_ranks = {}
    if passages is not None:
        texts = [text for q in qids for text in passages[q]]
        hyp_vectors = encode_cached(model, texts, str(cache_dir / f'{split}_hyde_vectors.npy'), is_query=False)
        n_passages = DESIGN['hyde']['num_return_sequences']
        hyp_vectors = hyp_vectors.reshape(len(qids), n_passages, -1)
        averaged = (question_vectors + hyp_vectors.sum(axis=1)) / (n_passages + 1)
        averaged /= np.linalg.norm(averaged, axis=1, keepdims=True)
        hyde_ids, _ = index.search(averaged, DESIGN['extra_depth'])
        hyde_ranks = dict(zip(qids, hyde_ids, strict=True))
    backfill_ids = []
    if backfill_qids:
        full_question_vectors = np.load(vectors / f'emb_{run.MODEL}_fp32_queries.npy')[backfill_qids]
        backfill_ids, _ = index.search(full_question_vectors, BUDGET_RULE['backfill_depth'])
    return sub_ranks, hyde_ranks, dict(zip(backfill_qids, backfill_ids, strict=True))


def score_stage(split, design_sha, *, cache_dir=CACHE, include_hyde=True):
    queries, chunks = load_queries(), load_chunks()
    qids = partition(queries, split)
    generations = {}
    forms = ('decomp', 'hyde') if include_hyde else ('decomp',)
    for form in forms:
        cache = read_json(cache_dir / f'{split}_{form}.json')
        if cache['design_sha256'] != design_sha or set(map(int, cache['rows'])) != set(qids):
            raise ValueError(f'Incomplete or mismatched {split} {form} generation cache')
        generations[form] = {int(q): row for q, row in cache['rows'].items()}
    outlets = sorted(outlet_names(chunks), key=len, reverse=True)
    questions = {q: queries[q]['query'] for q in qids}
    rule_claims = {q: run.claims(questions[q], outlets) for q in qids}
    generated = {q: valid_lines(generations['decomp'][q]['texts'][0]) for q in qids}
    subquestions = {q: generated[q] or rule_claims[q] for q in qids}
    passages = {q: generations['hyde'][q]['texts'] for q in qids} if include_hyde else None
    if include_hyde and any(len(texts) != DESIGN['hyde']['num_return_sequences'] for texts in passages.values()):
        raise ValueError('Each question requires four generated passages')

    cache = Path(run.paths.CACHE) / f'dense_ranking_{run.MODEL}_fp32.json'
    dense = read_json(cache)['ids']
    bm = run.BM25([run.chunk_doc(c) for c in chunks], DESIGN['bm25_k1'], DESIGN['bm25_b'])
    whole_depth, backfill_depth = DESIGN['whole_depth'], BUDGET_RULE['backfill_depth']
    whole = {q: run.rrf([dense[q][:whole_depth], bm.search(questions[q], whole_depth)], k=DESIGN['rrf_k'])
             for q in qids}
    max_k = max(DESIGN['cuts'])
    backfill_qids = [q for q in qids if len(run.pick([whole[q]], max_k, chunks, DESIGN['cap'])) < max_k]
    print(f'Embedding {split} sub-questions and passages with BGE', flush=True)
    sub_ranks, hyde_ranks, backfill_dense = dense_extras(split, qids, subquestions, passages, backfill_qids, cache_dir)
    backfill = {q: run.rrf([backfill_dense[q], bm.search(questions[q], backfill_depth)], k=DESIGN['rrf_k'])
                for q in backfill_qids}

    def bm25_list(text):
        return bm.search(text, DESIGN['extra_depth'])

    def fuse(lists, k):
        return run.rrf(lists, k=DESIGN['rrf_k'])[:k]

    rows = []
    for q in qids:
        sub_bm = [bm25_list(text) for text in subquestions[q]]
        extras = {
            'A': [], 'B': [bm25_list(text) for text in rule_claims[q]],
            'C': sub_bm,
            'C_hybrid': [fuse([sub_ranks[text], ids], DESIGN['extra_depth']) for text, ids in
                         zip(subquestions[q], sub_bm, strict=True)],
        }
        if include_hyde:
            extras.update(D=[hyde_ranks[q]], D_bm25=[hyde_ranks[q], bm25_list('\n'.join(passages[q]))],
                          E=[*sub_bm, hyde_ranks[q]])
        result = {}
        for name, extra in extras.items():
            result[name] = {}
            for k in DESIGN['cuts']:
                ids = run.pick([whole[q], *extra], k, chunks, DESIGN['cap'])
                result[name][f'backfilled@{k}'] = k - len(ids)
                if len(ids) < k:
                    ids = run.pick([ids + backfill[q]], k, chunks, DESIGN['cap'])
                if len(ids) != k or len(set(ids)) != k or max(Counter(chunks[c]['article_id'] for c in ids).values()) > DESIGN['cap']:
                    raise ValueError(f'{name} qid {q} violated the {k}-chunk budget or cap')
                flags = facts_found([chunks[c]['text'] for c in ids], gold_facts(queries[q]), k)
                result[name][f'complete@{k}'] = int(all(flags))
                result[name][f'recall@{k}'] = sum(flags) / len(flags)
                result[name][f'ids@{k}'] = ids
        rows.append({
            'qid': q, 'type': queries[q]['question_type'], 'system': result,
            'fallback': not bool(generated[q]), 'subquestions': subquestions[q],
            'passages': passages[q] if include_hyde else [],
            'llm': {form: {key: generations[form][q][key] for key in ('seconds', 'tokens')} for form in generations},
        })
    return rows


def metrics(rows, name):
    return {key: float(np.mean([row['system'][name][key] for row in rows])) for key in METRICS}


def ci(rows, name, baseline):
    differences = np.array([[r['system'][name][key] - r['system'][baseline][key] for key in METRICS] for r in rows])
    rng = np.random.default_rng(13)
    means = [differences[rng.integers(len(rows), size=(250, len(rows)))].mean(axis=1) for _ in range(40)]
    bounds = np.quantile(np.concatenate(means), (0.025, 0.975), axis=0)
    return {key: [float(differences[:, i].mean()), float(bounds[0, i]), float(bounds[1, i])]
            for i, key in enumerate(METRICS)}


def summarize(rows):
    return {name: metrics(rows, name) for name in SYSTEMS}


def comparisons(rows):
    return {name: {ref: ci(rows, name, ref) for ref in ('A', 'B') if name != ref} for name in SYSTEMS}


def quote(text):
    return '\n'.join('> ' + line for line in text.strip().splitlines()) or '> (empty output)'


def write_report(dev, test, decisions):
    rollup = {'overall': {}, 'by_type': {}, 'test_intervals': {}, 'test_intervals_by_type': {}, 'cost': {}}
    partitions = {'dev': dev, 'test': test}
    for part, rows in partitions.items():
        rollup['overall'][part] = summarize(rows)
        rollup['by_type'][part] = {kind: {'n': len(group), 'metrics': summarize(group)} for kind in TYPES
                                   if (group := [r for r in rows if r['type'] == kind])}
        rollup['cost'][part] = {
            form: {key: float(np.mean([r['llm'][form][key] for r in rows])) for key in ('seconds', 'tokens')}
            for form in ('decomp', 'hyde')
        }
        rollup['cost'][part]['decomp'].update(
            fallback=float(np.mean([r['fallback'] for r in rows])),
            subquestions=float(np.mean([len(r['subquestions']) for r in rows])),
        )
        rollup['cost'][part]['combined'] = {key: sum(rollup['cost'][part][form][key]
                                                  for form in ('decomp', 'hyde')) for key in ('seconds', 'tokens')}
    rollup['test_intervals'] = comparisons(test)
    rollup['test_intervals_by_type'] = {kind: comparisons(group) for kind in TYPES
                                       if (group := [r for r in test if r['type'] == kind])}
    c, d = (decisions['selected'][k] for k in ('decomp', 'hyde'))
    rollup['selected_decomp_vs_hyde'] = ci(test, c, d)
    write_json(OUT / 'results.json', rollup)

    def fmt(result):
        return ' | '.join(f'{result[key]:.2%}' for key in METRICS)

    def fmt_ci(v):
        d, lo, hi = (100 * x for x in v)
        return f'{d:+.2f} [{lo:+.2f}, {hi:+.2f}]'

    def ci_table(intervals):
        return ['| System | Reference | Complete@10 | Complete@6 | Recall@10 |', '|---|---|---:|---:|---:|', *[
            f'| {SYSTEMS[name]} | {ref} | ' + ' | '.join(fmt_ci(changes[key]) for key in METRICS) + ' |'
            for name, by_ref in intervals.items() for ref, changes in by_ref.items()
        ]]

    overall = rollup['overall']['test']
    lines = [
        '# LLM Query Decomposition and HyDE in Hybrid Retrieval', '',
        f"On the 1,574 test questions, whole-question hybrid search covered every fact on {overall['A']['complete@10']:.2%} "
        f"of questions at K=10. Rule-based claims reached {overall['B']['complete@10']:.2%}. The variants selected "
        f"on dev reached {overall[c]['complete@10']:.2%} for decomposition and {overall[d]['complete@10']:.2%} for HyDE. "
        'The uncertainty and cost tables give the paired differences and generation overhead.', '',
        '## Setup', '',
        'The same whole-question hybrid list goes first in each system, followed by extra lists in their original order. '
        'Each final list contains exactly K unique chunks and at most two per article. The whole-question BGE and BM25 lists '
        'each have depth 100, additional lists have depth 60, and reciprocal rank fusion uses k=60.', '',
        'The separate audit found that the original candidate lists sometimes run out below K. In this experiment, '
        'after a system exhausts its original lists, it fills the remaining slots from a shared deeper whole-question '
        'RRF list (depth 2,000 per retriever). This preserves the original picks and keeps the article cap. The '
        'verification report measures the original implementation separately.', '',
        'Qwen2.5-3B-Instruct used bf16 and left-padded batches. Decomposition used greedy decoding with a 128-token cap. '
        'It kept up to four lines with at least three letter-containing tokens after BM25 stopword filtering and '
        'used the rule-based claim texts as fallback when all lines failed. HyDE sampled four excerpts per question at '
        'temperature 0.7 with up to 150 tokens per excerpt. Their BGE passage '
        'embeddings use no query instruction. The normalized mean gives the four passages and the whole-question query '
        'embedding equal weight. The additional BM25 variant searches the four excerpts as one concatenated query.', '',
        'The combined system visits the decomposition BM25 lists, then the HyDE dense list, on each round after the '
        'common first list.', '',
        f"Dev selected **{SYSTEMS[c]}** and **{SYSTEMS[d]}** within their respective families by complete@10. "
        'The simpler variant wins a tie. The prompts, tie rule and system variants were fixed before test generation.', '',
        '## Overall coverage', '',
        '| System | Dev complete@10 | Dev complete@6 | Dev recall@10 | Test complete@10 | Test complete@6 | Test recall@10 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, title in SYSTEMS.items():
        lines.append(f"| {title} | {fmt(rollup['overall']['dev'][name])} | {fmt(overall[name])} |")
    lines += ['', '| System | Dev rows backfilled at K=10 / K=6 | Test rows backfilled at K=10 / K=6 |',
              '|---|---:|---:|']
    for name, title in SYSTEMS.items():
        counts = [' / '.join(str(sum(r['system'][name][f'backfilled@{k}'] > 0 for r in rows)) for k in (10, 6))
                  for rows in (dev, test)]
        lines.append(f'| {title} | {counts[0]} | {counts[1]} |')
    lines += ['', '## Test differences with paired 95% confidence intervals', '',
              'Differences are in percentage points. Intervals use 10,000 paired question resamples with seed 13.', '',
              *ci_table(rollup['test_intervals'])]
    selected_delta = ' | '.join(fmt_ci(rollup['selected_decomp_vs_hyde'][key]) for key in METRICS)
    lines.append(f'| {SYSTEMS[c]} (dev choice) | {SYSTEMS[d]} (dev choice) | {selected_delta} |')

    lines += ['', '## By question type', '']
    for kind, title in TYPES.items():
        dv, ts = (rollup['by_type'][part][kind] for part in ('dev', 'test'))
        lines += [f"### {title} ({dv['n']} dev, {ts['n']} test)", '',
                  '| System | Dev complete@10 | Dev complete@6 | Dev recall@10 | Test complete@10 | Test complete@6 | Test recall@10 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name, label in SYSTEMS.items():
            lines.append(f"| {label} | {fmt(dv['metrics'][name])} | {fmt(ts['metrics'][name])} |")
        lines += ['', 'Paired 95% confidence intervals on this type’s test questions, in percentage points.', '',
                  *ci_table(rollup['test_intervals_by_type'][kind]), '']

    lines += ['## LLM generation cost', '',
              'Generation seconds include prompt formatting, tokenization, decoding and transfer of the output. '
              'They are batch wall time divided by the number of questions, excluding model loading, BGE and retrieval. '
              'Token counts include the generated stop token when present and exclude batch padding. Both retrieval '
              'variants of each LLM family share the same generations. A and B have zero LLM cost. '
              'E combines the generation costs of C and D.', '',
              '| Split | Family | Seconds/question | Generated tokens/question | Fallback | Mean final sub-questions |',
              '|---|---|---:|---:|---:|---:|']
    for part in ('dev', 'test'):
        for form, title in (('decomp', 'Decomposition'), ('hyde', 'HyDE'), ('combined', 'Combined')):
            cost = rollup['cost'][part][form]
            parse = f"{cost['fallback']:.2%} | {cost['subquestions']:.2f}" if form == 'decomp' else '— | —'
            lines.append(f"| {part} | {title} | {cost['seconds']:.3f} | {cost['tokens']:.1f} | {parse} |")

    lines += ['', '## Ten random dev examples', '',
              'The four HyDE passages for each question are generated search text and can contain factual inventions.', '']
    queries = load_queries()
    for row in random.Random(20261119).sample(dev, 10):
        qid = row['qid']
        lines += [f'### Question {qid}', '', quote(queries[qid]['query']), '',
                  '**C used rule-based fallback**' if row['fallback'] else '**C sub-questions**', '']
        lines += [f'- {sub}' for sub in row['subquestions']]
        for i, passage in enumerate(row['passages'], 1):
            lines += ['', f'**D passage {i}**', '', quote(passage)]
        lines.append('')
    lines += ['[Frozen prompts](design.json) · [Retrieval design](retrieval_design.json) · '
              '[Dev decisions](dev_decisions.json) · [Numeric results](results.json)', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('dev', 'test'))
    args = parser.parse_args()
    signature = frozen_design()
    decisions_path = OUT / 'dev_decisions.json'
    plan_path = OUT / 'retrieval_design.json'
    if args.split == 'dev' and not plan_path.exists():
        write_json(plan_path, RETRIEVAL_DESIGN)
    if read_json(plan_path) != RETRIEVAL_DESIGN:
        raise ValueError('The frozen retrieval settings differ from the experiment')
    if args.split == 'test':
        decisions = read_json(decisions_path)
        if decisions['design_sha256'] != signature or decisions['retrieval_design'] != RETRIEVAL_DESIGN:
            raise ValueError('Test requires dev decisions from this frozen design')
    rows = score_stage(args.split, signature)
    write_json(OUT / f'{args.split}_per_question.json', rows)
    totals = summarize(rows)
    print(json.dumps(totals, indent=2), flush=True)
    if args.split == 'dev':
        selected = {form: max(choices, key=lambda name: totals[name]['complete@10']) for form, choices in
                    {'decomp': ('C', 'C_hybrid'), 'hyde': ('D', 'D_bm25')}.items()}
        write_json(decisions_path, {'design_sha256': signature, 'retrieval_design': RETRIEVAL_DESIGN, 'selected': selected,
                                    'dev': totals})
        print(f'Locked dev selections: {selected}', flush=True)
    else:
        write_report(read_json(OUT / 'dev_per_question.json'), rows, decisions)


if __name__ == '__main__':
    main()
