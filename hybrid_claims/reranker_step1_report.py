"""Score saved Step 1 reranker rankings and report paired retrieval comparisons."""

import argparse
from collections import Counter
import math
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import paths
from common.data import gold_facts, is_answerable, load_chunks, load_json, load_queries, split_queries
from common.metrics import facts_found
from hybrid_claims.llm_generate import write_json


OUTPUT = Path(paths.OUT) / 'reranker'
CV_FOLDS = Path(paths.OUT) / 'split_training' / 'cv' / 'folds.json'
UPSTREAM = Path(paths.OUT) / 'hybrid_claims' / 'verification' / 'upstream_3dd4d4e' / 'retrieval_evaluate.py'
SYSTEMS = {'A': 'Round-robin baseline', 'B': 'Whole-query reranking',
           'C': 'Rerank each list, then round-robin', 'D': 'Maximum reranker score'}
METRICS = ('complete@10', 'complete@6', 'recall@10', 'Hits@10', 'MRR@10', 'Hits@4')
COMPARISONS = (('B', 'A'), ('C', 'A'), ('C', 'B'))
TYPE_NAMES = {'all': 'Overall', 'inference_query': 'Inference',
              'comparison_query': 'Comparison', 'temporal_query': 'Temporal'}
SPLIT_NAMES = {'seed13_test': 'Fixed test split', 'cv_eligible': 'Pooled CV eligible questions'}
COST_FIELDS = ('seconds', 'reranker_seconds', 'selection_seconds', 'pairs')
AUDIT_KEYS = {'seed13_test': 'step1_same_test', 'cv_eligible': 'step1_all_eligible'}


def evaluation_sets(queries, cv_folds):
    answerable = {q['qid'] for q in queries if is_answerable(q)}
    test = sorted(answerable.intersection(split_queries(queries, seed=13)[1]))
    demo_ids = cv_folds['excluded_demo_qids']
    eligible = sorted(answerable.difference(demo_ids))
    if (len(answerable), len(test), len(eligible)) != (2255, 1574, 2251):
        raise ValueError('Unexpected MultiHop-RAG question counts')
    if eligible != sorted(cv_folds['eligible_qids']):
        raise ValueError('CV manifest and answerable benchmark questions disagree')
    pooled = sorted(cv_folds['folds']['fold_1']['test'] + cv_folds['folds']['fold_2']['test'])
    if pooled != eligible:
        raise ValueError('CV folds must exhaust the eligible questions once each')
    if not set(test) <= set(eligible):
        raise ValueError('The fixed test split must be a subset of the pooled CV questions')
    return {'seed13_test': test, 'cv_eligible': eligible}


def validated_texts(qid, name, ids, chunks, k):
    if not 0 < len(ids) <= k or any(type(cid) is not int or not 0 <= cid < len(chunks) for cid in ids):
        raise ValueError(f'Qid {qid}, system {name} has invalid ranked chunk IDs at K={k}')
    if len(set(ids)) != len(ids) or max(Counter(chunks[cid]['article_id'] for cid in ids).values()) > 2:
        raise ValueError(f'Qid {qid}, system {name}, K={k} violates uniqueness or the article cap')
    return [chunks[cid]['text'] for cid in ids]


def audit_pool(row, candidate, chunks):
    texts, lists = candidate['texts'], candidate['lists']
    if (row['qid'] != candidate['qid'] or row['question_type'] != candidate['question_type']
            or row['candidate_seconds'] != candidate['candidate_seconds']
            or len(texts) != len(lists) or len(texts) != row['claims'] + 1):
        raise ValueError(f'Candidate identity differs for qid {row["qid"]}')
    union = set().union(*lists)
    if any(type(cid) is not int or not 0 <= cid < len(chunks) for cid in union):
        raise ValueError(f'Candidate union has invalid chunk IDs for qid {row["qid"]}')
    capacity = sum(min(2, count) for count in Counter(chunks[cid]['article_id'] for cid in union).values())
    c_pairs = len({(text, cid) for text, ids in zip(texts, lists, strict=True) for cid in ids})
    if row['union_chunks'] != len(union) or row['capacity'] != capacity:
        raise ValueError(f'Candidate union size or capped capacity differs for qid {row["qid"]}')
    return union, capacity, {'A': 0, 'B': len(union), 'C': c_pairs, 'D': len(set(texts)) * len(union)}


def score_rankings(rankings, candidates, queries, chunks, requested_qids, calculate_metrics):
    answerable = {q['qid'] for q in queries if is_answerable(q)}
    by_qid = {row['qid']: row for row in candidates}
    if len(by_qid) != len(candidates) or set(by_qid) != set(requested_qids):
        raise ValueError('Candidate records must cover exactly the requested questions')
    scored = {}
    if not rankings:
        raise ValueError('No saved rankings to score')
    system_names = tuple(name for name in SYSTEMS if name in rankings[0]['systems'])
    if system_names not in (('A', 'B', 'C'), ('A', 'B', 'C', 'D')):
        raise ValueError('Rankings must contain A, B and C, with optional D')
    for row in rankings:
        qid = row['qid']
        if type(qid) is not int or qid not in answerable or qid not in by_qid or qid in scored:
            raise ValueError(f'Duplicate or non-answerable qid {qid}')
        query = queries[qid]
        if row['question_type'] != query['question_type'] or set(row['systems']) != set(system_names):
            raise ValueError(f'Systems or question type do not match the design for qid {qid}')
        gold = gold_facts(query)
        if not gold:
            raise ValueError(f'Answerable qid {qid} lacks gold facts')
        candidate_seconds = row['candidate_seconds']
        if not math.isfinite(candidate_seconds) or candidate_seconds < 0:
            raise ValueError(f'Invalid candidate search time for qid {qid}')
        union, capacity, expected_pairs = audit_pool(row, by_qid[qid], chunks)
        systems = {}
        for name in system_names:
            item = row['systems'][name]
            ten = validated_texts(qid, name, item['ids@10'], chunks, 10)
            six = validated_texts(qid, name, item['ids@6'], chunks, 6)
            if len(ten) != min(10, capacity) or len(six) != min(6, capacity):
                raise ValueError(f'Systems did not fill the same achievable K for qid {qid}')
            if not set(item['ids@10']) <= union or item['ids@6'] != item['ids@10'][:6]:
                raise ValueError(f'Final rankings differ from fixed candidates or the K=6 prefix for {name}, qid {qid}')
            cost = {key: item[key] for key in COST_FIELDS}
            if any(not math.isfinite(value) or value < 0 for value in cost.values()) or type(cost['pairs']) is not int:
                raise ValueError(f'Invalid timing or pair allocation for {name}, qid {qid}')
            if cost['pairs'] != expected_pairs[name]:
                raise ValueError(f'Invalid reranker pair count for {name}, qid {qid}')
            tracked = candidate_seconds + cost['reranker_seconds'] + cost['selection_seconds']
            if not math.isclose(cost['seconds'], tracked, abs_tol=1e-8):
                raise ValueError(f'Total time differs from logged stages for {name}, qid {qid}')
            metrics = calculate_metrics([ten], [gold])
            found = facts_found(ten, gold, 10)
            systems[name] = {
                **{key: metrics[key] for key in ('Hits@10', 'MRR@10', 'Hits@4')},
                'complete@10': int(all(found)), 'complete@6': int(all(facts_found(six, gold, 6))),
                'recall@10': sum(found) / len(found), **cost,
            }
        scored[qid] = {'question_type': row['question_type'], 'systems': systems,
                       'candidate_seconds': candidate_seconds, 'delivered@6': min(6, capacity),
                       'delivered@10': min(10, capacity)}
    if missing := set(requested_qids) - scored.keys():
        raise ValueError(f'Missing {len(missing)} scored qids, starting with {sorted(missing)[:10]}')
    return scored, system_names


def paired_intervals(differences, *, draws=10_000, seed=13):
    n = len(differences)
    rng = np.random.default_rng(seed)
    samples = np.concatenate([
        differences[rng.integers(n, size=(min(250, draws - start), n))].mean(axis=1)
        for start in range(0, draws, 250)
    ])
    lower, upper = np.quantile(samples, (0.025, 0.975), axis=0)
    delta = differences.mean(axis=0)
    return {key: [float(d), float(low), float(high)] for key, d, low, high in zip(METRICS, delta, lower, upper)}


def summarize(rows, names):
    arrays = {name: np.array([[r['systems'][name][m] for m in METRICS] for r in rows]) for name in names}
    return {
        'n': len(rows),
        'short_counts': {str(k): sum(r[f'delivered@{k}'] < k for r in rows) for k in (10, 6)},
        'candidate_seconds_per_question': sum(r['candidate_seconds'] for r in rows) / len(rows),
        'systems': {name: {
            'metrics': dict(zip(METRICS, array.mean(axis=0).tolist())),
            'complete_counts': {k: int(array[:, METRICS.index(k)].sum()) for k in ('complete@10', 'complete@6')},
            'cost_per_question': {key: sum(r['systems'][name][key] for r in rows) / len(rows) for key in COST_FIELDS},
        } for name, array in arrays.items()},
        'paired95': {f'{a}/{b}': paired_intervals(arrays[a] - arrays[b]) for a, b in COMPARISONS},
    }


def rollup(rows, names, splits):
    result = {}
    for split, qids in splits.items():
        group = [rows[q] for q in qids]
        strata = {'all': group}
        for kind in TYPE_NAMES.keys() - {'all'}:
            strata[kind] = [r for r in group if r['question_type'] == kind]
        if sum(len(part) for kind, part in strata.items() if kind != 'all') != len(group):
            raise ValueError('Encountered an unexpected question type')
        result[split] = {key: summarize(strata[key], names) for key in TYPE_NAMES if strata[key]}
    return result


def report_tables(data):
    lines = []
    metric_names = ' | '.join(METRICS)
    for split, groups in data.items():
        overall = groups['all']
        lines += [f'## {SPLIT_NAMES[split]}', '',
                  f'| Group | Questions | System | {metric_names} |',
                  '|---|---:|---|' + '---:|' * len(METRICS)]
        for kind, group in groups.items():
            for name, item in group['systems'].items():
                cells = ' | '.join(f'{item["metrics"][m]:.4f}' for m in METRICS)
                lines.append(f'| {TYPE_NAMES[kind]} | {group["n"]:,} | {name}. {SYSTEMS[name]} | {cells} |')
        lines += ['', f'| Group | Comparison | {metric_names} |', '|---|---|' + '---:|' * len(METRICS)]
        for kind, group in groups.items():
            for comparison, interval in group['paired95'].items():
                cells = ' | '.join(f'{100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]'
                                   for d, lo, hi in (interval[m] for m in METRICS))
                name, ref = comparison.split('/')
                lines.append(f'| {TYPE_NAMES[kind]} | {name} vs {ref} | {cells} |')
        short = overall['short_counts']
        lines += ['', f'Under the cap, the fixed candidate union was short for {short["10"]:,} questions at K=10 '
                  f'and {short["6"]:,} at K=6. All systems delivered the same number of chunks for each question.', '',
                  'Mean cost per question in seconds, with the final column in query–chunk pairs.', '',
                  '| System | Candidate search | Reranking | Selection | Total | Pairs |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, item in overall['systems'].items():
            cost = item['cost_per_question']
            lines.append(f'| {name} | {overall["candidate_seconds_per_question"]:.6f} | '
                         f'{cost["reranker_seconds"]:.6f} | {cost["selection_seconds"]:.6f} | '
                         f'{cost["seconds"]:.6f} | {cost["pairs"]:.2f} |')
        lines.append('')
    return lines


def check_audit(audit, data):
    for split, key in AUDIT_KEYS.items():
        archived = audit[key]['systems']['A60_no_backfill']
        overall = data[split]['all']
        baseline = overall['systems']['A']
        checks = {'n': overall['n'], 'complete_count': baseline['complete_counts']['complete@10'],
                  'short_top10': overall['short_counts']['10'],
                  **{key: baseline['metrics'][key] for key in METRICS if key != 'complete@6'}}
        if any(not math.isclose(value, archived[key], abs_tol=1e-12) for key, value in checks.items()):
            raise ValueError(f'Step 1 system A differs from the independent candidate archive audit for {split}')


def historical_rows(audit):
    labels = {'archived_exact_k': 'Archived exact-K rule split', 'uncropped_no_backfill': 'Full fused list',
              'A60_no_backfill': 'Clipped fused list (A)'}
    lines = ['| Question set | Rule-split candidate setup | Complete questions | Complete@10 | Short at K=10 |',
             '|---|---|---:|---:|---:|']
    for split, key in AUDIT_KEYS.items():
        for system, label in labels.items():
            row = audit[key]['systems'][system]
            lines.append(f'| {SPLIT_NAMES[split]} | {label} | {row["complete_count"]:,}/{row["n"]:,} | '
                         f'{row["complete@10"]:.2%} | {row["short_top10"]:,} |')
    return lines


def render_report(ranked, data, audit):
    lines = ['# MultiHop-RAG reranking placement', '']
    design = ranked['design']
    summary = []
    for split, groups in data.items():
        overall = groups['all']
        scores = ', '.join(f'{name} {row["metrics"]["complete@10"]:.2%}'
                           for name, row in overall['systems'].items())
        summary.append(f'Complete@10 on {SPLIT_NAMES[split].lower()} was {scores} '
                       f'over {overall["n"]:,} questions.')
    lines += [' '.join(summary), '', '## Fixed comparison', '',
              'All systems use the same whole-question BGE+BM25 list and BM25 lists for the rule-based claims. '
              f"Whole-question fusion takes {design['whole_retriever_depth']} candidates from each retriever "
              f"and keeps {design['per_list_depth']} after reciprocal rank fusion with k={design['rrf_k']}. "
              f"Each claim gets one {design['per_list_depth']}-candidate BM25 list. "
              f"All BM25 searches use k1={design['bm25_k1']} and b={design['bm25_b']}. The article cap is two.", '',
              '- **A** uses the original list order and round-robin selection.',
              '- **B** scores the deduplicated candidate union with the whole question, sorts by score, and applies the cap.',
              '- **C** scores each list with its own query and round-robins over the reranked lists with the cap.', '']
    if 'D' in data['seed13_test']['all']['systems']:
        lines += ['D sorts chunks by their best reranker score across the whole question and claims, then applies the cap.', '']
    elif design.get('optional_D'):
        lines += [f"Optional D. {design['optional_D']}", '']
    lines += ['The fixed candidate union supplies both selection budgets. Results use the available chunks when '
              'the cap leaves fewer than K. Complete@6 uses the first six delivered chunks. Hits@4 uses the first '
              'four chunks of the K=10 ranking. Hits@10, MRR@10 and Hits@4 come from the upstream evaluator at '
              'commit `3dd4d4e`. [Archived evaluator](../hybrid_claims/verification/upstream_3dd4d4e/retrieval_evaluate.py)', '',
              'Intervals are 95% paired percentile intervals from 10,000 question resamples within the stated '
              'group, using the same sampled question for both systems. The comparison tables show mean differences '
              'and bounds on the 0–100 scale. The fixed test questions are also included in the pooled CV results.', '',
              'Candidate search is shared. Total time adds shared candidate search, the branch’s reranker cost '
              'and selection. The reranker cost allocates batch time by pair, including requested pairs whose '
              'scores were reused across branches in this experiment. Pairs count distinct query–chunk inputs '
              'within each system and question. These measurements use cached dense rankings. The per-question '
              'times exclude model loading, batch preparation by length, index setup, cache I/O, and scoring metrics.', '']
    lines += [f"The cross-encoder is `{design['reranker']}` at revision `{design['reranker_revision']}`. "
              f"It uses {design['dtype']} and a limit of {design['max_length']} tokens per pair. "
              f"{design['input']} {design['tie_rule']}", '']
    runtime = ranked['runtime']
    lines += [f"The run used {runtime['unique_pairs']:,} distinct pairs across B and C. The CUDA scoring loop "
              f"processed {runtime['new_pairs']:,} new pairs in {runtime['new_scoring_wall_seconds']:.2f} seconds. "
              f"BM25 index setup took {runtime['bm25_index_seconds']:.2f} seconds.", '']
    if 'final_batch_size' in runtime:
        lines += [f"The final batch size was {runtime['final_batch_size']} and peak reserved GPU memory was "
                  f"{runtime['peak_reserved_gib']:.2f} GiB. Attention implementation was `{runtime['attention']}`.", '']
    lines += report_tables(data)
    lines += ['## Earlier 31.68% baseline', '',
              'The archived rule-split rankings on these 1,574 fixed test questions give MAP@10 of 0.3167533 '
              '(31.68%) and complete@10 of 455/1,574 (28.91%). Under binary per-question scoring, the closest '
              'complete counts to 31.68% are 498 (31.64%) and 499 (31.70%). The 31.68% figure corresponds to '
              'MAP@10 in the archive.', '',
              'The historical rule-split baseline retained the full fused whole-question list. Step 1 clips '
              'the fused list to 60 and uses the resulting fixed union at both budgets. Its A row measures '
              'round-robin over this clipped pool. [Prior independent verification](../hybrid_claims/VERIFY.md)', '']
    if audit is not None:
        lines += historical_rows(audit)
        lines += ['', 'The archived dense-only cross-encoder reranked BGE’s top 20 and achieved 10.55% '
                      'complete@10 on the fixed test set. [Archived baseline audit](previous_ablation_audit.json)', '']
    lines += ['[Candidate lists](step1_candidates.json) · [Rankings and run settings](step1_rankings.json) · '
              '[Metrics and paired intervals](step1_metrics.json)', '',
              '[Retrieval runner](../../hybrid_claims/reranker_step1.py) · '
              '[Scoring code](../../hybrid_claims/reranker_step1_report.py)', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    path = args.out_dir / 'step1_rankings.json'
    ranked = load_json(path)
    candidates = load_json(args.out_dir / 'step1_candidates.json')
    if any(ranked['design'].get(key) != value for key, value in candidates['design'].items()):
        raise ValueError('The rankings and candidate lists use different retrieval settings')
    queries, chunks = load_queries(), load_chunks()
    splits = evaluation_sets(queries, load_json(CV_FOLDS))
    calculate_metrics = runpy.run_path(str(UPSTREAM))['calculate_metrics']
    scored, names = score_rankings(ranked['rows'], candidates['rows'], queries, chunks,
                                   set().union(*splits.values()), calculate_metrics)
    data = rollup(scored, names, splits)
    audit_path = args.out_dir / 'previous_ablation_audit.json'
    audit = load_json(audit_path) if audit_path.is_file() else None
    if audit is not None:
        check_audit(audit, data)
    write_json(args.out_dir / 'step1_metrics.json', {
        'design': ranked['design'], 'runtime': ranked['runtime'], 'upstream_evaluator_commit': '3dd4d4e',
        'ranked_questions': len(scored), 'bootstraps': 10_000, 'bootstrap_seed': 13,
        'split_qids': splits, 'results': data,
    })
    (args.out_dir / 'STEP1_REPORT.md').write_text(render_report(ranked, data, audit), encoding='utf-8')


if __name__ == '__main__':
    main()
