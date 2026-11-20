"""Score out-of-fold DPO decomposition inside the Step 1 C reranker pipeline."""

import argparse
from collections import Counter
from hashlib import file_digest
import math
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.data import gold_facts, load_chunks, load_json, load_queries
from common.metrics import facts_found
from hybrid_claims.llm_generate import write_json
from hybrid_claims.reranker_step1_report import (CV_FOLDS, OUTPUT, SPLIT_NAMES, UPSTREAM, audit_pool,
                                                evaluation_sets, validated_texts)


VARIANTS = {'zero_base': 'Zero-shot untrained', 'zero_dpo': 'Zero-shot DPO',
            'few_base': 'Few-shot untrained', 'few_dpo': 'Few-shot DPO'}
REFERENCE = 'step1_c'
SYSTEMS = {REFERENCE: 'Step 1 C rule split', **VARIANTS}
METRICS = ('complete@10', 'complete@6', 'recall@10', 'Hits@4', 'Hits@10', 'MAP@10', 'MRR@10')
COMPARISONS = (('zero_dpo', 'zero_base'), ('few_dpo', 'few_base'), *((name, REFERENCE) for name in VARIANTS))
COST_FIELDS = ('candidate_seconds', 'reranker_seconds', 'selection_seconds', 'seconds', 'pairs')
FOLD_NAMES = ('fold_1', 'fold_2')
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 13


def index_rows(rows, expected_qids, label):
    indexed = {row['qid']: row for row in rows}
    if (len(indexed) != len(rows) or indexed.keys() != expected_qids
            or any(type(q) is not int for q in indexed)):
        raise ValueError(f'{label} must cover each eligible question exactly once')
    return indexed


def check_candidate_lists(candidate, original, chunks):
    texts, lists = candidate['texts'], candidate['lists']
    if (len(texts) != len(lists) or not texts or texts[0] != original['texts'][0]
            or lists[0] != original['lists'][0]):
        raise ValueError('Candidate lists must start with the unchanged Step 1 whole-question list')
    if any(len(ids) != 60 or len(set(ids)) != len(ids) for ids in lists):
        raise ValueError('Each candidate list must contain 60 distinct chunk IDs')
    if type(candidate['fallback']) is not bool:
        raise ValueError('Missing candidate fallback flag')
    if candidate['fallback'] and (texts != original['texts'] or lists != original['lists']):
        raise ValueError('Fallback must use the archived Step 1 rule texts and lists')
    union = set().union(*lists)
    if any(type(cid) is not int or not 0 <= cid < len(chunks) for cid in union):
        raise ValueError('Candidate union contains invalid chunk IDs')
    capacity = sum(min(2, count) for count in Counter(chunks[cid]['article_id'] for cid in union).values())
    pair_count = len({(text, cid) for text, ids in zip(texts, lists, strict=True) for cid in ids})
    return union, capacity, pair_count


def score_system(qid, name, ranked, chunks, gold, union, capacity, expected_pairs, calculate_metrics):
    ten = validated_texts(qid, name, ranked['ids@10'], chunks, 10)
    six = validated_texts(qid, name, ranked['ids@6'], chunks, 6)
    if (len(ten) != min(10, capacity) or len(six) != min(6, capacity)
            or ranked['ids@6'] != ranked['ids@10'][:6] or not set(ranked['ids@10']) <= union):
        raise ValueError(f'Qid {qid}, {name} violates its candidate pool, prefix or feasible budget')
    costs = {field: ranked[field] for field in COST_FIELDS}
    if (type(costs['pairs']) is not int or costs['pairs'] != expected_pairs
            or any(not math.isfinite(value) or value < 0 for value in costs.values())):
        raise ValueError(f'Qid {qid}, {name} has invalid reranker pair counts or timing')
    accounted = costs['candidate_seconds'] + costs['reranker_seconds'] + costs['selection_seconds']
    if not math.isclose(costs['seconds'], accounted, abs_tol=1e-8):
        raise ValueError(f'Qid {qid}, {name} total time does not equal the sum of its stages')
    split_seconds = ranked.get('split_seconds')
    if split_seconds is not None and (not math.isfinite(split_seconds) or split_seconds < 0):
        raise ValueError(f'Qid {qid}, {name} has invalid archived generation timing')
    seconds_with_split = ranked.get('seconds_with_split')
    if ((split_seconds is None) != (seconds_with_split is None)
            or split_seconds is not None and not math.isclose(seconds_with_split, costs['seconds'] + split_seconds,
                                                             abs_tol=1e-8)):
        raise ValueError(f'Qid {qid}, {name} has inconsistent retrieval plus generation timing')
    found = facts_found(ten, gold, 10)
    return {
        **calculate_metrics([ten], [gold]),
        'complete@10': int(all(found)), 'complete@6': int(all(facts_found(six, gold, 6))),
        'recall@10': sum(found) / len(found),
        'delivered@10': len(ten), 'delivered@6': len(six), 'fallback': ranked.get('fallback'),
        'split_seconds': split_seconds, 'seconds_with_split': seconds_with_split, **costs,
    }


def score_rows(ranked_rows, candidate_rows, step1_rows, step1_candidate_rows, queries, chunks, qids,
               calculate_metrics, *, check_fallback_ranks=False):
    ranked, candidates, references, reference_pools = (
        index_rows(rows, qids, label)
        for rows, label in ((ranked_rows, 'DPO C rankings'), (candidate_rows, 'DPO C candidates'),
                            (step1_rows, 'Step 1 rankings'), (step1_candidate_rows, 'Step 1 candidates'))
    )
    scored = {}
    for q in sorted(qids):
        row, source = ranked[q], candidates[q]
        ref, ref_pool = references[q], reference_pools[q]
        query, gold = queries[q], gold_facts(queries[q])
        if (not gold or any(record['question_type'] != query['question_type'] for record in (row, source, ref))
                or row['systems'].keys() != VARIANTS.keys() or source['systems'].keys() != VARIANTS.keys()):
            raise ValueError(f'Qid {q} has inconsistent question type, evidence or systems')
        union, capacity, expected = audit_pool(ref, ref_pool, chunks)
        original = {**ref['systems']['C'], 'candidate_seconds': ref['candidate_seconds']}
        systems = {REFERENCE: score_system(q, REFERENCE, original, chunks, gold, union, capacity,
                                           expected['C'], calculate_metrics)}
        for name in VARIANTS:
            item, pool = row['systems'][name], source['systems'][name]
            union, capacity, pairs = check_candidate_lists(pool, ref_pool, chunks)
            if (item['capacity'] != capacity or item['fallback'] is not pool['fallback']
                    or item['candidate_seconds'] != pool['candidate_seconds']
                    or item['split_seconds'] != pool['split_seconds']):
                raise ValueError(f'Candidate pool and ranking metadata disagree for qid {q}, {name}')
            if check_fallback_ranks and item['fallback'] and item['ids@10'] != original['ids@10']:
                raise ValueError(f'Cached fallback scores changed the Step 1 ranking for qid {q}, {name}')
            systems[name] = score_system(q, name, item, chunks, gold, union, capacity, pairs, calculate_metrics)
        scored[q] = systems
    return scored


def paired_intervals(groups, *, draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """Preserve the number of evaluated questions from each fitted outer fold."""
    total = sum(map(len, groups))
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, draws, 250):
        batch = min(250, draws - start)
        sums = [group[rng.integers(len(group), size=(batch, len(group)))].sum(axis=1)
                for group in groups if len(group)]
        means.append(sum(sums) / total)
    low, high = np.quantile(np.concatenate(means), (0.025, 0.975), axis=0)
    delta = sum(group.sum(axis=0) for group in groups) / total
    return {metric: [float(d), float(lo), float(hi)] for metric, d, lo, hi in zip(METRICS, delta, low, high)}


def summarize(scored, split_ids, folds):
    fold_qids = [sorted(set(split_ids).intersection(folds[fold]['test'])) for fold in FOLD_NAMES]
    qids = [qid for group in fold_qids for qid in group]
    if sorted(qids) != sorted(split_ids):
        raise ValueError('The two CV folds do not cover this evaluation set exactly once')
    arrays = {name: np.array([[scored[q][name][metric] for metric in METRICS] for q in qids]) for name in SYSTEMS}
    endpoints = np.cumsum([len(group) for group in fold_qids])[:-1]
    rows = {}
    for name, array in arrays.items():
        records = [scored[q][name] for q in qids]
        recorded = [row for row in records if row['split_seconds'] is not None]
        rows[name] = {
            'metrics': dict(zip(METRICS, array.mean(axis=0).tolist())),
            'complete_counts': {metric: int(array[:, METRICS.index(metric)].sum())
                                for metric in ('complete@10', 'complete@6')},
            'short_counts': {str(k): sum(row[f'delivered@{k}'] < k for row in records) for k in (10, 6)},
            'fallback_count': sum(row['fallback'] is True for row in records) if name in VARIANTS else None,
            'cost_per_question': {key: sum(row[key] for row in records) / len(records) for key in COST_FIELDS},
            'split_seconds_coverage': len(recorded),
            'split_seconds_per_recorded_question': (sum(row['split_seconds'] for row in recorded) / len(recorded)
                                                    if recorded else None),
            'seconds_with_split_per_recorded_question': (sum(row['seconds_with_split'] for row in recorded) / len(recorded)
                                                         if recorded else None),
        }
    return {
        'n': len(qids), 'fold_counts': {fold: len(group) for fold, group in zip(FOLD_NAMES, fold_qids)}, 'systems': rows,
        'paired95': {f'{a}/{b}': paired_intervals(np.split(arrays[a] - arrays[b], endpoints))
                     for a, b in COMPARISONS},
    }


def check_reference(published, results):
    for split, current in results.items():
        reference = current['systems'][REFERENCE]
        prior_split = published['results'][split]['all']
        prior = prior_split['systems']['C']
        if (current['n'] != prior_split['n'] or reference['short_counts'] != prior_split['short_counts']
                or reference['complete_counts'] != prior['complete_counts']):
            raise ValueError(f'Published Step 1 C counts differ for {split}')
        for metric, value in prior['metrics'].items():
            if not math.isclose(value, reference['metrics'][metric], abs_tol=1e-12):
                raise ValueError(f'Published Step 1 C {metric} differs for {split}')


def check_design(ranked, candidates, step1, out_dir):
    design = ranked['design']
    if any(design.get(key) != value for key, value in candidates['design'].items()):
        raise ValueError('DPO C ranking settings differ from the candidate manifest')
    fixed_keys = ('reranker', 'reranker_revision', 'dtype', 'max_length', 'article_cap', 'per_list_depth',
                  'whole_retriever_depth', 'rrf_k', 'bm25_k1', 'bm25_b', 'backfill')
    if any(design[key] != step1['design'][key] for key in fixed_keys) or design['cuts'] != [10, 6]:
        raise ValueError('DPO C and Step 1 C use different retrieval or reranker settings')
    files = {'step1_candidates': out_dir / 'step1_candidates.json', 'folds': CV_FOLDS}
    for key, path in files.items():
        with path.open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != design['frozen_archive_sha256'][key]:
                raise ValueError(f'DPO C candidate archive does not match {path.name}')
    runtime = ranked['runtime']
    pairs = (runtime['reused_step1_pairs'], runtime['reused_current_run_pairs'], runtime['new_pairs'])
    if (any(type(n) is not int or n < 0 for n in pairs) or sum(pairs) != runtime['unique_pairs']
            or runtime['step1_logit_cache'] != step1['runtime']['logit_cache']):
        raise ValueError('Cross-encoder score provenance or pair count is inconsistent')


def cells(interval):
    return ' | '.join(f'{100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]'
                      for d, lo, hi in (interval[metric] for metric in METRICS))


def tables(summary):
    metric_names = ' | '.join(METRICS)
    lines = [f'| System | {metric_names} |', '|---|' + '---:|' * len(METRICS)]
    for name, row in summary['systems'].items():
        scores = ' | '.join(f'{row["metrics"][key]:.4f}' for key in METRICS)
        lines.append(f'| {SYSTEMS[name]} | {scores} |')
    lines += ['', f'| Paired comparison | {metric_names} |', '|---|' + '---:|' * len(METRICS)]
    for comparison, interval in summary['paired95'].items():
        a, b = comparison.split('/')
        lines.append(f'| {SYSTEMS[a]} vs {SYSTEMS[b]} | {cells(interval)} |')
    lines += ['', '| System | Short at K=10 | Short at K=6 | Rule fallback | Candidate prep (s) | Reranking (s) | '
              'Selection (s) | Total retrieval (s) | Distinct pairs |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['systems'].items():
        fallback = f"{row['fallback_count']:,}" if row['fallback_count'] is not None else '—'
        cost = row['cost_per_question']
        times = ' | '.join(f'{cost[key]:.6f}' for key in COST_FIELDS[:-1])
        lines.append(f'| {SYSTEMS[name]} | {row["short_counts"]["10"]:,} | {row["short_counts"]["6"]:,} | {fallback} | '
                     f'{times} | {cost["pairs"]:.2f} |')
    return lines


def generation_times(summary):
    lines = ['| System | Logged split times | Split (s) | Retrieval + split (s) |',
             '|---|---:|---:|---:|']
    for name, row in summary['systems'].items():
        count = row['split_seconds_coverage']
        if count:
            lines.append(f'| {SYSTEMS[name]} | {count:,}/{summary["n"]:,} | '
                         f'{row["split_seconds_per_recorded_question"]:.6f} | '
                         f'{row["seconds_with_split_per_recorded_question"]:.6f} |')
        elif name == REFERENCE:
            lines.append(f'| {SYSTEMS[name]} | — | — | {row["cost_per_question"]["seconds"]:.6f} |')
        else:
            lines.append(f'| {SYSTEMS[name]} | 0/{summary["n"]:,} | — | — |')
    return lines


def render_report(results, design, runtime):
    lines = ['# DPO decomposition with per-list reranking', '']
    for split in ('cv_eligible', 'seed13_test'):
        cohort = results[split]
        summaries = []
        for prompt in ('zero', 'few'):
            base, dpo = f'{prompt}_base', f'{prompt}_dpo'
            control = cohort['systems'][base]['metrics']['complete@10']
            trained = cohort['systems'][dpo]['metrics']['complete@10']
            d, lo, hi = cohort['paired95'][f'{dpo}/{base}']['complete@10']
            summaries.append(f'{SYSTEMS[dpo]} reached **{trained:.2%}** complete@10 from {control:.2%} with the '
                             f'matched untrained model ({100*d:+.2f} points, 95% CI {100*lo:+.2f} to {100*hi:+.2f}).')
        ref = cohort['systems'][REFERENCE]['metrics']['complete@10']
        lines += [f"{SPLIT_NAMES[split]} ({cohort['n']:,}). " + ' '.join(summaries) +
                  f' The Step 1 C rule split reached **{ref:.2%}**.', '']
    lines += ['## Evaluation setup', '',
              'The four decomposers reuse saved out-of-fold Qwen2.5-3B-Instruct generations. Zero-shot uses BF16. '
              'Few-shot uses 4-bit NF4 with BF16 compute and the fixed four demonstrations. Each DPO adapter was '
              'chosen on its fold’s held-out set under the earlier protocol. Both question sets use the same '
              'predictions. The 1,574 fixed test questions form a subset of the 2,251 eligible CV questions, which '
              'exclude the four demonstrations. [Fold IDs](../split_training/cv/folds.json) · '
              '[Adapter training and selection](../split_training/cv/REPORT.md)', '',
              'The whole-question hybrid list is copied from Step 1 C. Each rule claim or retained LLM '
              'sub-question gets its own BM25 top-60 list. The line filter removes numbering or bullets, keeps '
              'the first four lines with at least three content words, and falls back to the original rule '
              'claims when every LLM line fails. The cross-encoder scores each list with its own query, '
              'then round-robin picks from the sorted lists with at most two chunks per article.', '',
              f"The reranker is `{design['reranker']}` at revision `{design['reranker_revision']}`. "
              f"It uses {design['dtype']} with at most {design['max_length']} tokens per query–chunk pair. "
              'Each system uses its own candidate union. Some unions cannot fill a cutoff under the article cap. '
              'Short counts are shown for each system. Complete@6 uses the first six delivered chunks. Hits@4 uses '
              'the first four ranks from the top-10 list. Hits@4, Hits@10, MAP@10 and MRR@10 come from the upstream '
              'evaluator at commit `3dd4d4e`. [Step 1 report](STEP1_REPORT.md) · '
              '[Upstream scorer](../hybrid_claims/verification/upstream_3dd4d4e/retrieval_evaluate.py)', '',
              'The paired 95% percentile intervals use 10,000 question resamples within outer fold. Each '
              'comparison uses the same resampled questions for both systems. Differences and bounds use the '
              '0–100 scale. The intervals describe variation across test questions for the fitted fold models.', '',
              'Cost columns average logged seconds and distinct query–chunk pairs per question. Retrieval time '
              'adds candidate preparation, amortized reranking, and final selection. The raw generations are cached, '
              'and their generation time is reported separately when recorded. Split and retrieval-plus-split '
              'means use only questions with logged split times. Archive batches and precision varied across '
              'generations, so these times describe the recorded runs. A shared scored pair is credited to each '
              'system requesting it.', '',
              'Step 1 candidate preparation used cached BGE ranks and live BM25. The four LLM systems reuse the '
              'prepared whole-question list. Their candidate stage uses cached lexical lists when available and '
              'runs BM25 on misses.', '']
    lines += [f"The four LLM systems requested {runtime['unique_pairs']:,} distinct query–chunk pairs. The run "
              f"reused {runtime['reused_step1_pairs']:,} logits from Step 1 and "
              f"{runtime['reused_current_run_pairs']:,} from this experiment’s earlier batches. The final CUDA "
              f"scoring pass processed {runtime['new_pairs']:,} new pairs in "
              f"{runtime['new_scoring_wall_seconds']:.2f} seconds.", '']
    if 'final_batch_size' in runtime:
        lines += [f"The final pair batch size was {runtime['final_batch_size']}. Peak reserved GPU memory was "
                  f"{runtime['peak_reserved_gib']:.2f} GiB with `{runtime['attention']}` attention.", '']
    for split in ('cv_eligible', 'seed13_test'):
        data = results[split]
        lines += [f"## {SPLIT_NAMES[split]} ({data['n']:,})", '', *tables(data), '', *generation_times(data), '']
    lines += ['[Candidate lists](dpo_c_candidates.json) · [Saved rankings](dpo_c_rankings.json) · '
              '[Metrics and intervals](dpo_c_metrics.json) · [Step 1 reference rankings](step1_rankings.json)', '',
              '[Candidate preparation](../../hybrid_claims/reranker_dpo_c_candidates.py) · '
              '[Retrieval runner](../../hybrid_claims/reranker_dpo_c.py) · '
              '[Scoring code](../../hybrid_claims/reranker_dpo_c_report.py)', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    ranked = load_json(args.out_dir / 'dpo_c_rankings.json')
    candidates = load_json(args.out_dir / 'dpo_c_candidates.json')
    step1 = load_json(args.out_dir / 'step1_rankings.json')
    step1_candidates = load_json(args.out_dir / 'step1_candidates.json')
    check_design(ranked, candidates, step1, args.out_dir)
    queries, chunks, cv = load_queries(), load_chunks(), load_json(CV_FOLDS)
    splits = evaluation_sets(queries, cv)
    calculate_metrics = runpy.run_path(str(UPSTREAM))['calculate_metrics']
    scored = score_rows(ranked['rows'], candidates['rows'], step1['rows'], step1_candidates['rows'],
                        queries, chunks, set(splits['cv_eligible']), calculate_metrics, check_fallback_ranks=True)
    results = {split: summarize(scored, qids, cv['folds']) for split, qids in splits.items()}
    check_reference(load_json(args.out_dir / 'step1_metrics.json'), results)
    report = render_report(results, ranked['design'], ranked['runtime'])
    write_json(args.out_dir / 'dpo_c_metrics.json', {
        'design': ranked['design'], 'runtime': ranked['runtime'], 'upstream_evaluator_commit': '3dd4d4e',
        'bootstrap': {'draws': BOOTSTRAPS, 'seed': BOOTSTRAP_SEED, 'strata': 'outer CV fold'},
        'split_qids': splits, 'results': results,
    })
    (args.out_dir / 'DPO_C_REPORT.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
