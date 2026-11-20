"""Score the fixed split-training comparisons and write their experiment reports."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_claims.split_training_generation import DEMO_FILE, OUT, read_json, run_generation, write_json
from hybrid_claims.split_training_retrieval import METRICS, Retrieval, paired_ci, summarize


STEP1 = OUT / 'step1'
SOURCE7 = OUT.parent / 'llm_decomp_bigger' / 'qwen2.5-7b-instruct' / 'generation'
SOURCE3 = Path(OUT.parents[1]) / 'cache' / 'llm_decomp_vs_hyde'
BASE_NAMES = {'A': 'Whole-question hybrid', 'B': 'Rule split', 'zero': '7B zero-shot', 'few': '7B few-shot'}


def cached_raw(path, ids):
    rows = read_json(path)['rows']
    return {qid: rows[str(qid)]['texts'] for qid in ids}


def score_raw(retriever, raw):
    if any(len(completions) != 1 for completions in raw.values()):
        raise ValueError('Evaluation requires one completion per question')
    return {q: records[0] for q, records in retriever.score_samples(raw).items()}


def original_bge_records(retriever, part, qids):
    names = {'A': 'A', 'B': 'B', 'zero': 'C_hybrid'}
    originals = {row['qid']: row for row in read_json(SOURCE7.parent / f'{part}_per_question.json')}
    result = {name: {} for name in names}
    for q in qids:
        original = originals[q]
        for name, source in names.items():
            old = original['system'][source]
            row = retriever.score_contexts(q, {k: old[f'ids@{k}'] for k in (10, 6)},
                                           original['subquestions'] if name == 'zero' else (),
                                           original['fallback'] if name == 'zero' else False,
                                           backfilled_by_k={k: old[f'backfilled@{k}'] for k in (10, 6)})
            if any(row[metric] != old[metric] for metric in METRICS if metric != 'Hits@10'):
                raise ValueError(f'Saved baseline metrics changed for question {q}')
            result[name][q] = row
    return result


def measure(systems, comparisons, related=()):
    rows = list(systems.values())
    if any(record.keys() != rows[0].keys() for record in rows[1:]):
        raise ValueError('Evaluation systems have mismatched question IDs')
    scores = {key: summarize(data) for key, data in systems.items()}
    intervals = {f'{a}_vs_{b}': paired_ci(systems[a], systems[b]) for a, b in comparisons}
    by_type = {}
    for kind in sorted({row['type'] for row in rows[0].values()}):
        section = {key: {qid: row for qid, row in data.items() if row['type'] == kind} for key, data in systems.items()}
        by_type[kind] = {f'{a}_vs_{b}': paired_ci(section[a], section[b]) for a, b in comparisons}
    extra = {}
    if related:
        clear = {key: {qid: row for qid, row in data.items() if qid not in related} for key, data in systems.items()}
        extra = {key: summarize(data)['overall'] for key, data in clear.items()}
    return {'scores': scores, 'intervals': intervals, 'intervals_by_type': by_type, 'without_demo_related': extra}


def cells(metric_row):
    return ' | '.join(f'{metric_row[key]:.2%}' for key in METRICS)


def ci_cells(record):
    return ' | '.join(f'{100 * value[0]:+.2f} [{100 * value[1]:+.2f}, {100 * value[2]:+.2f}]'
                      for value in (record[key] for key in METRICS))


def table_metrics():
    return ' | '.join(METRICS)


def report_step1(all_results, split):
    demos = read_json(DEMO_FILE)['examples']
    lines = ['# Step 1 A few-shot decomposition comparison', '',
             f"The comparison uses {len(split['dev'])} eligible dev questions and {len(split['test']):,} test questions. "
             f"The dev evaluation excludes {len(split['excluded_dev'])} demonstration or evidence-set matches. "
             'All four demonstration questions were checked with the fixed BGE pipeline at K=10 before generation.', '',
             '| Example qid | Gold facts before | Gold facts after | Final decomposition |', '|---:|---:|---:|---|']
    for qid, demo in demos.items():
        old, new = demo['initial_hits'], demo['final_hits']
        final = '<br>'.join(demo['final']) if demo['initial'] != demo['final'] else 'As supplied'
        lines.append(f'| {qid} | {sum(old)}/{len(old)} | {sum(new)}/{len(new)} | {final} |')
    lines += ['', 'The few-shot prompt adds four user–assistant examples before the current question. The same generated '
              'texts, filter, fallback, BM25 settings, RRF, selection, and exact-K backfill are used for BGE and e5.', '',
              'The BGE baseline and zero-shot rows were rescored from their saved rankings. The e5 runs rerank the '
              'same chunks with e5-large-v2 using query and passage prefixes and mean pooling.', '',
              f"Test question IDs sharing a demonstration evidence set are {split.get('demo_related_test', [])}. "
              'The main test table keeps the fixed split. The sensitivity table excludes those IDs.', '']
    for backend, partitions in all_results.items():
        lines += [f'## {backend.upper()}', '']
        for part, result in partitions.items():
            lines += [f'### {part.title()}', '',
                      f'| System | Group | {table_metrics()} | Mean sub-questions | Fallback |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for key, scores in result['scores'].items():
                for kind, row in {'Overall': scores['overall'], **scores['by_type']}.items():
                    extras = (f"{row['mean_subquestions']:.2f} | {row['fallback_rate']:.2%}" if key in ('zero', 'few')
                              else '— | —')
                    lines.append(f'| {BASE_NAMES[key]} | {kind} | {cells(row)} | {extras} |')
            lines += ['', 'Paired 95% confidence intervals in percentage points over 10,000 question resamples.', '',
                      f'| Comparison | Group | {table_metrics()} |', '|---|---|---:|---:|---:|---:|']
            for kind, stats in {'Overall': result['intervals'], **result['intervals_by_type']}.items():
                for key, record in stats.items():
                    a, b = key.split('_vs_')
                    lines.append(f'| {BASE_NAMES[a]} vs {BASE_NAMES[b]} | {kind} | {ci_cells(record)} |')
            if result['without_demo_related']:
                lines += ['', 'Sensitivity with the demonstration-related test IDs excluded.', '',
                          f'| System | {table_metrics()} |', '|---|---:|---:|---:|---:|']
                for key, row in result['without_demo_related'].items():
                    lines.append(f'| {BASE_NAMES[key]} | {cells(row)} |')
            lines.append('')
    (STEP1 / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def step1_generation(split):
    for part in ('dev', 'test'):
        run_generation('7b', 'few', split[part], STEP1 / f'{part}_few_generations.json')


def step1_score(split, backend):
    result = {}
    with Retrieval(backend=backend) as retriever:
        for part in ('dev', 'test'):
            print(f'Scoring {backend} {part}', flush=True)
            qids = split[part]
            if backend == 'bge':
                records = original_bge_records(retriever, part, qids)
            else:
                baselines = retriever.score_baselines(qids)
                records = {key: {q: values[key] for q, values in baselines.items()} for key in ('A', 'B')}
                records['zero'] = score_raw(retriever, cached_raw(SOURCE7 / f'{part}_decomp.json', qids))
            records['few'] = score_raw(retriever, cached_raw(STEP1 / f'{part}_few_generations.json', qids))
            write_json(STEP1 / f'{backend}_{part}_rankings.json', records)
            related = split.get('demo_related_test', []) if part == 'test' else ()
            result[part] = measure(records, [('few', 'zero'), ('few', 'B')], related)
    write_json(STEP1 / f'{backend}_metrics.json', result)
    merged = {name: read_json(STEP1 / f'{name}_metrics.json') for name in ('bge', 'e5')
              if (STEP1 / f'{name}_metrics.json').exists()}
    report_step1(merged, split)


def main():
    from hybrid_claims.split_training_preferences import make_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate-few', 'score-bge', 'score-e5'))
    args = parser.parse_args()
    split = make_split()
    if args.command == 'generate-few':
        step1_generation(split)
    else:
        step1_score(split, args.command.removeprefix('score-'))


if __name__ == '__main__':
    main()
