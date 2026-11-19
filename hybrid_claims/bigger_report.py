"""Recount saved MultiHop-RAG runs and compare the dev-selected variants."""

import argparse
from collections import Counter
from math import isclose
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import paths
from common.data import load_chunks, load_queries, outlet_names
from hybrid_claims import run
from hybrid_claims.llm_compare import METRICS, TYPES, read_json
from hybrid_claims.llm_generate import DESIGN, OUT as OLD_OUTPUT, partition, write_json
from hybrid_claims.recount_complete import REMOVE, matching_ranks


MODELS = {
    '3B': ('llm_decomp_vs_hyde', 'Qwen/Qwen2.5-3B-Instruct', 'bf16'),
    '7B': ('qwen2.5-7b-instruct', 'Qwen/Qwen2.5-7B-Instruct', '4-bit NF4 with bf16 compute'),
    '9B': ('qwen3.5-9b', 'Qwen/Qwen3.5-9B', '4-bit NF4 with bf16 compute'),
}
FAMILIES = {'decomp': ('C', 'C_hybrid'), 'hyde': ('D', 'D_bm25')}
VARIANTS = {'C': 'BM25', 'C_hybrid': 'hybrid', 'D': 'dense', 'D_bm25': 'dense + BM25'}
COLUMN_NAMES = ('Complete@10', 'Complete@6', 'Recall@10')


def read_optional(path, default):
    return read_json(path) if path.is_file() else default


def average(rows, key):
    return {metric: sum(row['system'][key][metric] for row in rows) / len(rows) for metric in METRICS}


def independent_check(rows, qids, queries, chunks, normalized, base=None):
    by_qid = {row['qid']: row for row in rows}
    if set(by_qid) != set(qids) or len(by_qid) != len(rows):
        raise ValueError('Saved scores must use exactly the fixed answerable question IDs')
    systems = set(rows[0]['system'])
    for qid, row in by_qid.items():
        if set(row['system']) != systems or row['type'] != queries[qid]['question_type']:
            raise ValueError(f'Inconsistent system or question type in row {qid}')
        facts = [evidence['fact'] for evidence in queries[qid]['evidence_list']]
        for key, scores in row['system'].items():
            for k in (10, 6):
                ids = scores[f'ids@{k}']
                if (len(ids) != k or len(set(ids)) != k
                    or any(type(cid) is not int or cid < 0 or cid >= len(chunks) for cid in ids)):
                    raise ValueError(f'Invalid {key} top-{k} chunk IDs for question {qid}')
                articles = Counter(chunks[cid]['article_id'] for cid in ids)
                if max(articles.values()) > DESIGN['cap']:
                    raise ValueError(f'Article cap failed for {key}, question {qid}')
                texts = [normalized[cid] for cid in ids]
                count = sum(bool(matching_ranks(fact, texts)) for fact in facts)
                if scores[f'complete@{k}'] != int(count == len(facts)) or not isclose(
                    scores[f'recall@{k}'], count / len(facts)
                ):
                    raise ValueError(f'Independent coverage disagrees for {key}, question {qid}')
                if base is not None and key in ('A', 'B') and ids != base[qid]['system'][key][f'ids@{k}']:
                    raise ValueError(f'Baseline rankings changed for {key}, question {qid}')
    return by_qid


def dev_choices(rows, saved):
    if not rows:
        return {}
    keys = set(rows[0]['system'])
    selected = {}
    for family, choices in FAMILIES.items():
        if not set(choices) <= keys:
            if keys.intersection(choices):
                raise ValueError(f'Dev scoring is missing a {family} variant')
            continue
        selected[family] = max(choices, key=lambda key: average(rows, key)['complete@10'])
        recorded = saved.get(family)
        if recorded is not None and recorded != selected[family]:
            raise ValueError(f'Saved {family} selection disagrees with the dev scores')
    return selected


def paired_ci(rows, key, reference, baseline_key):
    differences = np.array([[row['system'][key][metric] - reference[row['qid']]['system'][baseline_key][metric]
                             for metric in METRICS] for row in rows])
    rng = np.random.default_rng(13)
    means = [differences[rng.integers(len(rows), size=(250, len(rows)))].mean(axis=1) for _ in range(40)]
    bounds = np.quantile(np.concatenate(means), (0.025, 0.975), axis=0)
    return {metric: [float(differences[:, i].mean()), float(bounds[0, i]), float(bounds[1, i])]
            for i, metric in enumerate(METRICS)}


def cost(rows, family):
    generation = [row['llm'][family] for row in rows if family in row['llm']]
    if not generation:
        return {}
    if len(generation) != len(rows):
        raise ValueError(f'Incomplete generation costs for {family}')
    result = {metric: sum(row[metric] for row in generation) / len(rows) for metric in ('seconds', 'tokens')}
    if family == 'decomp':
        result['fallback_rate'] = sum(row['fallback'] for row in rows) / len(rows)
        result['mean_subquestions'] = sum(len(row['subquestions']) for row in rows) / len(rows)
    return result


def collect(output):
    queries, chunks = load_queries(), load_chunks()
    normalized = [chunk['text'].translate(REMOVE) for chunk in chunks]
    qids = {split: partition(queries, split) for split in ('dev', 'test')}
    models = {}
    for size, (directory, model, quantization) in MODELS.items():
        folder = OLD_OUTPUT if size == '3B' else output / directory
        status = read_optional(folder / 'status.json', {})
        saved = read_optional(folder / 'dev_decisions.json', {}).get('selected', {})
        rows = {split: read_optional(folder / f'{split}_per_question.json', []) for split in qids}
        if size == '3B' and not all(rows.values()):
            raise FileNotFoundError('The original 3B dev and test rows are required')
        indexed = {}
        for split, split_rows in rows.items():
            if split_rows:
                baseline = models['3B']['indexed'][split] if size != '3B' else None
                indexed[split] = independent_check(split_rows, qids[split], queries, chunks, normalized, baseline)
        choices = dev_choices(rows['dev'], saved)
        if rows['test']:
            for family, variants in FAMILIES.items():
                present = set(variants).intersection(rows['test'][0]['system'])
                if present and (family not in choices or set(variants) != present):
                    raise ValueError(f'{model} test results need both {family} variants and a dev choice')
        models[size] = {
            'model': status.get('model', model), 'quantization': status.get('quantization', quantization),
            'status': status, 'rows': rows, 'indexed': indexed, 'choices': choices,
        }

    baseline = models['3B']
    table = [('A. Whole-question hybrid', baseline, 'A'), ('B. Rule split', baseline, 'B')]
    for family, title in (('decomp', 'C. Decomposition'), ('hyde', 'D. HyDE')):
        for size, model in models.items():
            choice = model['choices'].get(family)
            label = f'{title}, {size}' + (f' ({VARIANTS[choice]})' if choice else '')
            table.append((label, model, choice))

    selected = {}
    for label, model, choice in table:
        selected[label] = {}
        for split, rows in model['rows'].items():
            if choice and rows and choice in rows[0]['system']:
                groups = {kind: [row for row in rows if row['type'] == kind] for kind in TYPES}
                selected[label][split] = {'overall': average(rows, choice),
                                          **{kind: average(group, choice) for kind, group in groups.items()}}

    comparisons = {}
    for size in ('7B', '9B'):
        model = models[size]
        choice, rows = model['choices'].get('decomp'), model['rows']['test']
        if choice and rows and choice in rows[0]['system']:
            comparisons[size] = {name: paired_ci(rows, choice, baseline['indexed']['test'], key)
                                 for name, key in (('Rule split', 'B'), ('3B decomposition', baseline['choices']['decomp']))}

    variants = {}
    for size, model in models.items():
        variants[size] = {split: {key: average(rows, key) for key in VARIANTS if key in rows[0]['system']}
                          for split, rows in model['rows'].items() if rows}

    generation = {}
    for size, model in models.items():
        generation[size] = {split: {family: cost(rows, family) for family in FAMILIES}
                            for split, rows in model['rows'].items() if rows}

    outlets = sorted(outlet_names(chunks), key=len, reverse=True)
    examples = []
    for qid in random.Random(20261119).sample(qids['dev'], 10):
        question = queries[qid]['query']
        systems = {'Rule split': run.claims(question, outlets)}
        for size, model in models.items():
            row = model['indexed'].get('dev', {}).get(qid)
            systems[size] = (None if row is None else {'subquestions': row['subquestions'], 'fallback': row['fallback']})
        examples.append({'qid': qid, 'question': question, 'systems': systems})

    return {'n': {split: len(ids) for split, ids in qids.items()}, 'scores': selected, 'variants': variants,
            'comparisons': comparisons, 'generation': generation, 'examples': examples,
            'models': {size: {key: model[key] for key in ('model', 'quantization', 'status', 'choices')}
                       for size, model in models.items()}}


def cell(text):
    return str(text).replace('|', r'\|').replace('\r', '').replace('\n', ' ')


def report(data):
    lines = ['# Larger models for MultiHop-RAG decomposition and HyDE', '',
             f"The comparison uses {data['n']['dev']:,} dev questions and {data['n']['test']:,} test questions "
             'from seed 13. The original BGE+BM25 setup, two chunks per article, exact-K backfill, prompts, '
             'decoding settings and sub-question filter define the protocol. Model precision and any runtime '
             'batch-size constraints are listed below. The larger runs use an NVIDIA RTX 4070 Laptop GPU with 8 GB VRAM.', '',
             '## Run status', '',
             '| Model | Quantization | Decomposition | HyDE |', '|---|---|---|---|']
    for size, model in data['models'].items():
        statuses = []
        for family in FAMILIES:
            stage = model['status'].get(family, {})
            evaluated = family in model['choices'] and model['choices'][family] in data['variants'][size].get('test', {})
            status = stage.get('status') or ('Evaluated' if evaluated else 'Pending')
            reason = stage.get('reason')
            statuses.append(cell(f'{status} ({reason})' if reason else status))
        lines.append(f"| {cell(model['model'])} | {cell(model['quantization'])} | {' | '.join(statuses)} |")

    def metric_cells(metrics):
        return ' | '.join(f'{metrics[key]:.2%}' if key in metrics else '—' for key in METRICS)

    lines += ['', '## Test coverage', '',
              'Decomposition and HyDE search variants are selected independently by dev complete@10. '
              'The BM25 decomposition variant and dense-only HyDE win ties. Complete coverage requires every gold fact.', '']
    for key, title in {'overall': 'Overall', **TYPES}.items():
        lines += [f'### {title}', '', '| System | ' + ' | '.join(COLUMN_NAMES) + ' |', '|---|---:|---:|---:|']
        for label, partitions in data['scores'].items():
            metrics = partitions.get('test', {}).get(key, {})
            lines.append(f'| {label} | {metric_cells(metrics)} |')
        lines.append('')

    lines += ['## Dev selection', '', '| Model | Family | Retrieval variant | Dev complete@10 | Selected |',
              '|---|---|---|---:|---|']
    for size, model in data['models'].items():
        for family, keys in FAMILIES.items():
            for key in keys:
                metrics = data['variants'][size].get('dev', {}).get(key)
                if metrics is not None:
                    selection = 'Yes' if model['choices'].get(family) == key else '—'
                    lines.append(f"| {size} | {family} | {VARIANTS[key]} | {metrics['complete@10']:.2%} | {selection} |")

    lines += ['', '## Paired test differences for decomposition', '',
              'Point estimates and 95% intervals are in percentage points. Each interval resamples the same test '
              'questions for both systems, with 10,000 draws and seed 13.', '',
              '| Model | Reference | ' + ' | '.join(COLUMN_NAMES) + ' |', '|---|---|---:|---:|---:|']
    for size in ('7B', '9B'):
        rows = data['comparisons'].get(size, {})
        for label in ('Rule split', '3B decomposition'):
            baseline = rows.get(label, {})
            cells = []
            for metric in METRICS:
                if metric in baseline:
                    delta, low, high = (value * 100 for value in baseline[metric])
                    cells.append(f'{delta:+.2f} [{low:+.2f}, {high:+.2f}]')
                else:
                    cells.append('—')
            lines.append(f"| {size} | {label} | {' | '.join(cells)} |")

    lines += ['', '## Generation cost', '',
              'Seconds per question use measured generation batch time divided across the questions in that batch. '
              'The historical 3B run recorded time and token counts. Peak VRAM is shown where a run recorded it. '
              'The 7B decomposition resumed after a memory stall with 881 questions already cached. Its VRAM peak '
              'covers the resumed process, and its timing averages the saved successful batches from both processes.', '',
              '| Model | Family | Split | Seconds/question | Generated tokens/question | Fallback | '
              'Mean sub-questions | Peak VRAM (GiB) |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    measurements = []
    for size, model in data['models'].items():
        for family in FAMILIES:
            stage = model['status'].get(family, {})
            peak = stage.get('peak_vram_gib')
            peak_text = '—' if peak is None else f'{peak:.2f}'
            if stage.get('vram_measurement'):
                measurements.append(f"- {size} {family} VRAM measurement — {stage['vram_measurement']}")
            if batch := stage.get('batch_sizes', stage.get('batch_size')):
                measurements.append(f'- {size} {family} generation batch size — {batch} questions')
            if checked := stage.get('continuations_checked_for_thinking'):
                measurements.append(f'- {size} {family} — {checked:,} saved continuations passed the thinking-tag check')
            if allocator := stage.get('allocator'):
                measurements.append(f'- {size} {family} CUDA allocator — `{allocator}`')
            for split in ('dev', 'test'):
                values = data['generation'][size].get(split, {}).get(family, {})
                seconds = f"{values['seconds']:.3f}" if values else '—'
                tokens = f"{values['tokens']:.1f}" if values else '—'
                fallback = f"{values['fallback_rate']:.2%}" if 'fallback_rate' in values else '—'
                count = f"{values['mean_subquestions']:.2f}" if 'mean_subquestions' in values else '—'
                lines.append(f'| {size} | {family} | {split} | {seconds} | {tokens} | {fallback} | {count} | {peak_text} |')
    if measurements:
        lines.extend(['', *measurements])

    lines += ['', '## Ten random dev examples', '',
              'A fixed seed of 20261119 selects question IDs once for all models. Each cell shows the text '
              'passed to retrieval, including rule claims when all generated lines failed the filter.', '']
    for example in data['examples']:
        systems = example['systems']
        lines += [f"### Question {example['qid']}", '', f"> {cell(example['question'])}", '',
                  '| Rule split | 3B | 7B | 9B |', '|---|---|---|---|']

        def list_cell(items):
            return '<br>'.join(f'{i}. {cell(text)}' for i, text in enumerate(items, 1))

        columns = [list_cell(systems['Rule split'])]
        for size in MODELS:
            entry = systems[size]
            if entry is None:
                columns.append('—')
            else:
                prefix = '**Rule fallback**<br>' if entry['fallback'] else ''
                columns.append(prefix + list_cell(entry['subquestions']))
        lines += ['| ' + ' | '.join(columns) + ' |', '']

    lines += ['[Numeric results](results.json)', '']
    baseline = data['scores']['B. Rule split']['test']['overall']['complete@10']
    scored = {label: split['test']['overall']['complete@10'] for label, split in data['scores'].items()
              if label.startswith(('C.', 'D.')) and 'test' in split}
    best = max(scored, key=scored.get)
    if scored[best] > baseline:
        lines.append(f'Yes, {best} beat the rule split ({baseline:.2%}) with {scored[best]:.2%} test complete@10.')
    else:
        lines.append(f'Among the evaluated models, the rule split leads at {baseline:.2%} test complete@10.')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(paths.OUT) / 'llm_decomp_bigger')
    args = parser.parse_args()
    data = collect(args.output)
    write_json(args.output / 'results.json', data)
    (args.output / 'REPORT.md').write_text(report(data), encoding='utf-8')
    print(args.output / 'REPORT.md')


if __name__ == '__main__':
    main()
