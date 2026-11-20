"""Score and report the frozen MultiHop-RAG DPO cross-validation predictions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.data import gold_facts, is_answerable, load_chunks, load_queries
from common.metrics import all_metrics
from hybrid_claims.llm_generate import DESIGN, write_json
from hybrid_claims.split_training_generation import OUT, demo_messages, read_json
from hybrid_claims.split_training_preferences import EXAMPLES, ensure_samples
from hybrid_claims.split_training_retrieval import Retrieval


CV = OUT / 'cv'
FOLDS = ('fold_1', 'fold_2')
OPTIONAL = 'article_disjoint'
CASES = ('3b_zero', '3b_few')
PROMPTS = {'3b_zero': '3B zero-shot', '3b_few': '3B few-shot'}
METRICS = ('complete@10', 'recall@10', 'Hits@10', 'MRR@10', 'Hits@4')
SYSTEMS = {'whole': 'Whole-question hybrid', 'rule': 'Rule split',
           'untrained': 'Untrained', 'trained': 'DPO'}
SUBSETS = {'all': 'All questions', 'shared_article_test': 'Shares a gold article with train pool',
           'no_shared_article_test': 'Shares no gold article with train pool',
           'shared_pair_article_test': 'Shares a gold article with a DPO pair',
           'no_shared_pair_article_test': 'Shares no gold article with a DPO pair'}
EXPOSURE_SUBSETS = {'train_pool': ('shared_article_test', 'no_shared_article_test'),
                    'dpo_pairs': ('shared_pair_article_test', 'no_shared_pair_article_test')}


def gold_articles(queries):
    return {q['qid']: {fact['url'] for fact in q['evidence_list']} for q in queries if is_answerable(q)}


def article_exposed(qids, other_qids, articles):
    other_articles = set().union(*(articles[q] for q in other_qids))
    return sorted(q for q in qids if articles[q] & other_articles)


def validate_folds(manifest, articles):
    eligible = articles.keys() - set(EXAMPLES)
    folds = manifest['folds']
    for name, fold in folds.items():
        for part in ('train', 'heldout', 'test'):
            ids = fold[part]
            if not ids or len(set(ids)) != len(ids) or not set(ids) <= eligible:
                raise ValueError(f'Invalid {name}/{part} qids')
        train, heldout, test = (set(fold[part]) for part in ('train', 'heldout', 'test'))
        if train & heldout or train & test or heldout & test or train | heldout | test != eligible:
            raise ValueError(f'{name} partitions must cover all eligible qids without overlap')
        article_sets = [{frozenset(articles[q]) for q in ids} for ids in (train, heldout, test)]
        if any(article_sets[i] & article_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f'{name} crosses partitions with identical evidence sets')
        shared = set(article_exposed(test, train, articles))
        for subset, expected in (('shared_article_test', shared), ('no_shared_article_test', test - shared)):
            ids = fold[subset]
            if len(ids) != len(expected) or set(ids) != expected:
                raise ValueError(f'{name} article-overlap flags disagree with the corpus')
        if name == OPTIONAL and article_exposed(test, train | heldout, articles):
            raise ValueError('Optional article-disjoint direction has gold article overlap')
    left, right = (set(folds[name]['test']) for name in FOLDS)
    if left & right or left | right != eligible:
        raise ValueError('The two balanced test folds must predict each eligible qid once')
    return folds


def fixed_rankings(cv, backend, qids):
    ranks = {}
    for partition in ('dev', 'test'):
        if backend == 'bge':
            path = cv.parents[1] / 'llm_decomp_vs_hyde' / f'{partition}_per_question.json'
            rows = read_json(path)
            local = {row['qid']: {name: row['system'][name]['ids@10'] for name in ('A', 'B')} for row in rows}
            if len(local) != len(rows):
                raise ValueError(f'Duplicate qid in {path}')
        else:
            path = cv.parent / 'step1' / f'e5_{partition}_rankings.json'
            rows = read_json(path)
            if rows['A'].keys() != rows['B'].keys():
                raise ValueError(f'Baseline coverage differs in {path}')
            local = {int(q): {name: rows[name][q]['ids@10'] for name in ('A', 'B')} for q in rows['A']}
        if ranks.keys() & local.keys():
            raise ValueError(f'Duplicate archive qids in {path}')
        ranks.update(local)
    if backend == 'e5':
        recovered = {int(q): row for q, row in read_json(cv / 'baselines.json')['recovered_e5_ranks'].items()}
        if ranks.keys() & recovered.keys():
            raise ValueError('Recovered e5 qids collide with the archive')
        ranks.update(recovered)
    return {q: ranks[q] for q in sorted(qids)}


def raw_greedy(path, qids, case, precision, checkpoint=None):
    data = read_json(path)
    settings = data['settings']
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    model, prompt = case.split('_')
    adapter = str(Path(checkpoint).resolve()) if checkpoint is not None else None
    expected = {'model': model, 'prompt': prompt, 'examples': demo_messages() if prompt == 'few' else [],
                'instruction': DESIGN['decomp']['prompt'], 'n': 1, 'temperature': None, 'top_p': None,
                'max_new_tokens': DESIGN['decomp']['max_new_tokens'], 'adapter': adapter}
    expected_qids = set(qids)
    if (data['key'] != key or any(settings[name] != value for name, value in expected.items())
            or (settings['base_precision'] or 'bf16') != precision
            or len(settings['qids']) != len(qids) or set(settings['qids']) != expected_qids
            or data['rows'].keys() != set(map(str, expected_qids))):
        raise ValueError(f'Generation settings do not match the frozen evaluation for {path}')
    raw = {int(q): row['texts'] for q, row in data['rows'].items()}
    ensure_samples(raw, qids, 1)
    return {q: texts[0] for q, texts in raw.items()}


def benchmark_record(qid, ids, queries, chunks):
    if len(ids) != 10 or len(set(ids)) != 10 or any(not 0 <= c < len(chunks) for c in ids):
        raise ValueError(f'Qid {qid} needs exactly 10 unique and valid chunk IDs')
    if max(Counter(chunks[c]['article_id'] for c in ids).values()) > DESIGN['cap']:
        raise ValueError(f'Qid {qid} exceeded the article cap')
    query = queries[qid]
    if not is_answerable(query) or not gold_facts(query):
        raise ValueError(f'Qid {qid} has no gold evidence')
    values = all_metrics([[chunks[c]['text'] for c in ids]], [gold_facts(query)], ks=(10,))
    return {'ids@10': ids, **{metric: values[metric] for metric in METRICS}}


def score_fold(cv, name, fold, case, articles, fixed, retriever):
    folder = cv / name / case
    decision = read_json(folder / 'selection.json')
    selection = decision['selected']
    preferences = read_json(folder / 'pairs.json')
    pair_qids = {pair['qid'] for pair in preferences['pairs']}
    if len(pair_qids) != len(preferences['pairs']) or not pair_qids <= set(fold['train']):
        raise ValueError(f'Preference-pair qids escape the training pool for {name}/{case}')
    if selection is None:
        return {'status': 'no_selected_checkpoint', 'pair_count': len(pair_qids),
                'trials': [{key: trial.get(key) for key in ('status', 'lr', 'precision')}
                           for trial in decision.get('trials', [])]}
    if not pair_qids:
        raise ValueError(f'{name}/{case} selected a checkpoint without preference pairs')

    qids = fold['test']
    precision = selection['precision']
    base_raw = raw_greedy(folder / 'test_base.json', qids, case, precision)
    trained_raw = raw_greedy(folder / 'test_trained.json', qids, case, precision, selection['path'])
    scored = retriever.score_samples({q: [base_raw[q], trained_raw[q]] for q in qids})
    if scored.keys() != set(qids):
        raise ValueError(f'Retrieval coverage differs for {name}/{case}')
    rows = {}
    for q, candidates in scored.items():
        untrained, trained = candidates
        ranks = {'whole': fixed[q]['A'], 'rule': fixed[q]['B'],
                 'untrained': untrained['ids@10'], 'trained': trained['ids@10']}
        records = {key: benchmark_record(q, ids, retriever.queries, retriever.chunks) for key, ids in ranks.items()}
        for key, source in (('untrained', untrained), ('trained', trained)):
            if any(records[key][m] != source[m] for m in ('complete@10', 'recall@10', 'Hits@10')):
                raise ValueError(f'Independent evidence recount changed a retrieval metric for qid {q}')
        rows[str(q)] = records

    pair_shared = article_exposed(qids, pair_qids, articles)
    return {
        'status': 'scored', 'selection': selection, 'pair_count': len(pair_qids),
        'samples_per_train_question': preferences.get('samples_per_train_question'),
        'subsets': {
            'all': sorted(qids),
            'shared_article_test': fold['shared_article_test'],
            'no_shared_article_test': fold['no_shared_article_test'],
            'shared_pair_article_test': pair_shared,
            'no_shared_pair_article_test': sorted(set(qids) - set(pair_shared)),
        },
        'heldout_article_test_qids': article_exposed(qids, fold['heldout'], articles),
        'demo_article_test_qids': article_exposed(qids, EXAMPLES, articles),
        'rows': rows,
    }


def resampled_paired_means(folds, reference, rng, draws):
    parts = [np.array([[row['trained'][m] - row[reference][m] for m in METRICS] for row in rows])
             for rows in folds if rows]
    total = sum(len(x) for x in parts)
    samples = []
    for start in range(0, draws, 250):
        take = min(250, draws - start)
        sums = [x[rng.integers(len(x), size=(take, len(x)))].sum(axis=1) for x in parts]
        samples.append(sum(sums) / total)
    delta = sum(x.sum(axis=0) for x in parts) / total
    return delta, np.concatenate(samples)


def percentile_intervals(delta, samples):
    lower, upper = np.quantile(samples, (0.025, 0.975), axis=0)
    return {metric: [float(d), float(lo), float(hi)] for metric, d, lo, hi in zip(METRICS, delta, lower, upper)}


def paired_bootstrap(folds, reference, *, draws=10_000, seed=13):
    """Resample paired question deltas within test folds, keeping fold weights fixed."""
    if not any(folds):
        return None
    return percentile_intervals(*resampled_paired_means(folds, reference, np.random.default_rng(seed), draws))


def exposure_gain_difference(shared, unshared, *, draws=10_000, seed=13):
    """Subtract the pooled unshared gain from the pooled shared gain on each draw."""
    if not any(shared) or not any(unshared):
        return None
    rng = np.random.default_rng(seed)
    shared_delta, shared_samples = resampled_paired_means(shared, 'untrained', rng, draws)
    unshared_delta, unshared_samples = resampled_paired_means(unshared, 'untrained', rng, draws)
    return percentile_intervals(shared_delta - unshared_delta, shared_samples - unshared_samples)


def summarize_case(results, *, optional=False):
    """Pool question records, preserving pair alignment and the fitted fold identity."""
    if any(fold['status'] != 'scored' for fold in results):
        return None
    seen = set()
    for fold in results:
        if (seen.intersection(fold['rows']) or set(fold['rows']) != set(map(str, fold['subsets']['all']))):
            raise ValueError('Predictions must cover every pooled question exactly once')
        seen.update(fold['rows'])
    out = {}
    by_subset = {subset: [[fold['rows'][str(q)] for q in fold['subsets'][subset]] for fold in results]
                 for subset in SUBSETS}
    for subset, groups in by_subset.items():
        rows = [row for group in groups for row in group]
        if not rows:
            out[subset] = {'n': 0}
            continue
        scores = {name: {m: float(np.mean([row[name][m] for row in rows])) for m in METRICS}
                  for name in SYSTEMS}
        references = ('untrained', 'rule') if subset == 'all' and not optional else ('untrained',)
        out[subset] = {'n': len(rows), 'scores': scores,
                       'paired95': {ref: paired_bootstrap(groups, ref) for ref in references}}
    if not optional:
        out['exposure_gain_differences'] = {
            exposure: {'shared_n': out[shared]['n'], 'unshared_n': out[unshared]['n'],
                       'paired95': exposure_gain_difference(by_subset[shared], by_subset[unshared])}
            for exposure, (shared, unshared) in EXPOSURE_SUBSETS.items()
        }
    return out


def score_backend(cv, backend, device, include_optional):
    queries = load_queries()
    articles = gold_articles(queries)
    manifest = cv / 'folds.json'
    signature = hashlib.sha256(manifest.read_bytes()).hexdigest()
    folds = validate_folds(read_json(manifest), articles)
    names = (*FOLDS, OPTIONAL) if include_optional and OPTIONAL in folds else FOLDS
    qids = set().union(*(folds[name]['test'] for name in names))
    fixed = fixed_rankings(cv, backend, qids)
    path = cv / f'{backend}_cv_metrics.json'
    results = {}
    with Retrieval(backend=backend, device=device, out_dir=cv / 'retrieval') as retriever:
        for name in names:
            results[name] = {}
            cases = ('3b_zero',) if name == OPTIONAL else CASES
            for case in cases:
                folder = cv / name / case
                required = [folder / key for key in ('selection.json', 'pairs.json')]
                if name == OPTIONAL and not all(file.exists() for file in required):
                    results[name][case] = {'status': 'optional_run_unavailable'}
                    continue
                if name == OPTIONAL and read_json(required[0])['selected']:
                    files = [folder / f'test_{role}.json' for role in ('base', 'trained')]
                    if not all(file.exists() and len(read_json(file)['rows']) == len(folds[name]['test'])
                               for file in files):
                        results[name][case] = {'status': 'optional_generation_incomplete'}
                        continue
                print(f'Scoring {backend} {name}/{case}', flush=True)
                result = score_fold(cv, name, folds[name], case, articles, fixed, retriever)
                if result['status'] == 'scored':
                    rankings_path = folder / f'{backend}_cv_test_rankings.json'
                    write_json(rankings_path, result)
                    result = {'status': result['status'], 'rankings': rankings_path.relative_to(cv).as_posix()}
                results[name][case] = result
    write_json(path, {'backend': backend, 'device': device, 'folds_sha256': signature, 'folds': results})
    print(f'Saved {path}', flush=True)


def read_rankings(cv, source):
    return read_json(cv / source['rankings']) if source['status'] == 'scored' else source


def interval_cells(interval):
    return ' | '.join(f'{100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]'
                      for d, lo, hi in (interval[m] for m in METRICS))


def append_score_table(lines, summaries, subsets):
    labels = ' | '.join(METRICS)
    lines += [f'| Prompt | Test questions | n | System | {labels} |',
              '|---|---|---:|---|' + '---:|' * len(METRICS)]
    for case, summary in summaries.items():
        if summary is None:
            lines.append(f'| {PROMPTS[case]} | Result unavailable | — | — |' + ' — |' * len(METRICS))
            continue
        for subset in subsets:
            group = summary[subset]
            for key, scores in group.get('scores', {}).items():
                cells = ' | '.join(f'{scores[m]:.4f}' for m in METRICS)
                lines.append(f'| {PROMPTS[case]} | {SUBSETS[subset]} | {group["n"]:,} | {SYSTEMS[key]} | {cells} |')
            if group['n'] == 0:
                lines.append(f'| {PROMPTS[case]} | {SUBSETS[subset]} | 0 | — |' + ' — |' * len(METRICS))
    lines += ['', f'| Prompt | Test questions | Comparison | {labels} |',
              '|---|---|---|' + '---:|' * len(METRICS)]
    for case, summary in summaries.items():
        if summary is None:
            continue
        for subset in subsets:
            for reference, interval in summary[subset].get('paired95', {}).items():
                cells = interval_cells(interval)
                lines.append(f'| {PROMPTS[case]} | {SUBSETS[subset]} | DPO vs {SYSTEMS[reference]} | {cells} |')
    lines.append('')


def append_exposure_table(lines, results):
    labels = {'train_pool': 'Training pool', 'dpo_pairs': 'Questions that contributed DPO pairs'}
    lines += ['## Difference in DPO gains by article exposure', '',
              'Each cell subtracts the unshared group’s DPO gain over its matched untrained model from the '
              'shared group’s gain. The paired 95% intervals keep counts fixed within each combination of '
              'test fold and exposure status. This is a descriptive comparison. Questions sharing gold '
              'articles can be dependent, and the groups can differ in other ways.', '',
              '| Encoder | Prompt | Exposure measured against | Shared n | Unshared n | ' + ' | '.join(METRICS) + ' |',
              '|---|---|---|---:|---:|' + '---:|' * len(METRICS)]
    for backend, data in results.items():
        for case, summary in data['pooled'].items():
            if summary is None:
                continue
            for exposure, result in summary['exposure_gain_differences'].items():
                interval = result['paired95']
                cells = interval_cells(interval) if interval else ' | '.join(['—'] * len(METRICS))
                lines.append(f'| {backend.upper()} | {PROMPTS[case]} | {labels[exposure]} | '
                             f'{result["shared_n"]:,} | {result["unshared_n"]:,} | {cells} |')
    lines.append('')


def render_report(cv):
    queries = load_queries()
    manifest = cv / 'folds.json'
    signature = hashlib.sha256(manifest.read_bytes()).hexdigest()
    folds = validate_folds(read_json(manifest), gold_articles(queries))
    available = {backend: read_json(cv / f'{backend}_cv_metrics.json') for backend in ('bge', 'e5')}
    if any(run['backend'] != backend or run['folds_sha256'] != signature for backend, run in available.items()):
        raise ValueError('Saved scoring outputs use a different backend or fold manifest')
    statuses = [{fold: {case: item['status'] for case, item in rows.items()}
                 for fold, rows in run['folds'].items()} for run in available.values()]
    if statuses[0] != statuses[1]:
        raise ValueError('BGE and e5 runs disagree on fold/case completion status')
    audit_path = cv / 'fold_audit.json'
    audit = read_json(audit_path)
    nearest = audit['closest_article_disjoint_assignment']['folds']
    n = audit['eligible_questions']
    lines = ['# MultiHop-RAG DPO cross-validation', '',
             f'The two balanced test folds cover **{n:,} answerable questions** once each, excluding the four '
             'fixed demonstrations. Relevance uses the benchmark’s gold-fact matching. Hits@4 reads the first '
             'four chunks of the same top-10 list.', '',
             '## Fold audit', '',
             f'Strict article separation yielded {audit["components"]} connected components. The largest '
             f'contains {nearest[0]["n"]:,} questions ({nearest[0]["n"]/n:.1%}), leaving {nearest[1]["n"]:,} '
             'for the closest article-disjoint assignment. The approved balanced assignment keeps identical '
             'gold-article sets together. Individual gold articles can cross folds. '
             '[Original audit and closest disjoint assignment](fold_audit.json) · [Final fold IDs](folds.json)', '',
             '| Fold | Training pool | Held-out checkpoint selection | Test | Test with a train-pool article | '
             'Test without a train-pool article |', '|---|---:|---:|---:|---:|---:|']
    for name, fold in folds.items():
        lines.append(f'| {name} | {len(fold["train"]):,} | {len(fold["heldout"]):,} | {len(fold["test"]):,} | '
                     f'{len(fold["shared_article_test"]):,} | {len(fold["no_shared_article_test"]):,} |')
    lines += ['', 'The train-pool strata use the same question IDs for both prompts. The pair strata count only '
              'training questions that contributed a DPO pair and can differ by prompt. Held-out and fixed-demo '
              'article exposure are counted independently of the train-pool and pair strata.', '',
              '## Preference pairs and checkpoint selection', '',
              'Both arms used eight draws per training question at temperature 0.8 and top-p 0.95. Every '
              'retained pair has one completion with all gold facts in the BGE hybrid top ten and one '
              'with incomplete coverage. Zero-shot sampling and fitting used BF16. Few-shot sampling and '
              'fitting used NF4 with BF16 compute. The earlier Step 2 few-shot preference draws used BF16. '
              'Compatible archived untrained generations were reused. The NF4 sampling batch limit was '
              'increased after a throughput check, which changes how seeded draws are grouped. '
              'For each fold and prompt, two learning rates were run for three epochs. The selected checkpoint '
              'maximized held-out complete@10, with ties favoring an earlier epoch and then a smaller '
              'learning rate. Only the selected adapter generated the outer-fold predictions.', '',
              '| Fold | Prompt | Pairs | Samples / train question | Precision | LR | Epoch | Held-out complete@10 | '
              'Test with held-out article | Test with demo article |',
              '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    provenance = []
    for fold, cases in available['bge']['folds'].items():
        for case, item in cases.items():
            result = read_rankings(cv, item)
            folder = f'{fold}/{case}'
            if result['status'] != 'scored':
                selected_path = cv / folder / 'selection.json'
                label = result['status'].replace('_', ' ')
                selection_text = f'[{label}]({folder}/selection.json)' if selected_path.exists() else label
                lines.append(f'| {fold} | {PROMPTS[case]} | {result.get("pair_count", "—")} | — | '
                             f'{selection_text} | — | — | — | — | — |')
                continue
            selected = result['selection']
            lines.append(f'| {fold} | {PROMPTS[case]} | {result["pair_count"]:,} | '
                         f'{result["samples_per_train_question"] or "—"} | {selected["precision"]} | '
                         f'{selected["lr"]:g} | {selected["epoch"]} | {selected["complete@10"]:.4f} | '
                         f'{len(result["heldout_article_test_qids"]):,} | {len(result["demo_article_test_qids"]):,} |')
            provenance.append(f'- {fold}, {PROMPTS[case]} [Pairs]({folder}/pairs.json) · '
                              f'[Held-out checkpoint selection]({folder}/selection.json)')
    lines += ['', *provenance, '',
              '## Pooled out-of-fold results', '',
              'Scores weight each test question equally. The paired 95% intervals use 10,000 question resamples '
              'within each test fold and keep each selected adapter fixed. Interval columns show the paired '
              'mean difference and bounds on the 0–100 scale. These intervals describe uncertainty across test '
              'questions for the fitted fold models.', '']
    result_data = {}
    for backend, run in available.items():
        lines += [f'### {backend.upper()}', '']
        main = {case: summarize_case([read_rankings(cv, run['folds'][fold][case]) for fold in FOLDS])
                for case in CASES}
        result_data[backend] = {'pooled': main}
        append_score_table(lines, main, ('all', 'shared_article_test', 'no_shared_article_test'))
        lines += [f'#### {backend.upper()} sensitivity using DPO pair articles', '']
        append_score_table(lines, main, ('shared_pair_article_test', 'no_shared_pair_article_test'))
        if OPTIONAL in run['folds']:
            lines += [f'### {backend.upper()} one-way article-disjoint evaluation', '',
                      'This direction uses a separate model and test subset. Every test question has '
                      'zero train-pool article exposure, leaving one group for this evaluation.', '']
            optional = {case: summarize_case([read_rankings(cv, item)], optional=True)
                        for case, item in run['folds'][OPTIONAL].items()}
            result_data[backend]['article_disjoint'] = optional
            append_score_table(lines, optional, ('all',))
    if OPTIONAL not in available['bge']['folds']:
        lines += ['The optional one-way article-disjoint evaluation has no scored predictions in this report.', '']
    append_exposure_table(lines, result_data)
    lines += ['The independent archive baselines also include the pooled original BF16 3B zero-shot run. '
              '[Archive baseline scores and coverage](baselines.json)', '',
              '[Pooled metrics JSON](pooled_cv_metrics.json) · [BGE fold files](bge_cv_metrics.json) · '
              '[e5 fold files](e5_cv_metrics.json) · [PowerShell run script](run_cv.ps1)', '']
    few_e5 = result_data['e5']['pooled']['3b_few']
    if few_e5 is not None:
        scores = few_e5['all']['scores']
        delta, lower, upper = few_e5['all']['paired95']['untrained']['complete@10']
        shared = few_e5['shared_article_test']['paired95']['untrained']['complete@10'][0]
        clear = few_e5['no_shared_article_test']['paired95']['untrained']['complete@10'][0]
        _, overlap_low, overlap_high = few_e5['exposure_gain_differences']['train_pool']['paired95']['complete@10']
        lines[4:4] = [
            f'With e5, few-shot DPO reached **{scores["trained"]["complete@10"]:.2%} complete@10** '
            f'from {scores["untrained"]["complete@10"]:.2%} before DPO and '
            f'{scores["rule"]["complete@10"]:.2%} for the rule split. The gain over the untrained '
            f'model was {delta*100:+.2f} percentage points (paired 95% CI {lower*100:+.2f} to {upper*100:+.2f}).', '',
            f'The complete@10 gain was {shared*100:+.2f} points with a training-pool article and '
            f'{clear*100:+.2f} points without one. The paired interval for the difference between '
            f'those gains was {overlap_low*100:+.2f} to {overlap_high*100:+.2f} points.', '',
        ]
    write_json(cv / 'pooled_cv_metrics.json', result_data)
    (cv / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-dir', default=CV, type=Path)
    commands = parser.add_subparsers(dest='command', required=True)
    scorer = commands.add_parser('score')
    scorer.add_argument('backend', choices=('bge', 'e5'))
    scorer.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    scorer.add_argument('--include-article-disjoint', action='store_true')
    commands.add_parser('report')
    args = parser.parse_args()
    if args.command == 'score':
        score_backend(args.cv_dir, args.backend, args.device, args.include_article_disjoint)
    else:
        render_report(args.cv_dir)


if __name__ == '__main__':
    main()
