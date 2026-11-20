"""Collect preferences, select DPO checkpoints on held-out dev, and score test."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import paths
from hybrid_claims.split_training_eval import (STEP1, cached_raw, cells, ci_cells, measure, score_raw,
                                               table_metrics)
from hybrid_claims.split_training_generation import OUT, read_json, run_generation, write_json
from hybrid_claims.split_training_preferences import cached_zero_shot, collect_preferences, make_split
from hybrid_claims.split_training_retrieval import Retrieval, summarize


STEP2 = OUT / 'step2'
CASES = ('3b_zero', '3b_few', '7b_zero', '7b_few')
LRS = (5e-6, 2e-5)


def diagnostics(cases, split):
    lines = ['# Step 2 Retrieval preference diagnostics', '',
             f"The grouped dev split has {len(split['train'])} train and {len(split['heldout'])} held-out questions.", '',
             '| Model and prompt | Split | Greedy complete@10 | Mean of 8 | Best of 8 | Pairs (8) | Pairs (16) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for case in cases:
        file = OUT / 'preferences' / case / 'diagnostics.json'
        if not file.exists():
            continue
        data = read_json(file)
        for part, row in data['eight'].items():
            final = data['pairs16'] if data['pairs16'] is not None else '—'
            lines.append(f"| {case} | {part} | {row['greedy']:.2%} | {row['mean']:.2%} | {row['best']:.2%} | "
                         f"{data['pairs8']} | {final} |")
    STEP2.mkdir(parents=True, exist_ok=True)
    (STEP2 / 'PREFERENCES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def preferences(cases, split):
    for case in cases:
        model, prompt = case.split('_')
        greedy = None
        if case == '7b_few':
            greedy = cached_raw(STEP1 / 'dev_few_generations.json', split['dev'])
        collect_preferences(model, prompt, greedy=greedy)
        diagnostics(CASES, split)


def trial_directory(case, lr):
    return STEP2 / case / f'lr_{lr:g}'


def train_trial(case, lr):
    model, prompt = case.split('_')
    folder = trial_directory(case, lr)
    folder.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, '-u', str(Path(__file__).with_name('split_training_dpo.py')), '--model', model,
            '--prompt', prompt, '--pairs', str(OUT / 'preferences' / case / 'pairs.json'),
            '--split-file', str(OUT / 'dev_split.json'), '--output-dir', str(folder), '--lr', str(lr)]
    print(f'Training {case} at lr {lr:g}. Logs at {folder / "training.log"}', flush=True)
    with (folder / 'training.log').open('a', encoding='utf-8') as log:
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=paths.ROOT)
    return read_json(folder / 'status.json')


def train_and_select(cases, split):
    for case in cases:
        model, prompt = case.split('_')
        folder = STEP2 / case
        decision_path = folder / 'selection.json'
        if decision_path.exists():
            continue
        trials = []
        for lr in LRS:
            status = train_trial(case, lr)
            trial = {'lr': lr, **status, 'validation': []}
            trials.append(trial)
            if status['status'] != 'complete':
                print(f"{case} lr {lr:g} ended with {status['status']}: {status.get('reason', '')}", flush=True)
                continue
            for point in status['checkpoints']:
                root = trial_directory(case, lr)
                checkpoint = root / point['path']
                raw = run_generation(model, prompt, split['heldout'], root / f"heldout_{point['step']}.json",
                                     adapter=checkpoint, base_precision=status['precision'])
                with Retrieval() as retriever:
                    score = summarize(score_raw(retriever, raw))['overall']['complete@10']
                trial['validation'].append({**point, 'path': str(checkpoint), 'complete@10': score,
                                            'lr': lr, 'precision': status['precision']})
                print(f"{case} lr {lr:g} epoch {point['epoch']} held-out complete@10 {score:.2%}", flush=True)
        candidates = [point for trial in trials for point in trial['validation']]
        selected = max(candidates, key=lambda row: (row['complete@10'], -row['epoch'], -row['lr'])) if candidates else None
        write_json(decision_path, {'selected': selected, 'trials': trials})


def generate_test(cases, split):
    for case in cases:
        model, prompt = case.split('_')
        folder = STEP2 / case
        selected = read_json(folder / 'selection.json')['selected']
        if selected is None:
            continue
        run_generation(model, prompt, split['test'], folder / 'test_trained.json', adapter=selected['path'],
                       base_precision=selected['precision'])
        if model == '3b' and (prompt == 'few' or selected['precision'] == 'nf4'):
            run_generation(model, prompt, split['test'], folder / 'test_base.json', base_precision=selected['precision'])


def test_untrained(case, qids):
    file = STEP2 / case / 'test_base.json'
    if file.exists():
        return cached_raw(file, qids)
    if case == '7b_few':
        return cached_raw(STEP1 / 'test_few_generations.json', qids)
    base = cached_zero_shot(case.split('_')[0], 'test')
    return {q: [base[q]] for q in qids}


def score_test(cases, split, backend):
    qids = split['test']
    related = set(split['demo_related_test']) | set(split['train_related_test'])
    result = {}
    with Retrieval(backend=backend) as retriever:
        baselines = retriever.score_baselines(qids)
        rules = {q: row['B'] for q, row in baselines.items()}
        for case in cases:
            folder = STEP2 / case
            selected = read_json(folder / 'selection.json')['selected']
            if selected is None:
                continue
            print(f'Scoring {case} on {backend}', flush=True)
            base = score_raw(retriever, test_untrained(case, qids))
            trained = score_raw(retriever, cached_raw(folder / 'test_trained.json', qids))
            systems = {'rule': rules, 'untrained': base, 'trained': trained}
            write_json(folder / f'{backend}_test_rankings.json', systems)
            result[case] = measure(systems, [('trained', 'untrained'), ('trained', 'rule')], related)
    write_json(STEP2 / f'{backend}_metrics.json', result)
    report(CASES, split)


def report(cases, split):
    related = set(split['demo_related_test']) | set(split['train_related_test'])
    lines = ['# Step 2 DPO with retrieval feedback', '',
             f"Training and held-out selection use {len(split['train'])} and {len(split['heldout'])} eligible dev questions. "
             f"The original test split has {len(split['test']):,} questions. {len(related)} share an evidence-article set "
             'with a demo or training question across the inherited split. The sensitivity results omit those IDs.', '',
             '[Preference diagnostics before training](PREFERENCES.md)', '',
             '| Model and prompt | LR | Precision | Status | Peak reserved VRAM (GiB) | Selected epoch | Held-out complete@10 |',
             '|---|---:|---|---|---:|---:|---:|']
    for case in cases:
        file = STEP2 / case / 'selection.json'
        if not file.exists():
            continue
        decision = read_json(file)
        point = decision['selected']
        for trial in decision['trials']:
            chosen = point and point['lr'] == trial['lr']
            peak = f"{trial['peak_vram_gib']:.2f}" if 'peak_vram_gib' in trial else '—'
            score = f"{point['complete@10']:.2%}" if chosen else '—'
            epoch = point['epoch'] if chosen else '—'
            lines.append(f"| {case} | {trial['lr']:g} | {trial.get('precision', '—')} | {trial['status']} | "
                         f"{peak} | {epoch} | {score} |")
            if trial.get('reason'):
                reason = trial['reason'].split('. GPU 0', 1)[0] if trial['status'] == 'oom' else trial['reason']
                log = trial_directory(case, trial['lr']).relative_to(STEP2).as_posix() + '/training.log'
                lines += ['', f"{case} at {trial['lr']:g} ended with {reason.rstrip('.')}. [Training log]({log})", '']
    fallback_runs = []
    for case in cases:
        file = STEP2 / case / 'selection.json'
        if file.exists():
            for trial in read_json(file)['trials']:
                if trial.get('precision') == 'nf4' and trial.get('attempts', {}).get('bf16', {}).get('status') == 'oom':
                    fallback_runs.append(f"{case} at `{trial['lr']:g}`")
    if fallback_runs:
        lines += ['', f"BF16 ran out of GPU memory for {', '.join(fallback_runs)}. Those trials switched to 4-bit NF4. "
                  'Their preference samples came from the original untrained BF16 model. The paired test baseline '
                  'uses the precision of the selected adapter.', '']
    for backend in ('bge', 'e5'):
        file = STEP2 / f'{backend}_metrics.json'
        if not file.exists():
            continue
        lines += ['', f'## {backend.upper()} test results', '']
        for case, result in read_json(file).items():
            lines += [f'### {case}', '', f'| System | Group | {table_metrics()} | Mean sub-questions | Fallback |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for system, scores in result['scores'].items():
                for kind, row in {'Overall': scores['overall'], **scores['by_type']}.items():
                    lines.append(f"| {system} | {kind} | {cells(row)} | {row['mean_subquestions']:.2f} | "
                                 f"{row['fallback_rate']:.2%} |")
            lines += ['', 'Paired test differences with 95% confidence intervals in percentage points.', '',
                      f'| Comparison | Group | {table_metrics()} |', '|---|---|---:|---:|---:|---:|']
            for kind, intervals in {'Overall': result['intervals'], **result['intervals_by_type']}.items():
                for key, ci in intervals.items():
                    lines.append(f"| {key.replace('_vs_', ' vs ')} | {kind} | {ci_cells(ci)} |")
            lines += ['', f"Sensitivity complete@10 over {len(split['test']) - len(related):,} questions with different "
                      'evidence-article sets. ' + ', '.join(f"{system} {row['complete@10']:.2%}"
                                                         for system, row in result['without_demo_related'].items()) + '.', '']
    (STEP2 / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preferences', 'train', 'generate-test', 'score-bge', 'score-e5', 'report'))
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    split = make_split()
    cases = (args.case,) if args.case else CASES
    if not all((STEP1 / f'{backend}_metrics.json').is_file() for backend in ('bge', 'e5')):
        raise ValueError('Finish the two Step 1 evaluations first')
    if args.command == 'preferences':
        preferences(cases, split)
    elif args.command == 'train':
        train_and_select(cases, split)
        report(CASES, split)
    elif args.command == 'generate-test':
        generate_test(cases, split)
    elif args.command.startswith('score-'):
        score_test(cases, split, args.command.removeprefix('score-'))
    else:
        report(CASES, split)


if __name__ == '__main__':
    main()
