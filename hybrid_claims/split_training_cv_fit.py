"""Sample preferences, select adapters, and generate outer-fold decompositions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import paths
from common.data import load_queries
from hybrid_claims.split_training_dpo import CV_PRECISIONS, LEARNING_RATES, verified_cv_fold
from hybrid_claims.split_training_eval import score_raw
from hybrid_claims.split_training_generation import (DESIGN, OUT, demo_messages, read_json,
                                                    run_generation, user_turn, write_json)
from hybrid_claims.split_training_preferences import TEMPERATURE, TOP_P, choose_pairs, ensure_samples
from hybrid_claims.split_training_retrieval import Retrieval


CV_ROOT = OUT / 'cv'
FOLDS = ('fold_1', 'fold_2', 'article_disjoint')
MODEL = '3b'
SAMPLE_COUNT = 8


def sample_preferences(train, arm, prompt):
    run_generation(MODEL, prompt, train, arm / 'samples8.json', n=SAMPLE_COUNT,
                   temperature=TEMPERATURE, top_p=TOP_P, base_precision=CV_PRECISIONS[prompt])


def score_preferences(train, arm, prompt, queries):
    sample_path = arm / 'samples8.json'
    payload = sample_path.read_bytes()
    saved = json.loads(payload)
    settings = saved['settings']
    history = demo_messages() if prompt == 'few' else []
    expected = {'model': MODEL, 'prompt': prompt, 'examples': history,
                'instruction': DESIGN['decomp']['prompt'], 'qids': sorted(train, key=lambda q: (len(queries[q]['query']), q)),
                'n': SAMPLE_COUNT, 'temperature': TEMPERATURE, 'top_p': TOP_P,
                'max_new_tokens': DESIGN['decomp']['max_new_tokens'], 'adapter': None,
                'base_precision': CV_PRECISIONS[prompt]}
    key = sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if settings != expected or saved['key'] != key:
        raise ValueError('Sampling cache does not match the fixed training questions or generation settings')
    samples = {int(qid): row['texts'] for qid, row in saved['rows'].items()}
    ensure_samples(samples, train, SAMPLE_COUNT)
    sample_hash = sha256(payload).hexdigest()
    scores_path = arm / 'scores8.json'
    if scores_path.exists():
        scored = read_json(scores_path)
        if scored['samples_sha256'] != sample_hash:
            raise ValueError('Existing scores refer to a different sample cache')
        scores = {int(qid): row for qid, row in scored['scores'].items()}
    else:
        with Retrieval(backend='bge', out_dir=CV_ROOT / 'retrieval') as retrieval:
            scores = retrieval.score_samples(samples)
        write_json(scores_path, {'samples_sha256': sample_hash, 'scores': scores})

    def messages(qid):
        return [*history, user_turn(queries[qid]['query'])]

    pairs = choose_pairs(samples, scores, train, messages)
    write_json(arm / 'pairs.json', {
        'model': MODEL, 'prompt': prompt, 'base_precision': CV_PRECISIONS[prompt],
        'samples_per_train_question': SAMPLE_COUNT, 'temperature': TEMPERATURE, 'top_p': TOP_P,
        'train_questions': len(train), 'samples_sha256': sample_hash, 'pairs': pairs,
    })
    print(f'{arm.parent.name} {prompt} produced {len(pairs)} strict pairs from {len(train)} training questions',
          flush=True)


def trial_dir(arm, lr):
    return arm / f'lr_{lr:g}'


def fit_trial(fold_name, arm, prompt, lr, heldout):
    trial = trial_dir(arm, lr)
    trial.mkdir(parents=True, exist_ok=True)
    log_path = trial / 'training.log'
    args = [sys.executable, '-u', '-m', 'hybrid_claims.split_training_dpo', '--model', MODEL,
            '--prompt', prompt, '--pairs', str(arm / 'pairs.json'), '--split-file', str(CV_ROOT / 'folds.json'),
            '--output-dir', str(trial), '--lr', str(lr), '--epochs', '3', '--precision', CV_PRECISIONS[prompt],
            '--cv-fold', fold_name]
    print(f'Training {fold_name} {prompt} at {lr:g}. Logs at {log_path}', flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=paths.ROOT)

    status = read_json(trial / 'status.json')
    validated = {'lr': lr, **status, 'validation': []}
    if status['status'] == 'complete':
        for point in status['checkpoints']:
            checkpoint = trial / point['path']
            raw = run_generation(MODEL, prompt, heldout, trial / f"heldout_{point['step']}.json",
                                 adapter=checkpoint, base_precision=status['precision'])
            ensure_samples(raw, heldout, 1)
            with Retrieval(backend='bge', out_dir=CV_ROOT / 'retrieval') as retrieval:
                records = score_raw(retrieval, raw)
            score = sum(record['complete@10'] for record in records.values()) / len(heldout)
            validated['validation'].append({**point, 'path': str(checkpoint), 'complete@10': score,
                                            'lr': lr, 'precision': status['precision']})
            print(f"{fold_name} {prompt} LR {lr:g} epoch {point['epoch']} held-out complete@10 {score:.2%}",
                  flush=True)
    else:
        print(f"{fold_name} {prompt} at {lr:g} ended with {status['status']}. {status.get('reason', '')}",
              flush=True)
    write_json(trial / 'validation.json', validated)


def fit_and_select(fold_name, arm, prompt, heldout, *, lr=None):
    for rate in LEARNING_RATES if lr is None else (lr,):
        fit_trial(fold_name, arm, prompt, rate, heldout)
    paths = [trial_dir(arm, rate) / 'validation.json' for rate in LEARNING_RATES]
    if not all(path.is_file() for path in paths):
        return
    trials = [read_json(path) for path in paths]
    candidates = [point for trial in trials for point in trial['validation']]
    selected = (max(candidates, key=lambda row: (row['complete@10'], -row['epoch'], -row['lr']))
                if candidates else None)
    write_json(arm / 'selection.json', {'selected': selected, 'trials': trials})


def generate_test(test, arm, prompt, *, only_base=False):
    precision = CV_PRECISIONS[prompt]
    if not only_base:
        selected = read_json(arm / 'selection.json')['selected']
        if selected is None:
            raise ValueError('No checkpoints were eligible for selection')
        checkpoint = Path(selected['path']).resolve()
        if selected['precision'] != precision or not checkpoint.is_relative_to(arm):
            raise ValueError('Selected checkpoint must belong to the matching fold, prompt and precision')
        trained = run_generation(MODEL, prompt, test, arm / 'test_trained.json',
                                 adapter=checkpoint, base_precision=precision)
        ensure_samples(trained, test, 1)
    base = run_generation(MODEL, prompt, test, arm / 'test_base.json', base_precision=precision)
    ensure_samples(base, test, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('sample', 'pairs', 'fit', 'generate', 'generate-base'))
    parser.add_argument('--fold-dir', required=True, type=Path)
    parser.add_argument('--prompt', required=True, choices=CV_PRECISIONS)
    parser.add_argument('--lr', type=float, choices=LEARNING_RATES)
    args = parser.parse_args()
    fold_dir = args.fold_dir.resolve()
    if fold_dir.parent != CV_ROOT.resolve() or fold_dir.name not in FOLDS:
        parser.error('--fold-dir must be a supported fold under outputs/split_training/cv')
    if args.lr is not None and args.command != 'fit':
        parser.error('--lr only applies to fit')
    queries = load_queries()
    fold = verified_cv_fold(read_json(CV_ROOT / 'folds.json'), fold_dir.name, queries)
    arm = fold_dir / f'{MODEL}_{args.prompt}'
    if args.command == 'sample':
        sample_preferences(fold['train'], arm, args.prompt)
    elif args.command == 'pairs':
        score_preferences(fold['train'], arm, args.prompt, queries)
    elif args.command == 'fit':
        fit_and_select(fold_dir.name, arm, args.prompt, fold['heldout'], lr=args.lr)
    else:
        generate_test(fold['test'], arm, args.prompt, only_base=args.command == 'generate-base')


if __name__ == '__main__':
    main()
