"""Build retrieval-reward preference pairs from the fixed MultiHop-RAG dev split."""

import argparse
from collections import Counter
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import paths
from common.data import load_queries
from hybrid_claims.llm_compare import read_json
from hybrid_claims.llm_generate import CACHE as OLD_GENERATIONS, DESIGN, partition, write_json


OUT = Path(paths.OUT) / 'split_training'
EXAMPLES = (1006, 164, 654, 1966)
MODELS = ('3b', '7b')
PROMPTS = ('zero', 'few')
TEMPERATURE = 0.8
TOP_P = 0.95
MIN_PAIRS = 200


def article_set(row):
    """Use the article titles that identify groups in the existing benchmark split."""
    return tuple(sorted({evidence['title'] for evidence in row['evidence_list']}))


def make_split(queries=None, out_dir=OUT):
    from sklearn.model_selection import StratifiedGroupKFold

    queries = load_queries() if queries is None else queries
    dev, test = (partition(queries, split) for split in ('dev', 'test'))
    if (len(dev), len(test)) != (681, 1574) or not set(EXAMPLES).issubset(dev):
        raise ValueError('Unexpected fixed benchmark partitions or demonstration IDs')

    article_sets = {qid: article_set(queries[qid]) for qid in dev}
    demo_groups = {article_sets[qid] for qid in EXAMPLES}
    excluded = [qid for qid in dev if article_sets[qid] in demo_groups]
    remaining = [qid for qid in dev if article_sets[qid] not in demo_groups]
    labels = [queries[qid]['question_type'] for qid in remaining]
    group_ids = {key: i for i, key in enumerate(sorted({article_sets[qid] for qid in remaining}))}
    groups = [group_ids[article_sets[qid]] for qid in remaining]
    targets = {kind: count / 4 for kind, count in Counter(labels).items()}
    stratifier = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=DESIGN['split_seed'])

    def distance(indexes):
        counts = Counter(labels[i] for i in indexes)
        return sum(((counts[kind] - target) / target) ** 2 for kind, target in targets.items())

    # Use only dev group membership and labels to choose the fold nearest 25% of each type.
    fold, (train_idx, heldout_idx) = min(
        enumerate(stratifier.split(remaining, labels, groups)), key=lambda row: distance(row[1][1]),
    )
    train = sorted(remaining[i] for i in train_idx)
    heldout = sorted(remaining[i] for i in heldout_idx)
    train_groups = {article_sets[qid] for qid in train}
    if train_groups.intersection(article_sets[qid] for qid in heldout) or sorted(train + heldout) != remaining:
        raise ValueError('Train and held-out must partition dev without splitting an article group')

    test_groups = {qid: article_set(queries[qid]) for qid in test}
    manifest = {
        'seed': DESIGN['split_seed'], 'fold': fold, 'examples': list(EXAMPLES), 'excluded_dev': excluded,
        'dev': remaining, 'train': train, 'heldout': heldout, 'test': test,
        # The inherited benchmark split is kept. These IDs document overlap for reporting only.
        'demo_related_test': [qid for qid in test if test_groups[qid] in demo_groups],
        'train_related_test': [qid for qid in test if test_groups[qid] in train_groups],
        'dev_by_type': {split: dict(Counter(queries[qid]['question_type'] for qid in ids))
                        for split, ids in (('train', train), ('heldout', heldout))},
    }
    file = Path(out_dir) / 'dev_split.json'
    if file.exists():
        if read_json(file) != manifest:
            raise ValueError('Saved dev split differs from the deterministic experiment split')
    else:
        write_json(file, manifest)
    return manifest


def cached_zero_shot(model, partition_name):
    """Read original 3B or 7B greedy outputs, preserving the raw completion text."""
    if model not in MODELS or partition_name not in ('dev', 'test'):
        raise ValueError(f'Unknown original generation {model}/{partition_name}')
    if model == '3b':
        cache = OLD_GENERATIONS
    else:
        cache = Path(paths.OUT) / 'llm_decomp_bigger' / 'qwen2.5-7b-instruct' / 'generation'
    data = read_json(cache / f'{partition_name}_decomp.json')
    rows = {int(qid): row['texts'][0] for qid, row in data['rows'].items()}
    if set(rows) != set(partition(load_queries(), partition_name)):
        raise ValueError(f'Incomplete original generation cache for {model}/{partition_name}')
    return rows


def ensure_samples(samples, qids, count):
    if set(samples) != set(qids) or any(len(texts) != count for texts in samples.values()):
        raise ValueError(f'Expected {count} completion(s) for each requested dev question')
    if any(not isinstance(text, str) for texts in samples.values() for text in texts):
        raise ValueError('Every sampled completion must be raw text')


def choose_pairs(samples, scores, train_qids, messages):
    """Use exact full recall and lowest recall, preserving sampled text and order."""
    pairs = []
    for qid in train_qids:
        rows = scores[qid]
        if not rows or len(rows) != len(samples[qid]):
            raise ValueError(f'Mismatched samples and scores for dev question {qid}')
        complete = [i for i, row in enumerate(rows) if row['recall@10'] == 1.0]
        incomplete = [i for i, row in enumerate(rows) if row['recall@10'] < 1.0]
        if not complete or not incomplete:
            continue
        chosen = min(complete, key=lambda i: len(rows[i]['subquestions']))
        rejected = min(incomplete, key=lambda i: rows[i]['recall@10'])
        pairs.append({
            'qid': qid, 'prompt': messages(qid), 'chosen': samples[qid][chosen], 'rejected': samples[qid][rejected],
            'chosen_recall': rows[chosen]['recall@10'], 'rejected_recall': rows[rejected]['recall@10'],
            'chosen_subquestions': len(rows[chosen]['subquestions']),
            'rejected_subquestions': len(rows[rejected]['subquestions']),
            'chosen_index': chosen, 'rejected_index': rejected,
        })
    return pairs


def diagnostic_row(qids, greedy, samples):
    size = len(qids)
    return {
        'n': size,
        'greedy': sum(greedy[qid][0]['complete@10'] for qid in qids) / size,
        'mean': sum(sum(row['complete@10'] for row in samples[qid]) / len(samples[qid]) for qid in qids) / size,
        'best': sum(any(row['complete@10'] for row in samples[qid]) for qid in qids) / size,
    }


def write_diagnostics(out_dir, data):
    write_json(out_dir / 'diagnostics.json', data)
    lines = [f"# {data['model'].upper()} {data['prompt']} prompt pre-training diagnostics", '',
             'All retrieval scores use the BGE hybrid decomposition pipeline and complete@10.', '',
             '| Split | Questions | Greedy | Mean of 8 | Best of 8 |', '|---|---:|---:|---:|---:|']
    for key, row in data['eight'].items():
        lines.append(f"| {key} | {row['n']} | {row['greedy']:.2%} | {row['mean']:.2%} | {row['best']:.2%} |")
    lines += ['', f"The initial 8 samples per training question yielded **{data['pairs8']} strict pairs**."]
    if data.get('pairs16') is not None:
        row = data['train16']
        lines.append(f"The run resampled 16 completions per train question and yielded **{data['pairs16']} pairs**. "
                     f"Mean-of-16 complete@10 was {row['mean']:.2%} and best-of-16 was {row['best']:.2%}.")
        if data.get('samples16_prefix_cache'):
            lines.append('The 16-draw pass reused the shared token prefix on GPU. It has a separate generation cache '
                         'and uses the same prompt, temperature, top-p, and token limit.')
    lines += ['', 'Best-of-N measures whether at least one sampled split fetched every gold fact. '
              'Ties use generation order. The chosen sample has the fewest effective sub-questions, '
              'including rule-fallback queries when the line filter rejected every generated line.', '']
    (out_dir / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def collect_preferences(model, prompt, *, out_dir=OUT, greedy=None):
    from hybrid_claims.split_training_generation import build_messages, run_generation
    from hybrid_claims.split_training_retrieval import Retrieval

    if model not in MODELS or prompt not in PROMPTS:
        raise ValueError(f'Unknown model/prompt {model}/{prompt}')
    out_dir = Path(out_dir)
    manifest = make_split(out_dir=out_dir)
    work = out_dir / 'preferences' / f'{model}_{prompt}'
    work.mkdir(parents=True, exist_ok=True)
    qids = manifest['dev']
    if greedy is None:
        if prompt == 'zero':
            original = cached_zero_shot(model, 'dev')
            greedy = {qid: [original[qid]] for qid in qids}
        else:
            greedy = run_generation(model, prompt, qids, work / 'greedy.json')
    ensure_samples(greedy, qids, 1)
    samples8 = run_generation(model, prompt, qids, work / 'samples8.json',
                              n=8, temperature=TEMPERATURE, top_p=TOP_P)
    ensure_samples(samples8, qids, 8)

    with Retrieval(backend='bge', device='cuda', out_dir=out_dir / 'retrieval') as retrieval:
        scored = retrieval.score_samples({qid: [*greedy[qid], *samples8[qid]] for qid in qids})
    greedy_scores = {qid: rows[:1] for qid, rows in scored.items()}
    scores8 = {qid: rows[1:] for qid, rows in scored.items()}
    queries = load_queries()

    def messages(qid):
        return build_messages(prompt, queries[qid]['query'])

    pairs = choose_pairs(samples8, scores8, manifest['train'], messages)
    data = {
        'model': model, 'prompt': prompt, 'temperature': TEMPERATURE, 'top_p': TOP_P,
        'pairs8': len(pairs), 'pairs16': None,
        'eight': {key: diagnostic_row(manifest[key], greedy_scores, scores8) for key in ('train', 'heldout')},
    }
    write_json(work / 'scores8.json', {'greedy': greedy_scores, 'samples': scores8})
    write_diagnostics(work, data)
    print(f'{model}/{prompt}: {len(pairs)} strict pairs from 8 samples per train question', flush=True)

    count = 8
    if len(pairs) < MIN_PAIRS:
        count = 16
        train_qids = manifest['train']
        prefix_sample = prompt == 'few' and os.environ.get('SPLIT_TRAINING_CACHE_PREFIX') == '1'
        cache16 = 'samples16_prefix.json' if prefix_sample else 'samples16.json'
        samples16 = run_generation(model, prompt, train_qids, work / cache16,
                                   n=16, temperature=TEMPERATURE, top_p=TOP_P)
        data['samples16_prefix_cache'] = prefix_sample
        ensure_samples(samples16, train_qids, 16)
        with Retrieval(backend='bge', device='cuda', out_dir=out_dir / 'retrieval') as retrieval:
            scores16 = retrieval.score_samples(samples16)
        pairs = choose_pairs(samples16, scores16, train_qids, messages)
        write_json(work / 'scores16.json', scores16)
        data.update(pairs16=len(pairs), train16=diagnostic_row(train_qids, greedy_scores, scores16))
        print(f'{model}/{prompt}: resampling with 16 splits yielded {len(pairs)} strict pairs', flush=True)

    write_diagnostics(work, data)
    write_json(work / 'pairs.json', {'model': model, 'prompt': prompt, 'samples_per_train_question': count, 'pairs': pairs})
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--prompt', choices=PROMPTS)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    parser.add_argument('--split-only', action='store_true')
    args = parser.parse_args()
    if args.split_only:
        split = make_split(out_dir=args.out_dir)
        print(f"Dev exclusions {len(split['excluded_dev'])}, train {len(split['train'])}, held-out {len(split['heldout'])}")
        return
    if args.model is None or args.prompt is None:
        parser.error('Give --model and --prompt, or use --split-only')
    collect_preferences(args.model, args.prompt, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
