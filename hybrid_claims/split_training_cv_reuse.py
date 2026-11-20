"""Seed CV generation caches with matching, untrained Qwen completions."""

import argparse
import hashlib
import json
from pathlib import Path

from common import paths
from common.data import load_queries
from hybrid_claims.llm_generate import CACHE, DESIGN, OUT as ORIGINAL_LLM_OUT
from hybrid_claims.split_training_generation import demo_messages, read_json, write_json
from hybrid_claims.split_training_preferences import TEMPERATURE, TOP_P


OUT = Path(paths.OUT) / 'split_training'
CV = OUT / 'cv'


def copy_rows(sources, count, prompt, precision):
    rows = {}
    for path in sources:
        archive = read_json(path)
        if 'settings' in archive:
            settings = archive['settings']
            expected_precision = (None, 'bf16') if precision == 'bf16' else ('nf4',)
            expected_history = demo_messages() if prompt == 'few' else []
            if (settings['model'] != '3b' or settings['prompt'] != prompt or settings['n'] != count
                    or settings['adapter'] is not None or settings['base_precision'] not in expected_precision
                    or settings['temperature'] != (TEMPERATURE if count > 1 else None)
                    or settings['top_p'] != (TOP_P if count > 1 else None)
                    or settings['max_new_tokens'] != 128 or settings['examples'] != expected_history
                    or settings['instruction'] != DESIGN['decomp']['prompt']):
                raise ValueError(f'Archived generation settings do not match CV at {path}')
        else:
            signature = hashlib.sha256((ORIGINAL_LLM_OUT / 'design.json').read_bytes()).hexdigest()
            if count != 1 or prompt != 'zero' or precision != 'bf16' or archive['design_sha256'] != signature:
                raise ValueError(f'Archived design does not match CV at {path}')
            if read_json(ORIGINAL_LLM_OUT / 'design.json') != DESIGN:
                raise ValueError('Archived decomposition design changed')
        for qid, row in archive['rows'].items():
            texts = row['texts']
            if qid in rows or len(texts) != count or any(not isinstance(text, str) for text in texts):
                raise ValueError(f'Invalid archived completions for {qid} in {path}')
            rows[qid] = row
    return rows


def seed_cache(target, qids, queries, source_paths, pool, count, prompt, precision):
    if target.exists():
        return
    order = sorted(qids, key=lambda qid: (len(queries[qid]['query']), qid))
    settings = {
        'model': '3b', 'prompt': prompt, 'examples': demo_messages() if prompt == 'few' else [],
        'instruction': DESIGN['decomp']['prompt'], 'qids': order, 'n': count,
        'temperature': TEMPERATURE if count > 1 else None, 'top_p': TOP_P if count > 1 else None,
        'max_new_tokens': 128, 'adapter': None, 'base_precision': precision,
    }
    rows = {str(qid): pool[str(qid)] for qid in order if str(qid) in pool}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    write_json(target, {'key': key, 'settings': settings, 'rows': rows,
                       'reused_sources': [str(path.relative_to(paths.ROOT)) for path in source_paths],
                       'reused_qids': sorted(map(int, rows))})
    print(f'{target.relative_to(CV)}: {len(rows)}/{len(qids)} matched archived completions', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merge-disjoint-samples', action='store_true')
    args = parser.parse_args()
    queries = load_queries()
    folds = read_json(CV / 'folds.json')['folds']
    banks = [
        ('3b_zero', 'samples8.json', 'train', 8, 'zero', 'bf16',
         [OUT / 'preferences' / '3b_zero' / 'samples8.json']),
        ('3b_zero', 'test_base.json', 'test', 1, 'zero', 'bf16',
         [CACHE / f'{split}_decomp.json' for split in ('dev', 'test')]),
        ('3b_few', 'test_base.json', 'test', 1, 'few', 'nf4',
         [OUT / 'step2' / '3b_few' / 'test_base.json']),
    ]
    for arm, filename, part, count, prompt, precision, source in banks:
        pool = copy_rows(source, count, prompt, precision)
        for fold, split in folds.items():
            seed_cache(CV / fold / arm / filename, split[part], queries, source, pool, count, prompt, precision)

    if args.merge_disjoint_samples:
        source_paths = [CV / fold / '3b_zero' / 'samples8.json' for fold in ('fold_1', 'fold_2')]
        pool = copy_rows(source_paths, 8, 'zero', 'bf16')
        if set(map(int, pool)) != set(folds['fold_1']['train']) | set(folds['fold_2']['train']):
            raise ValueError('Optional sample reuse requires both complete main zero-shot sampling caches')
        target = CV / 'article_disjoint' / '3b_zero' / 'samples8.json'
        saved = read_json(target)
        ids = [str(qid) for qid in folds['article_disjoint']['train']]
        added = {qid: pool[qid] for qid in ids if qid not in saved['rows'] and qid in pool}
        if added:
            saved['rows'].update(added)
            saved['additional_reused_sources'] = [str(path.relative_to(paths.ROOT)) for path in source_paths]
            saved['reused_qids'] = sorted(set(saved['reused_qids']) | set(map(int, added)))
            write_json(target, saved)
        print(f'Article-disjoint sampling: {len(saved["rows"])}/{len(ids)} available after untrained cache reuse',
              flush=True)


if __name__ == '__main__':
    main()
