"""Freeze the two DPO cross-validation folds and the article-disjoint holdout."""

from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from common import paths
from common.data import is_answerable, load_json, load_queries
from hybrid_claims.llm_generate import write_json
from hybrid_claims.split_training_preferences import EXAMPLES


OUT = Path(paths.OUT) / 'split_training' / 'cv'
MANIFEST = OUT / 'folds.json'
SEED = 13
VALIDATION_PERCENT = 15


def article_set(question):
    return tuple(sorted({evidence['url'] for evidence in question['evidence_list']}))


def type_counts(qids, queries):
    return dict(sorted(Counter(queries[qid]['question_type'] for qid in qids).items()))


def grouped_folds(qids, queries, article_sets, n_splits):
    keys = {key: i for i, key in enumerate(sorted({article_sets[qid] for qid in qids}))}
    groups = [keys[article_sets[qid]] for qid in qids]
    labels = [queries[qid]['question_type'] for qid in qids]
    stratifier = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for train, test in stratifier.split(qids, labels, groups):
        yield [qids[i] for i in train], [qids[i] for i in test]


def validation_targets(pool, queries):
    counts = type_counts(pool, queries)
    targets = {kind: count * VALIDATION_PERCENT // 100 for kind, count in counts.items()}
    n_extra = round(len(pool) * VALIDATION_PERCENT / 100) - sum(targets.values())
    by_remainder = sorted(counts, key=lambda kind: (-(counts[kind] * VALIDATION_PERCENT % 100), kind))
    for kind in by_remainder[:n_extra]:
        targets[kind] += 1
    return targets


def validation_split(pool, queries, article_sets):
    """Approximate 15% per type, moving only complete evidence-set groups."""
    targets = validation_targets(pool, queries)

    def distance(counts):
        return sum(((counts.get(kind, 0) - target) / max(target, 1)) ** 2 for kind, target in targets.items())

    initial = min((heldout for _, heldout in grouped_folds(pool, queries, article_sets, n_splits=7)),
                  key=lambda qids: distance(type_counts(qids, queries)))
    heldout_groups = {article_sets[qid] for qid in initial}
    counts = Counter(type_counts(initial, queries))
    by_group = defaultdict(Counter)
    for qid in pool:
        by_group[article_sets[qid]][queries[qid]['question_type']] += 1
    keys = sorted(by_group)

    while (score := distance(counts)) > 0:
        candidates = ((counts - by_group[key] if key in heldout_groups else counts + by_group[key], key)
                      for key in keys)
        candidate, key = min(candidates, key=lambda entry: distance(entry[0]))
        if distance(candidate) >= score:
            break
        counts = candidate
        heldout_groups.symmetric_difference_update({key})

    heldout = [qid for qid in pool if article_sets[qid] in heldout_groups]
    train = sorted(set(pool) - set(heldout))
    return train, heldout, targets


def fold_details(pool, test, queries, article_sets, demo_ids):
    train, heldout, targets = validation_split(pool, queries, article_sets)
    parts = {'train': train, 'heldout': heldout, 'test': test}
    groups = {key: {article_sets[qid] for qid in qids} for key, qids in parts.items()}
    if sum(map(len, parts.values())) != len(set(pool) | set(test)) or any(
        groups[left] & groups[right] for left, right in (('train', 'heldout'), ('train', 'test'), ('heldout', 'test'))
    ):
        raise ValueError('The fold splits a full evidence-set group or reuses a question')

    def articles(qids):
        return {url for qid in qids for url in article_sets[qid]}

    train_urls, test_urls, demo_urls = (articles(ids) for ids in (train, test, demo_ids))
    shared_train = [qid for qid in test if train_urls.intersection(article_sets[qid])]
    shared_demo = [qid for qid in test if demo_urls.intersection(article_sets[qid])]
    no_shared_train = sorted(set(test) - set(shared_train))
    pool_urls = train_urls | articles(heldout)
    return {
        **parts,
        'shared_article_test': shared_train,
        'no_shared_article_test': no_shared_train,
        'shared_demo_article_test': shared_demo,
        'no_shared_train_or_demo_article_test': sorted(set(no_shared_train) - set(shared_demo)),
        'shared_outer_pool_article_test': [qid for qid in test if pool_urls.intersection(article_sets[qid])],
        'shared_article_counts': {'train_test': len(train_urls & test_urls),
                                  'outer_pool_test': len(pool_urls & test_urls),
                                  'demo_test': len(demo_urls & test_urls)},
        'counts_by_type': {key: type_counts(qids, queries) for key, qids in parts.items()},
        'full_article_set_counts': {key: len(group) for key, group in groups.items()},
        'heldout_type_targets': targets,
    }


def build_manifest():
    queries = load_queries()
    answerable = {q['qid'] for q in queries if is_answerable(q)}
    demo_ids = sorted(EXAMPLES)
    demos = load_json(OUT.parent / 'examples.json')['examples']
    if len(answerable) != 2255 or len(demo_ids) != 4 or not set(demo_ids) <= answerable:
        raise ValueError('Unexpected benchmark size or demonstration IDs')
    if sorted(map(int, demos)) != demo_ids or any(demos[str(q)]['question'] != queries[q]['query'] for q in demo_ids):
        raise ValueError('Saved demonstrations do not match the benchmark')
    qids = sorted(answerable.difference(demo_ids))
    article_sets = {qid: article_set(queries[qid]) for qid in answerable}
    if any(not keys or not all(keys) for keys in article_sets.values()):
        raise ValueError('Every answerable question must have gold evidence article URLs')

    outer_folds = list(grouped_folds(qids, queries, article_sets, n_splits=2))
    if sorted(qid for _, test in outer_folds for qid in test) != qids:
        raise ValueError('Two-fold test partitions must cover each eligible question exactly once')
    folds = {f'fold_{i}': fold_details(pool, test, queries, article_sets, demo_ids)
             for i, (pool, test) in enumerate(outer_folds, start=1)}

    audit = load_json(OUT / 'fold_audit.json')
    audited = audit['closest_article_disjoint_assignment']['folds']
    pool, test = (sorted(row['qids']) for row in sorted(audited, key=lambda row: -row['n']))
    if audit['excluded_demo_qids'] != demo_ids or sorted(pool + test) != qids:
        raise ValueError('Article-disjoint audit does not cover the same eligible questions')
    disjoint = fold_details(pool, test, queries, article_sets, demo_ids)
    if disjoint['shared_article_counts']['outer_pool_test']:
        raise ValueError('An evidence article is shared by the article-disjoint pool and test set')
    folds['article_disjoint'] = disjoint
    return {
        'seed': SEED,
        'article_key': 'exact gold evidence URL',
        'outer_grouping': 'identical full sets of gold evidence URLs',
        'heldout_target_percent': VALIDATION_PERCENT,
        'heldout_grouping': 'identical full sets of gold evidence URLs',
        'excluded_demo_qids': demo_ids,
        'eligible_qids': qids,
        'folds': folds,
    }


def main():
    manifest = build_manifest()
    if MANIFEST.exists():
        if load_json(MANIFEST) != manifest:
            raise ValueError(f'Existing fold manifest differs from the deterministic design at {MANIFEST}')
    else:
        write_json(MANIFEST, manifest)
    for name, fold in manifest['folds'].items():
        print(f"{name}: train {len(fold['train']):,}, validation {len(fold['heldout']):,}, "
              f"test {len(fold['test']):,}, test questions sharing a gold article with train "
              f"{len(fold['shared_article_test']):,}")


if __name__ == '__main__':
    main()
