"""Consolidate completed MultiHop-RAG runs on the fixed seed-13 test split."""

import gzip
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from common.data import gold_facts, is_answerable, load_chunks, load_queries, split_queries
from common.metrics import calculate_metrics


METRICS = ('Hits@10', 'MRR@10', 'MAP@10', 'Hits@4')
AUDIT = 'outputs/hybrid_claims/verification/chunk_ids.json.gz'
UPSTREAM = 'outputs/hybrid_claims/verification/upstream_3dd4d4e/retrieval_evaluate.py'
AUDIT_NAMES = {
    'bge_uncapped': 'BGE, no cap',
    'bge': 'BGE + cap 2, original candidate budget',
    'hybrid': 'BGE + BM25 hybrid + cap 2, no claims',
    'claims': 'Hybrid + rule-split BM25 claims + cap 2',
    'rank_bm25': 'Hybrid + rule splits, rank_bm25 implementation',
    'no_bm25': 'BGE whole question and BGE rule splits, no BM25',
    'bge_claims': 'Hybrid whole question + BGE rule splits',
    'filtered_claims': 'Hybrid + rule splits, drop back-references',
    'no_cap': 'Hybrid + rule splits, no article cap',
    'cap1': 'Hybrid + rule splits, cap 1',
    'cap3': 'Hybrid + rule splits, cap 3',
}
LLM_NAMES = {
    'C': 'decomposition + BM25',
    'C_hybrid': 'decomposition + hybrid',
    'D': 'HyDE dense',
    'D_bm25': 'HyDE dense + BM25',
    'E': 'decomposition (BM25) + HyDE dense',
}
VECTOR_NAMES = {
    'dense BGE': 'BGE, no cap',
    'dense BGE capped': 'BGE + cap 2',
    'pooled control': 'Four-vector pooled control + cap 2',
    'token heads': 'Four-vector token pooling + cap 2',
    'outlet filter': 'Outlet filter',
}
GROUPS = (
    'Dense baselines',
    'Idea 1 — residual retrieval',
    'Idea 2 — arrows and outlet filter',
    'Idea 2 — masked-outlet diagnostics',
    'Idea 3 — four-vector retrieval',
    'Idea 3 — masked-outlet diagnostics',
    'Idea 3 — news-adapted checkpoints',
    'Original hybrid claims and ablations',
    'Original hybrid claims — masked-outlet diagnostics',
    'LLM decomposition and HyDE',
    'Bundled 7B decomposition',
    'Few-shot and e5 comparisons',
    'DPO and precision-matched controls',
)


def read(path):
    path = ROOT / path
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def archived(group, method, path, key, value):
    return {'group': group, 'method': method, 'source': path, 'source_key': key,
            'n': value['n'], **{metric: value[metric] for metric in METRICS}}


class RankingEvaluator:
    def __init__(self):
        queries = load_queries()
        self.answers = {q['qid'] for q in queries if is_answerable(q)}
        self.qids = [qid for qid in split_queries(queries, seed=13)[1] if qid in self.answers]
        assert len(self.qids) == 1574
        self.gold = [gold_facts(queries[qid]) for qid in self.qids]
        self.chunks = [chunk['text'] for chunk in load_chunks()]
        self.signatures = {}
        spec = spec_from_file_location('upstream_retrieval_evaluate', ROOT / UPSTREAM)
        upstream = module_from_spec(spec)
        spec.loader.exec_module(upstream)
        self.upstream = upstream.calculate_metrics

    def row(self, group, method, path, key, by_qid, *, allow_short=False, all_answerable=False):
        expected = self.answers if all_answerable else set(self.qids)
        if set(by_qid) != expected:
            raise ValueError(f'Question IDs do not match {path}, {key}')
        rankings = [by_qid[qid] for qid in self.qids]
        for ids in rankings:
            if not ids or len(ids) > 10 or len(ids) != len(set(ids)):
                raise ValueError(f'Invalid ranks in {path}, {key}: {ids}')
            if any(not isinstance(i, int) or not 0 <= i < len(self.chunks) for i in ids):
                raise ValueError(f'Invalid chunk ID in {path}, {key}: {ids}')
        short = sum(len(ids) < 10 for ids in rankings)
        if short and not allow_short:
            raise ValueError(f'{path}, {key}: {short} ranked lists under ten')
        texts = [[self.chunks[i] for i in ids] for ids in rankings]
        metrics = calculate_metrics(texts, self.gold)
        if metrics != self.upstream(texts, self.gold):
            raise ValueError(f'Scorer mismatch in {path}, {key}')
        signature = hashlib.sha256(json.dumps(rankings, separators=(',', ':')).encode()).hexdigest()
        row = {'group': group, 'method': method, 'source': path, 'source_key': key,
               'n': len(rankings), 'short_at_10': short, 'ranking_hash': signature, **metrics}
        same = self.signatures.setdefault((method, signature), row)
        if same is not row:
            row['duplicate_of'] = {'group': same['group'], 'source': same['source'],
                                   'source_key': same['source_key']}
        return row


def hybrid_audit(score):
    rows = []
    for key, cutoffs in read(AUDIT).items():
        mode, system = key.split(':')
        group = 'Original hybrid claims and ablations'
        if mode == 'masked':
            group = 'Original hybrid claims — masked-outlet diagnostics'
        rows.append(score.row(group, AUDIT_NAMES[system], AUDIT, f'{key}.10',
                              {int(qid): ids for qid, ids in cutoffs['10'].items()},
                              allow_short=True, all_answerable=True))
    return rows


def nested_rankings(score, group, path, methods):
    data = read(path)
    if len({row['qid'] for row in data}) != len(data) or set(data[0]['system']) != set(methods):
        raise ValueError(f'Unexpected question or system keys in {path}')
    return [score.row(group, method, path, f'{key}.ids@10',
                      {r['qid']: r['system'][key]['ids@10'] for r in data})
            for key, method in methods.items()]


def flat_rankings(score, group, path, methods):
    data = read(path)
    if set(data) != set(methods):
        raise ValueError(f'Unexpected system keys in {path}')
    rows = []
    for key, method in methods.items():
        if any(int(qid) != r['qid'] for qid, r in data[key].items()):
            raise ValueError(f'Mismatched question ID in {path}, {key}')
        row = score.row(group, method, path, f'{key}.ids@10',
                        {r['qid']: r['ids@10'] for r in data[key].values()})
        if row['Hits@10'] != sum(r['Hits@10'] for r in data[key].values()) / row['n']:
            raise ValueError(f'Archived hits mismatch in {path}, {key}')
        rows.append(row)
    return rows


def collect():
    score = RankingEvaluator()
    rows = read(OUT / 'early.json')
    for mode, group in [('original', 'Idea 3 — four-vector retrieval'),
                        ('masked', 'Idea 3 — masked-outlet diagnostics')]:
        path = f'outputs/idea3/eval_{mode}.json'
        key = 'splits.test.10.table'
        rows.extend(archived(group, VECTOR_NAMES[row['system']], path, f'{key}.{row["system"]}', row)
                    for row in read(path)['splits']['test']['10']['table'])
    path = 'outputs/test_retrieval_metrics/adapted_idea3.json'
    adapted = read(path)
    if adapted['test_qids'] != score.qids:
        raise ValueError('Adapted Idea 3 test IDs do not match the fixed split')
    rows.extend(archived('Idea 3 — news-adapted checkpoints', row['system'], path,
                         f'test.{row["system"]}', row) for row in adapted['test'])
    rows.extend(hybrid_audit(score))

    base = {'A': 'BGE whole-question hybrid + cap 2, exact K',
            'B': 'BGE hybrid + rule splits + cap 2, exact K'}
    decomp = lambda size: f'Qwen2.5-{size} decomposition + hybrid'
    group = 'LLM decomposition and HyDE'
    for size, directory in [('3B', 'llm_decomp_vs_hyde'), ('7B', 'llm_decomp_bigger/qwen2.5-7b-instruct')]:
        methods = {**base, **{key: f'Qwen2.5-{size} {name}' for key, name in LLM_NAMES.items()}}
        rows.extend(nested_rankings(score, group, f'outputs/{directory}/test_per_question.json', methods))
    rows.extend(nested_rankings(score, 'Bundled 7B decomposition',
                                'outputs/llm_decomp_bigger/bundled/test_per_question.json',
                                {'B': base['B'], 'C_hybrid': decomp('7B'),
                                 'concat': 'Concat sub-questions + hybrid',
                                 'mean': 'Mean dense scores via mean query vector + joined BM25',
                                 'max': 'Max dense and BM25 scores across sub-questions + RRF'}))

    e5_base = {'A': 'e5 whole-question hybrid + cap 2, exact K',
               'B': 'e5 hybrid + rule splits + cap 2, exact K'}
    for encoder, baselines in [('bge', base), ('e5', e5_base)]:
        label = 'BGE' if encoder == 'bge' else 'e5'
        zero = decomp('7B') if encoder == 'bge' else 'Qwen2.5-7B zero-shot decomposition + e5 hybrid'
        rows.extend(flat_rankings(score, 'Few-shot and e5 comparisons',
                                  f'outputs/split_training/step1/{encoder}_test_rankings.json',
                                  {**baselines, 'zero': zero,
                                   'few': f'Qwen2.5-7B few-shot decomposition + {label} hybrid'}))
        for arm, size, prompt, precision in [('3b_zero', '3B', 'zero-shot', 'BF16'),
                                             ('3b_few', '3B', 'few-shot', 'NF4'),
                                             ('7b_zero', '7B', 'zero-shot', 'NF4')]:
            stem = f'Qwen2.5-{size} {prompt} ({precision})'
            untrained = f'{stem}, untrained + {label} hybrid'
            if prompt == 'zero-shot' and encoder == 'bge':
                untrained = decomp(size)
            elif arm == '7b_zero':
                untrained = zero
            rows.extend(flat_rankings(score, 'DPO and precision-matched controls',
                                      f'outputs/split_training/step2/{arm}/{encoder}_test_rankings.json',
                                      {'rule': baselines['B'], 'untrained': untrained,
                                       'trained': f'{stem}, DPO + {label} hybrid'}))
    for row in rows:
        if row['n'] != len(score.qids) or row['group'] not in GROUPS:
            raise ValueError(f'Unexpected cohort or group: {row}')
        if any(not 0 <= row[key] <= 1 for key in METRICS):
            raise ValueError(f'Unexpected metric value: {row}')
    return score.qids, rows


def link(path):
    return Path(os.path.relpath(ROOT / path, OUT)).as_posix()


def render(rows):
    visible = [row for row in rows if 'duplicate_of' not in row]
    main = [r for r in visible if 'masked-outlet' not in r['group'] and 'oracle' not in r['method'].lower()]
    leads = {metric: max(main, key=lambda row: row[metric]) for metric in METRICS}
    lines = [
        '# MultiHop-RAG retrieval metrics on the fixed test split', '',
        f'The tables cover **{len(visible)} system and condition rows** on the same **1,574 answerable '
        'test questions** from `split_queries(seed=13)`. Ranked archives were rescored with '
        'the MultiHop-RAG evaluator. For archives that retained only aggregate scores, the tables '
        'use their existing test metrics. The four news-adapted vector checkpoints were evaluated '
        'on this test split for the first time, using the saved weights.', '',
        '## Highest scores on the original questions', '',
        '| Metric | System | Score |', '|---|---|---:|',
    ]
    for metric, row in leads.items():
        lines.append(f'| {metric} | {row["method"]} ({row["group"]}) | {row[metric]:.4f} |')
    lines += [
        '',
        'Hits@10 and Hits@4 are the fractions of questions with at least one gold fact in the '
        'first ten and four chunks, respectively. MRR@10 rewards an early first hit. MAP@10 uses '
        'the benchmark’s formula, which credits each gold fact at its first match. Values are '
        'fractions on a 0–1 scale. Complete@10, used in the earlier reports, requires every gold fact. '
        '[Upstream evaluator at commit 3dd4d4e](https://github.com/yixuantt/MultiHop-RAG/blob/'
        '3dd4d4e79fc9843008b8f832da99086a82f1a805/retrieval_evaluate.py)', '',
        'The original hybrid audit kept short ranked lists as returned. The LLM comparison, bundled '
        'variants, and split-training runs used exact-K backfill. The two versions of a baseline '
        'can therefore score differently. The JSON records the number of short lists where ranked '
        'IDs were available. Masked-outlet runs and gold-count diagnostics are labeled in their '
        'tables.', '',
    ]
    for group in GROUPS:
        items = [row for row in visible if row['group'] == group]
        if not items:
            continue
        lines += [f'## {group}', '',
                  '| System | Hits@10 | MRR@10 | MAP@10 | Hits@4 | Source |',
                  '|---|---:|---:|---:|---:|---|']
        for row in items:
            method = row['method'].replace('|', r'\|')
            if row.get('short_at_10'):
                method += f' ({row["short_at_10"]} short lists)'
            cells = [method] + [f'{row[k]:.4f}' for k in METRICS]
            cells.append(f'[Run data]({link(row["source"])})')
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += [
        '## Coverage and caveats', '',
        '- Repeated copies of the same named system with identical ordered chunk IDs appear once '
        'in the tables. [Metrics and archive keys](metrics.json) include every copy checked.',
        '- Hits@4 uses the first four picks of each K=10 run. Standalone K=6 runs, smoke tests, '
        'alternative split seeds, and the HippoRAG benchmark corpora use different evaluation '
        'conditions.',
        '- Qwen3.5-9B failed to load, 7B few-shot DPO ran out of GPU memory, and ChatGPT distillation '
        'stopped before teacher generation. The handoff’s TF-IDF figure has no saved rankings or '
        'verified test split. The capped Idea 1 always-expand controls were evaluated on dev only.',
        '- The split-training rows retain the full inherited test split. The earlier reports document '
        'test questions that share sets of evidence articles with demonstrations or training questions '
        'of a different question type. [Split-training report](../split_training/REPORT.md)', '',
        '- None of the test question IDs appeared in Idea 3 adaptation training. **171 test questions** '
        'share their full normalized gold-fact set with an adaptation training question. This overlap '
        'limits what the adapted checkpoint’s MRR lead says about retrieval on new evidence. '
        '[Adaptation evaluation and overlap audit](adapted_idea3.json)', '',
        '## Reproducing the tables', '',
        'From the project root in PowerShell, run the following commands. The middle command '
        'reevaluates the existing checkpoint on CUDA and is optional when its JSON is already present.', '',
        '```powershell',
        r'& .\.venv\Scripts\python.exe outputs\test_retrieval_metrics\collect_early.py',
        r'& .\.venv\Scripts\python.exe outputs\test_retrieval_metrics\evaluate_adapted_idea3.py',
        r'& .\.venv\Scripts\python.exe outputs\test_retrieval_metrics\collect_and_report.py',
        '```', '',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    qids, rows = collect()
    (OUT / 'metrics.json').write_text(json.dumps({'split_seed': 13, 'test_qids': qids, 'rows': rows},
                                                 indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (OUT / 'REPORT.md').write_text(render(rows), encoding='utf-8')
    print(f'Wrote {sum("duplicate_of" not in row for row in rows)} report rows ({len(rows)} source rows).')
