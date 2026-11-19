"""Replay the frozen training-free runs and collect independent audit evidence."""

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from rank_bm25 import BM25Okapi

from hybrid_claims import run
from common.bge import BGE_QUERY_INSTRUCTION, _texts_hash


OUT = Path(run.paths.OUT) / 'hybrid_claims'
AUDIT = OUT / 'verification'
CACHE = Path(run.paths.CACHE)
PAPER = ('Hits@10', 'Hits@4', 'MRR@10', 'MAP@10')
BACKREF = re.compile(r'\b(?:these|those|they|them|their|theirs|both|it|its|aforementioned|the\s+(?:former|latter))\b', re.I)
AS_IS = {
    'bge_uncapped': 'BGE (paper setup)', 'bge': 'BGE + cap2', 'hybrid': 'Hybrid BGE+BM25 + cap2',
    'claims': 'Hybrid + claim searches + cap2 (method)',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, separators=(',', ':')), encoding='utf-8')


class RankBm25:
    """Cache library-computed term contributions to speed up repeated searches."""

    def __init__(self, docs):
        self.model = BM25Okapi([run.tokenize(doc) for doc in docs], k1=0.9, b=0.4)
        self.postings = defaultdict(list)
        for cid, counts in enumerate(self.model.doc_freqs):
            for term in counts:
                self.postings[term].append(cid)
        self.weights = {}

    def scores(self, terms):
        scores = np.zeros(self.model.corpus_size)
        for term in terms:
            if term not in self.postings:
                continue
            if term not in self.weights:
                ids = np.array(self.postings[term])
                weights = self.model.get_batch_scores([term], ids)
                self.weights[term] = (ids, weights)
            ids, weights = self.weights[term]
            scores[ids] += weights
        return scores

    def search(self, text, k):
        scores = self.scores(run.tokenize(text) + run.date_tokens(text))
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top], kind='stable')].tolist()


def dense_cache(tag):
    return CACHE / f'emb_{run.MODEL}_fp32_{tag}.npy'


def original_inputs(queries, chunks, qids, masked, bm):
    outlets = sorted(run.outlet_names(chunks), key=len, reverse=True)
    if masked:
        texts = {q: run.mask_outlets(queries[q]['query'], outlets) for q in qids}
        vectors = np.load(dense_cache('queries_masked'))
        scores = vectors[qids] @ np.load(dense_cache('chunks')).T
        dense = {q: np.argsort(-scores[i])[:100].tolist() for i, q in enumerate(qids)}
    else:
        texts = {q: queries[q]['query'] for q in qids}
        rows = read_json(CACHE / f'dense_ranking_{run.MODEL}_fp32.json')['ids']
        dense = {q: rows[q][:100] for q in qids}
    claims = {q: run.claims(texts[q], [] if masked else outlets) for q in qids}
    whole_lists = {q: run.rrf([dense[q], bm.search(texts[q], 100)]) for q in qids}
    extra_lists = {q: [bm.search(claim, 60) for claim in claims[q]] for q in qids}
    return texts, dense, claims, whole_lists, extra_lists


def selections(qids, chunks, dense, whole_lists, extra_lists):
    return {
        name: {str(k): {str(q): fn(q, k) for q in qids} for k in (10, 6)} for name, fn in {
            'bge_uncapped': lambda q, k: dense[q][:k],
            'bge': lambda q, k: run.pick([dense[q]], k, chunks, 2),
            'hybrid': lambda q, k: run.pick([whole_lists[q]], k, chunks, 2),
            'claims': lambda q, k: run.pick([whole_lists[q], *extra_lists[q]], k, chunks, 2),
        }.items()
    }


def ablation(qids, chunks, first, extra, cap=2):
    return {str(k): {str(q): run.pick([first[q], *extra[q]], k, chunks, cap) for q in qids} for k in (10, 6)}


def bge_claim_ranks(qids, claims):
    from common.bge import BGEEncoder, DenseIndex, encode_cached

    cache = CACHE / 'hybrid_claims_verify'
    file = cache / 'bge_claim_top60.json'
    texts = list(dict.fromkeys(text for q in qids for text in claims[q]))
    if file.exists():
        ranks = read_json(file)
        if set(ranks) == set(texts):
            return ranks
    encoder = BGEEncoder(dtype='fp32')
    vectors = encode_cached(encoder, texts, str(cache / 'claim_embeddings.npy'), is_query=True)
    index = DenseIndex(np.load(dense_cache('chunks')))
    ids, _ = index.search(vectors, 60)
    ranks = dict(zip(texts, ids, strict=True))
    write_json(file, ranks)
    return ranks


def score_export(qids, chunks, queries, selections, label):
    file = AUDIT / f'{label}_retrieval_test.json'
    export = []
    for q in qids:
        row = queries[q]
        export.append({
            'query': row['query'], 'answer': row['answer'], 'question_type': row['question_type'],
            'retrieval_list': [{'text': chunks[c]['text'], 'score': None} for c in selections[str(q)]],
            'gold_list': row['evidence_list'],
        })
    write_json(file, export)
    evaluator = AUDIT / 'upstream_3dd4d4e' / 'retrieval_evaluate.py'
    source = evaluator.read_bytes()
    blob = hashlib.sha1(f'blob {len(source)}\0'.encode() + source).hexdigest()
    if blob != 'de33c1bac074b11320ab03f90969d9295a7a3adc':
        raise ValueError('Upstream evaluator does not match the GitHub blob')
    output = subprocess.check_output([sys.executable, str(evaluator), '--file', str(file)], text=True)
    (AUDIT / f'{label}_upstream.txt').write_text(output, encoding='utf-8')
    return {metric: float(re.search(rf'^{re.escape(metric)}: (\d+\.\d+)', output, re.M)[1]) for metric in PAPER}


def audit_split_and_cache(queries, chunks, splits):
    model = 'BAAI/bge-large-en-v1.5'
    all_text = [q['query'] for q in queries]
    outlets = run.outlet_names(chunks)
    datasets = {'chunks': [c['text'] for c in chunks], 'queries': all_text,
                'queries_masked': [run.mask_outlets(text, outlets) for text in all_text]}
    keys = {}
    for tag, texts in datasets.items():
        key = _texts_hash(texts, f'{model}|fp32|{tag != "chunks"}|{BGE_QUERY_INSTRUCTION}')
        saved = read_json(dense_cache(tag).with_suffix('.npy.json'))
        emb = np.load(dense_cache(tag), mmap_mode='r')
        keys[tag] = {'key': key, 'matches': saved['key'] == key and len(emb) == len(texts),
                     'shape': emb.shape, 'dtype': str(emb.dtype)}
    prior = Path(run.paths.OUT) / 'idea1_cap'
    matches = {}
    for split, qids in splits[13].items():
        rows = read_json(prior / f'word_K10_cap2_{split}_per_question.json')
        matches[split] = sorted(row['qid'] for row in rows) == qids
    return keys, matches


def sample_audit(queries, chunks, dev, claims, selection, coverage):
    samples = {}
    for q in random.Random(20261119).sample(dev, 10):
        rows = [chunks[c] for c in selection[str(q)]]
        samples[str(q)] = {
            'question': queries[q]['query'], 'claims': claims[q],
            'chunks': [{key: row[key] for key in ('title', 'source', 'article_id')} for row in rows],
            'facts': [{'fact': ev['fact'], 'found_at': ranks} for ev, ranks in zip(
                queries[q]['evidence_list'], coverage[str(q)], strict=True)],
        }
    write_json(AUDIT / 'dev_examples.json', samples)
    return samples


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    chunks, queries = run.load_chunks(), run.load_queries()
    qids = [q for q, row in enumerate(queries) if row['question_type'] != 'null_query']
    splits = {seed: {part: [q for q in ids if queries[q]['question_type'] != 'null_query']
                     for part, ids in zip(('dev', 'test'), run.split_queries(queries, seed=seed), strict=True)}
              for seed in (13, 1, 2, 3, 4, 5)}
    keys, split_matches = audit_split_and_cache(queries, chunks, splits)
    docs = [run.chunk_doc(c) for c in chunks]
    bm = run.BM25(docs, 0.9, 0.4)
    runs, papers = {}, {}
    for label, masked in (('original', False), ('masked', True)):
        print(f'Replaying {label}', flush=True)
        texts, dense, claims, whole, extra = original_inputs(queries, chunks, qids, masked, bm)
        rows = selections(qids, chunks, dense, whole, extra)
        runs.update({f'{label}:{n}': data for n, data in rows.items()})
        papers[label] = score_export(qids, chunks, queries, rows['claims']['10'], f'hybrid_claims_{label}')
        if masked:
            outlets = '|'.join(re.escape(o) for o in sorted(run.outlet_names(chunks), key=len, reverse=True))
            residue = re.compile(rf'(?<!\w)(?:{outlets})(?!\w)', re.I)
            residual = {q: residue.search(text)[0] for q, text in texts.items() if residue.search(text)}
        else:
            original = texts, dense, claims, whole, extra

    texts, dense, claims, whole, extra = original
    print('Scoring independent rank_bm25 queries', flush=True)
    okapi = RankBm25(docs)
    probes = [texts[q] for q in random.Random(0).sample(qids, 12)]
    probes += [c for q in random.Random(1).sample(qids, 12) for c in claims[q][:1]]
    for probe in probes:
        terms = run.tokenize(probe) + run.date_tokens(probe)
        if not np.array_equal(okapi.scores(terms), okapi.model.get_scores(terms)):
            raise ValueError('Cached library scores differ from BM25Okapi.get_scores')
    _, _, _, ok_whole, ok_extra = original_inputs(queries, chunks, qids, False, okapi)
    runs['ablation:rank_bm25'] = ablation(qids, chunks, ok_whole, ok_extra)
    claim_bge = bge_claim_ranks(qids, claims)
    bge_extra = {q: [claim_bge[c] for c in claims[q]] for q in qids}
    filtered = {q: [row for c, row in zip(claims[q], extra[q], strict=True) if not BACKREF.search(c)] for q in qids}
    runs['ablation:no_bm25'] = ablation(qids, chunks, dense, bge_extra)
    runs['ablation:bge_claims'] = ablation(qids, chunks, whole, bge_extra)
    runs['ablation:filtered_claims'] = ablation(qids, chunks, whole, filtered)
    for name, cap in (('no_cap', None), ('cap1', 1), ('cap3', 3)):
        runs[f'ablation:{name}'] = ablation(qids, chunks, whole, extra, cap)

    rankings = AUDIT / 'chunk_ids.json.gz'
    with gzip.open(rankings, 'wt', encoding='utf-8') as f:
        json.dump(runs, f, separators=(',', ':'))
    print('Recounting from raw evidence and chunk text', flush=True)
    recount_file = AUDIT / 'recount.json'
    subprocess.run([sys.executable, str(Path(__file__).with_name('recount_complete.py')), str(rankings),
                    run.paths.CHUNKS_JSONL, run.paths.QUERIES_JSON, str(recount_file)], check=True)
    recount = read_json(recount_file)

    scores, sanity = {}, {}
    for name, cuts in recount.items():
        scores[name], sanity[name] = {}, {}
        for k, audit in cuts.items():
            found = audit['fact_ranks']
            scores[name][k] = {seed: {part: {'n': len(ids), 'complete': sum(all(found[str(q)]) for q in ids),
                                              'recall': sum(sum(map(bool, found[str(q)])) / len(found[str(q)])
                                                            for q in ids) / len(ids)}
                                      for part, ids in split.items()} for seed, split in splits.items()}
            sanity[name][k] = {field: audit[field] for field in ('issues', 'example_qids')}
    disagreements = []
    for label, suffix in (('original', ''), ('masked', '_masked')):
        result = read_json(OUT / f'results{suffix}.json')
        for name, title in AS_IS.items():
            for k, by_seed in scores[f'{label}:{name}'].items():
                row = result[f'{title}|K{k}']
                checks = {'dev_complete': by_seed[13]['dev']['complete'] / len(splits[13]['dev']),
                          'complete': by_seed[13]['test']['complete'] / len(splits[13]['test']),
                          'recall': by_seed[13]['test']['recall']}
                disagreements.extend((label, name, k, check) for check, actual in checks.items()
                                     if not np.isclose(row[check], actual, atol=1e-12, rtol=0))
        disagreements.extend((label, key) for key in PAPER if papers[label][key] != round(
            result[f'{AS_IS["claims"]}|paper_metrics'][key], 4))

    example = sample_audit(queries, chunks, splits[13]['dev'], claims, runs['original:claims']['10'],
                           recount['original:claims']['10']['fact_ranks'])
    write_json(AUDIT / 'summary.json', {
        'cache': keys, 'same_qids_as_idea1': split_matches, 'splits': splits, 'scores': scores,
        'sanity': sanity, 'recount_disagreements': disagreements, 'upstream': papers,
        'rank_bm25_probes': len(probes), 'backref_pattern': BACKREF.pattern,
        'backref_drops': sum(len(extra[q]) - len(filtered[q]) for q in qids),
        'backref_total': sum(map(len, extra.values())), 'masked_residual_names': residual,
        'rank_bm25_changed_top10': sum(runs['original:claims']['10'][str(q)] !=
                                       runs['ablation:rank_bm25']['10'][str(q)] for q in qids),
        'dev_examples': list(example),
    })
    print(f'Wrote verification evidence to {AUDIT}', flush=True)


if __name__ == '__main__':
    main()
