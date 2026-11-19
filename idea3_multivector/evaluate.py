"""Evaluate matched query encoders with max cosine fusion and a shared article cap."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from common import paths
from common.bge import DenseIndex, default_device
from common.data import gold_facts, load_chunks, load_queries, mask_outlets, outlet_names, split_queries
from common.metrics import facts_found, format_table, metrics_by_type
from common.selection import deal_quotas, largest_remainder
from idea2_arrows.eval_multihop import cap_per_article, outlets_in
from idea3_multivector.model import SYSTEMS, QuestionEncoder, make_models
from idea3_multivector.train import CHECKPOINT, MODEL
from replicate.main import chunk_cache_path


DENSE = 'dense BGE'
CAPPED = 'dense BGE capped'
OUTLET = 'outlet filter'
DEPTH = 60
BUDGETS = (6, 10)


def top_ids(scores, k):
    k = min(k, len(scores))
    ids = np.argpartition(-scores, k - 1)[:k]
    return ids[np.argsort(-scores[ids], kind='stable')]


def fused_scores(index, vectors):
    n, heads, d = vectors.shape
    return index.scores(vectors.reshape(n * heads, d)).reshape(n, heads, -1).max(axis=1)


def cap_ranking(candidates, scores, chunks, k):
    selected = cap_per_article(candidates, chunks, k, False)
    if len(selected) < k:
        selected = cap_per_article(np.argsort(-scores, kind='stable'), chunks, k, False)
    if len(selected) != k:
        raise ValueError(f'The article cap could not fill the budget of {k} chunks')
    return selected


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def retrieval_heads(models, suffix=''):
    pooled, token = (models[name] for name in SYSTEMS)
    return {f'{SYSTEMS[0]}{suffix}': lambda cls, hidden, mask: pooled(cls),
            f'{SYSTEMS[1]}{suffix}': lambda cls, hidden, mask: token(hidden, mask)}


@torch.inference_mode()
def rank_queries(models, encoder, index, queries, chunks, qids, masked, batch_size):
    device = torch.device(encoder.device)
    outlets = outlet_names(chunks)
    source_ids = defaultdict(list)
    for cid, chunk in enumerate(chunks):
        source_ids[chunk['source']].append(cid)
    source_ids = {name: np.array(ids) for name, ids in source_ids.items()}
    selected = {k: {name: {} for name in (DENSE, CAPPED, *models, OUTLET)} for k in BUDGETS}
    candidates = {name: {} for name in (DENSE, CAPPED, *models)}
    timing = {name: {'encode_seconds': 0.0, 'search_seconds': 0.0} for name in (DENSE, *models)}

    for start in tqdm(range(0, len(qids), batch_size), desc='masked' if masked else 'original', mininterval=5):
        batch = qids[start:start + batch_size]
        texts = [queries[q]['query'] for q in batch]
        if masked:
            texts = [mask_outlets(q, outlets) for q in texts]
        t0 = time.perf_counter()
        cls, hidden, mask = encoder.features(texts)
        sync(device)
        encode_time = time.perf_counter() - t0

        for name in (DENSE, *models):
            t0 = time.perf_counter()
            if name == DENSE:
                vectors = cls[:, None]
            else:
                vectors = models[name](cls, hidden, mask)
            vectors = vectors.cpu().numpy()
            timing[name]['encode_seconds'] += encode_time + time.perf_counter() - t0

            t0 = time.perf_counter()
            scores = fused_scores(index, vectors)
            tops = [top_ids(row, DEPTH).tolist() for row in scores]
            timing[name]['search_seconds'] += time.perf_counter() - t0

            for q, text, row, top in zip(batch, texts, scores, tops):
                capped_name = CAPPED if name == DENSE else name
                candidates[capped_name][q] = top
                capped = cap_ranking(top, row, chunks, max(BUDGETS))
                for k in BUDGETS:
                    selected[k][capped_name][q] = capped[:k]
                if name != DENSE:
                    continue
                candidates[DENSE][q] = top
                allowed = [source_ids[o] for o in outlets_in(text, outlets)]
                lists = [ids[top_ids(row[ids], DEPTH)].tolist() for ids in allowed]
                for k in BUDGETS:
                    selected[k][DENSE][q] = top[:k]
                    quota = largest_remainder([1] * len(lists) + [0], k)
                    selected[k][OUTLET][q] = deal_quotas(lists + [top], quota, chunks, k, False)

    latency = {name: {key.replace('seconds', 'ms_per_query'): value * 1000 / len(qids)
                      for key, value in times.items()} for name, times in timing.items()}
    return selected, candidates, latency


def summarize(splits, queries, chunks, selected, candidates):
    out = {}
    for split, qids in splits.items():
        out[split] = {}
        gold = [gold_facts(queries[q]) for q in qids]
        candidate_complete = {name: float(np.mean([
            all(facts_found([chunks[c]['text'] for c in ranking[q]], g, DEPTH))
            for q, g in zip(qids, gold)])) for name, ranking in candidates.items() if name != CAPPED}
        candidate_complete[CAPPED] = candidate_complete[DENSE]
        for k, systems in selected.items():
            rows, by_type = [], {}
            for name, rankings in systems.items():
                rows_ = [rankings[q] for q in qids]
                assert all(len(row) == k for row in rows_), (split, k, name)
                records = [{'question_type': queries[q]['question_type'], 'gold': g,
                            'retrieved': [chunks[c]['text'] for c in row]}
                           for q, g, row in zip(qids, gold, rows_)]
                by_type[name] = metrics_by_type(records, ks=(k,))
                article_count = np.mean([len({chunks[c]['article_id'] for c in row}) for row in rows_])
                if k < 10:
                    for group in by_type[name].values():
                        for metric in ('Hits@10', 'MRR@10', 'MAP@10'):
                            group.pop(metric, None)
                rows.append({'system': name, **by_type[name]['all'], 'candidate_complete@60': candidate_complete.get(name),
                             'distinct_articles': float(article_count)})
            out[split][str(k)] = {'table': rows, 'by_type': by_type}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    ap.add_argument('--out', type=Path, default=Path(paths.OUT) / 'idea3')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--device', default=default_device())
    a = ap.parse_args()
    device = torch.device(a.device)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['model'] != MODEL:
        raise ValueError('The checkpoint encoder must match the fixed BGE passage index')
    models = make_models(**checkpoint['config'], device=device)
    for name, m in models.items():
        m.load_state_dict(checkpoint['states'][name])
        m.eval()
    encoder = QuestionEncoder(MODEL, device=str(device), dtype='fp32')
    chunks, queries = load_chunks(), load_queries()
    index = DenseIndex(np.load(chunk_cache_path(MODEL, 'fp32')), device=str(device))
    qids = [q['qid'] for q in queries if q['question_type'] != 'null_query']
    answerable = set(qids)
    splits = {name: [q for q in ids if q in answerable]
              for name, ids in zip(('dev', 'test'), split_queries(queries))}
    a.out.mkdir(parents=True, exist_ok=True)
    for masked in (False, True):
        selected, candidates, latency = rank_queries(retrieval_heads(models), encoder, index, queries, chunks, qids,
                                                      masked, a.batch_size)
        results = summarize(splits, queries, chunks, selected, candidates)
        for k in BUDGETS:
            print(f'\nK={k} held-out test (mask_outlets={masked})')
            cols = [f'complete@{k}', f'recall@{k}', 'candidate_complete@60', 'distinct_articles']
            print(format_table(results['test'][str(k)]['table'], cols))
        label = 'masked' if masked else 'original'
        path = a.out / f'eval_{label}.json'
        report = {'checkpoint': str(a.checkpoint), 'train_size': checkpoint['train_size'],
                  'train_limit': checkpoint['train_limit'], 'n_chunks': len(chunks), 'mask_outlets': masked,
                  'device': str(device), 'batch_size': a.batch_size, 'latency': latency, 'splits': results}
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
