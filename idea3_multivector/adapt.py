"""Adapt the two existing encoders and evaluate a disjoint MultiHop-RAG validation fold."""

import argparse
from collections import Counter
from itertools import chain
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common import paths
from common.bge import DenseIndex, default_device
from common.data import load_chunks, load_json, load_queries
from common.metrics import format_table
from idea2_arrows.train import make_batch
from idea3_multivector.adapt_data import (NEWS_DATASET, NEWS_REVISION, adaptation_split, gold_keys, local_examples,
                                          news_examples, training_tensors)
from idea3_multivector.evaluate import rank_queries, retrieval_heads, summarize
from idea3_multivector.model import QuestionEncoder, make_models, matched_loss, query_vectors
from idea3_multivector.train import CHECKPOINT, MODEL
from replicate.main import chunk_cache_path, ranking_cache_path


def from_checkpoint(checkpoint, device):
    models = make_models(**checkpoint['config'], device=device)
    for name, model in models.items():
        model.load_state_dict(checkpoint['states'][name])
    return models


def training_batch(encoder, dataset, positions, passages, documents):
    features = encoder.features([dataset['questions'][i] for i in positions.tolist()])
    batch = make_batch(dataset, positions.to(passages.device), passages)
    batch['gold_documents'] = documents[batch['gpid'].clamp_min(0)]
    batch['distractor_documents'] = documents[batch['dpid'].clamp_min(0)]
    return features, batch


def train_arms(arms, encoder, passages, documents, local, news, args):
    optimizers = {arm: torch.optim.AdamW(chain.from_iterable(m.parameters() for m in models.values()),
                                         lr=args.lr, weight_decay=0.0) for arm, models in arms.items()}
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        local_batches = torch.randperm(len(local['questions']), generator=generator).split(args.batch_size)
        news_batches = torch.randperm(len(news['questions']), generator=generator).tensor_split(len(local_batches))
        if any(len(b) == 0 for b in news_batches):
            raise ValueError('Use a larger batch size so each step has auxiliary data')
        losses = {arm: dict.fromkeys(models, 0.0) for arm, models in arms.items()}
        for rows, auxiliary in zip(local_batches, news_batches):
            features, batch = training_batch(encoder, local, rows, passages, documents)
            news_features, news_batch = training_batch(encoder, news, auxiliary, passages, documents)
            for arm, models in arms.items():
                loss = {name: matched_loss(V, batch) for name, V in query_vectors(models, features).items()}
                if arm == 'MultiHop + news':
                    for name, vectors in query_vectors(models, news_features).items():
                        aux = matched_loss(vectors, news_batch)
                        loss[name] = (loss[name] + args.news_weight * aux) / (1 + args.news_weight)
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                sum(loss.values()).backward()
                optimizer.step()
                for name, value in loss.items():
                    losses[arm][name] += value.item() / len(local_batches)
        history.append({'epoch': epoch, 'mean_training_loss': losses})
        print(json.dumps(history[-1]))
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', type=Path, default=CHECKPOINT)
    ap.add_argument('--out', type=Path, default=Path(paths.OUT) / 'idea3_adapted')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--news_weight', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--device', default=default_device())
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.news_weight <= 0:
        ap.error('Epochs, batch size, learning rate and news weight must be positive')
    device = torch.device(args.device)
    queries, chunks = load_queries(), load_chunks()
    train, validation = adaptation_split(queries, seed=args.seed)
    rankings = load_json(ranking_cache_path(MODEL, 'fp32'))['ids']
    local, local_docs = local_examples(queries, chunks, rankings, train, validation)

    checkpoint = torch.load(args.start, map_location='cpu', weights_only=True)
    if checkpoint['model'] != MODEL or checkpoint['train_limit'] is not None:
        raise ValueError('Start from the full out-of-domain BGE checkpoint')
    encoder = QuestionEncoder(MODEL, device=str(device), dtype='fp32')
    news, news_vectors, news_docs, news_qids = news_examples(encoder, queries, len(chunks))
    vecs = np.load(chunk_cache_path(MODEL, 'fp32'), mmap_mode='c')
    if len(vecs) != len(chunks):
        raise ValueError('The cached embeddings must match the current index')
    P = torch.cat((torch.from_numpy(vecs).to(device), torch.from_numpy(news_vectors).to(device)))
    all_docs = local_docs + news_docs
    id_map = {doc: i for i, doc in enumerate(dict.fromkeys(all_docs))}
    documents = torch.tensor([id_map[doc] for doc in all_docs], device=device)
    sizes = {name: {'count': len(ids), 'by_type': dict(Counter(queries[q]['question_type'] for q in ids)),
                    'gold_articles': len({key for q in ids for kind, key in gold_keys(queries[q]) if kind == 'title'})}
             for name, ids in [('train', train), ('validation', validation)]}
    print('MultiHop split:', sizes)
    print(f'Continual-Eval: {len(news)} auxiliary questions, {len(news_vectors)} distinct evidence passages')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': args.seed, 'n_chunks': len(chunks), 'split_counts': sizes, 'train_qids': train,
                'validation_qids': validation, 'news_dataset': NEWS_DATASET, 'news_revision': NEWS_REVISION,
                'news_qids': news_qids, 'n_news_passages': len(news_vectors)}
    (args.out / 'split.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    arms = {arm: from_checkpoint(checkpoint, device) for arm in ('MultiHop only', 'MultiHop + news')}
    history = train_arms(arms, encoder, P, documents, training_tensors(local, device), training_tensors(news, device), args)
    torch.save({'model': MODEL, 'config': checkpoint['config'],
                'states': {arm: {name: model.state_dict() for name, model in models.items()}
                           for arm, models in arms.items()}}, args.out / 'adapted_encoders.pt')

    index = DenseIndex(vecs, device=str(device))
    heads = retrieval_heads(from_checkpoint(checkpoint, device), ' before adaptation')
    for arm, models in arms.items():
        heads.update(retrieval_heads(models, f' ({arm})'))
    selected, candidates, latency = rank_queries(heads, encoder, index, queries, chunks, validation, False, 32)
    results = summarize({'validation': validation}, queries, chunks, selected, candidates)['validation']
    for k in (10, 6):
        print(f'\nHeld-out MultiHop validation K={k}')
        print(format_table(results[str(k)]['table'], [f'complete@{k}', f'recall@{k}', 'candidate_complete@60',
                                                       'distinct_articles']))
    report = {'starting_checkpoint': str(args.start), 'epochs': args.epochs, 'lr': args.lr,
              'batch_size': args.batch_size, 'news_weight': args.news_weight, 'device': str(device),
              'history': history, 'results': results, 'latency': latency}
    (args.out / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {args.out / "validation.json"}')


if __name__ == '__main__':
    main()
