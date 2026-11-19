"""Train two matched four-vector query encoders on the prepared arrow datasets."""

import argparse
from itertools import chain
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from common import paths
from common.bge import default_device
from idea2_arrows.train import load_split, make_batch
from idea3_multivector.model import QuestionEncoder, make_models, matched_loss, query_vectors


MODEL = 'BAAI/bge-large-en-v1.5'
CHECKPOINT = Path(paths.CACHE) / 'multivector' / 'encoders.pt'


def load_questions(split, index):
    with (Path(paths.ARROWS_DATA) / f'{split}.jsonl').open(encoding='utf-8') as f:
        questions = []
        for line, row in zip(f, index, strict=True):
            data = json.loads(line)
            if (data['qid'], data['dataset']) != (row['qid'], row['dataset']):
                raise ValueError(f'{split} questions and paragraph index are out of order')
            questions.append(data['question'])
    return questions


@torch.no_grad()
def validate(models, encoder, data, paragraphs, questions, batch_size):
    ks = (2, 4, 6)
    totals = {name: dict.fromkeys(ks, 0) for name in ('dense BGE', *models)}
    for start in range(0, len(questions), batch_size):
        stop = min(start + batch_size, len(questions))
        gold = data['gpid'][start:stop]
        ids = torch.cat((gold, data['dpid'][start:stop]), dim=1)
        candidates = paragraphs[ids.clamp_min(0)].float()
        features = encoder.features(questions[start:stop])
        all_vectors = {'dense BGE': features[0][:, None], **query_vectors(models, features)}
        for name, vectors in all_vectors.items():
            scores = (vectors @ candidates.transpose(1, 2)).amax(dim=1).masked_fill(ids < 0, -torch.inf)
            retrieved = ids.gather(1, scores.topk(max(ks), dim=1).indices)
            for k in ks:
                covered = (retrieved[:, :k, None] == gold[:, None, :]).any(dim=1) | (gold < 0)
                totals[name][k] += covered.all(dim=1).sum().item()
    return {name: {f'complete@{k}': v / len(questions) for k, v in scores.items()}
            for name, scores in totals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_limit', type=int, default=None, help='Use a small subset for a smoke test')
    ap.add_argument('--device', default=default_device())
    ap.add_argument('--out', type=Path, default=CHECKPOINT)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    paragraph_path = Path(paths.ARROWS_DATA) / 'paragraphs.npy'
    P = torch.from_numpy(np.load(paragraph_path, mmap_mode='c')).to(device)
    train, dev = (load_split(split, None, device) for split in ('train', 'dev'))
    train_questions, dev_questions = (load_questions(split, data['index'])
                                      for split, data in (('train', train), ('dev', dev)))
    del train['q'], dev['q']
    size = min(a.train_limit or len(train_questions), len(train_questions))
    if not (size > 0 and a.epochs > 0 and a.batch_size > 0):
        ap.error('Training size, epochs and batch size must be positive')

    encoder = QuestionEncoder(MODEL, device=str(device), dtype='fp16' if device.type == 'cuda' else 'fp32')
    config = {'d': P.shape[1], 'pooled_rank': 16, 'token_rank': 64}
    models = make_models(**config, device=device)
    params = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    print(f'{size:,} train questions and {len(dev_questions):,} MuSiQue validation questions on {device}')
    print('Trainable parameters:', params)
    optimizer = torch.optim.AdamW(chain.from_iterable(m.parameters() for m in models.values()),
                                  lr=a.lr, weight_decay=0.0)
    history, best = [], {}
    for epoch in range(1, a.epochs + 1):
        start = time.perf_counter()
        loss_sum = dict.fromkeys(models, 0.0)
        order = torch.randperm(size).split(a.batch_size)
        for rows in tqdm(order, desc=f'epoch {epoch}', mininterval=5):
            features = encoder.features([train_questions[i] for i in rows.tolist()])
            batch = make_batch(train, rows.to(device), P)
            batch['G'], batch['D'] = batch['G'].float(), batch['D'].float()
            vectors = query_vectors(models, features)
            losses = {name: matched_loss(V, batch, a.temperature) for name, V in vectors.items()}
            optimizer.zero_grad(set_to_none=True)
            sum(losses.values()).backward()
            optimizer.step()
            for name, loss in losses.items():
                loss_sum[name] += loss.item() * len(rows)

        scores = validate(models, encoder, dev, P, dev_questions, a.batch_size)
        history.append({'epoch': epoch, 'seconds': time.perf_counter() - start,
                        'train_loss': {name: loss / size for name, loss in loss_sum.items()}, 'dev': scores})
        print(json.dumps(history[-1]))
        for name, model in models.items():
            score = scores[name]['complete@4']
            if name not in best or score > best[name]['score']:
                best[name] = {'epoch': epoch, 'score': score,
                              'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': MODEL, 'config': config, 'states': {k: v['state'] for k, v in best.items()},
                'train_size': size, 'train_limit': a.train_limit}, a.out)
    report = {'seed': a.seed, 'device': str(device), 'embedding_dtype': encoder.dtype,
              'epochs': a.epochs, 'batch_size': a.batch_size, 'lr': a.lr, 'temperature': a.temperature,
              'train_size': size, 'dev_size': len(dev_questions), 'trainable_parameters': params,
              'history': history, 'selected': {k: {f: v[f] for f in ('epoch', 'score')} for k, v in best.items()}}
    a.out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {a.out}')


if __name__ == '__main__':
    main()
