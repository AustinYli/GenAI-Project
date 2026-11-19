"""Article-disjoint MultiHop-RAG splits and auxiliary news evidence spans."""

from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
import random
from urllib.parse import urlsplit

import torch

from common.bge import DenseIndex
from common.data import is_answerable, split_queries
from common.metrics import _norm


NEWS_DATASET = 'ruggsea/continual-eval'
NEWS_REVISION = '94f1cb577499d9b9351f7ffbb32c67a32f537d14'


def normalized(text):
    return ' '.join(text.casefold().split())


def gold_keys(query):
    for evidence in query['evidence_list']:
        yield 'title', normalized(evidence['title'])
        yield 'fact', _norm(evidence['fact']).casefold()


def adaptation_split(queries, seed=13, validation_fraction=0.25):
    """Keep connected gold-article and exact-fact groups within one fold."""
    old_dev, _ = split_queries(queries)
    eligible = [q for q in old_dev if is_answerable(queries[q])]
    parent = {q: q for q in eligible}

    def root(q):
        while q != parent[q]:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    owners, groups = {}, defaultdict(list)
    for q in eligible:
        for key in gold_keys(queries[q]):
            other = owners.setdefault(key, q)
            parent[root(q)] = root(other)
    for q in eligible:
        groups[root(q)].append(q)

    target_n = round(len(eligible) * validation_fraction)
    candidates = [g for g in groups.values() if len(g) <= target_n]
    if sum(map(len, candidates)) < target_n:
        raise ValueError('Not enough independent evidence groups for an article-disjoint validation split')
    totals = Counter(queries[q]['question_type'] for q in eligible)
    rng = random.Random(seed)
    best = None
    for _ in range(512):
        rng.shuffle(candidates)
        chosen = []
        for g in candidates:
            if len(chosen) + len(g) <= target_n:
                chosen.extend(g)
        counts = Counter(queries[q]['question_type'] for q in chosen)
        cost = (target_n - len(chosen),
                sum(abs(counts[t] - n * validation_fraction) for t, n in totals.items()))
        if best is None or cost < best[0]:
            best = cost, set(chosen)
    validation = best[1]
    train = sorted(set(eligible) - validation)
    validation = sorted(validation)
    train_keys = {key for q in train for key in gold_keys(queries[q])}
    validation_keys = {key for q in validation for key in gold_keys(queries[q])}
    if not train or not validation or train_keys & validation_keys:
        raise ValueError('Adaptation training and validation must be evidence-disjoint')
    return train, validation


def select_negatives(ranks, document_ids, blocked, count=8):
    seen = set(blocked)
    chosen = []
    for cid in ranks:
        doc = document_ids[cid]
        if doc not in seen:
            chosen.append(cid)
            seen.add(doc)
        if len(chosen) == count:
            return chosen
    return chosen


def local_examples(queries, chunks, rankings, train, validation):
    doc_ids = [normalized(c['title']) for c in chunks]
    val_titles = {key for q in validation for kind, key in gold_keys(queries[q]) if kind == 'title'}
    by_title = defaultdict(list)
    normalized_chunks = [_norm(c['text']) for c in chunks]
    for cid, key in enumerate(doc_ids):
        by_title[key].append(cid)
    examples = []
    for q in train:
        query = queries[q]
        first_pass = rankings[q]
        rank = {cid: i for i, cid in enumerate(first_pass)}
        facts_by_title = defaultdict(list)
        for evidence in query['evidence_list']:
            facts_by_title[normalized(evidence['title'])].append(_norm(evidence['fact']))
        gold = []
        for title, facts in facts_by_title.items():
            hits = Counter(c for fact in facts for c in by_title[title] if fact in normalized_chunks[c])
            if not hits:
                raise ValueError(f'No indexed chunk contains a gold fact for qid {q}')
            gold.append(min(hits, key=lambda c: (-hits[c], rank.get(c, len(rank)), c)))
        if not 2 <= len(gold) <= 4:
            raise ValueError(f'Expected two to four distinct gold articles for qid {q}')
        distractors = select_negatives(chain(first_pass, range(len(chunks))), doc_ids,
                                       set(facts_by_title) | val_titles)
        if len(distractors) != 8 or any(doc_ids[c] in val_titles for c in gold + distractors):
            raise ValueError(f'Invalid training candidate set for qid {q}')
        examples.append((query['query'], gold, distractors))
    return examples, ['multihop:' + doc for doc in doc_ids]


def news_examples(encoder, queries, offset):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(NEWS_DATASET, repo_type='dataset', revision=NEWS_REVISION,
                                      allow_patterns=['multihop/*.parquet'], local_files_only=True))
    parquet = sorted((snapshot / 'multihop').glob('*.parquet'))
    if len(parquet) != 3:
        raise FileNotFoundError('Download the three pinned Continual-Eval multi-hop Parquet files first')
    fields = ['item_id', 'question', 'language', 'judge_answer_correct', 'supporting_facts']
    seen = {normalized(q['query']) for q in queries}
    passages, doc_ids, rows = {}, [], []
    for f in parquet:
        for row in pq.read_table(f, columns=fields).to_pylist():
            q = row['question']
            if row['language'] != 'en' or not row['judge_answer_correct'] or normalized(q) in seen:
                continue
            by_doc = defaultdict(list)
            for g in row['supporting_facts']:
                by_doc[g['doc_id']].append(g)
            if not 2 <= len(by_doc) <= 4 or any(not (g['span'] or '').strip() or
                    (g['publication_date'] or '') < '2024-01-01' for gs in by_doc.values() for g in gs):
                continue
            gold = []
            for doc, spans in by_doc.items():
                text = '\n'.join(dict.fromkeys(g['span'].strip() for g in spans))
                source = urlsplit(doc).hostname or doc
                date = spans[0]['publication_date']
                passage = (f'[Excerpt from document]\npublished_at: {date}\nsource: {source}\n'
                           f'Excerpt:\n-----\n{text}\n-----\n')
                key = doc, passage
                if key not in passages:
                    passages[key] = len(doc_ids)
                    doc_ids.append('news:' + doc)
                gold.append(passages[key])
            rows.append((row['item_id'], q, gold))
            seen.add(normalized(q))
    if not rows:
        raise ValueError('No eligible auxiliary news examples')
    vectors = encoder.encode([passage for _, passage in passages], is_query=False)
    qvec = encoder.encode([q for _, q, _ in rows], is_query=True)
    idx = DenseIndex(vectors, device=encoder.device)
    rankings, _ = idx.search(qvec, len(vectors))
    examples = []
    for (_, q, gold), order in zip(rows, rankings):
        neg = select_negatives(order, doc_ids, {doc_ids[g] for g in gold})
        examples.append((q, [g + offset for g in gold], [d + offset for d in neg]))
    return examples, vectors, doc_ids, [qid for qid, _, _ in rows]


def training_tensors(examples, device):
    def pad(ids, n):
        return ids + [-1] * (n - len(ids))

    neg_count = max(len(neg) for _, _, neg in examples)
    return {'questions': [q for q, _, _ in examples],
            'gpid': torch.tensor([pad(g, 4) for _, g, _ in examples], device=device),
            'dpid': torch.tensor([pad(d, neg_count) for _, _, d in examples], device=device),
            'm': torch.tensor([len(g) for _, g, _ in examples], device=device)}
