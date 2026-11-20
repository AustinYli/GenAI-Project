"""The fixed MultiHop-RAG retrieval pipeline for split training and evaluation."""

from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.bge import BGEEncoder, DenseIndex, _texts_hash
from common.data import gold_facts, load_chunks, load_queries, outlet_names
from common.metrics import facts_found
from hybrid_claims import run
from hybrid_claims.llm_compare import BUDGET_RULE, read_json, valid_lines
from hybrid_claims.llm_generate import DESIGN
from hipporag_public.baselines import E5, E5_REVISION


OUT = Path(run.paths.OUT) / 'split_training' / 'retrieval'
METRICS = ('complete@10', 'complete@6', 'recall@10', 'Hits@10')
CACHE = Path(run.paths.CACHE)


class _RankCache:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute('CREATE TABLE IF NOT EXISTS rankings (scope TEXT, text TEXT, ranking TEXT, '
                        'PRIMARY KEY (scope, text))')

    def get(self, scope, texts):
        results = {}
        for start in range(0, len(texts), 400):
            batch = texts[start:start + 400]
            placeholders = ','.join('?' for _ in batch)
            for text, payload in self.db.execute(
                f'SELECT text, ranking FROM rankings WHERE scope = ? AND text IN ({placeholders})',
                [scope, *batch],
            ):
                results[text] = json.loads(payload)
        return results

    def store(self, scope, rankings):
        with self.db:
            self.db.executemany('INSERT OR REPLACE INTO rankings VALUES (?, ?, ?)',
                                ((scope, text, json.dumps(ranks, separators=(',', ':')))
                                 for text, ranks in rankings.items()))


class Retrieval:
    """Score raw split outputs with the existing filter, fusion, and exact-K selection."""

    def __init__(self, backend='bge', device='cuda', out_dir=None):
        if backend not in ('bge', 'e5'):
            raise ValueError(f'Unsupported embedding backend {backend!r}')
        self.backend = backend
        self.device = device
        self.out_dir = Path(out_dir) if out_dir is not None else OUT
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.queries = load_queries()
        self.chunks = load_chunks()
        self.outlets = sorted(outlet_names(self.chunks), key=len, reverse=True)
        protocol = (f'v1|{run.MODEL}|{E5_REVISION}|{DESIGN["whole_depth"]}|{DESIGN["extra_depth"]}|'
                    f'{DESIGN["rrf_k"]}|{DESIGN["bm25_k1"]}|{DESIGN["bm25_b"]}|'
                    f'{BUDGET_RULE["backfill_depth"]}|{DESIGN["cap"]}')
        self.corpus_key = _texts_hash((run.chunk_doc(c) for c in self.chunks), protocol)
        self.cache = _RankCache(self.out_dir / f'rankings_{self.corpus_key}.sqlite3')
        self._encoder = self._index = self._bm25 = self._bge_whole = None
        self.whole = {}
        self.backfill = {}
        self._gold = {}
        self._rules = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.cache.db.close()
        self._encoder = self._index = self._bm25 = self._bge_whole = None
        torch = sys.modules.get('torch')
        if torch is not None and str(self.device).startswith('cuda') and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def _scope(self, key):
        return f'{self.backend}|{self.device}|{key}'

    def _encoder_model(self):
        if self._encoder is None:
            self._encoder = (BGEEncoder(device=self.device, dtype='fp32') if self.backend == 'bge'
                             else E5(self.device))
        return self._encoder

    def _index_model(self):
        if self._index is None:
            if self.backend == 'bge':
                path = CACHE / f'emb_{run.MODEL}_fp32_chunks.npy'
                vectors = np.load(path)
            else:
                path = self.out_dir / f'e5_passages_{self.corpus_key}.npy'
                if path.exists():
                    vectors = np.load(path)
                else:
                    print(f'Embedding {len(self.chunks):,} MultiHop-RAG chunks with e5-large-v2', flush=True)
                    vectors = self._encoder_model().embed([c['text'] for c in self.chunks], 'passage: ', 48)
                    temporary = path.with_suffix('.tmp.npy')
                    np.save(temporary, vectors)
                    temporary.replace(path)
            if vectors.shape != (len(self.chunks), 1024) or vectors.dtype != np.float32:
                raise ValueError(f'Unexpected dimensions or dtype in {path}')
            self._index = DenseIndex(vectors, device=self.device)
        return self._index

    def _lexical(self, text, depth):
        if self._bm25 is None:
            self._bm25 = run.BM25([run.chunk_doc(c) for c in self.chunks], DESIGN['bm25_k1'], DESIGN['bm25_b'])
        return self._bm25.search(text, depth)

    def _query_vectors(self, texts):
        model = self._encoder_model()
        if self.backend == 'bge':
            return model.encode(texts, is_query=True)
        return model.embed(texts, 'query: ', 48)

    def _fuse(self, text, dense_ids, depth):
        return run.rrf([dense_ids, self._lexical(text, depth)], k=DESIGN['rrf_k'])

    def prepare(self, qids):
        """Prepare only the requested question IDs and their common whole-question lists."""
        pending = list(dict.fromkeys(int(q) for q in qids if int(q) not in self.whole))
        if not pending:
            return
        for q in pending:
            if q < 0 or q >= len(self.queries) or self.queries[q]['question_type'] == 'null_query':
                raise ValueError(f'Question {q} is outside the answerable benchmark')
            self._gold[q] = gold_facts(self.queries[q])
        texts = {q: self.queries[q]['query'] for q in pending}
        keys = {q: f'{q}\n{texts[q]}' for q in pending}
        whole_scope = self._scope('whole')
        cached = self.cache.get(whole_scope, list(keys.values()))
        need = [q for q in pending if keys[q] not in cached]
        vectors = {}
        depth = DESIGN['whole_depth']
        if need:
            if self.backend == 'bge':
                if self._bge_whole is None:
                    self._bge_whole = read_json(CACHE / f'dense_ranking_{run.MODEL}_fp32.json')['ids']
                dense = [self._bge_whole[q][:depth] for q in need]
            else:
                index = self._index_model()
                embeddings = self._query_vectors([texts[q] for q in need])
                vectors = dict(zip(need, embeddings, strict=True))
                dense, _ = index.search(embeddings, depth)
            whole = {keys[q]: self._fuse(texts[q], ids, depth) for q, ids in zip(need, dense, strict=True)}
            self.cache.store(whole_scope, whole)
            cached.update(whole)
        max_k = max(DESIGN['cuts'])
        short = [q for q in pending
                 if len(run.pick([cached[keys[q]]], max_k, self.chunks, DESIGN['cap'])) < max_k]
        self._prepare_backfill(short, texts, keys, vectors)
        self.whole.update((q, cached[keys[q]]) for q in pending)

    def _prepare_backfill(self, qids, texts, keys, vectors):
        scope = self._scope('backfill')
        backfill = self.cache.get(scope, [keys[q] for q in qids])
        needed = [q for q in qids if keys[q] not in backfill]
        if needed:
            if self.backend == 'bge':
                query_vectors = np.load(CACHE / f'emb_{run.MODEL}_fp32_queries.npy', mmap_mode='r')[needed]
            else:
                missing = [q for q in needed if q not in vectors]
                if missing:
                    embeddings = self._query_vectors([texts[q] for q in missing])
                    vectors.update(zip(missing, embeddings, strict=True))
                query_vectors = np.stack([vectors[q] for q in needed])
            depth = BUDGET_RULE['backfill_depth']
            ids, _ = self._index_model().search(query_vectors, depth)
            built = {keys[q]: self._fuse(texts[q], ranks, depth) for q, ranks in zip(needed, ids, strict=True)}
            self.cache.store(scope, built)
            backfill.update(built)
        self.backfill.update((q, backfill[keys[q]]) for q in qids)

    def _rule_claims(self, qid):
        if qid not in self._rules:
            self._rules[qid] = run.claims(self.queries[qid]['query'], self.outlets)
        return self._rules[qid]

    def _claim_lists(self, texts, hybrid):
        scope = self._scope('claim_hybrid') if hybrid else 'claim_lexical'
        distinct = list(dict.fromkeys(texts))
        by_text = self.cache.get(scope, distinct)
        pending = [t for t in distinct if t not in by_text]
        depth = DESIGN['extra_depth']
        for start in range(0, len(pending), 1024):
            batch = pending[start:start + 1024]
            if hybrid:
                embeddings = self._query_vectors(batch)
                dense, _ = self._index_model().search(embeddings, depth)
                built = {text: self._fuse(text, ids, depth)[:depth]
                         for text, ids in zip(batch, dense, strict=True)}
            else:
                built = {text: self._lexical(text, depth) for text in batch}
            self.cache.store(scope, built)
            by_text.update(built)
        return by_text

    def _select(self, qid, extras, subquestions, fallback):
        ids_by_k = {}
        filled = {}
        for k in DESIGN['cuts']:
            ids = run.pick([self.whole[qid], *extras], k, self.chunks, DESIGN['cap'])
            filled[k] = k - len(ids)
            if filled[k]:
                ids = run.pick([ids + self.backfill[qid]], k, self.chunks, DESIGN['cap'])
            ids_by_k[k] = ids
        return self.score_contexts(qid, ids_by_k, subquestions, fallback, backfilled_by_k=filled)

    def score_contexts(self, qid, ids_by_k, subquestions=(), fallback=False, *, backfilled_by_k=None):
        """Independently score previously saved contexts, using the same support matching."""
        facts = self._gold.get(qid) or gold_facts(self.queries[qid])
        if not facts:
            raise ValueError(f'Question {qid} has no gold evidence')
        row = {'qid': qid, 'type': self.queries[qid]['question_type'],
               'subquestions': list(subquestions), 'fallback': bool(fallback)}
        for k in DESIGN['cuts']:
            ids = ids_by_k[k]
            if (len(ids) != k or len(set(ids)) != k or any(not 0 <= c < len(self.chunks) for c in ids)
                    or max(Counter(
                        self.chunks[c]['article_id'] for c in ids).values()) > DESIGN['cap']):
                raise ValueError(f'Budget or article cap violated on question {qid}')
            found = facts_found([self.chunks[c]['text'] for c in ids], facts, k)
            row[f'ids@{k}'] = ids
            row[f'complete@{k}'] = int(all(found))
            row[f'recall@{k}'] = sum(found) / len(found)
            if k == 10:
                row['Hits@10'] = int(any(found))
            if backfilled_by_k is not None:
                row[f'backfilled@{k}'] = backfilled_by_k[k]
        return row

    def score_samples(self, samples):
        """Return lists of scored rows in the same order as each question's completions."""
        samples = {int(q): completions for q, completions in samples.items()}
        self.prepare(samples)
        parsed = {}
        for q, completions in samples.items():
            parsed[q] = []
            for raw in completions:
                lines = valid_lines(raw)
                parsed[q].append((lines or self._rule_claims(q), not lines))
        rankings = self._claim_lists((text for group in parsed.values() for lines, _ in group for text in lines), True)
        return {q: [self._select(q, [rankings[text] for text in lines], lines, fallback)
                    for lines, fallback in group] for q, group in parsed.items()}

    def score_baselines(self, qids):
        """A uses only the whole question. B adds lexical searches for the rule-based claims."""
        qids = [int(q) for q in qids]
        self.prepare(qids)
        rules = {q: self._rule_claims(q) for q in qids}
        rankings = self._claim_lists((text for texts in rules.values() for text in texts), False)
        return {q: {'A': self._select(q, [], [], False),
                    'B': self._select(q, [rankings[text] for text in lines], lines, False)}
                for q, lines in rules.items()}


def _record_list(records):
    return list(records.values()) if isinstance(records, dict) else list(records)


def summarize(records):
    """Aggregate scorer records overall and within each question type."""
    rows = _record_list(records)
    if not rows:
        raise ValueError('Cannot summarize an empty set of questions')

    def stats(group):
        values = {key: float(np.mean([row[key] for row in group])) for key in METRICS}
        return dict(values, n=len(group), mean_subquestions=float(np.mean([len(r['subquestions']) for r in group])),
                    fallback_rate=float(np.mean([r['fallback'] for r in group])))

    by_type = {kind: stats([r for r in rows if r['type'] == kind])
               for kind in sorted({r['type'] for r in rows})}
    return {'overall': stats(rows), 'by_type': by_type}


def paired_ci(records, reference):
    """Per-metric paired deltas and 95% confidence intervals, using the old bootstrap seed."""
    rows, refs = _record_list(records), _record_list(reference)
    qids = [row['qid'] for row in rows]
    by_id = {row['qid']: row for row in refs}
    if not rows or len(set(qids)) != len(rows) or len(by_id) != len(refs) or set(qids) != by_id.keys():
        raise ValueError('Paired comparisons require the same unique question IDs')
    differences = np.array([[row[key] - by_id[row['qid']][key] for key in METRICS] for row in rows])
    rng = np.random.default_rng(13)
    means = [differences[rng.integers(len(rows), size=(250, len(rows)))].mean(axis=1) for _ in range(40)]
    bounds = np.quantile(np.concatenate(means), (0.025, 0.975), axis=0)
    return {key: [float(differences[:, i].mean()), float(bounds[0, i]), float(bounds[1, i])]
            for i, key in enumerate(METRICS)}
