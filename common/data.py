"""Loading the MultiHop-RAG corpus, chunks and queries, plus the dev/test split."""
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from . import paths


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunks(path: str = paths.CHUNKS_JSONL) -> List[Dict]:
    """Chunks made by replicate/make_chunks.py (original llama-index 0.9.40 pipeline)."""
    return load_jsonl(path)


def load_queries(path: str = paths.QUERIES_JSON) -> List[Dict]:
    """All 2,556 MultiHop-RAG queries, with a stable integer 'qid' (= position in the file)."""
    qs = load_json(path)
    for i, q in enumerate(qs):
        q["qid"] = i
    return qs


def load_corpus(path: str = paths.CORPUS_JSON) -> List[Dict]:
    return load_json(path)


def gold_facts(q: Dict) -> List[str]:
    return [e["fact"] for e in q.get("evidence_list", [])]


def is_answerable(q: Dict) -> bool:
    return q["question_type"] != "null_query"


def outlet_names(chunks_or_corpus) -> List[str]:
    """The news outlets in the corpus (49 of them appear in the evidence)."""
    return sorted({c["source"] for c in chunks_or_corpus})


def split_queries(queries: List[Dict], dev_frac: float = 0.3, seed: int = 13) -> Tuple[List[int], List[int]]:
    """Stratified dev/test split by question type.

    Near-duplicate questions (same type and the same set of evidence articles)
    are kept in the same split, so tuning on dev can't leak into test.
    Returns (dev_qids, test_qids). Null queries are included (they are skipped
    by the retrieval metrics anyway).
    """
    groups = defaultdict(list)  # (type, evidence titles) -> qids
    for q in queries:
        titles = tuple(sorted({e["title"] for e in q.get("evidence_list", [])}))
        key = (q["question_type"], titles) if titles else (q["question_type"], ("__q%d" % q["qid"],))
        groups[key].append(q["qid"])
    by_type = defaultdict(list)
    for key, qids in groups.items():
        by_type[key[0]].append(qids)
    rng = random.Random(seed)
    dev, test = [], []
    for t in sorted(by_type):
        gs = sorted(by_type[t])
        rng.shuffle(gs)
        n_total = sum(len(g) for g in gs)
        n_dev = 0
        for g in gs:
            if n_dev < dev_frac * n_total:
                dev.extend(g)
                n_dev += len(g)
            else:
                test.extend(g)
    return sorted(dev), sorted(test)


def mask_outlets(text: str, outlets: List[str], replacement: str = "a news outlet") -> str:
    """Replace outlet names in a question (used for the shortcut check)."""
    out = text
    for name in sorted(outlets, key=len, reverse=True):
        out = out.replace(name, replacement)
    return out
