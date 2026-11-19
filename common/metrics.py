"""Retrieval metrics.

`calculate_metrics` is copied line for line from MultiHop-RAG's
retrieval_evaluate.py (commit 3dd4d4e), so replicated numbers are computed
exactly the way the paper's repo computes them:
  * a chunk is relevant if a gold fact string (spaces/newlines removed) is a
    substring of the chunk text (spaces/newlines removed);
  * Hits@k = at least ONE gold fact in the top k;
  * MAP@10 is the repo's non-standard version.

The extra metrics below are what the two ideas are judged on:
  * complete@K  = ALL gold facts of the question appear in the top K
  * recall@K    = fraction of gold facts that appear in the top K
"""
from collections import defaultdict
from typing import Dict, List, Sequence


def calculate_metrics(retrieved_lists, gold_lists):
    hits_at_10_count = 0
    hits_at_4_count = 0
    map_at_10_list = []
    mrr_list = []

    for retrieved, gold in zip(retrieved_lists, gold_lists):
        hits_at_10_flag = False
        hits_at_4_flag = False
        average_precision_sum = 0
        first_relevant_rank = None
        find_gold = []

        gold = [item.replace(" ", "").replace("\n", "") for item in gold]
        retrieved = [item.replace(" ", "").replace("\n", "") for item in retrieved]

        for rank, retrieved_item in enumerate(retrieved[:11], start=1):
            if any(gold_item in retrieved_item for gold_item in gold):
                if rank <= 10:
                    hits_at_10_flag = True
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    if rank <= 4:
                        hits_at_4_flag = True
                    # Compute precision at this rank for this query
                    count = 0
                    for gold_item in gold:
                        if gold_item in retrieved_item and not gold_item in find_gold:
                            count = count + 1
                            find_gold.append(gold_item)
                    precision_at_rank = count / rank
                    average_precision_sum += precision_at_rank

        # Calculate metrics for this query
        hits_at_10_count += int(hits_at_10_flag)
        hits_at_4_count += int(hits_at_4_flag)
        map_at_10_list.append(average_precision_sum / min(len(gold), 10))
        mrr_list.append(1 / first_relevant_rank if first_relevant_rank else 0)

    # Calculate average metrics over all queries
    hits_at_10 = hits_at_10_count / len(gold_lists)
    hits_at_4 = hits_at_4_count / len(gold_lists)
    map_at_10 = sum(map_at_10_list) / len(gold_lists)
    mrr_at_10 = sum(mrr_list) / len(gold_lists)

    return {
        "Hits@10": hits_at_10,
        "Hits@4": hits_at_4,
        "MAP@10": map_at_10,
        "MRR@10": mrr_at_10,
    }


def _norm(s: str) -> str:
    return s.replace(" ", "").replace("\n", "")


def facts_found(retrieved_texts: Sequence[str], gold: Sequence[str], k: int) -> List[bool]:
    top = [_norm(t) for t in retrieved_texts[:k]]
    return [any(_norm(g) in t for t in top) for g in gold]


def coverage_metrics(retrieved_lists: List[List[str]], gold_lists: List[List[str]], ks=(4, 6, 10)) -> Dict[str, float]:
    out = {}
    for k in ks:
        complete, recall = [], []
        for r, g in zip(retrieved_lists, gold_lists):
            if not g:
                continue
            f = facts_found(r, g, k)
            complete.append(all(f))
            recall.append(sum(f) / len(f))
        out[f"complete@{k}"] = sum(complete) / max(1, len(complete))
        out[f"recall@{k}"] = sum(recall) / max(1, len(recall))
    return out


def all_metrics(retrieved_lists, gold_lists, ks=(4, 6, 10)) -> Dict[str, float]:
    m = calculate_metrics(retrieved_lists, gold_lists)
    m.update(coverage_metrics(retrieved_lists, gold_lists, ks))
    m["n"] = len(gold_lists)
    return m


def metrics_by_type(records: List[Dict], ks=(4, 6, 10)) -> Dict[str, Dict[str, float]]:
    """records: dicts with 'question_type', 'retrieved' (texts), 'gold' (facts). Null queries skipped."""
    groups = defaultdict(lambda: ([], []))
    for r in records:
        if r["question_type"] == "null_query":
            continue
        for key in ("all", r["question_type"]):
            groups[key][0].append(r["retrieved"])
            groups[key][1].append(r["gold"])
    return {k: all_metrics(v[0], v[1], ks) for k, v in groups.items()}


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Probability that a random positive gets a higher score than a random negative (ties = 0.5)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks with ties
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[t] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def format_table(rows: List[Dict], cols: List[str], name_key: str = "system") -> str:
    head = [name_key] + cols
    lines = [" | ".join(head), " | ".join(["---"] * len(head))]
    for r in rows:
        vals = [str(r.get(name_key, ""))]
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(" | ".join(vals))
    return "\n".join(lines)
