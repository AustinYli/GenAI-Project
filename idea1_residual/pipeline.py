"""Idea 1 decision + selection logic (no models in here, so it is cheap to re-run for many thresholds).

Per question:
  1. D1 = dense top-K for the question (first search).
  2. Every span gets a coverage gain from a scorer (see scorers.py).
  3. Targets with gain < tau are "missing", the rest "found".
     Descriptive spans with gain < tau only matter if no target is missing.
  4. Nothing missing            -> stop: answer from D1 (identical to dense RAG).
     Some targets missing        -> one second search per missing target:
                                    query = that target + all descriptive spans (question order);
                                    found targets and the other missing targets are left out.
     Only descriptive missing    -> one keyword-form search (all spans), given m0 slots.
     All second searches exclude D1 and run as one batch (2 rounds total, never a loop).
  5. Card dealing for the K slots: each round, every missing target's list gets
     one turn (first), then the D1 list gets one turn per found target.
"""
from typing import Dict, List, Optional

from common.selection import Picker, deal_turns, take_top

from .spans import Span, keyword_query


def candidate_queries(spans: List[Span]) -> Dict[str, str]:
    """All second-search queries a question could need, whatever tau is.

    key "t<i>"  -> query for target span i
    key "kw"    -> keyword form of the whole question
    """
    desc = [s for s in spans if s.kind == "descriptive"]
    out = {}
    for i, s in enumerate(spans):
        if s.kind == "target":
            keep = [s] + desc
            out[f"t{i}"] = " ".join(x.text for x in sorted(keep, key=lambda x: x.start))
    if spans:
        out["kw"] = keyword_query(spans)
    return out


def decide(spans: List[Span], gains: List[float], tau: float):
    targets = [i for i, s in enumerate(spans) if s.kind == "target"]
    missing_t = [i for i in targets if gains[i] < tau]
    found_t = [i for i in targets if gains[i] >= tau]
    missing_d = [i for i, s in enumerate(spans) if s.kind == "descriptive" and gains[i] < tau]
    return found_t, missing_t, missing_d


def active_query_keys(spans, gains, tau, m0):
    _, missing_t, missing_d = decide(spans, gains, tau)
    if missing_t:
        return [f"t{i}" for i in missing_t]
    return ["kw"] if missing_d and m0 > 0 else []


def select(d1_ranking: List[int], second: Dict[str, List[int]], spans: List[Span], gains: List[float], tau: float,
           chunks, K: int, m0: int = 2, d1_min_turns: int = 0, dedup_overlap: bool = False,
           *, max_per_article: Optional[int] = None, first_pass: Optional[List[int]] = None):
    """Final K chunk ids for one question, plus what happened."""
    found_t, missing_t, missing_d = decide(spans, gains, tau)
    first = d1_ranking if first_pass is None else first_pass
    picker = Picker(chunks, K, dedup_overlap, max_per_article=max_per_article)
    if missing_t:
        lists = [second[f"t{i}"] for i in missing_t] + [first]
        turns = [1] * len(missing_t) + [max(len(found_t), d1_min_turns)]
        if turns[-1] == 0 and all(len(l) == 0 for l in lists[:-1]):
            turns[-1] = 1
        picked = deal_turns(lists, turns, chunks, K, dedup_overlap, picker=picker)
        if not picker.full():  # lists ran out: top up from the same dense ranking
            for c in d1_ranking:
                if picker.full():
                    break
                picker.try_add(c)
        return picked, {"path": "expand_targets", "n_missing": len(missing_t), "n_found": len(found_t),
                        "queries": [f"t{i}" for i in missing_t]}
    if missing_d and "kw" in second and m0 > 0:
        p = picker
        for c in first:
            if len(p.picked) >= K - m0:
                break
            p.try_add(c)
        for c in second["kw"]:
            if p.full():
                break
            p.try_add(c)
        for c in d1_ranking:
            if p.full():
                break
            p.try_add(c)
        return p.picked, {"path": "expand_details", "n_missing": 0, "n_found": len(found_t), "queries": ["kw"]}
    picked = (first_pass if first_pass is not None else
              take_top(d1_ranking, chunks, K, dedup_overlap, max_per_article=max_per_article))
    return picked, {"path": "stop", "n_missing": 0, "n_found": len(found_t), "queries": []}


def question_signal(spans: List[Span], gains: List[float]) -> float:
    """One number per question for the H1 test: the lowest target gain (lower = more likely incomplete)."""
    t = [g for s, g in zip(spans, gains) if s.kind == "target"]
    if t:
        return min(t)
    return min(gains) if gains else 0.0
