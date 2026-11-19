"""Turning several ranked lists into one final context of K chunks.

Both ideas end the same way: some ranked lists of chunk ids, a budget of K
slots, and a rule for who gets which slot. Every system (including the dense
baseline) goes through `Picker`, so duplicate handling is identical.
"""
from typing import Dict, List, Optional, Sequence


class Picker:
    """Accepts chunks one at a time, rejecting exact repeats and (optionally) near-duplicates.

    With 256-token chunks and 200-token overlap, neighbouring chunks of the same
    article share ~86% of their text. dedup_overlap=True rejects a chunk if it
    overlaps an already-picked chunk of the same article by more than
    `overlap_frac` of the shorter one.
    `max_per_article` counts accepted chunks across all ranked lists.
    """

    def __init__(self, chunks: List[Dict], k: int, dedup_overlap: bool = False, overlap_frac: float = 0.5,
                 *, max_per_article: Optional[int] = None):
        self.chunks = chunks
        self.k = k
        self.dedup = dedup_overlap
        self.frac = overlap_frac
        self.picked: List[int] = []
        self._ids = set()
        self.max_per_article = max_per_article
        self._article_counts = {}

    def full(self) -> bool:
        return len(self.picked) >= self.k

    def _near_dup(self, cid: int) -> bool:
        c = self.chunks[cid]
        for p in self.picked:
            d = self.chunks[p]
            if d["article_id"] != c["article_id"]:
                continue
            ov = min(c["end_char"], d["end_char"]) - max(c["start_char"], d["start_char"])
            short = max(1, min(c["end_char"] - c["start_char"], d["end_char"] - d["start_char"]))
            if ov / short > self.frac:
                return True
        return False

    def try_add(self, cid: int) -> bool:
        if self.full() or cid in self._ids:
            return False
        if self.max_per_article is not None:
            article = self.chunks[cid]["article_id"]
            count = self._article_counts.get(article, 0)
            if count >= self.max_per_article:
                return False
        if self.dedup and self._near_dup(cid):
            return False
        if self.max_per_article is not None:
            self._article_counts[article] = count + 1
        self.picked.append(cid)
        self._ids.add(cid)
        return True


def take_top(ranked: Sequence[int], chunks, k: int, dedup_overlap: bool = False,
             *, max_per_article: Optional[int] = None) -> List[int]:
    p = Picker(chunks, k, dedup_overlap, max_per_article=max_per_article)
    for c in ranked:
        if p.full():
            break
        p.try_add(c)
    return p.picked


def deal_turns(lists: List[Sequence[int]], turns: List[int], chunks, k: int, dedup_overlap: bool = False,
               picker: Optional[Picker] = None) -> List[int]:
    """Card dealing with a fixed number of turns per round for each list.

    Each round, list i gets `turns[i]` picks (lists are visited in order, so put
    the lists that must not be starved first). On a turn, a list gives its next
    best chunk that the picker accepts. Stops when K chunks are picked or every
    list is exhausted.
    """
    p = picker or Picker(chunks, k, dedup_overlap)
    pos = [0] * len(lists)
    while not p.full():
        progressed = False
        for i, lst in enumerate(lists):
            for _ in range(turns[i]):
                if p.full():
                    break
                while pos[i] < len(lst):
                    c = lst[pos[i]]
                    pos[i] += 1
                    if p.try_add(c):
                        progressed = True
                        break
        if not progressed:
            break
    return p.picked


def largest_remainder(weights: Sequence[float], k: int, min_one: Optional[Sequence[bool]] = None) -> List[int]:
    """Split k slots in proportion to weights (rounded so they sum to k)."""
    tot = sum(weights)
    if tot <= 0:
        base = [0] * len(weights)
        base[0] = k
        return base
    raw = [k * w / tot for w in weights]
    q = [int(x) for x in raw]
    rem = k - sum(q)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - q[i], reverse=True)
    for i in order[:rem]:
        q[i] += 1
    if min_one:
        for i, need in enumerate(min_one):
            if need and q[i] == 0:
                j = max(range(len(q)), key=lambda t: q[t])
                if q[j] > 1:
                    q[j] -= 1
                    q[i] += 1
    return q


def deal_quotas(lists: List[Sequence[int]], quotas: List[int], chunks, k: int, dedup_overlap: bool = False) -> List[int]:
    """Round-robin over lists, each up to its quota; leftover slots (exhausted lists) go round-robin to all."""
    p = Picker(chunks, k, dedup_overlap)
    pos = [0] * len(lists)
    used = [0] * len(lists)

    def give(i):
        while pos[i] < len(lists[i]):
            c = lists[i][pos[i]]
            pos[i] += 1
            if p.try_add(c):
                return True
        return False

    while not p.full():
        progressed = False
        for i in range(len(lists)):
            if used[i] < quotas[i] and not p.full() and give(i):
                used[i] += 1
                progressed = True
        if not progressed:
            break
    while not p.full():
        progressed = False
        for i in range(len(lists)):
            if not p.full() and give(i):
                progressed = True
        if not progressed:
            break
    return p.picked
