"""Checks that the slot rules do what the design says (no models needed).

Run:  python tests/test_selection.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.metrics import calculate_metrics  # noqa: E402
from common.selection import deal_quotas, largest_remainder, take_top  # noqa: E402
from idea1_residual.pipeline import candidate_queries, select  # noqa: E402
from idea1_residual.spans import Span  # noqa: E402

CH = [{"article_id": i, "start_char": 0, "end_char": 10} for i in range(1000)]
D1, G, A = list(range(0, 100)), list(range(100, 200)), list(range(200, 300))


def count(picked):
    return (sum(c < 100 for c in picked), sum(100 <= c < 200 for c in picked), sum(200 <= c < 300 for c in picked))


def test_idea1_slots():
    sp = [Span("Netflix", 4, 11, "ORG", "target"), Span("Google", 15, 21, "ORG", "target"),
          Span("revenue", 29, 36, "NP", "descriptive")]
    assert candidate_queries(sp) == {"t0": "Netflix revenue", "t1": "Google revenue", "kw": "Netflix Google revenue"}
    # Netflix found, Google missing -> 3 + 3
    assert count(select(D1, {"t1": G}, sp, [5.0, 0.1, 3.0], 1.0, CH, 6)[0]) == (3, 3, 0)
    # both missing -> 3 each from their own searches
    assert count(select(D1, {"t0": A, "t1": G}, sp, [0.1, 0.2, 3.0], 1.0, CH, 6)[0]) == (0, 3, 3)
    # nothing missing -> stop, identical to dense
    picked, info = select(D1, {}, sp, [5, 5, 5], 1.0, CH, 6)
    assert info["path"] == "stop" and picked == D1[:6]
    # only a detail missing -> m0 = 2 slots for the keyword search
    assert count(select(D1, {"kw": G}, sp, [5, 5, 0.1], 1.0, CH, 6, m0=2)[0]) == (4, 2, 0)
    # 1 found, 2 missing -> 2 each
    sp3 = sp[:2] + [Span("Amazon", 23, 29, "ORG", "target")] + sp[2:]
    assert count(select(D1, {"t1": G, "t2": A}, sp3, [5.0, 0.1, 0.2, 3.0], 1.0, CH, 6)[0]) == (2, 2, 2)


def test_soft_slots():
    assert largest_remainder([1, 1, 0.4, 0.1], 6) == [3, 2, 1, 0]
    assert deal_quotas([D1, G, A], [3, 2, 1], CH, 6) == [0, 100, 200, 1, 101, 2]


def test_dedup():
    ch = [{"article_id": 0, "start_char": i * 10, "end_char": i * 10 + 100} for i in range(50)]
    assert take_top(list(range(50)), ch, 5, dedup_overlap=True) == [0, 5, 10, 15, 20]


def test_metrics():
    m = calculate_metrics([["xx fact one yy", "zz"], ["nothing"]], [["fact one"], ["fact two"]])
    assert m["Hits@10"] == 0.5 and m["MRR@10"] == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
