"""Regression checks for the capped residual experiment."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.selection import take_top
from idea1_residual.cap_experiment import paired_interval
from idea1_residual.pipeline import active_query_keys, select
from idea1_residual.query_forms import queries_for_question, trimmed_question, valid_rewrite
from idea1_residual.spans import Span


def span(question, name, kind="target"):
    i = question.index(name)
    return Span(name, i, i + len(name), "ORG", kind)


def test_cap_is_shared_by_dense_targets_details_and_fallback():
    chunks = [{"article_id": x // 3} for x in range(15)]
    dense = list(range(15))
    first = take_top(dense, chunks, 6, max_per_article=2)
    assert first == [0, 1, 3, 4, 6, 7]
    q = "Did Alpha and Beta grow revenue?"
    spans = [span(q, "Alpha"), span(q, "Beta"), span(q, "revenue", "descriptive")]
    second = {"t0": [2, 5, 8], "t1": [2, 5, 8], "kw": [2, 5, 8]}
    for gains in ([1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]):
        keys = active_query_keys(spans, gains, 0.5, 2)
        picked, info = select(dense, {k: second[k] for k in keys}, spans, gains, 0.5, chunks, 6,
                              max_per_article=2, first_pass=first)
        assert info["queries"] == keys
        assert len(set(picked)) == 6
        assert max(Counter(chunks[i]["article_id"] for i in picked).values()) <= 2
        if not keys:
            assert picked == first


def test_trimmed_removes_list_separators_and_keeps_other_words():
    q = "Did Amazon, Google, or Netflix report higher revenue for 2023, according to The Verge?"
    spans = [span(q, name) for name in ("Amazon", "Google", "Netflix")]
    assert trimmed_question(q, spans, 1) == "Did Google report higher revenue for 2023, according to The Verge?"


def test_rewrite_requests_are_batched_and_invalid_ones_fall_back():
    q = "Did Google and Amazon report higher revenue?"
    spans = [span(q, "Google"), span(q, "Amazon"), span(q, "revenue", "descriptive")]

    class FakeRewriter:
        def rewrite(self, question, requests):
            assert question == q
            assert requests == [("Google", ["Amazon"]), ("Amazon", ["Google"])]
            return ["Did Google report higher revenue?", "Did Amazon outpace Google?"]

    got = queries_for_question(q, spans, "rewrite", keys=["t0", "t1", "kw"], rewriter=FakeRewriter())
    assert got["t0"] == {"query": "Did Google report higher revenue?", "rewrite_fallback": False}
    assert got["t1"] == {"query": "Amazon revenue", "rewrite_fallback": True}
    assert got["kw"] == queries_for_question(q, spans, "keywords")["kw"]
    assert not valid_rewrite("Did Alphabet grow?", "Google", ["Amazon"])
    assert not valid_rewrite("Did Google Cloud grow?", "Google", ["Google Cloud"])
    assert valid_rewrite("Did Google Cloud grow?", "Google Cloud", ["Google"])


def test_paired_bootstrap_of_a_constant_difference():
    ci = paired_interval(np.full(100, 0.25), draws=1000)
    assert ci == {"delta": 0.25, "lo": 0.25, "hi": 0.25}
