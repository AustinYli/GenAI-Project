"""Collect fixed-split retrieval metrics from the replication and Ideas 1 and 2."""

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.data import gold_facts, load_chunks, load_queries, split_queries  # noqa: E402
from common.metrics import calculate_metrics  # noqa: E402
from idea1_residual.pipeline import select  # noqa: E402
from idea1_residual.spans import Span  # noqa: E402


METRICS = ("Hits@10", "MRR@10", "MAP@10", "Hits@4")
DENSE = "Dense baselines"
RESIDUAL = "Idea 1 — residual retrieval"
ARROWS = "Idea 2 — arrows and outlet filter"
MASKED = "Idea 2 — masked-outlet diagnostics"
DENSE_RANKING = "cache/dense_ranking_BAAI__bge-large-en-v1.5_fp32.json"


def read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def result(group, method, source, source_key, n, scores, rankings=None, note=None):
    row = {
        "group": group,
        "method": method,
        "source": source,
        "source_key": source_key,
        "n": n,
        **{metric: scores[metric] for metric in METRICS},
    }
    if rankings is not None:
        assert len(rankings) == n
        assert all(len(ids) == len(set(ids)) == 10 for ids in rankings)
        payload = json.dumps(rankings, separators=(",", ":")).encode("utf-8")
        row["ranking_hash"] = hashlib.sha256(payload).hexdigest()
    if note:
        row["note"] = note
    return row


def assert_matches(scored, archived):
    for metric in METRICS:
        assert math.isclose(scored[metric], archived[metric], rel_tol=1e-12, abs_tol=1e-14), metric


def replication(queries, chunks, test, score, dense):
    by_text = {chunk["text"]: chunk["id"] for chunk in chunks}
    assert len(by_text) == len(chunks)
    for name, suffix in (("BGE dense", ""), ("BGE + BGE reranker", "_rerank")):
        path = f"outputs/replicate/bge-large-en-v1.5{suffix}_retrieval_test.json"
        archive = read_json(path)
        assert len(archive) == len(queries)
        for entry, question in zip(archive, queries):
            assert entry["query"] == question["query"]
            assert entry["question_type"] == question["question_type"]
            assert entry["gold_list"] == question["evidence_list"]
        rankings = [[by_text[row["text"]] for row in archive[q]["retrieval_list"]] for q in test]
        if not suffix:
            assert rankings == [dense[q][:10] for q in test]
        yield result(
            DENSE, name, path, "retrieval_list indexed by fixed test qids", len(test),
            score(rankings), rankings,
            "Rescored on the fixed 1,574 questions from the saved all-question export.",
        )


def idea1_uncapped(queries, chunks, test, score, dense):
    files = {
        "word_K10": "Word overlap",
        "llm_inplace__Qwen__Qwen2.5-3B-Instruct_K10": "Qwen in-place",
        "llm_probe__Qwen__Qwen2.5-3B-Instruct_K10": "Qwen probe",
        "mlm__answerdotai__ModernBERT-large_K10": "ModernBERT masked LM",
    }
    span_cache = read_json("cache/idea1/spans.json")
    spans = {q: [Span.from_dict(entry) for entry in span_cache[str(q)]] for q in test}
    search_cache = read_json("cache/idea1/second_K10.json")
    second = {q: {key: entry["ids"] for key, entry in search_cache[str(q)].items()} for q in test}
    shared_control = None
    for stem, label in files.items():
        path = f"outputs/idea1/{stem}.json"
        archive = read_json(path)
        logs = read_json(f"outputs/idea1/{stem}_per_question.json")
        assert archive["n_test"] == len(test)
        assert [entry["qid"] for entry in logs] == test
        assert [entry["query"] for entry in logs] == [queries[q]["query"] for q in test]
        gains = read_json(f"cache/idea1/gains_{stem}.json")
        for index, threshold in ((1, float(archive["tau"])), (2, math.inf)):
            archived_row = archive["table"][index]
            rankings = [
                select(
                    dense[q], second[q], spans[q], gains[str(q)]["gains"], threshold, chunks, 10,
                    m0=archive["args"]["m0"], dedup_overlap=archive["args"]["dedup_overlap"],
                )[0]
                for q in test
            ]
            scores = score(rankings)
            assert_matches(scores, archived_row)
            if index == 2 and shared_control is not None:
                assert rankings == shared_control
                continue
            if index == 2:
                shared_control = rankings
            name = f"{label}, tuned" if index == 1 else "Always expand, uncapped"
            note = (
                "Replayed from cached rankings, spans and gains. All four scorers produced identical control rankings."
                if index == 2 else "Replayed from cached rankings, spans, gains and the frozen dev threshold."
            )
            yield result(RESIDUAL, name, path, f"table[{index}]", len(test), scores, rankings, note)


def idea1_capped(queries, test, score):
    forms = (
        ("word_K10_cap2_test", "Word overlap + cap 2, keyword list"),
        ("word_K10_cap2_trimmed_test", "Word overlap + cap 2, trimmed sentence"),
        ("word_K10_cap2_rewrite_Qwen__Qwen2.5-1.5B-Instruct_test", "Word overlap + cap 2, Qwen 1.5B rewrite"),
    )
    baseline = None
    for stem, label in forms:
        base = f"outputs/idea1_cap/{stem}"
        path = f"{base}_per_question.json"
        log = read_json(path)
        archive = read_json(f"{base}.json")
        assert archive["n_test"] == len(test)
        assert [entry["qid"] for entry in log] == test
        assert [entry["query"] for entry in log] == [queries[q]["query"] for q in test]
        dense_rankings = [entry["d1_chunk_ids"] for entry in log]
        assert (baseline is None) or (dense_rankings == baseline)
        if baseline is None:
            baseline = dense_rankings
            scores = score(baseline)
            assert_matches(scores, archive["table"][0])
            yield result(DENSE, "BGE dense + cap 2", path, "d1_chunk_ids", len(test), scores, baseline)
        rankings = [entry["selected_chunk_ids"] for entry in log]
        scores = score(rankings)
        assert_matches(scores, archive["table"][1])
        yield result(RESIDUAL, label, path, "selected_chunk_ids", len(test), scores, rankings)


def idea2(n, rows):
    labels = {
        "dense top-K": "BGE dense",
        "dense <=2 per article": "BGE dense + cap 2",
        "outlet filter (rival)": "Outlet-restricted search",
        "arrows (soft slots)": "Arrows, soft slots",
        "arrows (hard count)": "Arrows, predicted count",
        "arrows (oracle count)": "Arrows, oracle count",
    }
    baseline = {row["method"]: row for row in rows if row["group"] == DENSE}
    for masked in (False, True):
        path = f"outputs/idea2/eval_K10{'_masked' if masked else ''}.json"
        archive = read_json(path)
        assert archive["args"]["K"] == 10
        assert archive["args"]["mask_outlets"] == masked
        for index, row in enumerate(archive["splits"]["test"]["table"]):
            assert row["n"] == n
            label = labels[row["system"]]
            if not masked and label in ("BGE dense", "BGE dense + cap 2"):
                assert_matches(row, baseline[label])
                continue
            note = None
            if row["system"] == "arrows (oracle count)":
                note = "Gold-informed diagnostic. Uses the true number of evidence articles."
            elif masked and row["system"] == "outlet filter (rival)":
                note = "The masked questions produced the same scores as masked BGE dense."
            yield result(MASKED if masked else ARROWS, label, path, f"splits.test.table[{index}]", n, row, note=note)


def main():
    queries = load_queries()
    chunks = load_chunks()
    assert all(chunk["id"] == i for i, chunk in enumerate(chunks))
    test = [q for q in split_queries(queries, seed=13)[1] if queries[q]["question_type"] != "null_query"]
    assert len(test) == 1574 and test == sorted(set(test))
    gold = [gold_facts(queries[q]) for q in test]
    dense = read_json(DENSE_RANKING)["ids"]

    def score(rankings):
        assert len(rankings) == len(test)
        return calculate_metrics([[chunks[c]["text"] for c in ids] for ids in rankings], gold)

    rows = list(replication(queries, chunks, test, score, dense))
    rows.extend(idea1_capped(queries, test, score))
    rows.extend(idea1_uncapped(queries, chunks, test, score, dense))
    rows.extend(idea2(len(test), rows))
    path = Path(__file__).with_name("early.json")
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} results for {len(test)} questions to {path}")


if __name__ == "__main__":
    main()
