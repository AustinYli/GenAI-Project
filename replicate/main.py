"""Replicate MultiHop-RAG Table 5 (retrieval) for bge-large-en-v1.5.

Paper (Tang & Yang 2024, Table 5), bge-large-en-v1.5:
                    MRR@10   MAP@10   Hits@10  Hits@4
  without reranker  0.4298   0.3423   0.6718   0.5221
  with reranker     0.563    0.4759   0.7183   0.6364

Pipeline = the repo's simple_retrieval.py (commit 3dd4d4e) with llama-index 0.9.40:
  chunks: data/multihop/chunks.jsonl (made by replicate/make_chunks.py with the real library)
  embed:  common/bge.py (same tokenisation, CLS pooling, normalisation, query instruction)
  search: exact cosine, top 10  (with --rerank: top 20, then bge-reranker-large keeps 10)
  eval:   common/metrics.calculate_metrics, copied from retrieval_evaluate.py

Usage (from the project folder):
  python replicate/main.py
  python replicate/main.py --rerank
Outputs go to outputs/replicate/. The chunk/query embeddings and the dense
rankings are cached in cache/ and reused by idea1_residual and idea2_arrows.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths  # noqa: E402
from common.bge import BGEEncoder, DenseIndex, encode_cached  # noqa: E402
from common.data import gold_facts, load_chunks, load_queries, split_queries  # noqa: E402
from common.metrics import calculate_metrics, coverage_metrics, format_table, metrics_by_type  # noqa: E402

PAPER = {
    "BAAI/bge-large-en-v1.5": {
        False: {"MRR@10": 0.4298, "MAP@10": 0.3423, "Hits@10": 0.6718, "Hits@4": 0.5221},
        True: {"MRR@10": 0.563, "MAP@10": 0.4759, "Hits@10": 0.7183, "Hits@4": 0.6364},
    }
}
RANKING_DEPTH = 100  # how many dense results per query to keep for the two ideas


def chunk_cache_path(model, dtype):
    return os.path.join(paths.CACHE, f"emb_{paths.model_slug(model)}_{dtype}_chunks.npy")


def query_cache_path(model, dtype, tag="queries"):
    return os.path.join(paths.CACHE, f"emb_{paths.model_slug(model)}_{dtype}_{tag}.npy")


def ranking_cache_path(model, dtype):
    return os.path.join(paths.CACHE, f"dense_ranking_{paths.model_slug(model)}_{dtype}.json")


def build_dense(model, dtype, batch_size=32, device=None):
    """Embed chunks + queries (cached) and return (chunks, queries, encoder, index, q_emb)."""
    paths.ensure(paths.CACHE)
    chunks = load_chunks()
    queries = load_queries()
    enc = BGEEncoder(model, device=device, dtype=dtype)
    t0 = time.time()
    c_emb = encode_cached(enc, [c["text"] for c in chunks], chunk_cache_path(model, dtype), is_query=False,
                          batch_size=batch_size)
    q_emb = encode_cached(enc, [q["query"] for q in queries], query_cache_path(model, dtype), is_query=True,
                          batch_size=batch_size)
    print(f"embeddings ready: {c_emb.shape[0]} chunks, {q_emb.shape[0]} queries ({time.time() - t0:.0f}s)")
    return chunks, queries, enc, DenseIndex(c_emb, device=device), q_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                    help="fp32 = what the original ran; fp16 is faster")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-large")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    chunks, queries, enc, index, q_emb = build_dense(a.model, a.dtype, a.batch_size, a.device)
    del enc

    # dense rankings for everything downstream (depth 100), cached
    ids, scores = index.search(q_emb, RANKING_DEPTH)
    with open(ranking_cache_path(a.model, a.dtype), "w") as f:
        json.dump({"model": a.model, "dtype": a.dtype, "ids": ids, "scores": scores}, f)

    if a.rerank:
        from common.rerank import CrossEncoderReranker
        rr = CrossEncoderReranker(a.reranker, device=a.device)
        final = []
        from tqdm import tqdm
        for q, row in tqdm(list(zip(queries, ids)), desc="rerank"):
            cand = row[:20]
            final.append(rr.rerank(q["query"], cand, [chunks[c]["text"] for c in cand], a.topk))
        final_scores = [[None] * len(r) for r in final]
    else:
        final = [row[:a.topk] for row in ids]
        final_scores = [row[:a.topk] for row in scores]

    # save in the original repo's output format
    out_dir = paths.ensure(os.path.join(paths.OUT, "replicate"))
    name = a.model.split("/")[-1] + ("_rerank" if a.rerank else "") + ("" if a.dtype == "fp32" else "_" + a.dtype)
    save = []
    for q, row, sc in zip(queries, final, final_scores):
        save.append({
            "query": q["query"], "answer": q["answer"], "question_type": q["question_type"],
            "retrieval_list": [{"text": chunks[c]["text"], "score": s} for c, s in zip(row, sc)],
            "gold_list": q["evidence_list"],
        })
    out_file = os.path.join(out_dir, f"{name}_retrieval_test.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save, f)

    # evaluate exactly like retrieval_evaluate.py (null queries skipped)
    retrieved, gold = [], []
    for d in save:
        if d["question_type"] == "null_query":
            continue
        retrieved.append([m["text"] for m in d["retrieval_list"]])
        gold.append([m["fact"] for m in d["gold_list"]])
    m = calculate_metrics(retrieved, gold)
    extra = coverage_metrics(retrieved, gold, ks=(4, 10))

    paper = PAPER.get(a.model, {}).get(a.rerank)
    print(f"\n=== {name}: {len(gold)} answerable queries ===")
    rows = [{"system": "ours", **m}]
    if paper:
        rows.insert(0, {"system": "paper (Table 5)", **paper})
        rows.append({"system": "difference", **{k: m[k] - paper[k] for k in paper}})
    table = format_table(rows, ["MRR@10", "MAP@10", "Hits@10", "Hits@4"])
    print(table)
    print("\nextra (not in the paper): " + ", ".join(f"{k}={v:.4f}" for k, v in extra.items()))

    recs = [{"question_type": d["question_type"], "retrieved": [m_["text"] for m_ in d["retrieval_list"]],
             "gold": [g["fact"] for g in d["gold_list"]]} for d in save]
    by_type = metrics_by_type(recs, ks=(4, 10))
    dev, test = split_queries(queries)
    summary = {"model": a.model, "dtype": a.dtype, "rerank": a.rerank, "n_chunks": len(chunks),
               "metrics": m, "extra": extra, "paper": paper, "by_type": by_type,
               "split_sizes": {"dev": len(dev), "test": len(test)}, "output_file": out_file}
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, f"{name}_metrics.md"), "w") as f:
        f.write(f"# Replication: {name}\n\n{table}\n\nChunks: {len(chunks)}\n")
    print(f"\nsaved {out_file}")


if __name__ == "__main__":
    main()
