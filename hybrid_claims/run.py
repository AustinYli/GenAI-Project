"""Training-free, LLM-free retrieval for MultiHop-RAG: hybrid search + one search per claim + 2-per-article cap.

Method (per question):
  1. Hybrid list: BGE ranking (cached by replicate/main.py) fused with BM25 by reciprocal-rank fusion (k=60).
  2. Claim lists: split the question into its claims at commas / "and" / "while" / ";" etc. (MultiHop-RAG
     questions list one claim per evidence article). If the question names exactly one outlet, append it to
     claims that don't name one. One BM25 search per claim (top 60).
  3. Fill K slots round-robin: hybrid list, then each claim list, one pick each per round; skip repeats and
     any article that already has `cap` chunks (default 2).
BM25 runs on the chunk text (including its title/source/date header); question dates such as
"October 13, 2023" are matched to the chunk's published_at date.

Usage (from the project folder, CPU only, about a minute):
  python hybrid_claims\\run.py
  python hybrid_claims\\run.py --mask_outlets      (outlet names hidden; needs the masked query embeddings
                                                   from idea2_arrows\\eval_multihop.py --mask_outlets)
Writes outputs/hybrid_claims/REPORT.md and results.json.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer  # noqa: E402

from common import paths  # noqa: E402
from common.data import gold_facts, load_chunks, load_queries, mask_outlets, outlet_names, split_queries  # noqa: E402
from common.metrics import calculate_metrics, facts_found  # noqa: E402

MODEL = "BAAI__bge-large-en-v1.5"
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}
DATE_RX = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I)
MONTH_RX = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.I)
TOK = re.compile(r"[a-z0-9]+")
STOP = set(ENGLISH_STOP_WORDS) | {"article", "articles", "report", "reported", "reports", "according", "news"}
SPLIT_RX = re.compile(
    r",\s*(?:and\s+|but\s+|while\s+|whereas\s+|or\s+)?|;\s*|\s+(?:while|whereas|and also|as well as|but)\s+|"
    r"\s+and\s+(?=(?:the|a|an|did|does|do|was|is|has|have|had|its|their|his|her|in|on|after|before|"
    r"according|as|reported|accused|alleged|now|also)\b)", re.I)


def tokenize(t):
    return [w for w in TOK.findall(t.lower()) if w not in STOP]


def date_tokens(text):
    toks = []
    for m in DATE_RX.finditer(text):
        mo, d, y = MONTHS[m.group(1).lower()], int(m.group(2)), m.group(3)
        toks += [f"d{y}{mo:02d}{d:02d}", f"m{y}{mo:02d}"]
    for m in MONTH_RX.finditer(text):
        toks.append(f"m{m.group(2)}{MONTHS[m.group(1).lower()]:02d}")
    return toks


def chunk_doc(c):
    y, mo, d = c["published_at"][:4], c["published_at"][5:7], c["published_at"][8:10]
    return c["text"] + f" d{y}{mo}{d} m{y}{mo}"


class BM25:
    def __init__(self, docs, k1=0.9, b=0.4):
        self.cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        df = np.asarray((tf > 0).sum(0)).ravel()
        n = tf.shape[0]
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean())) * idf[tf.col]
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.vocab = self.cv.vocabulary_

    def scores(self, text):
        qv = np.zeros(self.W.shape[1], dtype=np.float32)
        for w in tokenize(text) + date_tokens(text):
            if w in self.vocab:
                qv[self.vocab[w]] += 1.0
        return np.asarray(self.W @ qv).ravel()

    def search(self, text, k):
        return top_ids(self.scores(text), k)


def top_ids(scores, k):
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top], kind="stable")].tolist()


def rrf(lists, k=60):
    sc = defaultdict(float)
    for lst in lists:
        for r, c in enumerate(lst):
            sc[c] += 1.0 / (k + r + 1)
    return [c for c, _ in sorted(sc.items(), key=lambda x: -x[1])]


def claims(q, outlets, min_words=3):
    t = DATE_RX.sub(lambda m: m.group(0).replace(",", ""), q)
    parts = [p.strip(" ?.") for p in SPLIT_RX.split(t) if p and p.strip()]
    parts = [p for p in parts if len(tokenize(p)) >= min_words]
    named = [o for o in outlets if o in q]
    named = [o for o in named if not any(o != p and o in p for p in named)]
    return [p if any(o in p for o in named) or len(named) != 1 else p + " " + named[0] for p in parts]


def pick(lists, K, chunks, cap=None):
    picked, seen, per_art, pos = [], set(), defaultdict(int), [0] * len(lists)
    while len(picked) < K:
        prog = False
        for i, lst in enumerate(lists):
            while pos[i] < len(lst) and len(picked) < K:
                c = lst[pos[i]]
                pos[i] += 1
                a = chunks[c]["article_id"]
                if c in seen or (cap and per_art[a] >= cap):
                    continue
                picked.append(c)
                seen.add(c)
                per_art[a] += 1
                prog = True
                break
        if not prog:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--mask_outlets", action="store_true")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    chunks, queries = load_chunks(), load_queries()
    outlets = sorted(outlet_names(chunks), key=len, reverse=True)
    dev, test = split_queries(queries)
    dev = [q for q in dev if queries[q]["question_type"] != "null_query"]
    test = [q for q in test if queries[q]["question_type"] != "null_query"]
    allq = sorted(dev + test)
    text = {q: (mask_outlets(queries[q]["query"], outlets) if a.mask_outlets else queries[q]["query"]) for q in allq}

    if a.mask_outlets:
        C = np.load(os.path.join(paths.CACHE, f"emb_{MODEL}_fp32_chunks.npy"))
        Q = np.load(os.path.join(paths.CACHE, f"emb_{MODEL}_fp32_queries_masked.npy"))
        S = Q[allq] @ C.T
        dense = {q: np.argsort(-S[i])[:100].tolist() for i, q in enumerate(allq)}
    else:
        with open(os.path.join(paths.CACHE, f"dense_ranking_{MODEL}_fp32.json")) as f:
            ids = json.load(f)["ids"]
        dense = {q: ids[q][:100] for q in allq}

    t0 = time.time()
    bm = BM25([chunk_doc(c) for c in chunks], a.k1, a.b)
    t_index = time.time() - t0
    t0 = time.time()
    bmr = {q: bm.search(text[q], 100) for q in allq}
    cl = {q: claims(text[q], [] if a.mask_outlets else outlets) for q in allq}
    clb = {q: [bm.search(c, 60) for c in cl[q]] for q in allq}
    ms_per_q = (time.time() - t0) / len(allq) * 1000

    systems = {
        "BGE (paper setup)": lambda q, K: dense[q][:K],
        f"BGE + cap{a.cap}": lambda q, K: pick([dense[q]], K, chunks, a.cap),
        f"Hybrid BGE+BM25 + cap{a.cap}": lambda q, K: pick([rrf([dense[q], bmr[q]])], K, chunks, a.cap),
        f"Hybrid + claim searches + cap{a.cap} (method)": lambda q, K: pick([rrf([dense[q], bmr[q]])] + clb[q], K, chunks, a.cap),
    }
    names = list(systems)

    def vecs(fn, K, qids):
        comp, rec = [], []
        for q in qids:
            f = facts_found([chunks[c]["text"] for c in fn(q, K)], gold_facts(queries[q]), K)
            comp.append(all(f))
            rec.append(sum(f) / len(f))
        return np.array(comp, float), np.array(rec)

    def ci(d):
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)]
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    types = np.array([queries[q]["question_type"] for q in test])
    res, lines = {"args": vars(a), "n_dev": len(dev), "n_test": len(test), "claims_per_question": float(np.mean([len(cl[q]) for q in allq])),
                  "bm25_index_seconds": t_index, "bm25_and_claims_ms_per_question": ms_per_q}, []
    for K in (10, 6):
        base_c, _ = vecs(systems[names[1]], K, test)
        hyb_c, _ = vecs(systems[names[2]], K, test)
        lines.append(f"\n### K = {K} (test, {len(test)} questions)\n")
        lines.append("| System | Complete@K | Recall@K | Dev complete@K | Gain vs BGE+cap [95% CI] | Inference | Comparison | Temporal |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for n in names:
            c, r = vecs(systems[n], K, test)
            cd, _ = vecs(systems[n], K, dev)
            lo, hi = ci(c - base_c)
            bt = [100 * c[types == t].mean() for t in ("inference_query", "comparison_query", "temporal_query")]
            lines.append(f"| {n} | {100*c.mean():.2f}% | {100*r.mean():.2f}% | {100*cd.mean():.2f}% | "
                         f"{100*(c-base_c).mean():+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {bt[0]:.1f}% | {bt[1]:.1f}% | {bt[2]:.1f}% |")
            res[f"{n}|K{K}"] = {"complete": c.mean(), "recall": r.mean(), "dev_complete": cd.mean(),
                                "gain_vs_bge_cap": (c - base_c).mean(), "ci": [lo, hi]}
        mc, _ = vecs(systems[names[3]], K, test)
        lo, hi = ci(mc - hyb_c)
        lines.append(f"\nClaim searches on top of hybrid+cap: {100*(mc-hyb_c).mean():+.2f} points [95% CI {100*lo:+.2f}, {100*hi:+.2f}].")
        res[f"claims_vs_hybrid|K{K}"] = {"gain": (mc - hyb_c).mean(), "ci": [lo, hi]}

    lines.append("\n### Paper metrics (MultiHop-RAG Table 5 style, all answerable questions)\n")
    lines.append("| System | Hits@10 | Hits@4 | MRR@10 | MAP@10 |\n|---|---:|---:|---:|---:|")
    for n in (names[0], names[3]):
        m = calculate_metrics([[chunks[c]["text"] for c in systems[n](q, 10)] for q in allq], [gold_facts(queries[q]) for q in allq])
        lines.append(f"| {n} | {m['Hits@10']:.4f} | {m['Hits@4']:.4f} | {m['MRR@10']:.4f} | {m['MAP@10']:.4f} |")
        res[f"{n}|paper_metrics"] = m

    head = (f"# Hybrid search + claim searches + {a.cap}-per-article cap\n\n"
            f"No training, no LLM, no reranker. Outlets masked: {a.mask_outlets}. "
            f"Claims per question: {res['claims_per_question']:.2f}. BM25 + claim split: {ms_per_q:.1f} ms/question on CPU.\n")
    out_dir = paths.ensure(os.path.join(paths.OUT, "hybrid_claims"))
    tag = "_masked" if a.mask_outlets else ""
    with open(os.path.join(out_dir, f"REPORT{tag}.md"), "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")
    with open(os.path.join(out_dir, f"results{tag}.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(head + "\n".join(lines))


if __name__ == "__main__":
    main()
