"""Evaluate the arrows on MultiHop-RAG (never trained on it) against dense retrieval and the outlet shortcut.

Systems (same BGE index, same K, same duplicate handling):
  dense top-K               the replication's retriever
  dense, <=2 per article    simple diversity cap
  outlet filter (rival)     outlet names found in the question -> one search per outlet, restricted to
                            that outlet's chunks, slots split evenly (rule-based Multi-Meta-RAG)
  arrows (soft slots)       all 4 arrows search; slots split by w = [1, 1, P(n>=3), P(n=4)]
  arrows (hard count)       only the first argmax(n) arrows, equal slots
  arrows (oracle count)     diagnostic: true number of evidence articles

Options:
  --mask_outlets   replace outlet names in the questions by "a news outlet" (shortcut check; re-embeds queries)
  --dedup_overlap  drop near-duplicate chunks for every system

Usage:  python idea2_arrows/eval_multihop.py [--K 10]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import paths  # noqa: E402
from common.bge import DenseIndex  # noqa: E402
from common.data import gold_facts, load_chunks, load_queries, mask_outlets, outlet_names, split_queries  # noqa: E402
from common.metrics import format_table, metrics_by_type  # noqa: E402
from common.selection import Picker, deal_quotas, largest_remainder, take_top  # noqa: E402
from idea2_arrows.model import ArrowAdapter, slot_weights  # noqa: E402


def outlets_in(question, outlets):
    found = []
    for o in sorted(outlets, key=len, reverse=True):
        if o in question and not any(o in f for f in found):
            found.append(o)
    return sorted(found, key=question.index)


def cap_per_article(ranking, chunks, K, dedup_overlap):
    picker = Picker(chunks, K, dedup_overlap)
    per_article = {}
    for cid in ranking:
        if picker.full():
            break
        article = chunks[cid]["article_id"]
        if per_article.get(article, 0) < 2 and picker.try_add(cid):
            per_article[article] = per_article.get(article, 0) + 1
    return picker.picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--embed_dtype", default="fp32")
    ap.add_argument("--adapter", default=os.path.join(paths.CACHE, "arrows", "adapter.pt"))
    ap.add_argument("--mask_outlets", action="store_true")
    ap.add_argument("--dedup_overlap", action="store_true")
    ap.add_argument("--depth", type=int, default=60)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from replicate.main import chunk_cache_path, query_cache_path
    chunks, queries = load_chunks(), load_queries()
    outlets = outlet_names(chunks)
    dev_q, test_q = split_queries(queries)
    ans = [q["qid"] for q in queries if q["question_type"] != "null_query"]
    c_emb = np.load(chunk_cache_path(a.embed_model, a.embed_dtype))
    index = DenseIndex(c_emb, device=a.device)
    texts = [q["query"] for q in queries]
    if a.mask_outlets:
        from common.bge import BGEEncoder, encode_cached
        enc = BGEEncoder(a.embed_model, device=a.device, dtype=a.embed_dtype)
        texts = [mask_outlets(text, outlets) for text in texts]
        q_emb = encode_cached(enc, texts, query_cache_path(a.embed_model, a.embed_dtype, "queries_masked"), is_query=True)
        del enc
    else:
        q_emb = np.load(query_cache_path(a.embed_model, a.embed_dtype))

    sd = torch.load(a.adapter, map_location="cpu")
    with open(a.adapter.replace(".pt", ".json")) as f:
        cfg = json.load(f)["config"]
    model = ArrowAdapter(**cfg)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        V, logits = model(torch.from_numpy(q_emb[ans]).float())
        W = slot_weights(logits).numpy()
        n_hat = (logits.argmax(-1) + 2).numpy()
    V = V.numpy()
    pos = {q: i for i, q in enumerate(ans)}

    K, dd = a.K, a.dedup_overlap
    dense_ids, _ = index.search(q_emb[ans], a.depth)
    arrow_ids = [index.search(V[:, j], a.depth)[0] for j in range(V.shape[1])]
    by_outlet = {}
    for i, c in enumerate(chunks):
        by_outlet.setdefault(c["source"], []).append(i)

    systems = {k: {} for k in ["dense top-K", "dense <=2 per article", "outlet filter (rival)",
                               "arrows (soft slots)", "arrows (hard count)", "arrows (oracle count)"]}
    named = []
    for q in ans:
        i = pos[q]
        systems["dense top-K"][q] = take_top(dense_ids[i], chunks, K, dd)
        diverse = cap_per_article(dense_ids[i], chunks, K, dd)
        if len(diverse) < K:
            deeper = index.search(q_emb[q], len(chunks))[0][0]
            diverse = cap_per_article(deeper, chunks, K, dd)
        systems["dense <=2 per article"][q] = diverse
        found = outlets_in(texts[q], outlets)
        named.append(len(found))
        if found:
            lists = [index.search(q_emb[q], a.depth, allowed=[np.array(by_outlet[o])])[0][0] for o in found]
            systems["outlet filter (rival)"][q] = deal_quotas(lists + [dense_ids[i]],
                                                              largest_remainder([1] * len(found) + [0], K), chunks, K, dd)
        else:
            systems["outlet filter (rival)"][q] = systems["dense top-K"][q]
        lists = [arrow_ids[j][i] for j in range(V.shape[1])]
        systems["arrows (soft slots)"][q] = deal_quotas(lists, largest_remainder(W[i].tolist(), K), chunks, K, dd)
        n = int(n_hat[i])
        systems["arrows (hard count)"][q] = deal_quotas(lists[:n], largest_remainder([1] * n, K), chunks, K, dd)
        n_true = max(2, min(4, len({e["title"] for e in queries[q]["evidence_list"]})))
        systems["arrows (oracle count)"][q] = deal_quotas(lists[:n_true], largest_remainder([1] * n_true, K), chunks, K, dd)

    # how different are the arrows from each other? (shared top-10 chunks between arrow 1 and 2)
    overlap12 = float(np.mean([len(set(arrow_ids[0][i][:10]) & set(arrow_ids[1][i][:10])) / 10 for i in range(len(ans))]))
    true_n = np.array([max(2, min(4, len({e['title'] for e in queries[q]['evidence_list']}))) for q in ans])
    count_acc = float((n_hat == true_n).mean())

    out = {"args": vars(a), "arrow12_top10_overlap": overlap12, "count_acc_vs_articles": count_acc,
           "share_naming_an_outlet": float(np.mean([x > 0 for x in named])), "splits": {}}
    cols = [f"complete@{K}", f"recall@{K}"]
    cols.extend(f"complete@{k}" for k in (4, 6) if k < K)
    if K >= 10:
        cols.extend(["Hits@10", "MRR@10", "MAP@10"])
    for split_name, qids in (("test", [q for q in test_q if q in pos]), ("dev", [q for q in dev_q if q in pos])):
        rows, det = [], {}
        for name, sel in systems.items():
            if any(len(sel[q]) != K for q in qids):
                raise RuntimeError(f"{name} could not fill the {K}-chunk budget for {split_name}")
            recs = [{"question_type": queries[q]["question_type"], "retrieved": [chunks[c]["text"] for c in sel[q]],
                     "gold": gold_facts(queries[q])} for q in qids]
            bt = metrics_by_type(recs, ks=sorted({4, 6, K}))
            rows.append({"system": name, **bt["all"]})
            det[name] = bt
        out["splits"][split_name] = {"table": rows, "by_type": det}
        print(f"\n{split_name.upper()} ({len(qids)} answerable questions, K={K}, mask_outlets={a.mask_outlets}, "
              f"dedup_overlap={dd})")
        print(format_table(rows, cols))
    print(f"\narrow 1 vs 2 share {overlap12:.2f} of their top-10; count head matches #evidence articles "
          f"{count_acc:.2f}; {out['share_naming_an_outlet']:.2f} of questions name an outlet")
    od = paths.ensure(os.path.join(paths.OUT, "idea2"))
    name = f"eval_K{K}{'_masked' if a.mask_outlets else ''}{'_dedup' if dd else ''}"
    with open(os.path.join(od, name + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(od, name + ".md"), "w") as f:
        f.write(f"# Idea 2 on MultiHop-RAG ({name})\n\n" + format_table(out["splits"]["test"]["table"], cols) + "\n")
    print(f"saved outputs/idea2/{name}.*")


if __name__ == "__main__":
    main()
