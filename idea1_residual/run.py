"""Run Idea 1 (reader-guided residual retrieval) on MultiHop-RAG and compare with dense retrieval.

Needs the dense rankings from replicate/main.py (run that first; same BGE settings).

Examples (from the project folder):
  python idea1_residual/run.py --scorer word                 # free word-absence baseline, seconds
  python idea1_residual/run.py --scorer llm_inplace          # answering LLM, ~1 s/question on a 4070
  python idea1_residual/run.py --scorer llm_probe
  python idea1_residual/run.py --scorer mlm                  # ModernBERT-large
  python idea1_residual/run.py --scorer llm_inplace --limit 50   # quick check

What it reports (dev split used only to pick tau; numbers reported on test):
  * dense top-K vs Idea 1 vs "always expand" (tau = +inf): complete@K, recall@K, Hits/MRR/MAP
  * H1: AUROC of the question-level signal for "D1 is missing evidence", vs the word-absence signal
  * H3: share of questions that stop after the first search, and what that costs
All spans, gains and second-search lists are cached, so re-running with another tau is instant.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from common import paths  # noqa: E402
from common.data import gold_facts, load_chunks, load_queries, outlet_names, split_queries  # noqa: E402
from common.metrics import auroc, facts_found, format_table, metrics_by_type  # noqa: E402
from common.selection import take_top  # noqa: E402
from idea1_residual.pipeline import active_query_keys, question_signal, select  # noqa: E402
from idea1_residual.query_forms import QwenRewriter, queries_for_question  # noqa: E402
from idea1_residual.spans import Span, SpanExtractor  # noqa: E402


def load_json(p, default=None):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(p, obj):
    p = os.fspath(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, p)


def get_dense(model, dtype):
    from replicate.main import ranking_cache_path
    r = load_json(ranking_cache_path(model, dtype))
    if r is None:
        sys.exit("No dense rankings found. Run `python replicate/main.py` first.")
    return r["ids"]


def report_tau(tau):
    return tau if tau is None or np.isfinite(tau) else str(tau)


def evaluate(sel, qids, queries, chunks, K):
    recs = []
    for q in qids:
        recs.append({"question_type": queries[q]["question_type"],
                     "retrieved": [chunks[c]["text"] for c in sel[q]], "gold": gold_facts(queries[q])})
    by_type = metrics_by_type(recs, ks=sorted({4, 6, K}))
    return by_type


def first_pass(a, dense, qids, chunks):
    from common.bge import DenseIndex
    from replicate.main import chunk_cache_path, query_cache_path

    def pick(q):
        return take_top(dense[q], chunks, a.K, a.dedup_overlap, max_per_article=a.max_per_article)

    d1 = {q: pick(q) for q in qids}
    short = [q for q in qids if len(d1[q]) < a.K]
    if short:
        index = DenseIndex(np.load(chunk_cache_path(a.embed_model, a.embed_dtype)), device=a.device)
        q_vecs = np.load(query_cache_path(a.embed_model, a.embed_dtype))
        for q in short:
            dense[q] = np.argsort(-index.scores(q_vecs[q])[0], kind="stable").tolist()
            d1[q] = pick(q)
            if len(d1[q]) != a.K:
                raise ValueError(f"The first pass could not fill the {a.K}-chunk budget for qid {q}")
        print(f"Dense search went beyond the cached ranking for {len(short)} questions")
    return d1


def build_second_searches(a, cache, tag, qids, queries, spans, gains, d1):
    from tqdm import tqdm

    from common.bge import BGEEncoder, DenseIndex, encode_cached
    from replicate.main import chunk_cache_path

    if a.split == "all":
        path = os.path.join(cache, f"second_{tag}.json")
        second = load_json(path, {})
        todo = [q for q in qids if str(q) not in second]
    else:
        fixed = ""
        if a.split == "test":
            from common.bge import default_device

            scorer_tag = a.scorer
            if a.scorer != "word":
                scorer_tag += "_" + paths.model_slug(a.mlm if a.scorer == "mlm" else a.llm)
            device_tag = (a.device or default_device()).replace(":", "_")
            fixed = f"_tau{a.tau}_m0{a.m0}_{scorer_tag}_{device_tag}"
        path = os.path.join(cache, f"timed_second_{tag}_{a.split}{fixed}.json")
        timed = load_json(path, {})
        todo = [q for q in qids if str(q) not in timed]

    if todo:
        enc = BGEEncoder(a.embed_model, device=a.device, dtype=a.embed_dtype)
        index = DenseIndex(np.load(chunk_cache_path(a.embed_model, a.embed_dtype)), device=a.device)
        rewriter = QwenRewriter(a.rewrite_model, a.device) if a.query_form == "rewrite" else None

        def formulate(q, keys):
            return queries_for_question(queries[q]["query"], spans[q], a.query_form, keys, rewriter)

        if a.split == "all":
            for q in tqdm(todo, desc="form queries", disable=a.query_form != "rewrite", mininterval=5):
                second[str(q)] = formulate(q, None)
            flat = [(q, entry) for q in todo for entry in second[str(q)].values()]
            if flat:
                q_emb = encode_cached(enc, [entry["query"] for _, entry in flat],
                                      os.path.join(cache, f"emb_residual_{tag}_{len(flat)}.npy"), is_query=True)
                ids, _ = index.search(q_emb, a.K + 40, exclude=[set(d1[q]) for q, _ in flat])
                for (_, entry), row in zip(flat, ids):
                    entry["ids"] = row
            save_json(path, second)
        else:
            last_save = time.monotonic()
            for q in tqdm(todo, desc=f"{a.split} queries ({a.query_form})", mininterval=5):
                t0 = time.perf_counter()
                keys = active_query_keys(spans[q], gains[q], a.tau, a.m0) if a.split == "test" else None
                entries = formulate(q, keys) if keys != [] else {}
                if entries:
                    q_emb = enc.encode([entry["query"] for entry in entries.values()], is_query=True,
                                       batch_size=len(entries), show_progress=False)
                    ids, _ = index.search(q_emb, a.K + 40, exclude=[set(d1[q])] * len(entries))
                    for entry, row in zip(entries.values(), ids):
                        entry["ids"] = row
                timed[str(q)] = {"entries": entries, "seconds": time.perf_counter() - t0}
                if time.monotonic() - last_save > 30:
                    save_json(path, timed)
                    last_save = time.monotonic()
            save_json(path, timed)
    if a.split == "all":
        return second, {}
    return ({q: row["entries"] for q, row in timed.items()},
            {q: row["seconds"] for q, row in timed.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="word",
                    choices=["word", "llm_inplace", "llm_inplace_mean", "llm_probe", "mlm"])
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--mlm", default="answerdotai/ModernBERT-large")
    ap.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--embed_dtype", default="fp32")
    ap.add_argument("--K", type=int, default=10, help="final context size (and size of D1)")
    ap.add_argument("--m0", type=int, default=2, help="slots for the keyword search when only details are missing")
    ap.add_argument("--tau", type=float, default=None, help="fixed threshold; default = tune on dev")
    ap.add_argument("--dedup_overlap", action="store_true", help="drop near-duplicate chunks (applied to every system)")
    ap.add_argument("--max_per_article", type=int, default=None, help="maximum chunks per article in any context")
    ap.add_argument("--query_form", choices=["keywords", "trimmed", "rewrite"], default="keywords")
    ap.add_argument("--rewrite_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all",
                    help="use dev to tune, then test with a frozen --tau and only the chosen searches")
    ap.add_argument("--limit", type=int, default=None, help="only the first N answerable questions (quick test)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.K < 1 or (a.limit is not None and a.limit < 1):
        ap.error("--K and --limit must be positive")
    if a.max_per_article is not None and a.max_per_article < 1:
        ap.error("--max_per_article must be positive")
    if a.split == "test" and (a.tau is None or np.isnan(a.tau)):
        ap.error("--split test requires an explicit dev-selected --tau")

    chunks = load_chunks()
    queries = load_queries()
    dense = get_dense(a.embed_model, a.embed_dtype)
    dev, test = split_queries(queries)
    answerable = [q["qid"] for q in queries if q["question_type"] != "null_query"]
    dev = [q for q in dev if queries[q]["question_type"] != "null_query"]
    test = [q for q in test if queries[q]["question_type"] != "null_query"]
    if a.split == "dev":
        dev, test = dev[:a.limit], []
    elif a.split == "test":
        dev, test = [], test[:a.limit]
    elif a.limit:
        keep = set(answerable[:a.limit])
        dev = [q for q in dev if q in keep]
        test = [q for q in test if q in keep]
    run_q = sorted(set(dev) | set(test))
    cache = paths.ensure(os.path.join(paths.CACHE, "idea1"))
    out_dir = paths.ensure(os.path.join(paths.OUT, "idea1_cap" if a.max_per_article else "idea1"))
    dd = ("_dedup" if a.dedup_overlap else "") + (f"_cap{a.max_per_article}" if a.max_per_article else "")
    form = "" if a.query_form == "keywords" else "_" + a.query_form
    if a.query_form == "rewrite":
        form += "_" + paths.model_slug(a.rewrite_model)
    # D1 = what the first search hands over (same rule the dense baseline uses)
    d1 = first_pass(a, dense, run_q, chunks)

    # 1. spans (cached)
    sp_path = os.path.join(cache, "spans.json")
    spans_all = load_json(sp_path, {})
    todo = [q for q in run_q if str(q) not in spans_all]
    if todo:
        ex = SpanExtractor(outlet_names(chunks))
        for q in todo:
            spans_all[str(q)] = [s.to_dict() for s in ex(queries[q]["query"])]
        save_json(sp_path, spans_all)
    spans = {q: [Span.from_dict(d) for d in spans_all[str(q)]] for q in run_q}
    n_t = [sum(s.kind == "target" for s in spans[q]) for q in run_q]
    print(f"spans: {np.mean([len(spans[q]) for q in run_q]):.1f} per question, {np.mean(n_t):.1f} targets; "
          f"{sum(x == 0 for x in n_t)} questions with no target")

    # 2. gains (cached per scorer and K)
    tag = a.scorer if a.scorer == "word" else f"{a.scorer}__{paths.model_slug(a.mlm if a.scorer == 'mlm' else a.llm)}"
    g_path = os.path.join(cache, f"gains_{tag}_K{a.K}{dd}.json")
    gains_all = load_json(g_path, {})
    todo = [q for q in run_q if str(q) not in gains_all]
    if todo:
        from idea1_residual.scorers import make_scorer
        scorer = make_scorer(a.scorer, a.llm, a.mlm, device=a.device)
        try:
            from tqdm import tqdm
            it = tqdm(todo, desc=f"scoring ({a.scorer})")
        except ImportError:
            it = todo
        last_save = time.time()
        for q in it:
            passages = [chunks[c]["text"] for c in d1[q]]
            t0 = time.time()
            g = scorer.score(queries[q]["query"], spans[q], passages) if spans[q] else []
            gains_all[str(q)] = {"gains": g, "sec": time.time() - t0}
            if time.time() - last_save > 60:
                save_json(g_path, gains_all)
                last_save = time.time()
        save_json(g_path, gains_all)
        del scorer
    gains = {q: gains_all[str(q)]["gains"] for q in run_q}
    secs = [gains_all[str(q)]["sec"] for q in run_q]

    # 3. Search all candidates for tuning, or just the selected queries at test time.
    second_all, search_seconds = build_second_searches(a, cache, f"K{a.K}{dd}{form}",
                                                       run_q, queries, spans, gains, d1)
    second = {q: {k: v["ids"] for k, v in second_all.get(str(q), {}).items()} for q in run_q}

    def run_tau(tau, qids):
        sel, info = {}, {}
        for q in qids:
            sel[q], info[q] = select(dense[q], second[q], spans[q], gains[q], tau, chunks, a.K,
                                     m0=a.m0, dedup_overlap=a.dedup_overlap,
                                     max_per_article=a.max_per_article, first_pass=d1[q] if a.max_per_article else None)
            if len(sel[q]) != a.K:
                raise ValueError(f"The final context could not fill the {a.K}-chunk budget for qid {q}")
        return sel, info

    def complete_rate(sel, qids):
        ok = [all(facts_found([chunks[c]["text"] for c in sel[q]], gold_facts(queries[q]), a.K)) for q in qids]
        return float(np.mean(ok)) if ok else 0.0

    # 4. pick tau on dev
    dev_gains = [g for q in dev for g in gains[q]]
    if a.tau is not None:
        tau, sweep = a.tau, []
    else:
        cands = sorted(set(np.quantile(dev_gains, np.linspace(0, 1, 41)).tolist())) if dev_gains else [0.0]
        cands.append(float("inf"))
        sweep = []
        for t in cands:
            sel, info = run_tau(t, dev)
            stop = np.mean([info[q]["path"] == "stop" for q in dev]) if dev else 0
            sweep.append({"tau": t, "dev_complete": complete_rate(sel, dev), "dev_stop_rate": float(stop)})
        best = max(sweep, key=lambda r: (round(r["dev_complete"], 4), r["dev_stop_rate"]))
        tau = best["tau"]
    print(f"tau = {tau:.4f}" + (" (tuned on dev)" if a.tau is None else ""))

    # 5. Test-only runs measure the chosen policy without precomputing unused queries.
    eval_q = dev if a.split == "dev" else test
    eval_name = "dev" if a.split == "dev" else "test"
    tuned_name = f"idea1 ({a.scorer}, tau={tau:.3g})"
    systems = {}
    systems["dense top-K"] = ({q: d1[q] for q in eval_q}, {q: {"path": "stop"} for q in eval_q})
    systems[tuned_name] = run_tau(tau, eval_q)
    if a.split != "test":
        systems["idea1 always-expand"] = run_tau(float("inf"), eval_q)
    rows, details = [], {}
    for name, (sel, info) in systems.items():
        bt = evaluate(sel, eval_q, queries, chunks, a.K)
        stop = float(np.mean([info[q]["path"] == "stop" for q in eval_q]))
        n2 = float(np.mean([len(info[q].get("queries", [])) for q in eval_q]))
        rows.append({"system": name, **{k: v for k, v in bt["all"].items() if k != "n"},
                     "stop_rate": stop, "2nd_searches": n2})
        details[name] = {"by_type": bt, "stop_rate": stop, "mean_second_searches": n2}

    cols = [f"complete@{a.K}", f"recall@{a.K}"]
    if a.K > 6:
        cols.append("complete@6")
    if a.K >= 10:
        cols.extend(["Hits@10", "MRR@10", "MAP@10"])
    cols.extend(["stop_rate", "2nd_searches"])
    table = format_table(rows, cols)
    print(f"\n{eval_name.upper()} ({len(eval_q)} answerable questions, K={a.K}, "
          f"max_per_article={a.max_per_article}, dedup_overlap={a.dedup_overlap})\n" + table)

    # 6. H1: does the signal predict that D1 is missing evidence?
    h1 = {}
    wp = os.path.join(cache, f"gains_word_K{a.K}{dd}.json")
    if a.scorer == "word":
        word_gains = gains
    else:
        from idea1_residual.scorers import WordScorer
        wg = load_json(wp, {})
        ws = WordScorer()
        for q in run_q:
            if str(q) not in wg:
                wg[str(q)] = {"gains": ws.score(queries[q]["query"], spans[q], [chunks[c]["text"] for c in d1[q]])
                              if spans[q] else [], "sec": 0}
        save_json(wp, wg)
        word_gains = {q: wg[str(q)]["gains"] for q in run_q}
    labels = [int(not all(facts_found([chunks[c]["text"] for c in d1[q]], gold_facts(queries[q]), a.K)))
              for q in eval_q]
    sig = [-question_signal(spans[q], gains[q]) for q in eval_q]
    h1[a.scorer] = auroc(sig, labels)
    wsig = [-question_signal(spans[q], word_gains[q]) for q in eval_q]
    h1["word"] = auroc(wsig, labels)
    covered_by_words = [i for i, q in enumerate(eval_q) if word_gains[q] and min(word_gains[q]) >= 1.0]
    if covered_by_words:
        subset_labels = [labels[i] for i in covered_by_words]
        h1[f"{a.scorer} | words all present"] = auroc([sig[i] for i in covered_by_words], subset_labels)
        h1["word | words all present"] = auroc([wsig[i] for i in covered_by_words], subset_labels)
    h1["n | words all present"] = len(covered_by_words)
    print("\nH1  AUROC for 'D1 is missing evidence' (0.5 = chance):")
    for k, v in h1.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  share of {eval_name} questions where D1 is incomplete: {np.mean(labels):.3f}")
    print(f"\nscoring time: {np.mean(secs):.3f} s/question (mean)")

    logged_sweep = [{**row, "tau": report_tau(row["tau"])} for row in sweep]
    logged_h1 = {k: v if np.isfinite(v) else None for k, v in h1.items()}
    mean_search = float(np.mean([search_seconds[str(q)] for q in eval_q])) if search_seconds else None
    res = {"args": {**vars(a), "tau": report_tau(a.tau)}, "tau": report_tau(tau), "sweep": logged_sweep,
           "table": rows, "details": details, "H1": logged_h1,
           "scoring_sec_mean": float(np.mean(secs)), "extra_search_sec_mean": mean_search,
           "n_test": len(test), "n_dev": len(dev)}
    stage = "" if a.split == "all" else "_" + a.split
    name = f"{tag}_K{a.K}{dd}{form}{stage}{'_limit' + str(a.limit) if a.limit else ''}"
    save_json(os.path.join(out_dir, name + ".json"), res)
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write(f"# Idea 1 — {a.scorer}, K={a.K}\n\ntau={tau:.4f}\n\n{table}\n\nH1: {json.dumps(logged_h1)}\n")
    # per-question log for inspection
    sel, info = systems[tuned_name]
    log = []
    for q in eval_q:
        gold = gold_facts(queries[q])

        def coverage(ids):
            found = facts_found([chunks[c]["text"] for c in ids], gold, a.K)
            return {"complete": all(found), "recall": sum(found) / len(found)}

        entries = [second_all[str(q)][k] for k in info[q].get("queries", [])]
        row = {"qid": q, "query": queries[q]["query"], "type": queries[q]["question_type"],
               "spans": [(s.text, s.kind, round(g, 3)) for s, g in zip(spans[q], gains[q])],
               "path": info[q]["path"], "second_queries": [entry["query"] for entry in entries],
               "rewrite_attempts": sum("rewrite_fallback" in e for e in entries),
               "rewrite_fallbacks": sum(e.get("rewrite_fallback", False) for e in entries),
               "selected_chunk_ids": sel[q], "d1_chunk_ids": d1[q], **coverage(sel[q])}
        if a.split == "test" and a.max_per_article:
            row["capped_dense"] = coverage(d1[q])
            row["uncapped_dense"] = coverage(take_top(dense[q], chunks, a.K, a.dedup_overlap))
        if search_seconds:
            row["extra_search_seconds"] = search_seconds[str(q)]
        log.append(row)
    save_json(os.path.join(out_dir, name + "_per_question.json"), log)
    print(f"\nsaved {os.path.join(out_dir, name)}.*")


if __name__ == "__main__":
    main()
