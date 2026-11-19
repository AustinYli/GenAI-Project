"""Run the two article-capped word-scoring experiments, with a locked dev choice."""

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import paths
from common.data import load_chunks
from idea1_residual.run import load_json, save_json


CONFIGS = ((10, "keywords"), (6, "keywords"), (10, "trimmed"), (10, "rewrite"))
FORMS = {"keywords": "Keyword list", "trimmed": "Trimmed sentence", "rewrite": "Qwen rewrite"}
BASELINES = {"capped_dense": "Capped dense", "uncapped_dense": "Uncapped dense"}


def paired_interval(differences, draws=10_000, seed=13):
    """Percentile interval over resampled paired question-level differences."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise ValueError("Paired differences must be a nonempty vector")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for start in range(0, draws, 256):
        stop = min(start + 256, draws)
        pairs = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        samples[start:stop] = differences[pairs].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"delta": float(differences.mean()), "lo": float(lo), "hi": float(hi)}


def stem(k, form, stage, rewrite_model):
    suffix = "" if form == "keywords" else f"_{form}"
    if form == "rewrite":
        suffix += "_" + paths.model_slug(rewrite_model)
    return f"word_K{k}_cap2{suffix}_{stage}"


def stage_run(k, form, stage, device, rewrite_model, out, tau=None):
    name = stem(k, form, stage, rewrite_model)
    script = Path(__file__).with_name("run.py")
    cmd = [sys.executable, "-u", str(script), "--scorer", "word", "--max_per_article", "2", "--K", str(k),
           "--query_form", form, "--rewrite_model", rewrite_model, "--split", stage, "--device", device]
    if tau is not None:
        cmd.extend(("--tau", str(tau)))
    log = Path(paths.OUT) / "logs" / f"16_{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {name}, log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=paths.ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    return load_json(out / (name + ".json")), load_json(out / (name + "_per_question.json"))


def write_report(out, decisions, results, records, rewrite_model):
    def pct(x):
        return f"{x:.2%}"

    def threshold(x):
        return f"{x:.4g}" if isinstance(x, float) else str(x)

    def mean(rows, baseline, metric):
        return float(np.mean([q[baseline][metric] for q in rows]))

    def interval_text(ci):
        return f"{100 * ci['delta']:+.2f} [{100 * ci['lo']:+.2f}, {100 * ci['hi']:+.2f}]"

    intervals = {f"K{k}_{form}": {
        b: {m: paired_interval([float(q[m]) - float(q[b][m]) for q in records[(k, form)]])
            for m in ("complete", "recall")}
        for b in BASELINES} for k, form in CONFIGS}
    save_json(out / "paired_intervals.json", intervals)
    winner = decisions["selected_form"]
    headline, capped_intervals = [], []
    for k in (10, 6):
        base = mean(records[k, "keywords"], "capped_dense", "complete")
        score = results[k, "keywords"]["table"][1][f"complete@{k}"]
        headline.append(f"K={k} went from {pct(base)} for capped dense to {pct(score)} with word-guided expansion")
        capped_intervals.append(intervals[f"K{k}_keywords"]["capped_dense"]["complete"])
    uncertainty = ""
    if all(c["lo"] <= 0 <= c["hi"] for c in capped_intervals):
        uncertainty = " Both paired confidence intervals for the change in complete coverage included zero."
    recall_k10 = intervals["K10_keywords"]["capped_dense"]["recall"]
    winner_score = results[10, winner]["table"][1]["complete@10"]
    winner_dev = decisions["dev"][f"K10_{winner}"]["complete"]
    form_selection = f"Dev selected **{FORMS[winner].lower()}** at {pct(winner_dev)} complete@10."
    if all(decisions["dev"][f"K10_{form}"]["complete"] == winner_dev for form in FORMS):
        form_selection = (f"All three forms tied at {pct(winner_dev)} complete@10 on dev, so the preset tie-break "
                          f"selected **{FORMS[winner].lower()}**.")
    lines = ["# Idea 1 with an Article Cap", "",
             f"{headline[0]}. {headline[1]}.{uncertainty} {form_selection} "
             f"That form reached {pct(winner_score)} complete@10 on test. The keyword run at K=10 also changed "
             f"fact recall by {100 * recall_k10['delta']:+.2f} percentage points over capped dense "
             f"(95% CI {100 * recall_k10['lo']:+.2f} to {100 * recall_k10['hi']:+.2f}).", "",
             "## Part 1 Word scoring with a two-chunk article cap", "",
             "The word scorer examines the capped first-pass context. Every final context in the capped experiment "
             "uses at most two chunks per article, including the fallback to dense results. Uncapped dense is shown "
             "as a reference. K=6 uses its own six-chunk first pass and its own threshold.", "",
             "| K | System | Complete | Fact recall | Stop rate | Mean extra searches |",
             "|---:|---|---:|---:|---:|---:|"]
    for k in (10, 6):
        rows = records[k, "keywords"]
        for key in ("uncapped_dense", "capped_dense"):
            lines.append(f"| {k} | {BASELINES[key]} | {pct(mean(rows, key, 'complete'))} | "
                         f"{pct(mean(rows, key, 'recall'))} | 100.00% | 0.00 |")
        score = results[k, "keywords"]["table"][1]
        tau = threshold(results[k, "keywords"]["tau"])
        lines.append(f"| {k} | Capped word-guided expansion (τ = {tau}) | {pct(score[f'complete@{k}'])} | "
                     f"{pct(score[f'recall@{k}'])} | {pct(score['stop_rate'])} | {score['2nd_searches']:.2f} |")
    lines.extend(["", "Differences are percentage points, with paired 95% confidence intervals in brackets.", "",
                  "| K | Comparison | Change in complete | Change in recall |",
                  "|---:|---|---:|---:|"])
    for k in (10, 6):
        for b in BASELINES:
            ci = intervals[f"K{k}_keywords"][b]
            lines.append(f"| {k} | Word-guided expansion vs {BASELINES[b].lower()} | "
                         f"{interval_text(ci['complete'])} | {interval_text(ci['recall'])} |")

    lines.extend(["", "## Part 2 Extra-search query form at K=10", "",
                  "Each form uses the same article cap and its own dev-selected word threshold. The word scorer "
                  "uses the existing target labels, which include outlet names. The details-only keyword search stays "
                  "the same for all forms.", "",
                  "| Query form | τ | Dev complete | Test complete | Test recall | Stop rate | Mean extra searches | Seconds per question |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for form, name in FORMS.items():
        r = results[10, form]
        score = r["table"][1]
        name += " (dev choice)" if form == winner else ""
        lines.append(f"| {name} | {threshold(r['tau'])} | {pct(decisions['dev'][f'K10_{form}']['complete'])} | "
                     f"{pct(score['complete@10'])} | {pct(score['recall@10'])} | {pct(score['stop_rate'])} | "
                     f"{score['2nd_searches']:.2f} | {r['extra_search_sec_mean']:.4f} |")
    lines.extend(["", "| Query form vs capped dense | Change in complete | Change in recall |",
                  "|---|---:|---:|"])
    for form, name in FORMS.items():
        ci = intervals[f"K10_{form}"]["capped_dense"]
        lines.append(f"| {name} | {interval_text(ci['complete'])} | {interval_text(ci['recall'])} |")
    rewrites = records[10, "rewrite"]
    attempts = sum(row["rewrite_attempts"] for row in rewrites)
    falls = sum(row["rewrite_fallbacks"] for row in rewrites)
    fallback_share = f" ({falls / attempts:.1%})" if attempts else ""
    lines.extend(["", f"The rewrite used `{rewrite_model}` with greedy decoding and one batch per question. "
                  f"On test, {falls:,} of {attempts:,} attempted rewrites{fallback_share} used the keyword fallback "
                  "after the name check.",
                  "",
                  "The time column measures query construction or generation, BGE encoding, and exact search for "
                  "the selected extra queries. It averages over all test questions, including stops, and excludes "
                  "model loading, the first pass, span scoring, and the final merge.",
                  "", "## Split and uncertainty", "",
                  "Thresholds were tuned on the 681 dev questions for complete coverage, with higher stop rate "
                  "breaking ties. Query forms were compared by dev complete@10, with keyword list, trimmed sentence, "
                  "then rewrite as the tie order. The setup was saved before scoring the 1,574 test questions. "
                  "Intervals use 10,000 paired question resamples, with the same question carrying both systems' "
                  "scores in each draw. Completeness requires every gold fact, and fact recall averages each question's "
                  "fraction of gold facts found.", "",
                  "[Frozen dev choices](dev_decisions.json) · [Paired intervals](paired_intervals.json)", ""])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rewrite_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    out = Path(paths.OUT) / "idea1_cap"
    dev = {}
    for k, form in CONFIGS:
        r, rows = stage_run(k, form, "dev", args.device, args.rewrite_model, out)
        if r["n_dev"] != 681 or r["n_test"] != 0 or len(rows) != 681:
            raise ValueError("The dev phase must contain exactly the 681 dev questions")
        score = r["table"][1]
        dev[f"K{k}_{form}"] = {"tau": r["tau"], "complete": score[f"complete@{k}"],
                               "stop_rate": score["stop_rate"]}
    winner = max(FORMS, key=lambda f: dev[f"K10_{f}"]["complete"])
    decisions = {"selected_form": winner, "dev": dev, "rewrite_model": args.rewrite_model}
    save_json(out / "dev_decisions.json", decisions)
    print(f"Dev choices locked: {decisions}", flush=True)

    results, records = {}, {}
    chunks = load_chunks()
    test_qids = None
    for k, form in CONFIGS:
        r, rows = stage_run(k, form, "test", args.device, args.rewrite_model, out, dev[f"K{k}_{form}"]["tau"])
        qids = [row["qid"] for row in rows]
        if r["n_test"] != 1574 or r["n_dev"] or len(set(qids)) != 1574 or (test_qids is not None and qids != test_qids):
            raise ValueError("Every system must use the same 1,574 test questions")
        test_qids = qids
        for row in rows:
            for field in ("d1_chunk_ids", "selected_chunk_ids"):
                ids = row[field]
                counts = Counter(chunks[c]["article_id"] for c in ids)
                if len(set(ids)) != k or len(ids) != k or max(counts.values()) > 2:
                    raise ValueError(f"Invalid capped context for qid {row['qid']}")
        results[k, form], records[k, form] = r, rows
    write_report(out, decisions, results, records, args.rewrite_model)
    print(f"Saved {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
