"""Score training-free retrieval on the public HippoRAG passage collections."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = HERE / "e5-large-v2"
OUT = HERE.parent / "outputs" / "hipporag_public"
RELEASE = "d5c8329422e0a0b834a15874545cb6a74b4f9b26"
E5_REVISION = "f169b11"
METRICS = ("R@5", "MAP@10", "nDCG@10", "FSR@10")
TABLE1 = {
    "musique": {"BM25": [34.58, 28.91, 39.18, 10.20], "e5-large-v2": [52.19, 46.03, 57.58, 27.10]},
    "hotpotqa": {"BM25": [66.75, 57.03, 66.61, 62.80], "e5-large-v2": [88.80, 78.73, 84.75, 89.20]},
    "2wikimultihopqa": {"BM25": [54.60, 47.71, 59.16, 28.70], "e5-large-v2": [72.15, 69.26, 78.12, 51.00]},
}
NAMES = {"musique": "MuSiQue", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2WikiMultihopQA"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_dataset(dataset):
    questions_path = DATA / f"{dataset}.json"
    corpus_path = DATA / f"{dataset}_corpus.json"
    samples = read_json(questions_path)
    corpus = read_json(corpus_path)
    if len(samples) != 1000:
        raise ValueError(f"Expected 1,000 {dataset} questions, got {len(samples)}")
    docs = [f"{row['title']}\n{row['text']}" for row in corpus]
    index = {doc: i for i, doc in enumerate(docs)}
    if len(index) != len(docs):
        raise ValueError(f"Duplicate passage in {corpus_path}")
    refs, kinds = [], []
    for sample in samples:
        if "supporting_facts" in sample:
            titles = {item[0] for item in sample["supporting_facts"]}
            separator = "" if dataset == "hotpotqa" else " "
            contexts = [(title, sentences) for title, sentences in sample["context"] if title in titles]
            if {title for title, _ in contexts} != titles:
                raise ValueError(f"Question {sample['_id']} has an unresolved supporting title")
            text = [title + "\n" + separator.join(sentences) for title, sentences in contexts]
            kinds.append(sample["type"])
        else:
            text = [p["title"] + "\n" + p["paragraph_text"] for p in sample["paragraphs"] if p["is_supporting"]]
            kinds.append(sample["id"].split("__", 1)[0])
        refs.append(frozenset(index[p] for p in text))
        if not refs[-1]:
            raise ValueError(f"No supporting passages for {sample['question']}")
    hashes = {"questions": file_hash(questions_path), "corpus": file_hash(corpus_path)}
    return samples, docs, refs, kinds, hashes


def measure(rankings, references):
    ranks = np.arange(1, 11)
    discount = 1 / np.log2(ranks + 1)
    rows = []
    for result, gold in zip(rankings, references, strict=True):
        if len(result) != 10 or len(set(result)) != 10:
            raise ValueError("Expected ten distinct passage IDs")
        hits = np.array([idx in gold for idx in result])
        denominator = min(len(gold), len(ranks))
        rows.append([
            hits[:5].sum() / len(gold),
            (hits * hits.cumsum() / ranks).sum() / denominator,
            (hits * discount).sum() / discount[:denominator].sum(),
            float(hits.sum() == len(gold)),
        ])
    return np.asarray(rows)


def top10(scores):
    return np.argsort(-scores, kind="stable")[:10].tolist()


class BM25:
    def __init__(self, docs):
        from rank_bm25 import BM25Okapi

        terms = [self.tokenize(doc) for doc in docs]
        self.index = BM25Okapi(terms)
        self.postings = defaultdict(list)
        for idx, freq in enumerate(self.index.doc_freqs):
            for term in freq:
                self.postings[term].append(idx)
        self.weights = {}

    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+", text.casefold())

    def search(self, text):
        scores = np.zeros(self.index.corpus_size)
        for term in self.tokenize(text):
            if term not in self.postings:
                continue
            if term not in self.weights:
                ids = np.array(self.postings[term])
                self.weights[term] = (ids, self.index.get_batch_scores([term], ids))
            ids, weights = self.weights[term]
            scores[ids] += weights
        return top10(scores)


class E5:
    def __init__(self, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL, local_files_only=True).to(device).eval()
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False

    def embed(self, texts, prefix, batch_size):
        from tqdm import tqdm

        torch = self.torch
        result = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        prefixed = [prefix + text for text in texts]
        lengths = self.tokenizer(prefixed, max_length=512, truncation=True, return_length=True)["length"]
        order = sorted(range(len(texts)), key=lengths.__getitem__, reverse=True)
        batches = []
        offset = 0
        while offset < len(order):
            padded_length = (lengths[order[offset]] + 7) // 8 * 8
            size = min(batch_size, max(1, 6_144 // padded_length))
            batches.append(order[offset:offset + size])
            offset += size
        with torch.inference_mode():
            for indices in tqdm(batches, desc=prefix.rstrip(": ")):
                tokens = self.tokenizer([prefixed[i] for i in indices], max_length=512, padding=True,
                                        pad_to_multiple_of=8, truncation=True, return_tensors="pt").to(self.device)
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"][..., None].bool()
                pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)
                result[indices] = torch.nn.functional.normalize(pooled.float(), dim=1).cpu().numpy()
        return result


def dense_search(queries, passages, device):
    import torch

    queries = torch.from_numpy(queries).to(device)
    passages = torch.from_numpy(passages).to(device)
    rankings = []
    with torch.inference_mode():
        for offset in range(0, len(queries), 128):
            scores = (queries[offset:offset + 128] @ passages.T).cpu().numpy()
            rankings.extend(top10(row) for row in scores)
    return rankings


def write_results(results):
    file = OUT / "baseline_results.json"
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    tmp.replace(file)


def summarize(ranks, refs, kinds):
    rows = measure(ranks, refs)
    return {
        "metrics_percent": dict(zip(METRICS, rows.mean(axis=0) * 100, strict=True)),
        "n_complete": int(rows[:, -1].sum()),
        "by_type_percent": {kind: dict(zip(METRICS, rows[np.array(kinds) == kind].mean(axis=0) * 100, strict=True))
                            for kind in sorted(set(kinds))},
        "top10_passage_indices": ranks,
    }


def run_baselines(args):
    OUT.mkdir(parents=True, exist_ok=True)
    file = OUT / "baseline_results.json"
    results = read_json(file) if file.exists() else {"hipporag_git_commit": RELEASE, "e5_revision": E5_REVISION,
                                                      "table1_percent": TABLE1, "datasets": {}}
    e5 = E5(args.device) if args.method != "bm25" else None
    for dataset in args.dataset:
        samples, docs, refs, kinds, hashes = load_dataset(dataset)
        row = results["datasets"].setdefault(dataset, {"n_questions": len(samples), "n_passages": len(docs),
                                                       "n_supports": sum(map(len, refs)), "sha256": hashes})
        if row["sha256"] != hashes:
            raise ValueError(f"{dataset} files differ from existing results")
        questions = [sample["question"] for sample in samples]
        if args.method != "e5" and ("BM25" not in row or args.force):
            tick = time.perf_counter()
            bm25 = BM25(docs)
            row["BM25"] = summarize([bm25.search(q) for q in questions], refs, kinds)
            row["BM25"]["seconds"] = time.perf_counter() - tick
            write_results(results)
            print(dataset, "BM25", row["BM25"]["metrics_percent"], flush=True)
        if e5 is not None and ("e5-large-v2" not in row or args.force):
            tick = time.perf_counter()
            scratch = OUT / f"{dataset}_e5_embeddings.npz"
            if scratch.exists() and not args.force:
                print(dataset, "using cached embeddings", flush=True)
                with np.load(scratch) as arrays:
                    passages, queries = arrays["passages"], arrays["queries"]
                if len(passages) != len(docs) or len(queries) != len(samples):
                    raise ValueError(f"Sizes of cached embeddings differ from {dataset}")
            else:
                print(dataset, "embedding passages and questions", flush=True)
                passages = e5.embed(docs, "passage: ", args.batch_size)
                queries = e5.embed(questions, "query: ", args.batch_size)
                tmp = scratch.with_suffix(".tmp.npz")
                np.savez(tmp, passages=passages, queries=queries)
                tmp.replace(scratch)
            row["e5-large-v2"] = summarize(dense_search(queries, passages, args.device), refs, kinds)
            row["e5-large-v2"]["seconds"] = time.perf_counter() - tick
            row["e5-large-v2"]["device"] = args.device
            write_results(results)
            print(dataset, "e5-large-v2", row["e5-large-v2"]["metrics_percent"], flush=True)
    return results


def write_report(results):
    datasets = results["datasets"]
    if any(dataset not in datasets or not {"BM25", "e5-large-v2"}.issubset(datasets[dataset]) for dataset in TABLE1):
        return
    e5_match = all(round(datasets[dataset]["e5-large-v2"]["metrics_percent"]["FSR@10"], 2) ==
                   TABLE1[dataset]["e5-large-v2"][-1] for dataset in TABLE1)
    outcome = ("The public HippoRAG files reproduced all three e5 FSR@10 values in Table 1." if e5_match else
               "The e5 FSR@10 reproduction check failed. The corpus and evaluation match remains unconfirmed, "
               "so the HippoRAG LLM stage stopped at this check.")
    lines = ["# HippoRAG public benchmark", "", outcome, "", "## Step 1 results", "",
             "Each metric cell shows the measured percentage and the Table 1 percentage, in that order. "
             "The last column gives the FSR difference in percentage points. "
             "[arXiv 2609.07050 Table 1](https://arxiv.org/html/2609.07050)", "",
             "| Dataset | Retriever | R@5 | MAP@10 | nDCG@10 | FSR@10 | FSR difference |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for dataset, title in NAMES.items():
        for retriever, reference in TABLE1[dataset].items():
            observed = [datasets[dataset][retriever]["metrics_percent"][metric] for metric in METRICS]
            scores = " | ".join(f"{a:.2f} / {b:.2f}" for a, b in zip(observed, reference, strict=True))
            lines.append(f"| {title} | {retriever} | {scores} | {observed[-1] - reference[-1]:+.2f} |")
    lines += ["", "## Corpus and scoring", "", "| Dataset | Questions | Passages | Gold passage references |",
              "|---|---:|---:|---:|"]
    for dataset, title in NAMES.items():
        record = datasets[dataset]
        lines.append(f"| {title} | {record['n_questions']:,} | {record['n_passages']:,} | {record['n_supports']:,} |")
    lines += ["",
              "All six data files came from the pinned public HippoRAG repository. Its paper reports the same corpus sizes. "
              "All gold titles and passage text resolved exactly to the release corpus using its evaluator’s dataset-specific "
              "sentence joins. Both retrievers indexed every released passage as the title followed by the passage text. "
              f"[Pinned data](https://github.com/OSU-NLP-Group/HippoRAG/tree/{RELEASE}/reproduce/dataset) "
              "[HippoRAG 2 corpus statistics](https://arxiv.org/html/2502.14802)", "",
              "e5 used the `query: ` and `passage: ` prefixes, attention-mask mean pooling, FP32 embeddings, L2 "
              "normalization, 512-token truncation, and exhaustive cosine ranking. BM25 used Okapi with k1=1.5, b=0.75, "
              "casefolded word tokens, and stable ties by source passage index. "
              f"[e5 model revision](https://huggingface.co/intfloat/e5-large-v2/tree/{E5_REVISION})", "",
              "All four metrics average per-question scores. Relevance is binary. MAP@10 divides the summed precision "
              "at relevant ranks by the number of supporting passages. FSR@10 counts a question only when all its "
              "supporting passages occur in the top 10. Table 1’s paper describes a common retrieval corpus and names "
              "the encoder, but its experiment section gives no exact question IDs or corpus revision. A discrepancy "
              "can therefore reflect the question set, corpus, formatting, or other retrieval details. "
              "One of its cited evaluations, GFM-RAG, reports a 9,221-passage HotpotQA corpus. "
              "[Paper’s experimental settings](https://arxiv.org/html/2609.07050) "
              "[GFM-RAG Table 6](https://proceedings.neurips.cc/paper_files/paper/2025/file/33ca0b1102b54c191a9a45a05adafaf4-Paper-Conference.pdf)", "",
              "[Per-type metrics, checksums, and top-10 passage IDs](baseline_results.json)", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("both", "bm25", "e5"), default="both")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--dataset", nargs="+", choices=tuple(TABLE1), default=list(TABLE1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_report(run_baselines(args))


if __name__ == "__main__":
    main()
