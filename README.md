# MultiHop-RAG replication + two retrieval ideas

Everything runs from this folder. GPU strongly recommended (written for an RTX 4070, 12 GB).

```
common/            shared code: data, BGE embedding + exact search, metrics, slot rules
replicate/         Table 5 replication (bge-large-en-v1.5, with and without the reranker)
idea1_residual/    Idea 1: reader-guided residual retrieval
idea2_arrows/      Idea 2: several search arrows from one question (trained on MuSiQue/2Wiki/HotpotQA)
tests/             logic checks for the slot rules and metrics (no models)
data/multihop/     corpus.json, MultiHopRAG.json (repo commit 3dd4d4e), chunks.jsonl
outputs/  cache/   created by the scripts
```

## Setup and running (Windows)

1. Double-click **setup.bat** once. It makes a `.venv` in this folder, from Anaconda's Python, so it can
   reuse packages you already have. It installs CUDA PyTorch only if missing, plus the packages in
   `requirements.txt` and the spaCy English model, and runs the logic tests.
   Log: `outputs\logs\00_setup.log`.
2. Double-click **run_all.bat**. It runs every step below in order and writes one log per step to
   `outputs\logs\` plus `status.txt`. Finished steps are skipped when you run it again, so if a step
   fails you can fix it and rerun.

To run one step by hand: `.venv\Scripts\python.exe replicate\main.py` (and so on, commands below).

Models are downloaded from Hugging Face on first use (into `%USERPROFILE%\.cache\huggingface`;
set `HF_HOME` to another drive if C: is short on space):

| Model | Used by | Size |
|---|---|---|
| BAAI/bge-large-en-v1.5 | everything | 1.3 GB |
| BAAI/bge-reranker-large | `replicate --rerank` only | 2.2 GB |
| Qwen/Qwen2.5-3B-Instruct | Idea 1 LLM scorers | 6.2 GB |
| answerdotai/ModernBERT-large | Idea 1 `mlm` scorer | 0.8 GB |

## 1. Replication (run this first; everything else reuses its embeddings)

```bat
python replicate\main.py
python replicate\main.py --rerank
```

Target (paper Table 5, bge-large-en-v1.5):

| | MRR@10 | MAP@10 | Hits@10 | Hits@4 |
|---|---|---|---|---|
| without reranker | 0.4298 | 0.3423 | 0.6718 | 0.5221 |
| with reranker | 0.563 | 0.4759 | 0.7183 | 0.6364 |

How faithful it is:
* **Chunks** were produced by the original pipeline itself (llama-index 0.9.40, `SentenceSplitter(chunk_size=256)`
  with its default 200-token overlap, tiktoken gpt-3.5 counting, the repo's `CustomExtractor`). That gives
  42,065 chunks, each with the template `[Excerpt from document]\ntitle: …\npublished_at: …\nsource: …\nExcerpt:\n-----\n…`.
  `replicate/make_chunks.py` rebuilds the file if you want to check it (needs `pip install llama-index==0.9.40`).
* **Embedding** matches llama-index 0.9.40's `HuggingFaceEmbedding`: CLS pooling, normalised, max 512 tokens,
  query prefix `"Represent this question for searching relevant passages: "`, fp32.
* **Metrics**: `common/metrics.calculate_metrics` is copied from the repo's `retrieval_evaluate.py`; the output
  file `outputs/replicate/*_retrieval_test.json` is in the repo's format, so the repo's own script gives the same numbers.
* Every gold evidence fact fits inside at least one chunk (checked), so nothing is lost at chunk boundaries.

Also reported (not in the paper): `complete@K` = all of a question's gold facts are in the top K,
`recall@K` = share of gold facts in the top K. These are what the two ideas are judged on.

## 2. Idea 1 — reader-guided residual retrieval

```bat
python idea1_residual\run.py --scorer word           :: word-absence baseline (seconds)
python idea1_residual\run.py --scorer llm_inplace    :: Qwen2.5-3B reads D1 + question, ~1 s/question
python idea1_residual\run.py --scorer llm_probe      :: span scored after a neutral probe sentence
python idea1_residual\run.py --scorer mlm            :: ModernBERT-large fill-in-the-blank
python idea1_residual\run.py --scorer llm_inplace --limit 50   :: quick check first
```

Steps per question: D1 = dense top-K → spaCy spans (targets = named things incl. the 49 outlet names;
descriptive = details) → coverage gain per span → targets below τ are missing → one second search per
missing target (that target + all details, D1 excluded), all in one batch → K slots dealt round-robin
(each missing target 1 turn per round, D1 one turn per found target). τ is tuned on a dev split
(30%, stratified by question type, near-duplicates kept together); numbers are reported on the other 70%.
Options: `--K 6`, `--dedup_overlap` (drop near-duplicate chunks for every system), `--tau 1.5`.

Output `outputs/idea1/<scorer>_K10.md/.json`: dense vs Idea 1 vs always-expand (complete@K, recall@K,
Hits/MRR/MAP, share of questions that stop after the first search), the H1 AUROC for
"D1 is missing evidence" (scorer vs word absence, and on questions where all words are present),
scoring time, and a per-question log.

## 3. Idea 2 — several search arrows

```bat
python idea2_arrows\prepare_data.py     :: streams MuSiQue + 2Wiki + HotpotQA from the Hub (~45k questions)
python idea2_arrows\embed_data.py       :: frozen BGE, fp16, ~30-60 min
python idea2_arrows\train.py            :: minutes; picks the epoch by MuSiQue-dev complete@4
python idea2_arrows\eval_multihop.py    :: MultiHop-RAG (never seen in training)
python idea2_arrows\eval_multihop.py --mask_outlets   :: shortcut check
```

* Training data never includes MultiHop-RAG. Dev = MuSiQue validation.
* Model: `v_j = normalise(e + A_j B_j^T e)`, 4 arrows, rank 16 (131k parameters), plus a count head
  (2/3/4 pieces). Loss: best one-to-one pairing of arrows with gold paragraphs (true count), contrastive
  softmax with the other gold paragraphs as negatives (weight `--beta`), the question's distractors and the
  rest of the batch; spare arrows pulled to the nearest gold (`--lam_extra`); class-weighted count loss.
* Evaluation systems: dense top-K, dense ≤2 chunks per article, a rule-based outlet filter (the
  Multi-Meta-RAG shortcut, as the rival to beat), arrows with soft slots `w = [1, 1, P(n≥3), P(n=4)]`,
  arrows with the predicted count, and arrows with the true count (diagnostic).
* If the Hub can't be reached, download the official files and pass `--hotpot_file`, `--musique_file`,
  `--musique_dev_file`, `--twowiki_file` to `prepare_data.py`.

## 4. Four query vectors from token representations

See the [experiment design](docs/four_vector_query_encoder.md). The matched experiment reuses the frozen
BGE checkpoint, the prepared arrow training data and paragraph vectors, and the FP32 MultiHop-RAG index.
Run the commands from the project folder after replication and arrow data embedding.

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'cache\hf'
$env:HF_HUB_OFFLINE = '1'
& .\.venv\Scripts\python.exe idea3_multivector\train.py --device cuda
& .\.venv\Scripts\python.exe idea3_multivector\evaluate.py --device cuda
```

Training saves the independently selected MuSiQue-validation checkpoints and curve in `cache/multivector/`.
Evaluation saves original and outlet-masked results for both budgets in `outputs/idea3/`.
See the [completed GPU results](outputs/idea3/REPORT.md).

### News adaptation experiment

This experiment starts at the saved four-vector checkpoints and splits the former MultiHop-RAG
dev questions into article-disjoint training and validation folds. It compares local adaptation
with and without auxiliary multi-document news evidence from Continual-Eval. The original test
questions are outside both folds.

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'cache\hf'
$env:HF_HUB_OFFLINE = '0'
& .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('ruggsea/continual-eval', repo_type='dataset', revision='94f1cb577499d9b9351f7ffbb32c67a32f537d14', allow_patterns=['multihop/*.parquet'])"
$env:HF_HUB_OFFLINE = '1'
& .\.venv\Scripts\python.exe idea3_multivector\adapt.py --device cuda
```

The split manifest, adapted heads and validation results are written to `outputs/idea3_adapted/`.
See the [held-out validation results](outputs/idea3_adapted/REPORT.md).

## Disk

About 5 GB for the Python environment with CUDA PyTorch, the models above, ~0.5–1 GB of
arrow training vectors and ~0.3 GB of caches. `--no-cache-dir` keeps pip from storing extra copies.
