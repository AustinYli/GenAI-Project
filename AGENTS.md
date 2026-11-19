# AGENTS.md — handoff for the coding agent (read this first)

This folder holds a student research project on retrieval for multi-hop RAG. The code is written and was
tested on CPU with tiny stand-in models. **It has not yet been run with the real models or the real
training data.** Your job is to run it on this machine's GPU, fix whatever breaks, and report results.

---------------------------------------------------------------------------------------------------

## 0. Your tasks, in priority order

1. **Set up the environment** (section 3) and confirm PyTorch sees the GPU.
2. **Replicate MultiHop-RAG Table 5** for `bge-large-en-v1.5`: `replicate/main.py`, then `--rerank` (section 5).
   Report ours vs paper, side by side.
3. **Idea 1**: run `idea1_residual/run.py` with `--scorer word` first, then `llm_inplace`, `llm_probe`
   and `mlm` (section 6).
4. **Idea 2**: run prepare → embed → train → eval, plus the `--mask_outlets` check (section 7).
5. Report every result table back to the user (section 10). Fix bugs as you find them, but **do not change
   the experimental protocol** (section 2) without telling the user.

`run_all.bat --nopause` runs steps 2–4 in order, one log per step in `outputs\logs\`. Finished steps are
skipped on rerun, because each writes a `.done` marker. You can also run each step by hand.

---------------------------------------------------------------------------------------------------

## 1. Background: what this project is

### 1.1 The proposal being reviewed
The user is reviewing a classmate's project proposal, due around **Dec 10**. That proposal:
- builds a cheap "graph-assisted, latency-aware two-pass retrieval" pipeline: dense first pass, pull
  missing terms and new "bridge clues" from the first results with an entity–passage graph, then a
  second retrieval and weighted fusion;
- compares it against HippoRAG 2, with a ~3B instruction-tuned answering model;
- names one numerical replication target: MultiHop-RAG Table 5, `bge-large-en-v1.5` **without**
  reranker, Hits@10 = 0.6718 and MRR@10 = 0.4298 (retrieval only).

The review found good experimental design but low novelty. The core loop is essentially
pseudo-relevance feedback. Also, MultiHop-RAG questions mostly *name* every target (comparison and
temporal questions), so "bridge clue" discovery has little to do. The two ideas below are the user's
proposed alternatives.

### 1.2 MultiHop-RAG (Tang & Yang, 2024, arXiv 2401.15391; repo yixuantt/MultiHop-RAG)
- 609 news articles (`data/multihop/corpus.json`) and 2,556 queries (`data/multihop/MultiHopRAG.json`):
  2,255 answerable plus 301 `null_query`. Types are inference, comparison, temporal and null. Both files
  come from repo commit `3dd4d4e` and match their Git LFS SHA-256.
- Each answerable query has 2–4 evidence facts from 2–4 different articles. Distinct articles per
  question: 1,169 have 2, 774 have 3 and 312 have 4.
- 93% of answerable questions name the outlet of every evidence article ("as reported by TechCrunch and
  The Verge"). Only 49 outlets appear in the evidence. In 598 questions, all the evidence comes from a
  single outlet.
- **Retrieval eval** (repo's `retrieval_evaluate.py`): a chunk is relevant if a gold `fact` string
  (spaces and newlines removed) is a substring of the chunk text. Hits@k means **at least one** fact in
  the top k. MAP@10 is a non-standard variant. Null queries are skipped.
- **Paper's generation eval:** top-6 chunks from voyage-02 + bge-reranker-large (2,048-token cap). A
  word-overlap rule counts an answer correct if it shares any word with the gold answer. That rule can be
  gamed: the constant answer "yes no Google Sam Bankman-Fried insufficient" scores 0.86, beating GPT-4's
  0.56. Do **not** use that rule. If you add QA evaluation later, use normalized exact match / token F1,
  count hedged answers (yes and no together, several entities) as wrong, and report the
  constant-answer floor.

### 1.3 The two ideas (what the code implements)
- **Idea 1 — reader-guided residual retrieval.** After the first search returns D1, measure how much D1
  "explains" each part (span) of the question. Use the answering LLM's token probabilities with vs.
  without D1, a pointwise-mutual-information style gain. Named things with low gain are "missing". Run
  one extra search per missing thing, with the covered names removed from the query. If nothing is
  missing, stop: the result is then identical to dense retrieval. The process is always at most 2
  rounds.
- **Idea 2 — several search arrows.** A tiny trainable adapter turns BGE's single question vector into
  up to 4 vectors ("arrows"), each meant to point at a different evidence article. It is trained on
  MuSiQue, 2WikiMultihopQA and HotpotQA, **never on MultiHop-RAG**, then evaluated on MultiHop-RAG.
  There is no LLM call at query time.

### 1.4 Closest published work (for the write-up; not implemented here)
- **EfficientRAG** (EMNLP 2024, arXiv 2408.04259): iterates new queries without LLM calls. A
  DeBERTa-v3-large Labeler/Filter keeps "the unknown parts" of the question. Idea 1 = replace the
  trained Labeler/Filter with a training-free span-PMI score from the answering model. Its synthesized
  data is released (no checkpoints). Recall: 81.84 HotpotQA, 84.08 2Wiki, 49.51 MuSiQue.
- **Question Decomposition + Reranking** (Ammann et al. 2025, arXiv 2507.00355): MultiHop-RAG with
  bge-large + bge-reranker-large. Naive Hits@10 is 0.781 and QD+RR is 0.872 (MRR 0.464 → 0.635), but
  QD costs 16.7 s/query. Its GitHub repo is empty, so it would need reimplementing.
- **Multi-Meta-RAG** (arXiv 2406.13213): an LLM extracts outlet/date and filters the database. With
  bge-large + reranker (chunk 256, overlap 32), Hits@10 goes from 0.7490 to 0.8909. This is the
  "outlet shortcut". `idea2_arrows/eval_multihop.py` includes a rule-based version as the rival.
- **AMER** (Chen et al., Nov 2025, arXiv 2511.02770; code at github.com/timchen0618/amer): generates
  several query embeddings autoregressively with a LoRA-tuned 1–8B LLM. It uses Hungarian matching,
  InfoNCE and round-robin merging, and was tested only on AmbigQA/QAMPARI. Idea 2 is essentially a cheap,
  non-autoregressive version: a 131k-parameter adapter on frozen BGE, with a predicted count, applied to
  multi-hop. The write-up must cite AMER.

---------------------------------------------------------------------------------------------------

## 2. Experimental protocol (do not change silently)

- **Same retriever everywhere:** `BAAI/bge-large-en-v1.5`, the same chunk index (42,065 chunks), fp32
  embeddings and exact cosine search. Second searches and arrows search the same index.
- **Same final context per question:** every system hands over exactly **K** chunks (default K = 10;
  also report K = 6). The cap is per question, never "on average". Candidate pools may differ.
- **Duplicates:** neighbouring chunks overlap about 86% (200-token overlap on 256-token chunks).
  `--dedup_overlap` drops near-duplicates **for every system at once**. Report it as a separate table,
  never mixed with the default one.
- **Split:** `common/data.split_queries` gives a stratified 30/70 dev/test split by question type, seed
  13. Near-duplicate questions (same type + same evidence titles) are kept together. Answerable
  questions: **681 dev, 1,574 test**. Tune thresholds and pick variants on dev only; report test.
- **No MultiHop-RAG in Idea 2 training.** Its dev set is MuSiQue validation.
- **Main metric for the ideas:** `complete@K`, meaning **all** gold facts of the question are in the top
  K. Also report `recall@K` (share of facts found) and the repo's Hits@10 / MRR@10 / MAP@10.
  For scale: a quick TF-IDF retriever gives Hits@10 0.726 but complete@10 only 0.141. Complete coverage
  is the real gap.

---------------------------------------------------------------------------------------------------

## 3. Machine and environment

- **Windows**, Anaconda at `%USERPROFILE%\anaconda3`, **RTX 4070 (12 GB VRAM)**.
- **Disk: drive C: has only ~13–14 GB free.** Needs: CUDA PyTorch ~4–5 GB (skipped if Anaconda already
  has it), BGE 1.3 GB, Qwen2.5-3B-Instruct 6.2 GB, ModernBERT-large 0.8 GB, bge-reranker-large 2.2 GB,
  arrow vectors ~0.5–1 GB. Check free space before the big downloads. If space runs short, set `HF_HOME`
  to another drive, or skip the reranker row (step 11, optional) first.
- **Setup:** `setup.bat --nopause`. It creates `.venv` from Anaconda's Python with
  `--system-site-packages`, installs CUDA torch from `https://download.pytorch.org/whl/cu128` only if
  `torch.cuda.is_available()` is False, then installs `requirements.txt`, `en_core_web_sm` and runs
  `tests\test_selection.py`. Log: `outputs\logs\00_setup.log`. Success = the log shows
  `CUDA available: True` and four `ok test_...` lines.
  - If cu128 doesn't fit the driver, try `cu126` (or what pytorch.org lists for Windows + pip).
- **Python to use:** `.venv\Scripts\python.exe` (PowerShell: `& .\.venv\Scripts\python.exe ...`). The
  folder path contains a space ("GenAI Project"), so quote paths.
- **Long jobs:** embedding, LLM scoring and data prep each take tens of minutes. Run them in the
  background or with long timeouts, and follow the log files. `PYTHONUNBUFFERED=1` is set by
  `run_all.bat`.
- Library notes: written for transformers 4.48+ and 5.x. `common/hf.py` passes `dtype=` or
  `torch_dtype=` depending on the version. ModernBERT needs transformers ≥ 4.48.

---------------------------------------------------------------------------------------------------

## 4. Repository map

```
AGENTS.md                this file
README.md                short user-facing guide
setup.bat / run_all.bat  Windows setup and full pipeline (accept --nopause)
requirements.txt         torch is installed separately (CUDA build)
common/
  paths.py               folder layout (data/, outputs/, cache/)
  data.py                load chunks/queries/corpus, gold facts, dev/test split, outlet masking
  bge.py                 BGEEncoder (exact llama-index 0.9.40 behaviour), encode_cached, DenseIndex (exact search;
                         exclude= and allowed= per query)
  metrics.py             calculate_metrics = verbatim copy of the repo's retrieval_evaluate.py;
                         complete@K / recall@K; per-type breakdown; AUROC; table formatting
  rerank.py              bge-reranker-large as llama-index's FlagEmbeddingReranker used it
  selection.py           Picker (dedup), take_top, deal_turns (Idea 1), deal_quotas + largest_remainder (Idea 2)
  hf.py                  transformers version shim
replicate/
  make_chunks.py         rebuilds data/multihop/chunks.jsonl with llama-index==0.9.40 (optional; file is shipped)
  main.py                Table 5 replication; writes the caches the ideas reuse
idea1_residual/
  spans.py               spaCy + outlet gazetteer -> target / descriptive spans
  scorers.py             word, llm_inplace, llm_inplace_mean, llm_probe, mlm
  pipeline.py            candidate second-search queries, decide(), select() with card dealing
  run.py                 end-to-end: caches, tau tuning on dev, test tables, H1 AUROC, per-question log
idea2_arrows/
  prepare_data.py        MuSiQue/2Wiki/HotpotQA -> data/arrows/{train,dev}.jsonl (Hub streaming, or local files)
  embed_data.py          frozen BGE vectors for questions and paragraphs (fp16)
  model.py               ArrowAdapter, best_pairing, arrow_loss, slot_weights
  train.py               training + MuSiQue-dev evaluation, saves cache/arrows/adapter.pt(+.json)
  eval_multihop.py       MultiHop-RAG evaluation vs dense, diversity cap, outlet-filter rival
tests/test_selection.py  slot-rule and metric checks (no models)
data/multihop/           corpus.json, MultiHopRAG.json, chunks.jsonl (+ .gz copy)
outputs/                 results (created)    cache/   embeddings, rankings, scorer caches (created)
```

---------------------------------------------------------------------------------------------------

## 5. Replication (`replicate/main.py`)

**Target** (paper Table 5, bge-large-en-v1.5):

| | MRR@10 | MAP@10 | Hits@10 | Hits@4 |
|---|---|---|---|---|
| no reranker | 0.4298 | 0.3423 | 0.6718 | 0.5221 |
| with bge-reranker-large | 0.563 | 0.4759 | 0.7183 | 0.6364 |

**What the original code did** (simple_retrieval.py @ 3dd4d4e + llama-index 0.9.40), and how we match it:
- `Document(text=body, metadata={title, published_at, source})` →
  `IngestionPipeline([SentenceSplitter(chunk_size=256), CustomExtractor()])`. The default
  `chunk_overlap=200` applies, and tokens are counted with tiktoken (gpt-3.5-turbo, cl100k_base). The
  splitter subtracts the metadata length from the chunk size.
- `CustomExtractor` is a `BaseExtractor`, so it switches every node to the template
  `"[Excerpt from document]\n{metadata}\nExcerpt:\n-----\n{content}\n-----\n"`. The embedded text and
  the evaluated text are the same string.
- **We ran that exact library code** to produce `data/multihop/chunks.jsonl`: **42,065 chunks**,
  deterministic, SHA-256 `5be1b322…c99c`. Every one of the 6,084 gold facts is contained in at least one
  chunk.
  - Don't re-chunk with another library. If you rerun `make_chunks.py`, it needs
    `llama-index==0.9.40` and `setuptools<70` (for `pkg_resources`). tiktoken downloads `cl100k_base`
    from openaipublic.blob.core.windows.net.
- Embedding (llama-index 0.9.40 `HuggingFaceEmbedding`):
  - `max_length` = `max_position_embeddings` (512), `padding=True, truncation=True`
  - `token_type_ids` dropped, **CLS pooling**, L2-normalised, fp32
  - query text = `f"{instr} {query}".strip()` with
    `instr = "Represent this question for searching relevant passages: "`
  - passage text = `text.strip()`, no instruction

  `common/bge.py` reproduces this; batches are sorted by length, which doesn't change CLS outputs.
- Search: exact cosine top-10. Rerank: top-20, then bge-reranker-large on `(raw query, chunk text)`
  pairs, max_length 512, keep 10. Sort key `-score if score else 0`, as in the library.
- Output `outputs/replicate/bge-large-en-v1.5[_rerank]_retrieval_test.json` is in the repo's format. The
  repo's own `retrieval_evaluate.py` gave identical numbers to `common/metrics.calculate_metrics` in our
  tests.

**Expected:** within about ±0.01 of the paper. If it is off by more, check in this order:
1. chunk count = 42,065;
2. fp32;
3. the query instruction is applied (model name must be exactly `BAAI/bge-large-en-v1.5`, since the
   instruction is keyed on it);
4. null queries are excluded;
5. the evaluation scripts were added to the repo in Aug 2024, after the paper, so a small residual gap
   is possible and should be reported, not "fixed".

The repo's *current* default overlap is 25 (changed Sep 2026). The paper-era code used 200, which is
what we use.

**Caches it writes (reused by both ideas):**
- `cache/emb_BAAI__bge-large-en-v1.5_fp32_chunks.npy` (+ .json key)
- `..._queries.npy`
- `cache/dense_ranking_BAAI__bge-large-en-v1.5_fp32.json` (top-100 ids and scores for all 2,556 queries)

---------------------------------------------------------------------------------------------------

## 6. Idea 1 — reader-guided residual retrieval (`idea1_residual/`)

**Per question:**
1. `D1 = take_top(dense ranking, K)`, so `--dedup_overlap` applies here too.
2. **Spans** (`spans.py`):
   - outlet names from the corpus (gazetteer, labelled `OUTLET`) first, then spaCy `en_core_web_sm`
     entities, then noun chunks (with determiners, pronouns and wh-words trimmed);
   - **target** = ORG, PERSON, PRODUCT, GPE, EVENT, NORP, WORK_OF_ART, FAC, LOC, LAW, OUTLET;
   - **descriptive** = everything else (dates, numbers, noun phrases);
   - on MultiHop-RAG this averages 14.4 spans and 4.4 targets per question; 7 questions have no
     target.
3. **Gain per span** (`scorers.py`; all use D1 = the K chunk texts, numbered `[1] …`):
   - `word`: fraction of the span's content words present in D1. This is the word-absence baseline the
     original proposal effectively uses.
   - `llm_inplace`:
     $\sum_{i \in s} \big[\log p(t_i \mid \text{instr}, D_1, t_{<i}) - \log p(t_i \mid \text{instr}, t_{<i})\big]$
     over the span's question tokens.
     - Prompt: the chat template applied to "instr + Passages + Question: q". Two forward passes, no
       generation.
     - Logits are computed only for question positions (`base model + lm_head`) to save memory.
   - `llm_inplace_mean`: the same, averaged over tokens. This was the original formula; averaging
     dilutes multi-token names, which is why sum is the default.
   - `llm_probe`: mean over probes ("The passages above discuss:", "The passages above contain
     information about:") of $\log p(\text{span} \mid D_1,\text{probe}) - \log p(\text{span} \mid \text{probe})$.
     - The prefix KV cache is reused per span (deepcopy).
     - This removes word-order dependence.
   - `mlm`: ModernBERT-large fill-in-the-blank,
     $\sum_{i\in s} [\log p(t_i \mid D_1, q_{\setminus s}) - \log p(t_i \mid q_{\setminus s})]$.
     - Sees both sides of the blank.
     - Input is `[CLS] D1 [SEP] masked question [SEP]`, up to 8,192 tokens.
   - LLM default: `Qwen/Qwen2.5-3B-Instruct` (ungated). The user may swap in the proposal's own ~3B
     model with `--llm`.
4. **Decision with threshold τ:** targets with gain < τ are missing, the others found. Descriptive spans
   below τ matter only if no target is missing.
5. **Second searches** (all candidates are precomputed once, independent of τ, and cached):
   - for each missing target: `that target + all descriptive spans`, in question order. Found targets
     and other missing targets are left out;
   - if only details are missing: the keyword form of the whole question (key `kw`), given `m0 = 2`
     slots;
   - D1 is excluded from these searches, and all run as one batch. **Never loop.**
6. **Card dealing** for K slots:
   - each round, every missing target's list gets 1 turn (first), then the D1 list gets one turn per
     found target;
   - each turn takes the next chunk the `Picker` accepts;
   - any shortfall is topped up from the dense ranking.
   - Checked cases (K = 6): 1 found / 1 missing → 3+3; 1 found / 2 missing → 2+2+2; both missing → 3+3
     from the two searches; 6 found / 1 missing → 5+1; nothing missing → identical to dense.
7. **τ** is tuned on dev over 41 quantiles of all observed gains (+∞), maximising dev complete@K with
   ties broken by the higher stop rate. `--tau` fixes it instead.

**Hypotheses reported by `run.py`:**
- **H1:** AUROC of the question-level signal (lowest target gain, negated) for "D1 misses some gold
  fact". Compared with the word-absence signal, and also on the subset where every span word appears in
  D1 (the "passing mention" cases where the LLM signal must beat word matching).
  - Main risk to watch: the LLM gain may just track word overlap (copying). Report it honestly either
    way.
- **H2:** complete@K / recall@K of Idea 1 vs dense top-K vs always-expand (τ = +∞), on test.
- **H3:** share of questions that stop after the first search, and the mean number of second searches.
  Scoring time per question is also logged.

**Outputs** go to `outputs/idea1/<tag>_K10[ _dedup].{json,md}` plus `_per_question.json`, which has the
spans and gains, the path taken, the residual queries and whether coverage was complete.

**Caches** (`cache/idea1/`): `spans.json`, `gains_<scorer>[__model]_K10.json` (saved every 60 s, so it
can resume), `second_K10.json`, and the residual-query embeddings.

**Expected cost on a 4070:**
- `word`: seconds.
- `llm_inplace`: ~1 s/question, so ~40 min for 2,255 questions (D1 is about 3.5k tokens).
- `llm_probe`: ~2×.
- `mlm`: ~1–2 s/question.
- Try `--limit 50` first.

**Worth trying after the main runs** (on dev only):
- `--K 6`;
- `--dedup_overlap`;
- a separate threshold for descriptive spans (currently one τ for both; questions are long, so details
  often fall below τ);
- the small-LLM rewrite variant of the residual query (not implemented: the rule-based keyword query is
  used).

---------------------------------------------------------------------------------------------------

## 7. Idea 2 — several search arrows (`idea2_arrows/`)

**Data** (`prepare_data.py`, streamed from the Hugging Face Hub, nothing large written):

| Dataset | Source | How gold is read | Default size |
|---|---|---|---|
| MuSiQue train | `dgslibisey/MuSiQue` | `is_supporting`, answerable only | all (~20k) |
| 2WikiMultihopQA train | `hf://datasets/xanhho/2WikiMultihopQA/train.parquet` | supporting_facts titles | 15k |
| HotpotQA train | `hotpotqa/hotpot_qa`, config `distractor`; fallback `…/distractor/train-*.parquet` | supporting_facts titles | 15k |
| **dev** | MuSiQue `validation` (fallback `dev`; else 1,500 held-out MuSiQue train questions) | | all |

- Keeps questions with 2–4 **distinct** gold paragraphs; gold paragraphs with identical text are
  merged.
- Training questions keep all gold paragraphs plus up to 8 random distractors. Dev keeps all of its
  paragraphs.
- The parsers accept HF dict format, official list format, JSON-string fields and list-of-dicts. Each
  source fails softly, so check the log for `FAILED` or `skipped a row`.
- **These Hub IDs and schemas were not tested against the real Hub** (it was blocked where the code was
  written). If a source fails, either:
  - fix its entry in `SOURCES`, or
  - download the official files and pass `--hotpot_file`, `--musique_file`, `--musique_dev_file`,
    `--twowiki_file`.
- Look at `data/arrows/stats.json`. You need a reasonable number of 3- and 4-gold questions; they come
  from MuSiQue, plus 2Wiki `bridge_comparison` (4 gold). HotpotQA is always 2.

**Vectors** (`embed_data.py`):
- frozen BGE, fp16;
- paragraphs are written in the MultiHop-RAG chunk layout (`[Excerpt from document]\ntitle: …`) to
  shrink the domain gap;
- questions get the BGE query instruction;
- files: `data/arrows/paragraphs.npy`, `{train,dev}_q.npy`, `{train,dev}_index.json`;
- expect roughly 30–60 min.

**Model** (`model.py`):
- arrows: $v_j = \mathrm{normalize}(e + A_j B_j^\top e)$, $J = 4$, rank 16, $A_j, B_j \in \mathbb{R}^{1024\times16}$;
- $A_j$ starts at 0, so every arrow starts equal to $e$;
- $B_j$ starts small, random and different per arrow, which breaks the tie and gives $A_j$ a gradient.
  (Starting both at zero would freeze training.)
- count head: Linear(1024 → 3) for "2/3/4 pieces".

**Loss** (`arrow_loss`):
- **Pairing:** the **true** gold count $m$ decides which arrows are trained (arrows $1..m$). They are
  paired one-to-one with the gold paragraphs by the best total similarity, trying all $m!$ orders,
  recomputed every step on detached similarities.
- **Arrow loss:** softmax at temperature $t = 0.05$ over the in-batch candidates.
  - Target = the paired gold paragraph.
  - The other gold paragraphs of the same question are negatives with weight $\beta$ (default 1); this
    is what makes the arrows split.
  - The question's distractors and all paragraphs of other questions in the batch are negatives.
  - Copies of the question's own gold paragraphs elsewhere in the batch are excluded.
- **Spare arrows** ($j > m$): weight 0.1, pulled to the nearest gold paragraph with duplicates allowed.
  If the count head over-predicts at test time, they then fetch a copy of a real piece, not noise.
- **Count loss:** cross-entropy with class weights from the training counts.

**Training** (`train.py`):
- AdamW, lr 1e-3, batch 256, 6 epochs;
- the checkpoint is chosen by MuSiQue-dev `arrows_complete@4`. Dev ranks each question's own
  candidates, and the log prints single-vector vs arrows complete@2/4/6 plus count accuracy;
- saves `cache/arrows/adapter.pt` and `adapter.json`, which records the settings and the full history;
- runs in minutes on GPU, and also works on CPU;
- knobs: `--beta`, `--t`, `--lam_extra`, `--rank`, `--epochs`, `--lr`. Tune them on MuSiQue dev only.
- On a synthetic stand-in, loss fell from 4.6 to 2.5 and dev complete@6 rose from 0.943 to 0.97, so
  the mechanism works. Real numbers are unknown.

**MultiHop-RAG evaluation** (`eval_multihop.py`, test and dev tables). Systems:
- dense top-K;
- dense with ≤ 2 chunks per article;
- **outlet filter (rival)**: outlet names in the question → one search restricted to each outlet's
  chunks, with equal slots. This is the rule-based Multi-Meta-RAG shortcut; arrows must beat it;
- **arrows (soft slots)**: all 4 search, slots from $w = [1, 1, P(n\ge3), P(n=4)]$ via largest remainder,
  then dealt round-robin;
- **arrows (hard count)**: only the first argmax(n) arrows;
- **arrows (oracle count)**: diagnostic only, using the number of evidence articles.

It also prints the top-10 overlap between arrows 1 and 2, and how often the count head matches the
number of evidence articles.

`--mask_outlets` replaces outlet names with "a news outlet" (and re-embeds) for the shortcut check.

**Risks:**
- Domain gap: training is Wikipedia bridge and comparison questions; testing is news questions that
  name several things.
- The count head never sees outlet names in training.
- Arrows can drift far from $e$. If arrows lose to dense, look at arrow 1-vs-2 overlap, and at
  `--lam_extra`, `--beta` and the learning rate.

---------------------------------------------------------------------------------------------------

## 8. Status: tested vs untested

**Tested (CPU, tiny random models of the same architectures: BERT, Qwen2, ModernBERT, BERT reranker):**
- every script runs end to end, including the reranker path;
- metric equivalence with the repo's script;
- slot rules (`tests/test_selection.py`);
- scorer bookkeeping: in-place log-probs equal a full forward pass, probe cache path equals a full
  forward pass, and MLM masks the right tokens;
- dataset parsers on every format;
- the arrow loss learns on a synthetic task.

**Untested:**
- real model downloads;
- CUDA on this machine;
- Hugging Face dataset streaming, IDs and splits;
- the real runtimes;
- Windows-specific issues (console encoding, paths with spaces);
- memory on 12 GB: Qwen2.5-3B in bf16 with a ~3.5k-token prompt should fit. If it OOMs, lower `--K` or
  use `--limit` to check first.

---------------------------------------------------------------------------------------------------

## 9. Run order (what `run_all.bat` does)

| Step | Command | Needs | Rough time |
|---|---|---|---|
| 01 | `replicate\main.py` | BGE | ~10–15 min (fp32) |
| 02 | `idea1_residual\run.py --scorer word` | 01 | ~1 min |
| 03 | `idea2_arrows\prepare_data.py` | Hub | 10–30 min |
| 04 | `idea2_arrows\embed_data.py` | 03 | 30–60 min |
| 05 | `idea2_arrows\train.py` | 04 | minutes |
| 06 | `idea2_arrows\eval_multihop.py` | 01, 05 | minutes |
| 07 | `idea2_arrows\eval_multihop.py --mask_outlets` | 06 | minutes |
| 08 | `idea1_residual\run.py --scorer llm_inplace` | 01, Qwen 6.2 GB | ~40 min |
| 09 | `idea1_residual\run.py --scorer llm_probe` | 08's model | ~80 min |
| 10 | `idea1_residual\run.py --scorer mlm` | ModernBERT | ~40–75 min |
| 11 | `replicate\main.py --rerank` | reranker 2.2 GB | ~10 min |

Logs: `outputs\logs\<step>.log`. Status lines are in `outputs\logs\status.txt`. A `<step>.done` file
marks success; delete it to rerun that step.

---------------------------------------------------------------------------------------------------

## 10. What to report back to the user

1. **Replication table:** ours vs paper for both rows, with differences, plus complete@4/10 and
   recall@4/10 (not in the paper).
2. **Idea 1:** for each scorer, the test table (dense / Idea 1 / always-expand: complete@10, recall@10,
   complete@6, Hits@10, MRR@10, MAP@10, stop rate, number of second searches), the tuned τ, the H1
   AUROCs (scorer, word, and the "words all present" subset), and seconds per question. Note which
   question types gain or lose (`details[...]["by_type"]` in the JSON).
3. **Idea 2:** `data/arrows/stats.json` summary, the dev training curve, and the MultiHop-RAG test table
   with and without `--mask_outlets`, including the outlet-filter rival.
4. Anything you changed in the code, and why.

**Next steps the user has in mind** (don't start without asking):
- QA evaluation with the answering model and normalized scoring;
- paired bootstrap confidence intervals on test;
- EfficientRAG / question-decomposition baselines.

---------------------------------------------------------------------------------------------------

## 11. Glossary

- **D1**: the K chunks from the first (dense) search.
- **Span**: a short phrase of the question. A *target* is a named thing; *descriptive* is a detail.
- **Gain**: how much D1 raises the model's probability of a span (log-ratio). Low gain = not covered.
- **τ (tau)**: the gain threshold.
- **Residual query**: one missing target plus the details.
- **Card dealing**: the round-robin slot rule that keeps every list from being starved.
- **Arrow**: one of up to 4 query vectors from the adapter.
- **complete@K**: all gold facts are in the top K. **recall@K**: the share of gold facts in the top K.
