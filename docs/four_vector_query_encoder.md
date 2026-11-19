# Four Vector Query Encoding for MultiHop Retrieval

## Research question

Can four query vectors pooled directly from BGE's token representations find more distinct evidence than one BGE vector when both systems use the same article cap? We propose to train four small pooling heads on frozen BGE, search the existing passage index, and compare the systems on held-out MultiHop-RAG questions. The primary outcome is complete@10, the share of questions for which the final ten chunks contain every gold fact.

## Evidence from the previous experiment

The original arrow adapter learned four transformations of a single pooled query embedding. On 1,574 held-out MultiHop-RAG questions with outlet names present, it slightly trailed plain BGE. A simple cap of two chunks per article produced a much larger gain.

| System | Complete@10 | Fact recall@10 |
|---|---:|---:|
| Plain BGE | 8.07% | 33.44% |
| Original arrows with soft slots | 7.81% | 32.66% |
| Plain BGE with two chunks per article | 14.80% | 39.54% |
| Outlet filter | 15.57% | 42.31% |

[Project experiment report](../outputs/REPORT.md#idea-2--multihop-rag-evaluation)

In a follow-up diagnostic with the number of unique candidates matched per question, the original arrow pool contained every gold fact for 37.23% of questions, compared with 39.64% for dense BGE. The ten chunks kept by the arrows still came from only 2.69 articles on average, compared with 5.68 for capped BGE. These results motivate evaluating candidate retrieval and final chunk selection separately.

## Query encoder

Run the question through `BAAI/bge-large-en-v1.5` once using the existing query instruction. Retain the final contextual representation of each question token before pooling. Give each of four learned attention heads its own scoring weights over these tokens. Each head takes a weighted average of the token representations, then applies a shared trainable projection and L2 normalization. Mask instruction, special, and padding tokens when computing the pooling weights. Each output is a full 1,024-dimensional vector, and training aligns those vectors with the frozen BGE passage space so the existing chunk index can be reused.

For example, in “Did Company A announce layoffs before Company B froze hiring?”, two heads could emphasize the layoff announcement and the hiring freeze, respectively. Their roles are learned from evidence matches and can vary by question.

## Training objective

Train on the existing MuSiQue, 2WikiMultihopQA, and HotpotQA preparation. Hold BGE and the passage embeddings fixed. For each question, assign its two to four distinct gold paragraphs one-to-one to the best-matching subset of the four heads. Train each assigned head to rank its matched paragraph above distractors and competing evidence using the contrastive loss from the original adapter. Unassigned heads receive no retrieval loss for that question. Use all four heads in the first evaluation to isolate the encoder change. If surplus heads frequently crowd out relevant candidates, test a learned per-head gate as a separate ablation and choose its operating point on validation data.

MuSiQue validation guides checkpoint and hyperparameter selection. MultiHop-RAG supplies evaluation queries and labels only.

## Retrieval and evaluation

Score the same 42,065 indexed chunks with exact cosine search. For a multi-vector query, rank each chunk by its best score across the four heads. Use a single fused ranking and apply the same two-chunks-per-article rule to each encoder. Scan deeper in the ranking when the cap leaves a list short of its exact chunk budget. This setup lets heads contribute evidence without reserving a fixed number of final slots for each head.

Compare single-vector BGE, four transformations of the pooled BGE vector, and the four token-pooling heads under that same selection rule. Retrain the pooled-vector control with the same assignment and loss as the token-pooling model. Run both with four heads and the same score fusion, and report their trainable parameter counts so readers can assess the effect of model capacity. Include the uncapped dense result and the outlet filter for context.

Use the existing 681-question dev and 1,574-question held-out test split. Report complete@10 and fact recall@10, then repeat at K=6. Measure complete coverage in the top 60 unique candidates before the cap and count distinct articles in the final context. Include query encoding and search latency because a shared encoder pass still produces four similarity searches. Repeat with outlet names masked as a diagnostic, with the original questions as the primary evaluation.

The primary comparison is the four-head encoder against capped single-vector BGE on complete@10. A gain before and after selection would indicate that the new vectors found additional evidence and retained it within the ten-chunk budget. A gain confined to the candidate pool would point to evidence selection as the next bottleneck.

## Related work

AMER also learns multiple query vectors against a shared document embedding space. It generates its vectors autoregressively and evaluates on multi-answer retrieval. This proposal tests four parallel pooling heads over one BGE pass for multi-hop evidence retrieval. [Chen et al., Beyond Single Embeddings, 2025](https://arxiv.org/abs/2511.02770). The target benchmark is [Tang and Yang, MultiHop-RAG, 2024](https://arxiv.org/abs/2401.15391).
