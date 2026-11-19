"""bge-reranker-large as llama-index 0.9.40's FlagEmbeddingReranker used it.

  pairs  = (query_str, node.get_content(MetadataMode.EMBED))   # raw query, no BGE instruction
  scores = FlagReranker.compute_score(pairs)  -> logits of the cross-encoder, max_length 512
  order  = sorted(nodes, key=lambda x: -x.score if x.score else 0)[:top_n]
"""
from common.hf import dtype_kwargs
from typing import List, Optional


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: Optional[str] = None,
                 fp16: bool = False, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **dtype_kwargs(torch.float16 if fp16 else torch.float32)).to(self.device).eval()
        self.max_length = max_length

    def score(self, query: str, passages: List[str], batch_size: int = 32) -> List[float]:
        import torch
        out = []
        for s in range(0, len(passages), batch_size):
            pairs = [[query, p] for p in passages[s:s + batch_size]]
            enc = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                out.extend(self.model(**enc, return_dict=True).logits.view(-1).float().cpu().tolist())
        return out

    def rerank(self, query: str, ids: List[int], texts: List[str], top_n: int) -> List[int]:
        scores = self.score(query, texts)
        order = sorted(range(len(ids)), key=lambda i: -scores[i] if scores[i] else 0)
        return [ids[i] for i in order[:top_n]]
