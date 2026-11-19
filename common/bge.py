"""BGE embedding exactly as llama-index 0.9.40's HuggingFaceEmbedding did it.

Checked against llama_index/embeddings/huggingface.py and huggingface_utils.py (0.9.40):
  * tokenizer(padding=True, truncation=True, max_length=model.config.max_position_embeddings)  -> 512
  * token_type_ids dropped, CLS pooling, L2-normalised
  * query text  = f"{instruction} {query}".strip(), instruction for BAAI/bge-*-en* =
                  "Represent this question for searching relevant passages: "
  * passage text = text.strip()  (no instruction)
  * cosine similarity (= dot product, vectors are normalised), top-k
The original ran in fp32; fp16 is offered for speed (tiny numeric differences).
"""
import hashlib
import json
import os
from typing import List, Optional, Sequence

import numpy as np

from common.hf import dtype_kwargs

BGE_EN = {
    "BAAI/bge-small-en", "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en", "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en", "BAAI/bge-large-en-v1.5",
}
BGE_QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "


def default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEEncoder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: Optional[str] = None,
                 dtype: str = "fp32", query_instruction: Optional[str] = None, max_length: Optional[int] = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, **dtype_kwargs(torch_dtype)).to(self.device).eval()
        self.dtype = dtype
        self.max_length = max_length or int(self.model.config.max_position_embeddings)
        if query_instruction is None:
            query_instruction = BGE_QUERY_INSTRUCTION if model_name in BGE_EN else ""
        self.query_instruction = query_instruction

    def format_query(self, q: str) -> str:
        return f"{self.query_instruction} {q}".strip()

    @staticmethod
    def format_passage(t: str) -> str:
        return f" {t}".strip()

    def _embed(self, texts: List[str]) -> np.ndarray:
        import torch
        enc = self.tokenizer(texts, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        enc.pop("token_type_ids", None)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model(**enc)[0][:, 0]
            out = torch.nn.functional.normalize(out.float(), p=2, dim=1)
        return out.cpu().numpy().astype(np.float32)

    def encode(self, texts: Sequence[str], is_query: bool, batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        fmt = self.format_query if is_query else self.format_passage
        texts = [fmt(t) for t in texts]
        # sort by length so batches have little padding; results are put back in order
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        it = range(0, len(order), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="embed " + ("queries" if is_query else "passages"))
            except ImportError:
                pass
        for s in it:
            idx = order[s:s + batch_size]
            out[idx] = self._embed([texts[i] for i in idx])
        return out


def _texts_hash(texts: Sequence[str], extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def encode_cached(encoder: BGEEncoder, texts: Sequence[str], cache_path: str, is_query: bool,
                  batch_size: int = 32) -> np.ndarray:
    """Encode texts, or load them from cache_path(.npy) if the same texts were already encoded."""
    key = _texts_hash(texts, f"{encoder.model_name}|{encoder.dtype}|{is_query}|{encoder.query_instruction}")
    meta_path = cache_path + ".json"
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("key") == key:
            return np.load(cache_path)
    emb = encoder.encode(texts, is_query=is_query, batch_size=batch_size)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, emb)
    with open(meta_path, "w") as f:
        json.dump({"key": key, "n": len(texts), "model": encoder.model_name, "dtype": encoder.dtype,
                   "is_query": is_query}, f)
    return emb


class DenseIndex:
    """Exact cosine search over normalised vectors (what llama-index's SimpleVectorStore does)."""

    def __init__(self, emb: np.ndarray, device: Optional[str] = None):
        self.emb = emb.astype(np.float32)
        self.device = device
        self._t = None
        try:
            import torch
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._t = torch.from_numpy(self.emb).to(dev)
            self.device = dev
        except ImportError:
            pass

    def scores(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q).astype(np.float32)
        if self._t is not None:
            import torch
            with torch.inference_mode():
                return (torch.from_numpy(q).to(self.device) @ self._t.T).cpu().numpy()
        return q @ self.emb.T

    def search(self, q: np.ndarray, k: int, exclude: Optional[List[set]] = None, allowed: Optional[List[np.ndarray]] = None,
               batch: int = 256):
        """Top-k ids and scores for each query row.

        exclude[i]: chunk ids that query i may not return.
        allowed[i]: if given, only these chunk ids are searched (used by the outlet-filter baseline).
        """
        q = np.atleast_2d(q)
        ids_out, sc_out = [], []
        for s in range(0, len(q), batch):
            S = self.scores(q[s:s + batch])
            for r in range(S.shape[0]):
                i = s + r
                row = S[r]
                if allowed is not None and allowed[i] is not None:
                    mask = np.full(row.shape, -np.inf, dtype=np.float32)
                    mask[allowed[i]] = 0.0
                    row = row + mask
                if exclude is not None and exclude[i]:
                    row = row.copy()
                    row[list(exclude[i])] = -np.inf
                kk = min(k, int(np.isfinite(row).sum()))
                top = np.argpartition(-row, kk - 1)[:kk] if kk > 0 else np.array([], dtype=int)
                top = top[np.argsort(-row[top], kind="stable")]
                ids_out.append(top.tolist())
                sc_out.append(row[top].tolist())
        return ids_out, sc_out
