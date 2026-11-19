"""Embed the arrow training data with the same frozen BGE used for MultiHop-RAG.

Paragraphs are written in the same layout as the MultiHop-RAG chunks
("[Excerpt from document]\\ntitle: ...\\nExcerpt:\\n-----\\n...") so the training
vectors look like the vectors the arrows will search at test time.

Writes data/arrows/:
  paragraphs.npy   [P, 1024] float16, every distinct paragraph once
  {split}_q.npy    [N, 1024] float16, questions (with the BGE query instruction)
  {split}_index.json  per question: paragraph ids, which positions are gold

Usage:  python idea2_arrows/embed_data.py   (fp16 on the GPU; ~30-60 min for the default data on a 4070)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from common import paths  # noqa: E402
from common.bge import BGEEncoder  # noqa: E402
from common.data import load_jsonl  # noqa: E402


def para_text(p, plain=False):
    if plain:
        return f"{p['title']}\n{p['text']}"
    return f"[Excerpt from document]\ntitle: {p['title']}\nExcerpt:\n-----\n{p['text']}\n-----\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--plain", action="store_true", help="embed 'title\\ntext' instead of the chunk layout")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    d = paths.ARROWS_DATA
    splits = {s: load_jsonl(os.path.join(d, f"{s}.jsonl")) for s in ("train", "dev")}
    pid = {}
    texts = []
    index = {}
    for s, recs in splits.items():
        index[s] = []
        for r in recs:
            ids = []
            for p in r["paragraphs"]:
                key = (p["title"], p["text"])
                if key not in pid:
                    pid[key] = len(texts)
                    texts.append(para_text(p, a.plain))
                ids.append(pid[key])
            index[s].append({"qid": r["qid"], "dataset": r["dataset"], "para_ids": ids, "gold": r["gold"]})
    print(f"{len(texts)} distinct paragraphs, " + ", ".join(f"{s}: {len(v)} questions" for s, v in splits.items()))

    enc = BGEEncoder(a.model, device=a.device, dtype=a.dtype)
    np.save(os.path.join(d, "paragraphs.npy"), enc.encode(texts, is_query=False, batch_size=a.batch_size).astype(np.float16))
    for s, recs in splits.items():
        q = enc.encode([r["question"] for r in recs], is_query=True, batch_size=a.batch_size)
        np.save(os.path.join(d, f"{s}_q.npy"), q.astype(np.float16))
        with open(os.path.join(d, f"{s}_index.json"), "w") as f:
            json.dump(index[s], f)
    with open(os.path.join(d, "embed_meta.json"), "w") as f:
        json.dump({"model": a.model, "dtype": a.dtype, "plain": a.plain, "n_paragraphs": len(texts)}, f)
    print("done")


if __name__ == "__main__":
    main()
