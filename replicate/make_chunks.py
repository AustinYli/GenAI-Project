"""Rebuild the MultiHop-RAG chunks exactly as the original code did.

Needs llama-index==0.9.40 (the version pinned in the MultiHop-RAG repo). The
output, data/multihop/chunks.jsonl, is already included, so you only need to
run this to check it.

Original pipeline (simple_retrieval.py at commit 3dd4d4e):
  Document(text=body, metadata={title, published_at, source})
  IngestionPipeline([SentenceSplitter(chunk_size=256), CustomExtractor()])
  -> default chunk_overlap = 200, tokens counted with tiktoken (gpt-3.5-turbo)
  -> CustomExtractor (a BaseExtractor) switches each node to the template
     "[Excerpt from document]\n{metadata}\nExcerpt:\n-----\n{content}\n-----\n"
  -> text that gets embedded = node.get_content(MetadataMode.EMBED)
  -> text the evaluation searches = node.get_content(MetadataMode.LLM) (same string)

Usage: python replicate/make_chunks.py [--chunk_size 256] [--chunk_overlap 200]
"""
import argparse
import json
import os
from typing import Dict, List

from llama_index.extractors import BaseExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.schema import Document, MetadataMode
from llama_index.text_splitter import SentenceSplitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CustomExtractor(BaseExtractor):
    """Same as the original: re-attaches title / source / published_at."""

    async def aextract(self, nodes) -> List[Dict]:
        return [
            {
                "title": n.metadata["title"],
                "source": n.metadata["source"],
                "published_at": n.metadata["published_at"],
            }
            for n in nodes
        ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=os.path.join(ROOT, "data", "multihop", "corpus.json"))
    p.add_argument("--out", default=os.path.join(ROOT, "data", "multihop", "chunks.jsonl"))
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--chunk_overlap", type=int, default=None,
                   help="default: llama-index default (200), as in the paper's code")
    a = p.parse_args()

    with open(a.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    docs = []
    for d in corpus:
        md = {"title": d["title"], "published_at": d["published_at"], "source": d["source"]}
        docs.append(Document(text=d["body"], metadata=md))
    doc2art = {doc.doc_id: i for i, doc in enumerate(docs)}

    kw = {"chunk_size": a.chunk_size}
    if a.chunk_overlap is not None:
        kw["chunk_overlap"] = a.chunk_overlap
    splitter = SentenceSplitter(**kw)
    print("chunk_size", splitter.chunk_size, "chunk_overlap", splitter.chunk_overlap)
    nodes = IngestionPipeline(transformations=[splitter, CustomExtractor()]).run(documents=docs)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, n in enumerate(nodes):
            art = doc2art[n.ref_doc_id]
            text = n.get_content(metadata_mode=MetadataMode.EMBED)
            assert text == n.get_content(metadata_mode=MetadataMode.LLM)
            rec = {
                "id": i,
                "article_id": art,
                "title": n.metadata["title"],
                "source": n.metadata["source"],
                "published_at": n.metadata["published_at"],
                "start_char": n.start_char_idx,
                "end_char": n.end_char_idx,
                # what gets embedded AND what the evaluation searches (identical here)
                "text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(nodes)} chunks from {len(docs)} articles -> {a.out}")


if __name__ == "__main__":
    main()
