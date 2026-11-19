import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MULTIHOP = os.path.join(DATA, "multihop")
ARROWS_DATA = os.path.join(DATA, "arrows")
OUT = os.path.join(ROOT, "outputs")
CACHE = os.path.join(ROOT, "cache")

CORPUS_JSON = os.path.join(MULTIHOP, "corpus.json")
QUERIES_JSON = os.path.join(MULTIHOP, "MultiHopRAG.json")
CHUNKS_JSONL = os.path.join(MULTIHOP, "chunks.jsonl")


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0] if len(dirs) == 1 else dirs


def model_slug(name: str) -> str:
    return name.replace("/", "__")
