"""Build the arrow training data from multi-hop datasets OTHER than MultiHop-RAG.

  train: MuSiQue-Ans (train, answerable) + 2WikiMultihopQA (train) + HotpotQA (train, distractor setting)
  dev:   MuSiQue-Ans validation (answerable) -- used to pick the checkpoint, never MultiHop-RAG

Each record: question, its candidate paragraphs (gold + distractors) and which are gold.
Only questions with 2-4 distinct gold paragraphs are kept (that is what the arrows learn to split).

Default source is the Hugging Face Hub, streamed (nothing big is written to disk):
  HotpotQA  hotpotqa/hotpot_qa (config "distractor")
  MuSiQue   dgslibisey/MuSiQue
  2Wiki     xanhho/2WikiMultihopQA (train.parquet / dev.parquet)
If the Hub is not reachable, download the official files yourself and pass them:
  --hotpot_file hotpot_train_v1.1.json  --musique_file musique_ans_v1.0_train.jsonl
  --musique_dev_file musique_ans_v1.0_dev.jsonl  --twowiki_file train.json

Usage:  python idea2_arrows/prepare_data.py            (defaults: 15k HotpotQA, 15k 2Wiki, all MuSiQue)
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths  # noqa: E402


# ---------------------------------------------------------------- parsing helpers
def _maybe_json(x):
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _pairs(x, a_keys=("title",), b_keys=("sentences", "content", "sent_id", "sentence", "text")):
    """Normalise [[a, b], ...] / {"title": [...], "sentences": [...]} / [{"title":..,"sentences":..}] to list of (a, b)."""
    x = _maybe_json(x)
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, dict):
        ak = next((k for k in a_keys if k in x), None)
        bk = next((k for k in b_keys if k in x), None)
        if ak is None or bk is None:
            raise ValueError(f"can't parse {list(x)[:5]}")
        return list(zip(list(x[ak]), list(x[bk])))
    out = []
    for item in x:
        item = _maybe_json(item)
        if isinstance(item, dict):
            ak = next((k for k in a_keys if k in item), None)
            bk = next((k for k in b_keys if k in item), None)
            out.append((item[ak], item[bk]))
        else:
            item = list(item)
            out.append((item[0], item[1]))
    return out


def _sent_text(sents):
    sents = _maybe_json(sents)
    if isinstance(sents, str):
        return sents.strip()
    if hasattr(sents, "tolist"):
        sents = sents.tolist()
    return " ".join(str(s).strip() for s in sents if str(s).strip())


def parse_context_style(ex, dataset):
    """HotpotQA and 2Wiki: context = paragraphs with titles, supporting_facts = (title, sentence id)."""
    q = ex.get("question")
    ctx = _pairs(ex["context"])
    sup = _pairs(ex["supporting_facts"])
    paras = [{"title": str(t), "text": _sent_text(s)} for t, s in ctx]
    gold_titles = []
    for t, _ in sup:
        if t not in gold_titles:
            gold_titles.append(t)
    title_idx = {}
    for i, p in enumerate(paras):
        title_idx.setdefault(p["title"], i)
    gold = [title_idx[t] for t in gold_titles if t in title_idx]
    if len(gold) != len(gold_titles):
        return None
    qid = ex.get("id") or ex.get("_id") or ex.get("qid")
    return {"qid": f"{dataset}:{qid}", "dataset": dataset, "question": q, "paragraphs": paras, "gold": gold,
            "type": ex.get("type")}


def parse_musique(ex):
    if ex.get("answerable") is False:
        return None
    ps = _maybe_json(ex["paragraphs"])
    if hasattr(ps, "tolist"):
        ps = ps.tolist()
    paras, gold = [], []
    for i, p in enumerate(ps):
        p = _maybe_json(p)
        paras.append({"title": str(p.get("title", "")), "text": str(p.get("paragraph_text", p.get("text", ""))).strip()})
        if p.get("is_supporting"):
            gold.append(i)
    return {"qid": f"musique:{ex['id']}", "dataset": "musique", "question": ex["question"], "paragraphs": paras,
            "gold": gold, "type": str(ex["id"]).split("__")[0]}


def dedup_gold(rec):
    """Merge gold paragraphs with identical text (they are one target for the arrows)."""
    seen, g = {}, []
    for i in rec["gold"]:
        key = (rec["paragraphs"][i]["title"], rec["paragraphs"][i]["text"])
        if key not in seen:
            seen[key] = i
            g.append(i)
    rec["gold"] = g
    return rec


def trim_distractors(rec, max_distractors, rng):
    if max_distractors is None:
        return rec
    gold = set(rec["gold"])
    dis = [i for i in range(len(rec["paragraphs"])) if i not in gold]
    rng.shuffle(dis)
    keep = sorted(list(gold) + dis[:max_distractors])
    remap = {old: new for new, old in enumerate(keep)}
    rec["paragraphs"] = [rec["paragraphs"][i] for i in keep]
    rec["gold"] = [remap[i] for i in rec["gold"]]
    return rec


# ---------------------------------------------------------------- loading
def iter_local(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.endswith(".parquet"):
        import pandas as pd
        for _, row in pd.read_parquet(path).iterrows():
            yield row.to_dict()
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        yield from (data if isinstance(data, list) else data.get("data", []))


def iter_hub(name, config=None, split="train", data_files=None, seed=0, shuffle=True):
    from datasets import load_dataset
    if data_files:
        ds = load_dataset("parquet", data_files={split: data_files}, split=split, streaming=True)
    else:
        ds = load_dataset(name, config, split=split, streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=5000)
    yield from ds


def collect(it, parser, cap, min_gold=2, max_gold=4):
    out, seen = [], 0
    for ex in it:
        seen += 1
        try:
            rec = parser(ex)
        except Exception as e:  # malformed row: skip, but say so once
            if seen < 3:
                print("  skipped a row:", repr(e)[:200])
            continue
        if rec is None:
            continue
        rec = dedup_gold(rec)
        if min_gold <= len(rec["gold"]) <= max_gold and rec["question"]:
            out.append(rec)
        if cap and len(out) >= cap:
            break
    return out


def hub_iter(candidates, seed, shuffle=True):
    """Try several ways of reading a dataset from the Hub; return an iterator over the first that works."""
    errors = []
    for c in candidates:
        try:
            it = iter_hub(c.get("name"), c.get("config"), c.get("split", "train"), c.get("data_files"), seed=seed,
                          shuffle=shuffle)
            first = next(iter(it))
            print(f"  using {c}")

            def chain(first=first, it=it):
                yield first
                yield from it
            return chain()
        except Exception as e:  # try the next source
            errors.append(f"{c}: {repr(e)[:300]}")
    raise RuntimeError("could not load from the Hub:\n  " + "\n  ".join(errors))


SOURCES = {
    "musique_train": [{"name": "dgslibisey/MuSiQue", "split": "train"}],
    "musique_dev": [{"name": "dgslibisey/MuSiQue", "split": "validation"},
                    {"name": "dgslibisey/MuSiQue", "split": "dev"}],
    "2wiki_train": [{"data_files": "hf://datasets/xanhho/2WikiMultihopQA/train.parquet"}],
    "hotpot_train": [{"name": "hotpotqa/hotpot_qa", "config": "distractor", "split": "train"},
                     {"data_files": "hf://datasets/hotpotqa/hotpot_qa/distractor/train-*.parquet"}],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_hotpot", type=int, default=15000)
    ap.add_argument("--n_2wiki", type=int, default=15000)
    ap.add_argument("--n_musique", type=int, default=0, help="0 = all answerable training questions")
    ap.add_argument("--n_dev", type=int, default=0, help="0 = all MuSiQue validation questions")
    ap.add_argument("--max_distractors", type=int, default=8, help="per training question (dev keeps all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hotpot_file")
    ap.add_argument("--musique_file")
    ap.add_argument("--musique_dev_file")
    ap.add_argument("--twowiki_file")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out_dir = paths.ensure(paths.ARROWS_DATA)

    def load(label, local, key, parser, cap, shuffle=True):
        print(f"{label} ...")
        try:
            it = iter_local(local) if local else hub_iter(SOURCES[key], a.seed, shuffle)
            recs = collect(it, parser, cap)
        except Exception as e:
            print(f"  FAILED: {e}\n  -> continuing without {label}")
            return []
        print(f"  {len(recs)} questions kept")
        return recs

    musique = load("MuSiQue train", a.musique_file, "musique_train", parse_musique, a.n_musique)
    twowiki = load("2WikiMultihopQA train", a.twowiki_file, "2wiki_train",
                   lambda ex: parse_context_style(ex, "2wiki"), a.n_2wiki)
    hotpot = load("HotpotQA train", a.hotpot_file, "hotpot_train",
                  lambda ex: parse_context_style(ex, "hotpotqa"), a.n_hotpot)
    dev = load("MuSiQue validation (dev)", a.musique_dev_file, "musique_dev", parse_musique, a.n_dev, shuffle=False)
    if not dev and len(musique) > 2000:
        print("  no MuSiQue validation set: holding out 1,500 MuSiQue training questions as dev instead")
        dev, musique = musique[:1500], musique[1500:]
    train = musique + twowiki + hotpot
    if not train or not dev:
        sys.exit("No training or dev data. See the instructions at the top of this file for local files.")

    train = [trim_distractors(r, a.max_distractors, rng) for r in train]
    rng.shuffle(train)
    for name, recs in (("train", train), ("dev", dev)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        name: {"n": len(recs), "by_dataset": Counter(r["dataset"] for r in recs),
               "n_gold": Counter(len(r["gold"]) for r in recs),
               "paragraphs_per_q": sum(len(r["paragraphs"]) for r in recs) / max(1, len(recs))}
        for name, recs in (("train", train), ("dev", dev))
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
