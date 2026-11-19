"""Train the arrow adapter + count head on precomputed BGE vectors (BGE stays frozen).

Validation = MuSiQue dev: for each question, rank its own ~20 candidate paragraphs
and check whether ALL gold paragraphs land in the top K (complete@K), comparing
the single BGE vector with the arrows (soft slots from the count head).

Usage:  python idea2_arrows/train.py                 (GPU if available; also fine on CPU, it's a 135k-parameter model)
        python idea2_arrows/train.py --beta 0.5 --epochs 8
Saves cache/arrows/adapter.pt (+ adapter.json with the settings and dev scores).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import paths  # noqa: E402
from common.selection import largest_remainder  # noqa: E402
from idea2_arrows.model import ArrowAdapter, arrow_loss, slot_weights  # noqa: E402


def load_split(split, P, device, max_d=None):
    d = paths.ARROWS_DATA
    q = torch.from_numpy(np.load(os.path.join(d, f"{split}_q.npy")).astype(np.float32))
    with open(os.path.join(d, f"{split}_index.json")) as f:
        idx = json.load(f)
    N = len(idx)
    M = max(len(r["para_ids"]) - len(r["gold"]) for r in idx)
    if max_d is not None:
        M = min(M, max_d)
    gpid = torch.full((N, 4), -1, dtype=torch.long)
    dpid = torch.full((N, M), -1, dtype=torch.long)
    for i, r in enumerate(idx):
        g = [r["para_ids"][k] for k in r["gold"]][:4]
        gpid[i, :len(g)] = torch.tensor(g)
        ds = [p for k, p in enumerate(r["para_ids"]) if k not in set(r["gold"])][:M]
        if ds:
            dpid[i, :len(ds)] = torch.tensor(ds)
    m = (gpid >= 0).sum(1)
    return {"q": q.to(device), "gpid": gpid.to(device), "dpid": dpid.to(device), "m": m.to(device), "index": idx}


def make_batch(S, rows, P):
    gpid, dpid = S["gpid"][rows], S["dpid"][rows]
    return {
        "G": P[gpid.clamp(min=0)], "gmask": gpid >= 0, "gpid": gpid,
        "D": P[dpid.clamp(min=0)], "dmask": dpid >= 0, "dpid": dpid,
        "m": S["m"][rows],
    }


def deal(lists, quotas, k):
    picked, pos, used = [], [0] * len(lists), [0] * len(lists)

    def give(i):
        while pos[i] < len(lists[i]):
            c = lists[i][pos[i]]
            pos[i] += 1
            if c not in picked:
                picked.append(c)
                return True
        return False

    for limit in (True, False):
        while len(picked) < k:
            prog = False
            for i in range(len(lists)):
                if len(picked) >= k:
                    break
                if (not limit or used[i] < quotas[i]) and give(i):
                    used[i] += 1
                    prog = True
            if not prog:
                break
    return picked


@torch.no_grad()
def evaluate(model, S, P, ks=(2, 4, 6)):
    """complete@K within each question's own candidates: single vector vs arrows."""
    model.eval()
    V, logits = model(S["q"])
    w = slot_weights(logits)
    res = {f"single_complete@{k}": [] for k in ks}
    res.update({f"arrows_complete@{k}": [] for k in ks})
    res["count_acc"] = ((logits.argmax(-1) + 2) == S["m"]).float().mean().item()
    for i, r in enumerate(S["index"]):
        cand = torch.tensor(r["para_ids"], device=P.device)
        gold = {r["para_ids"][g] for g in r["gold"]}
        E = P[cand]
        single = cand[(E @ S["q"][i]).argsort(descending=True)].tolist()
        per_arrow = [cand[(E @ V[i, j]).argsort(descending=True)].tolist() for j in range(V.shape[1])]
        for k in ks:
            res[f"single_complete@{k}"].append(gold <= set(single[:k]))
            quotas = largest_remainder(w[i].tolist(), k)
            res[f"arrows_complete@{k}"].append(gold <= set(deal(per_arrow, quotas, k)))
    model.train()
    return {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--J", type=int, default=4)
    ap.add_argument("--t", type=float, default=0.05, help="temperature")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the other gold paragraphs as negatives")
    ap.add_argument("--lam_extra", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select_k", type=int, default=4, help="checkpoint chosen by dev arrows_complete@K")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(paths.CACHE, "arrows", "adapter.pt"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    P = torch.from_numpy(np.load(os.path.join(paths.ARROWS_DATA, "paragraphs.npy")).astype(np.float32)).to(device)
    train = load_split("train", P, device)
    dev = load_split("dev", P, device)
    d = P.shape[1]
    counts = torch.bincount((train["m"] - 2).clamp(0, 2).cpu(), minlength=3).float()
    class_w = (counts.sum() / (3 * counts.clamp(min=1))).to(device)
    print(f"train {len(train['index'])} questions, dev {len(dev['index'])}; gold counts 2/3/4 = {counts.tolist()}; "
          f"class weights {class_w.tolist()}")

    model = ArrowAdapter(d=d, J=a.J, r=a.rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    base = evaluate(model, dev, P)
    print("dev before training:", json.dumps({k: round(v, 4) for k, v in base.items()}))
    best, best_score, history = None, -1, []
    N = len(train["index"])
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(N, device=device)
        t0, logs = time.time(), []
        for s in range(0, N, a.batch_size):
            rows = perm[s:s + a.batch_size]
            batch = make_batch(train, rows, P)
            V, logits = model(train["q"][rows])
            loss, info = arrow_loss(V, logits, batch, t=a.t, beta=a.beta, lam_extra=a.lam_extra, class_weights=class_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            logs.append(info)
        tr = {k: float(np.mean([x[k] for x in logs])) for k in logs[0]}
        ev = evaluate(model, dev, P)
        history.append({"epoch": ep, "train": tr, "dev": ev})
        print(f"epoch {ep} ({time.time() - t0:.0f}s) train {json.dumps({k: round(v, 4) for k, v in tr.items()})}")
        print(f"         dev {json.dumps({k: round(v, 4) for k, v in ev.items()})}")
        score = ev[f"arrows_complete@{a.select_k}"]
        if score > best_score:
            best_score, best = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(best, a.out)
    with open(a.out.replace(".pt", ".json"), "w") as f:
        json.dump({"config": model.config(), "args": vars(a), "dev_before": base, "history": history,
                   "best_dev_arrows_complete": best_score}, f, indent=2)
    print(f"saved {a.out} (best dev arrows_complete@{a.select_k} = {best_score:.4f}, "
          f"single vector = {base[f'single_complete@{a.select_k}']:.4f})")


if __name__ == "__main__":
    main()
