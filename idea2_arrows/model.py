"""Arrow adapter, count head and the training loss.

  v_j = normalise(e + A_j B_j^T e),   A_j, B_j in R^{1024 x 16},  j = 1..4
  A_j starts at 0 (every arrow starts equal to the plain BGE vector e);
  B_j starts small and random, different per arrow (breaks the tie, gives A_j a gradient).
  count head: softmax(W e + b) over "2, 3 or 4 pieces".

Loss for a question with m gold paragraphs (m = TRUE count from the labels):
  * pair arrows 1..m one-to-one with the m gold paragraphs, best total similarity (try all m! orders)
  * arrow j: softmax over candidates, target = its paired gold paragraph
      other gold paragraphs of the same question  -> weight beta in the denominator (beta=1: full push apart)
      the question's distractors                 -> weight 1
      every paragraph of the other questions in the batch -> weight 1 (copies of this question's gold are skipped)
  * arrows m+1..4: small weight lam_extra, pulled to the closest gold paragraph (duplicates allowed),
    so if the count head over-predicts at test time they fetch a copy of a real piece instead of noise
  * count head: class-weighted cross-entropy on m
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

PERMS = {m: torch.tensor(list(itertools.permutations(range(m)))) for m in (1, 2, 3, 4)}


class PooledQueryHeads(nn.Module):
    def __init__(self, d: int = 1024, J: int = 4, r: int = 16, init_std: float = 0.02):
        super().__init__()
        self.J, self.d, self.r = J, d, r
        self.A = nn.Parameter(torch.zeros(J, d, r))
        self.B = nn.Parameter(torch.randn(J, d, r) * init_std)

    def forward(self, e: torch.Tensor):
        z = torch.einsum("nd,jdr->njr", e, self.B)
        v = e[:, None, :] + torch.einsum("njr,jdr->njd", z, self.A)
        return F.normalize(v, dim=-1)

    def config(self):
        return {"d": self.d, "J": self.J, "r": self.r}


class ArrowAdapter(PooledQueryHeads):
    def __init__(self, d: int = 1024, J: int = 4, r: int = 16, init_std: float = 0.02, n_classes: int = 3):
        super().__init__(d=d, J=J, r=r, init_std=init_std)
        self.count = nn.Linear(d, n_classes)

    def forward(self, e: torch.Tensor):
        return super().forward(e), self.count(e)


def best_pairing(P: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """P: [N, J, 4] arrow-to-gold similarity (detached). Returns perm [N, 4]: arrow j -> gold index (j < m)."""
    N = P.shape[0]
    out = torch.full((N, 4), -1, dtype=torch.long)
    Pc = P.detach().float().cpu()
    mc = m.cpu()
    for mm in (1, 2, 3, 4):
        sel = (mc == mm).nonzero(as_tuple=True)[0]
        if len(sel) == 0:
            continue
        perms = PERMS[mm]  # [p, mm]
        sub = Pc[sel][:, :mm, :mm]  # [n, mm(arrows), mm(golds)]
        # total similarity of each ordering: sum_j sub[:, j, perm[j]]
        tot = sub[:, torch.arange(mm)[None, :], perms].sum(-1)  # [n, p]
        best = perms[tot.argmax(1)]  # [n, mm]
        out[sel, :mm] = best
    return out.to(P.device)


def contrastive_nll(V, batch, target_k, assigned, t: float = 0.05, beta: float = 1.0):
    """Per-head negative log likelihood for the supplied gold assignments."""
    G, gmask, gpid = batch["G"], batch["gmask"], batch["gpid"]
    D, dmask, dpid = batch["D"], batch["dmask"], batch["dpid"]
    N, J, d = V.shape
    M = D.shape[1]
    dev = V.device

    C = torch.cat([G.reshape(N * 4, d), D.reshape(N * M, d)], 0)  # all candidates in the batch
    cvalid = torch.cat([gmask.reshape(-1), dmask.reshape(-1)])
    cpid = torch.cat([gpid.reshape(-1), dpid.reshape(-1)])
    S = torch.einsum("njd,td->njt", V, C) / t  # [N, J, T]

    # base log-weights per question: 0 = negative, -inf = excluded
    neg_inf = torch.finfo(S.dtype).min
    LW = torch.zeros(N, C.shape[0], device=dev)
    LW[:, ~cvalid] = neg_inf
    # Exclude copies of this question's gold. For chunked corpora the caller
    # can additionally group overlapping chunks by their source document.
    gold_group = batch.get('gold_documents', gpid)
    candidate_group = (torch.cat((gold_group.flatten(), batch['distractor_documents'].flatten()))
                       if 'gold_documents' in batch else cpid)
    same_as_my_gold = (candidate_group[None, :, None] == gold_group[:, None, :]) & gmask[:, None, :]
    LW[same_as_my_gold.any(-1)] = neg_inf
    LW = LW[:, None, :].expand(N, J, -1).clone()

    onehot = F.one_hot(target_k, 4).bool() & gmask[:, None, :]

    log_beta = torch.log(torch.tensor(beta, device=dev)) if beta > 0 else torch.tensor(neg_inf, device=dev)
    own = torch.full((N, J, 4), neg_inf, device=dev)
    own = torch.where(onehot, torch.zeros_like(own), own)
    other_gold = gmask[:, None, :] & ~onehot & assigned[:, :, None]
    own = torch.where(other_gold, log_beta.expand_as(own), own)
    pos_idx = (torch.arange(N, device=dev) * 4)[:, None] + torch.arange(4, device=dev)[None, :]  # [N, 4]
    LW[torch.arange(N, device=dev)[:, None, None], torch.arange(J, device=dev)[None, :, None], pos_idx[:, None, :]] = own

    tgt = (torch.arange(N, device=dev)[:, None] * 4 + target_k)  # [N, J] index into C
    pos_score = S.gather(2, tgt[..., None]).squeeze(-1)
    lse = torch.logsumexp(S + LW, dim=-1)
    return lse - pos_score  # [N, J]


def arrow_loss(V, count_logits, batch, t: float = 0.05, beta: float = 1.0, lam_extra: float = 0.1,
               class_weights=None):
    """Pair the first m heads and use spare heads as backups for Idea 2."""
    m = batch["m"]
    J = V.shape[1]
    dev = V.device
    P = torch.einsum("njd,nkd->njk", V, batch["G"]).masked_fill(~batch["gmask"][:, None, :], -1e4)
    perm = best_pairing(P, m)
    assigned = torch.arange(J, device=dev)[None, :] < m[:, None]
    target_k = torch.where(assigned, perm[:, :J].clamp(min=0), P.argmax(-1))
    nll = contrastive_nll(V, batch, target_k, assigned, t, beta)
    main = nll[assigned].mean()
    extra = nll[~assigned].mean() if (~assigned).any() and lam_extra > 0 else torch.zeros((), device=dev)
    cls = F.cross_entropy(count_logits, (m - 2).clamp(0, 2), weight=class_weights)
    loss = main + lam_extra * extra + cls
    with torch.no_grad():
        acc = ((count_logits.argmax(-1) + 2) == m).float().mean()
    return loss, {"main": main.item(), "extra": extra.item(), "count_ce": cls.item(), "count_acc": acc.item()}


def slot_weights(count_logits: torch.Tensor) -> torch.Tensor:
    """w1 = w2 = 1, w3 = P(n >= 3), w4 = P(n = 4)."""
    p = count_logits.softmax(-1)  # [N, 3] for n = 2, 3, 4
    one = torch.ones_like(p[:, 0])
    return torch.stack([one, one, p[:, 1] + p[:, 2], p[:, 2]], -1)
