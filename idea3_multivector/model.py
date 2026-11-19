"""Query-only multi-vector retrieval over a frozen BGE passage space."""

from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F

from common.bge import BGEEncoder
from idea2_arrows.model import PooledQueryHeads, contrastive_nll


SYSTEMS = ('pooled control', 'token heads')
ASSIGNMENTS = {m: torch.tensor(list(permutations(range(4), m))) for m in (2, 3, 4)}


class QuestionEncoder(BGEEncoder):
    @torch.no_grad()
    def features(self, questions):
        """Return normalized CLS plus contextual tokens and their question-only mask."""
        texts = [self.format_query(q) for q in questions]
        starts = torch.tensor([len(t) - len(q.strip()) for t, q in zip(texts, questions)])
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                             return_offsets_mapping=True, return_special_tokens_mask=True, return_tensors='pt')
        offsets = enc.pop('offset_mapping')
        special = enc.pop('special_tokens_mask').bool()
        enc.pop('token_type_ids', None)
        valid = (offsets[..., 1] > starts[:, None]) & ~special & enc['attention_mask'].bool()
        if not valid.any(dim=1).all():
            raise ValueError('Every query needs a question token after the instruction')
        hidden = self.model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state.float()
        return F.normalize(hidden[:, 0], dim=-1), hidden, valid.to(self.device)


class TokenQueryHeads(nn.Module):
    def __init__(self, d=1024, rank=64):
        super().__init__()
        self.attention = nn.Parameter(torch.empty(4, d))
        nn.init.normal_(self.attention, std=d ** -0.5)
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden, mask):
        logits = torch.einsum('btd,hd->bht', hidden, self.attention)
        weights = logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(dim=-1)
        pooled = weights @ hidden
        return F.normalize(pooled + self.up(self.down(pooled)), dim=-1)


def make_models(d, pooled_rank, token_rank, device):
    return dict(zip(SYSTEMS, (PooledQueryHeads(d=d, r=pooled_rank).to(device),
                              TokenQueryHeads(d=d, rank=token_rank).to(device))))


def query_vectors(models, features):
    cls, hidden, mask = features
    return {SYSTEMS[0]: models[SYSTEMS[0]](cls), SYSTEMS[1]: models[SYSTEMS[1]](hidden, mask)}


@torch.no_grad()
def best_subset_assignment(similarity, gold_count):
    """Map four heads to gold indices, using -1 for unmatched heads."""
    assignments = torch.full(similarity.shape[:2], -1, device=similarity.device, dtype=torch.long)
    for m, candidates in ASSIGNMENTS.items():
        rows = (gold_count == m).nonzero(as_tuple=True)[0]
        candidates = candidates.to(similarity.device)
        gold = torch.arange(m, device=similarity.device)
        scores = similarity[rows][:, candidates, gold].sum(dim=-1)
        best = candidates[scores.argmax(dim=1)]
        assignments[rows[:, None], best] = gold
    return assignments


def matched_loss(vectors, batch, temperature=0.05):
    similarity = vectors.detach() @ batch['G'].transpose(1, 2)
    target = best_subset_assignment(similarity, batch['m'])
    assigned = target >= 0
    nll = contrastive_nll(vectors, batch, target.clamp_min(0), assigned, t=temperature)
    return nll[assigned].mean()
