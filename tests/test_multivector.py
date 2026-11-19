"""Regression checks for matching, token pooling, and the shared article cap."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from idea3_multivector.evaluate import cap_ranking, fused_scores
from idea3_multivector.model import TokenQueryHeads, best_subset_assignment, matched_loss
from idea3_multivector.adapt_data import adaptation_split, gold_keys, select_negatives


def test_matching_can_use_the_last_two_heads():
    scores = torch.tensor([[[0.1, 0.2, 0, 0], [0.15, 0.15, 0, 0],
                            [0.3, 0.95, 0, 0], [0.9, 0.3, 0, 0]]])
    assert best_subset_assignment(scores, torch.tensor([2])).tolist() == [[-1, -1, 1, 0]]


def test_unmatched_heads_have_no_retrieval_gradient():
    vectors = torch.tensor([[[-1., 0.], [0., -1.], [1., 0.], [0., 1.]]], requires_grad=True)
    batch = {'G': torch.tensor([[[1., 0.], [0., 1.], [0., 0.], [0., 0.]]]),
             'D': torch.tensor([[[-0.6, -0.8]]]), 'gpid': torch.tensor([[0, 1, -1, -1]]),
             'gmask': torch.tensor([[True, True, False, False]]), 'dpid': torch.tensor([[2]]),
             'dmask': torch.tensor([[True]]), 'm': torch.tensor([2])}
    matched_loss(vectors, batch, temperature=0.5).backward()
    assert vectors.grad[0, :2].abs().sum() == 0
    assert vectors.grad[0, 2:].abs().sum() > 0


def test_gold_document_versions_are_not_negatives():
    vectors = torch.tensor([[[-1., 0.], [0., -1.], [1., 0.], [0., 1.]]])
    batch = {'G': torch.tensor([[[1., 0.], [0., 1.], [0., 0.], [0., 0.]]]),
             'D': torch.tensor([[[1., 0.]]]), 'gpid': torch.tensor([[0, 1, -1, -1]]),
             'gmask': torch.tensor([[True, True, False, False]]), 'dpid': torch.tensor([[2]]),
             'dmask': torch.tensor([[True]]), 'm': torch.tensor([2])}
    original = matched_loss(vectors, batch)
    batch['gold_documents'] = torch.tensor([[10, 11, -1, -1]])
    batch['distractor_documents'] = torch.tensor([[10]])
    assert matched_loss(vectors, batch) < original


def test_pooling_ignores_masked_tokens():
    torch.manual_seed(0)
    model = TokenQueryHeads(d=8, rank=2)
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor([[False, True, True, False, False], [False, True, True, True, False]])
    changed = hidden.clone()
    changed[~mask] = 10_000
    vectors = model(hidden, mask)
    assert vectors.shape == (2, 4, 8)
    torch.testing.assert_close(vectors, model(changed, mask))
    torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(2, 4))


def test_max_fusion_and_article_cap():
    class TinyIndex:
        def scores(self, vectors):
            return np.einsum('nd,dc->nc', vectors, np.array([[1., 0., -1.], [0., 1., 0.]]))

    np.testing.assert_array_equal(fused_scores(TinyIndex(), np.array([[[1., 0.], [0., 1.]]])), [[1, 1, 0]])
    chunks = [{'article_id': i} for i in [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]]
    assert cap_ranking([0, 1, 2], -np.arange(11), chunks, 10) == [0, 1, *range(3, 11)]


def test_local_holdout_and_negative_exclusion():
    from common.data import load_queries, split_queries

    queries = load_queries()
    train, validation = adaptation_split(queries)
    _, original_test = split_queries(queries)
    assert not (set(train) | set(validation)) & set(original_test)
    train_gold = {key for q in train for key in gold_keys(queries[q])}
    val_gold = {key for q in validation for key in gold_keys(queries[q])}
    assert train and validation and not train_gold & val_gold
    assert select_negatives(range(5), ['heldout', 'allowed', 'allowed', 'gold', 'other'],
                            {'heldout', 'gold'}, 2) == [1, 4]


if __name__ == '__main__':
    for test in (test_matching_can_use_the_last_two_heads, test_unmatched_heads_have_no_retrieval_gradient,
                 test_gold_document_versions_are_not_negatives, test_pooling_ignores_masked_tokens,
                 test_max_fusion_and_article_cap, test_local_holdout_and_negative_exclusion):
        test()
        print(f'ok {test.__name__}')
