import torch

from transinterp.attention.induction import head_pattern_similarity, induction_head_score


def test_induction_head_score_rewards_the_induction_target():
    token_ids = torch.tensor([[1, 2, 3, 1]])
    attention = torch.zeros(1, 2, 4, 4)
    # Only query position 3 (the repeated token) has an earlier occurrence,
    # so only that row counts toward the score. Its induction target is
    # position 1 (the token that followed the first "1").
    attention[0, 0, 3, 1] = 1.0
    attention[0, 1, 3, :] = 0.25

    scores = induction_head_score(attention, token_ids)

    assert scores.shape == (1, 2)
    assert scores[0, 0].item() == 1.0
    assert scores[0, 1].item() == 0.25


def test_induction_head_score_requires_self_attention_shape():
    token_ids = torch.tensor([[1, 2, 3]])
    attention = torch.rand(1, 2, 3, 5)
    try:
        induction_head_score(attention, token_ids)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for query/source length mismatch")


def test_head_pattern_similarity_is_one_for_identical_heads():
    pattern = torch.rand(1, 1, 3, 3)
    attention = pattern.repeat(1, 2, 1, 1)
    similarity = head_pattern_similarity(attention)
    assert similarity.shape == (2, 2)
    assert torch.allclose(similarity, torch.ones(2, 2), atol=1e-5)


def test_head_pattern_similarity_diagonal_is_one_for_distinct_heads():
    torch.manual_seed(1)
    attention = torch.rand(2, 3, 4, 4)
    similarity = head_pattern_similarity(attention)
    assert torch.allclose(torch.diagonal(similarity), torch.ones(3), atol=1e-5)
    assert bool((similarity <= 1.0 + 1e-5).all()) and bool((similarity >= -1.0 - 1e-5).all())
