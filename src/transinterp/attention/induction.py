"""Detectors for well-characterized attention head behaviors.

These functions implement descriptive tests from the circuits literature —
for example, the induction-head pattern, where a head attends to the token
that previously followed the current token, which is associated with
in-context copying behavior. Each function reports a continuous score, not a
classification. Callers should pick thresholds appropriate to their model
and confirm any candidate head with the caller-supplied intervention
utilities in ``transinterp.attention.circuits`` before treating it as a
circuit component.
"""

from __future__ import annotations

import torch


def induction_head_score(attention: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Score how strongly each head attends to induction targets.

    For a query position whose token also appeared earlier in the sequence,
    the induction target is the token that immediately followed that earlier
    occurrence. This scores the attention mass each head places on that
    target, averaged over query positions that have at least one earlier
    repeat. Sequences with no repeated tokens will produce a score of zero
    for every head.

    Args:
        attention: Self-attention weights shaped
            ``(batch, heads, tokens, tokens)``.
        token_ids: Integer token ids shaped ``(batch, tokens)``.

    Returns:
        Tensor shaped ``(batch, heads)``.
    """
    if attention.ndim != 4:
        raise ValueError("attention must be shaped (batch, heads, tokens, tokens)")
    batch, heads, n_query, n_source = attention.shape
    if n_query != n_source:
        raise ValueError(
            "induction_head_score expects self-attention (query_tokens == source_tokens)"
        )
    if token_ids.shape != (batch, n_source):
        raise ValueError("token_ids must be shaped (batch, tokens)")

    scores = torch.zeros(batch, heads, dtype=attention.dtype)
    counts = torch.zeros(batch)
    for b in range(batch):
        ids = token_ids[b]
        for q in range(1, n_source):
            earlier_matches = (ids[:q] == ids[q]).nonzero(as_tuple=True)[0]
            targets = earlier_matches + 1
            targets = targets[targets < n_source]
            if targets.numel() == 0:
                continue
            scores[b] += attention[b, :, q, targets].sum(dim=-1)
            counts[b] += 1
    return scores / counts.clamp_min(1).unsqueeze(-1)


def head_pattern_similarity(attention: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between heads' flattened attention patterns.

    Args:
        attention: Attention weights shaped
            ``(batch, heads, query_tokens, source_tokens)``.

    Returns:
        A ``(heads, heads)`` similarity matrix averaged over the batch, with
        1.0 on the diagonal. High similarity is a hint that two heads may
        play a redundant or complementary role; it is not evidence of a
        shared circuit on its own and should be paired with an intervention.
    """
    if attention.ndim != 4:
        raise ValueError("attention must be shaped (batch, heads, query_tokens, source_tokens)")
    batch, heads, n_query, n_source = attention.shape
    flat = attention.reshape(batch, heads, n_query * n_source)
    normalized = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = torch.einsum("bhi,bgi->bhg", normalized, normalized)
    return similarity.mean(dim=0)
