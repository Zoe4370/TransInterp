"""Metrics for dissecting attention patterns."""

from __future__ import annotations

import torch


def attention_entropy(attention: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return Shannon entropy over the source-token axis.

    Args:
        attention: Tensor shaped ``(..., query_tokens, source_tokens)``.
        eps: Numerical floor applied before taking logarithms.
    """
    probabilities = attention.clamp_min(eps)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def topk_edges(attention: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k source indices and weights for every query position."""
    if attention.ndim < 2:
        raise ValueError("attention must have at least query and source dimensions")
    k = min(k, attention.shape[-1])
    return torch.topk(attention, k=k, dim=-1)


def attention_mass(attention: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Measure attention mass assigned to a boolean source-token mask."""
    if token_mask is None:
        raise ValueError("token_mask is required to define the target token set")
    if token_mask.shape[-1] != attention.shape[-1]:
        raise ValueError("token_mask and attention must agree on source-token dimension")
    return (attention * token_mask.to(dtype=attention.dtype)).sum(dim=-1)
