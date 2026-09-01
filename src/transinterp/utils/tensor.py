"""Shape-safe tensor helpers used by analysis modules."""

from __future__ import annotations

import torch


def flatten_tokens(hidden: torch.Tensor) -> torch.Tensor:
    """Flatten ``(batch, tokens, features)`` into ``(batch*tokens, features)``."""
    if hidden.ndim != 3:
        raise ValueError("expected a tensor shaped (batch, tokens, features)")
    return hidden.reshape(-1, hidden.shape[-1])


def normalize_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize the final feature axis while preserving all leading axes."""
    if features.ndim == 0:
        raise ValueError("features must have at least one dimension")
    return features / features.norm(dim=-1, keepdim=True).clamp_min(eps)
