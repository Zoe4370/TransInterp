"""Scalar behavior metrics for causal intervention experiments.

A patching experiment needs a single number summarizing model behavior, and
the choice of number materially changes the conclusion. Logit difference is
sensitive to a specific contrast the researcher names; KL divergence measures
whether the whole output distribution moved. A component can look important
under one and irrelevant under the other, so both are provided and neither is
treated as the default answer.

All functions take raw logits shaped ``(batch, tokens, vocab)`` or
``(batch, vocab)`` and reduce to a scalar per batch element.
"""

from __future__ import annotations

import torch

__all__ = [
    "kl_divergence",
    "logit_difference",
    "normalized_effect",
    "target_logprob",
    "target_probability",
]


def _select_position(logits: torch.Tensor, position: int) -> torch.Tensor:
    """Reduce ``(batch, tokens, vocab)`` to ``(batch, vocab)`` at one position."""
    if logits.ndim == 3:
        return logits[:, position, :]
    if logits.ndim == 2:
        return logits
    raise ValueError(
        "expected logits shaped (batch, tokens, vocab) or (batch, vocab), "
        f"got {tuple(logits.shape)}"
    )


def logit_difference(
    logits: torch.Tensor,
    correct_token: int,
    incorrect_token: int,
    *,
    position: int = -1,
) -> torch.Tensor:
    """Return ``logit[correct] - logit[incorrect]`` at one position.

    The workhorse metric for circuit analysis. Because it is a difference of
    logits at the same position, the softmax normalizer cancels, which makes
    it less sensitive to overall confidence shifts than a raw probability.
    """
    selected = _select_position(logits, position)
    return selected[:, correct_token] - selected[:, incorrect_token]


def target_probability(
    logits: torch.Tensor, target_token: int, *, position: int = -1
) -> torch.Tensor:
    """Return the softmax probability of a single token."""
    return _select_position(logits, position).softmax(dim=-1)[:, target_token]


def target_logprob(
    logits: torch.Tensor, target_token: int, *, position: int = -1
) -> torch.Tensor:
    """Return the log-probability of a single token, computed stably."""
    return _select_position(logits, position).log_softmax(dim=-1)[:, target_token]


def kl_divergence(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    position: int = -1,
) -> torch.Tensor:
    """Return ``KL(reference || logits)`` in nats at one position.

    Useful when no single token contrast captures the behavior of interest:
    it asks whether the intervention moved the output distribution at all,
    rather than whether it moved one particular contrast.
    """
    current = _select_position(logits, position).log_softmax(dim=-1)
    reference = _select_position(reference_logits, position).log_softmax(dim=-1)
    return (reference.exp() * (reference - current)).sum(dim=-1)


def normalized_effect(
    patched: float | torch.Tensor,
    clean: float | torch.Tensor,
    corrupted: float | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """Scale a patched metric onto the clean/corrupted interval.

    Returns roughly 0 when patching changed nothing relative to the corrupted
    run and roughly 1 when it fully restored clean behavior. This is the
    standard normalization for reporting patching results, and it is what
    makes effects comparable across prompts with different baseline logit
    gaps.

    Values outside ``[0, 1]`` are returned unclipped and are informative: a
    result above 1 means patching overshot clean behavior, and a negative
    value means it pushed further from it. Clipping would hide both.

    When the clean and corrupted baselines are nearly equal the denominator
    collapses and the ratio is meaningless, so ``0.0`` is returned rather than
    a large spurious number.
    """
    patched_value = float(patched)
    clean_value = float(clean)
    corrupted_value = float(corrupted)
    denominator = clean_value - corrupted_value
    if abs(denominator) < eps:
        return 0.0
    return (patched_value - corrupted_value) / denominator
