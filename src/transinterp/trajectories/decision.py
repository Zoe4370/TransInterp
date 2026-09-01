"""Represent and summarize how a model's decision evolves through depth."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DecisionTrajectory:
    """Layer-indexed class scores for one example."""

    layer: list[int]
    probabilities: torch.Tensor
    token_ids: list[int] | None = None
    labels: list[str] | None = None

    @property
    def predicted_class(self) -> torch.Tensor:
        """Return the most likely class at each layer.

        A single-example trajectory is squeezed to ``(layers,)`` for convenient
        reporting; batched trajectories retain ``(layers, batch)``.
        """
        predicted = self.probabilities.argmax(dim=-1)
        return predicted[:, 0] if predicted.ndim == 2 and predicted.shape[1] == 1 else predicted

    def rank(self, class_index: int) -> torch.Tensor:
        """Return the one-based rank of a class at every layer.

        Shape mirrors the leading axes of ``probabilities``: ``(layers,)`` for
        a single example and ``(layers, batch)`` for a batch.

        The previous implementation collected match positions with
        ``nonzero``, which flattens. For a batch of three it returned six
        values in one dimension instead of a ``(layers, batch)`` grid — wrong
        but shaped plausibly enough to go unnoticed. Comparing against the
        sorted order elementwise keeps the leading axes intact.
        """
        if not 0 <= class_index < self.probabilities.shape[-1]:
            raise IndexError(
                f"class_index {class_index} is outside the class axis of size "
                f"{self.probabilities.shape[-1]}"
            )
        order = self.probabilities.argsort(dim=-1, descending=True)
        ranks = (order == class_index).float().argmax(dim=-1) + 1
        if ranks.ndim == 2 and ranks.shape[1] == 1:
            return ranks[:, 0]
        return ranks


def from_logits(logits_by_layer: dict[int, torch.Tensor]) -> DecisionTrajectory:
    """Build a trajectory from ``layer -> class logits`` mappings."""
    if not logits_by_layer:
        raise ValueError("logits_by_layer cannot be empty")
    layers = sorted(logits_by_layer)
    stacked = torch.stack([logits_by_layer[layer] for layer in layers])
    return DecisionTrajectory(layer=layers, probabilities=stacked.softmax(dim=-1))
