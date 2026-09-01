"""Latent feature extraction over captured hidden-state tensors.

``transinterp.extraction.capture`` records raw activations. This module turns
those activations into lower-dimensional feature bases so a researcher can
ask which directions are active for a given input. Two strategies are
provided.

``fit_pca`` finds a dense, orthonormal basis that explains variance in a
batch of hidden states. It is fast, deterministic, and a reasonable first
pass for exploring a representation.

``SparseAutoencoder`` is a small, trainable dictionary-learning model whose
columns can be treated as candidate "features" in the spirit of prior
dictionary-learning work on language model activations. It is overcomplete
and sparse rather than orthonormal, which tends to produce more
monosemantic-looking directions at the cost of requiring training.

Neither method labels a discovered direction as semantically meaningful.
That judgment requires inspecting a direction's top-activating examples
(``top_activating_examples``) and testing it against counterexamples and,
ideally, an intervention (see ``transinterp.attention.circuits``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FeatureBasis:
    """A dense, orthonormal linear basis fit to a batch of hidden states."""

    components: torch.Tensor  # (n_components, d_model)
    mean: torch.Tensor  # (d_model,)
    explained_variance_ratio: torch.Tensor  # (n_components,)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project ``(..., d_model)`` hidden states onto the fitted basis."""
        if hidden.shape[-1] != self.mean.shape[-1]:
            raise ValueError("hidden's last dimension must match the fitted d_model")
        centered = hidden - self.mean
        return centered @ self.components.T

    def reconstruct(self, scores: torch.Tensor) -> torch.Tensor:
        """Approximately invert ``project``, for reconstruction-error checks."""
        return scores @ self.components + self.mean


def fit_pca(hidden: torch.Tensor, n_components: int) -> FeatureBasis:
    """Fit a PCA basis over ``(n_samples, d_model)`` hidden states.

    Uses a full SVD; callers should subsample large activation sets before
    fitting if memory is a concern. This is a descriptive decomposition, not
    a sparse dictionary: components are dense and mutually orthogonal, so a
    single semantic concept may be smeared across several components.
    """
    if hidden.ndim != 2:
        raise ValueError("hidden must be shaped (n_samples, d_model)")
    if not (1 <= n_components <= hidden.shape[-1]):
        raise ValueError("n_components must be between 1 and d_model")

    mean = hidden.mean(dim=0)
    centered = hidden - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    variance = singular_values.pow(2) / max(hidden.shape[0] - 1, 1)
    ratio = variance[:n_components] / variance.sum().clamp_min(1e-12)
    return FeatureBasis(components=components, mean=mean, explained_variance_ratio=ratio)


class SparseAutoencoder(nn.Module):
    """A minimal sparse autoencoder over hidden-state activations.

    This follows the standard dictionary-learning recipe used in prior work
    on decomposing language model activations into sparse, overcomplete
    feature directions: a linear encoder with a non-negative activation and
    an L1 sparsity penalty, and a linear decoder that reconstructs the input.
    The class trains on activations the caller has already captured; it does
    not hook into a model itself, keeping module boundaries the same as the
    rest of the package.
    """

    def __init__(self, d_model: int, n_features: int, *, tied_weights: bool = True) -> None:
        super().__init__()
        if n_features < d_model:
            raise ValueError("n_features should be >= d_model for an overcomplete dictionary")
        self.d_model = d_model
        self.n_features = n_features
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.tied_weights = tied_weights
        if tied_weights:
            self.decoder_weight = None
        else:
            self.decoder_weight = nn.Parameter(torch.empty(d_model, n_features))
            nn.init.kaiming_uniform_(self.decoder_weight)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))

    def _decoder_weight(self) -> torch.Tensor:
        return self.encoder.weight.T if self.decoder_weight is None else self.decoder_weight

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return non-negative feature activations shaped ``(..., n_features)``."""
        return torch.relu(self.encoder(hidden))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct hidden states from feature activations."""
        return codes @ self._decoder_weight().T + self.decoder_bias

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.encode(hidden)
        return self.decode(codes), codes

    def loss(
        self, hidden: torch.Tensor, *, l1_coefficient: float = 1e-3
    ) -> dict[str, torch.Tensor]:
        """Return reconstruction, sparsity, and total loss terms for one batch."""
        reconstruction, codes = self(hidden)
        reconstruction_loss = (reconstruction - hidden).pow(2).sum(dim=-1).mean()
        sparsity_loss = codes.abs().sum(dim=-1).mean()
        total = reconstruction_loss + l1_coefficient * sparsity_loss
        return {
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
            "total_loss": total,
            "l0_active": (codes > 0).float().sum(dim=-1).mean(),
        }


def top_activating_examples(
    scores: torch.Tensor,
    tokens: Sequence[str],
    *,
    feature_index: int,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-``k`` tokens by score for a single feature index.

    ``scores`` is shaped ``(n_tokens, n_features)`` and ``tokens`` must have
    matching length. This is a lookup helper for qualitative inspection, not
    a labeling procedure: a coherent top-k list is evidence, not proof, that
    a feature corresponds to one human concept, and it says nothing about
    what the feature does on inputs outside this sample.
    """
    if scores.shape[0] != len(tokens):
        raise ValueError("scores and tokens must have the same length")
    if not (0 <= feature_index < scores.shape[-1]):
        raise ValueError("feature_index is out of range")
    column = scores[:, feature_index]
    top_values, top_indices = torch.topk(column, k=min(k, column.shape[0]))
    pairs = zip(top_indices.tolist(), top_values.tolist(), strict=True)
    return [(tokens[index], float(value)) for index, value in pairs]
