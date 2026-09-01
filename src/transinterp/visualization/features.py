"""Visualization helpers for latent feature spaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from transinterp.extraction.features import FeatureBasis


def plot_feature_space(
    hidden: torch.Tensor,
    basis: FeatureBasis,
    *,
    color_by: list[float] | list[str] | None = None,
    labels: list[str] | None = None,
    output: str | Path | None = None,
    title: str = "Feature space (first two components)",
):
    """Scatter hidden states projected onto a fitted basis's first two axes.

    This is a two-dimensional summary of a higher-dimensional space; points
    that are close here are not guaranteed to be semantically close along
    the axes that were discarded. Use it for exploration, not as a
    stand-alone claim about representational structure.

    Args:
        hidden: Hidden states shaped ``(n_samples, d_model)``.
        basis: A ``FeatureBasis`` fit with at least two components.
        color_by: Optional per-point numeric values or category strings.
        labels: Optional per-point text annotations (e.g. token strings).
        output: Optional path to save the figure.
        title: Figure title.
    """
    if basis.components.shape[0] < 2:
        raise ValueError("basis must have at least two components to plot")
    scores = basis.project(hidden)[:, :2].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    numeric_color = color_by is not None and not isinstance(color_by[0], str)
    scatter = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=color_by if color_by is not None else None,
        cmap="viridis" if numeric_color else None,
        s=18,
        alpha=0.85,
    )
    if labels is not None:
        for (x, y), label in zip(scores, labels, strict=False):
            ax.annotate(label, (x, y), fontsize=7, alpha=0.7)
    if numeric_color:
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig, ax
