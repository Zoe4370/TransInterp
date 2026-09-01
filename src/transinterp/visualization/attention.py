"""Static visualization helpers for attention tensors."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_attention_heatmap(
    attention: torch.Tensor,
    *,
    query_tokens: list[str] | None = None,
    source_tokens: list[str] | None = None,
    output: str | Path | None = None,
    title: str = "Attention",
):
    """Plot a single head's attention matrix as a heatmap.

    Args:
        attention: Weights shaped ``(query_tokens, source_tokens)`` for one
            layer and head. Index a full ``(heads, query, source)`` or
            ``(batch, heads, query, source)`` tensor before calling this
            function so each figure documents exactly what it shows.
        query_tokens: Optional row labels.
        source_tokens: Optional column labels.
        output: Optional path to save the figure.
        title: Figure title.
    """
    if attention.ndim != 2:
        raise ValueError(
            "attention must be shaped (query_tokens, source_tokens); index a single head first"
        )
    values = attention.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(0.5 * values.shape[1] + 3, 0.5 * values.shape[0] + 3))
    vmax = float(values.max()) if values.size else 1.0
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=vmax or 1.0)
    if source_tokens is not None:
        ax.set_xticks(range(len(source_tokens)))
        ax.set_xticklabels(source_tokens, rotation=90)
    if query_tokens is not None:
        ax.set_yticks(range(len(query_tokens)))
        ax.set_yticklabels(query_tokens)
    ax.set_xlabel("Source token")
    ax.set_ylabel("Query token")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig, ax
