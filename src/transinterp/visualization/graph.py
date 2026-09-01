"""Graph export for attention edges, for use in interactive viewers.

The graph is only a display of caller-supplied edges; it does not compute
causal importance. Combine it with
``transinterp.attention.circuits.rank_interventions`` before treating an
edge as more than a descriptive attention weight. Output is plain
dictionaries and lists so it serializes directly to JSON for external
viewers (for example, a small D3 or Plotly front end).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AttentionGraph:
    """A lightweight directed graph of token-to-token attention edges."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for external viewers."""
        return {"nodes": self.nodes, "edges": self.edges}


def build_attention_graph(
    attention: torch.Tensor,
    tokens: list[str],
    *,
    layer: int,
    head: int,
    k: int = 3,
    min_weight: float = 0.0,
) -> AttentionGraph:
    """Build a graph of the top-``k`` source tokens attended to by each query.

    Args:
        attention: Weights shaped ``(query_tokens, source_tokens)`` for one
            layer and head (index a full tensor before calling this).
        tokens: Token strings aligned with the source-token axis.
        layer: Layer index, stored as edge metadata only.
        head: Head index, stored as edge metadata only.
        k: Number of source tokens to keep per query position.
        min_weight: Drop edges below this attention weight.
    """
    if attention.ndim != 2:
        raise ValueError("attention must be shaped (query_tokens, source_tokens)")
    if attention.shape[1] != len(tokens):
        raise ValueError("tokens must align with the source-token axis")

    nodes = [{"id": i, "token": token} for i, token in enumerate(tokens)]
    edges: list[dict[str, Any]] = []
    top_k = min(k, attention.shape[-1])
    values, indices = torch.topk(attention, k=top_k, dim=-1)
    for query_index in range(attention.shape[0]):
        row_values = values[query_index].tolist()
        row_indices = indices[query_index].tolist()
        for weight, source_index in zip(row_values, row_indices, strict=True):
            if weight < min_weight:
                continue
            edges.append(
                {
                    "source": source_index,
                    "target": query_index,
                    "weight": weight,
                    "layer": layer,
                    "head": head,
                }
            )
    return AttentionGraph(nodes=nodes, edges=edges)
