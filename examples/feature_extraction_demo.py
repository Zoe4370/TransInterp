"""Offline example of the latent feature extraction and induction-head utilities.

This script uses synthetic tensors so it runs without downloading a model,
mirroring ``examples/basic_usage.py``. Swap the synthetic hidden states for a
``HookCapture`` record and real token ids to use it on a live model.
"""

from __future__ import annotations

import torch

from transinterp.attention.induction import head_pattern_similarity, induction_head_score
from transinterp.extraction.features import SparseAutoencoder, fit_pca, top_activating_examples
from transinterp.visualization.graph import build_attention_graph


def main() -> None:
    torch.manual_seed(0)

    # --- Latent feature extraction -------------------------------------
    hidden = torch.randn(64, 16)  # (tokens, d_model)
    basis = fit_pca(hidden, n_components=4)
    print("PCA explained variance ratio:", basis.explained_variance_ratio.tolist())

    sae = SparseAutoencoder(d_model=16, n_features=32)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.05)
    for _ in range(100):
        optimizer.zero_grad()
        losses = sae.loss(hidden, l1_coefficient=1e-3)
        losses["total_loss"].backward()
        optimizer.step()
    print("SAE reconstruction loss after training:", round(losses["reconstruction_loss"].item(), 4))
    print("SAE mean active features per token:", round(losses["l0_active"].item(), 2))

    codes = sae.encode(hidden)
    tokens = [f"tok_{i}" for i in range(hidden.shape[0])]
    print("Top tokens for feature 0:", top_activating_examples(codes, tokens, feature_index=0, k=5))

    # --- Induction-head detection ---------------------------------------
    token_ids = torch.tensor([[1, 2, 3, 4, 2, 3]])  # "2, 3" repeats
    attention = torch.rand(1, 4, 6, 6)
    attention = attention / attention.sum(dim=-1, keepdim=True)
    scores = induction_head_score(attention, token_ids)
    print("Induction score per head:", scores.tolist())
    print("Head pattern similarity:\n", head_pattern_similarity(attention))

    # --- Attention graph export ------------------------------------------
    graph = build_attention_graph(
        attention[0, 0], [str(t) for t in token_ids[0].tolist()], layer=0, head=0, k=2
    )
    print("Graph nodes:", graph.nodes)
    print("Graph edges (first 3):", graph.edges[:3])


if __name__ == "__main__":
    main()
