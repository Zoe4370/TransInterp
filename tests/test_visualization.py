import matplotlib

matplotlib.use("Agg")

import torch

from transinterp.extraction.features import fit_pca
from transinterp.trajectories.decision import from_logits
from transinterp.visualization.attention import plot_attention_heatmap
from transinterp.visualization.features import plot_feature_space
from transinterp.visualization.graph import build_attention_graph
from transinterp.visualization.trajectory import plot_decision_trajectory


def test_plot_attention_heatmap_returns_figure():
    attention = torch.rand(4, 4)
    fig, ax = plot_attention_heatmap(
        attention, query_tokens=["a", "b", "c", "d"], source_tokens=["a", "b", "c", "d"]
    )
    assert fig is not None and ax is not None


def test_plot_feature_space_returns_figure():
    torch.manual_seed(0)
    hidden = torch.randn(20, 6)
    basis = fit_pca(hidden, n_components=2)
    fig, ax = plot_feature_space(hidden, basis, labels=[f"tok{i}" for i in range(20)])
    assert fig is not None and ax is not None


def test_plot_decision_trajectory_returns_figure():
    trajectory = from_logits({0: torch.tensor([[0.1, 0.9]]), 4: torch.tensor([[0.8, 0.2]])})
    fig, ax = plot_decision_trajectory(trajectory, class_labels=["neg", "pos"])
    assert fig is not None and ax is not None


def test_build_attention_graph_respects_top_k():
    attention = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    graph = build_attention_graph(attention, ["a", "b", "c"], layer=0, head=0, k=1)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2  # one edge per query position
    assert graph.edges[0]["source"] == 0  # top-1 source for query 0 is token "a"
    assert graph.edges[1]["source"] == 2  # top-1 source for query 1 is token "c"
