"""Static visualization helpers for decision trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from transinterp.trajectories.decision import DecisionTrajectory


def plot_decision_trajectory(
    trajectory: DecisionTrajectory,
    output: str | Path | None = None,
    *,
    class_labels: list[str] | None = None,
):
    """Plot class probabilities over model depth and optionally save a figure."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = (
        class_labels
        or trajectory.labels
        or [str(i) for i in range(trajectory.probabilities.shape[-1])]
    )
    values = trajectory.probabilities.detach().cpu().numpy()
    for class_index, label in enumerate(labels):
        ax.plot(
            trajectory.layer,
            values[:, 0, class_index] if values.ndim == 3 else values[:, class_index],
            label=label,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Decision trajectory")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.2)
    if output is not None:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig, ax
