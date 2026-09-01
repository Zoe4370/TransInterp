"""Basic, offline example of the TransInterp analysis primitives."""

import torch

from transinterp.attention.metrics import attention_entropy, topk_edges
from transinterp.trajectories.decision import from_logits


def main() -> None:
    attention = torch.tensor([[[[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]]]])
    print("Attention entropy:", attention_entropy(attention).flatten().tolist())
    weights, sources = topk_edges(attention, k=2)
    print("Top sources:", sources.flatten().tolist())
    print("Top weights:", weights.flatten().tolist())

    trajectory = from_logits(
        {
            0: torch.tensor([[0.2, 1.1, 0.0]]),
            4: torch.tensor([[0.4, 0.8, 0.1]]),
            8: torch.tensor([[1.4, 0.2, 0.0]]),
        }
    )
    print("Predicted class by layer:", trajectory.predicted_class.tolist())
    print("Rank of class 1:", trajectory.rank(1).tolist())


if __name__ == "__main__":
    main()
