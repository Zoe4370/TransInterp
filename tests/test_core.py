import torch
from torch import nn

from transinterp.attention.circuits import rank_interventions
from transinterp.attention.metrics import attention_entropy, attention_mass, topk_edges
from transinterp.extraction.capture import HookCapture
from transinterp.trajectories.decision import from_logits
from transinterp.utils.tensor import flatten_tokens, normalize_features


def test_attention_metrics_on_normalized_rows():
    attention = torch.tensor([[[[0.75, 0.25], [0.5, 0.5]]]])
    assert attention_entropy(attention).shape == (1, 1, 2)
    mass = attention_mass(attention, torch.tensor([True, False]))
    assert torch.allclose(mass, torch.tensor([[[0.75, 0.5]]]))
    values, indices = topk_edges(attention, k=1)
    assert values.shape == indices.shape == (1, 1, 2, 1)
    assert indices[0, 0, 0, 0].item() == 0


def test_decision_trajectory_orders_layers_and_ranks():
    trajectory = from_logits({4: torch.tensor([[0.0, 2.0]]), 1: torch.tensor([[3.0, 0.0]])})
    assert trajectory.layer == [1, 4]
    assert trajectory.predicted_class.tolist() == [0, 1]
    assert trajectory.rank(1).tolist() == [2, 1]


def test_interventions_rank_by_absolute_effect():
    scores = rank_interventions(1.0, {"head.2.3": 0.1, "mlp.4": 1.2, "residual.5": 0.7})
    assert [score.component for score in scores] == ["head.2.3", "residual.5", "mlp.4"]
    assert scores[0].absolute_effect == 0.9


def test_tensor_helpers_validate_shapes_and_preserve_features():
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    assert flatten_tokens(hidden).shape == (4, 3)
    normalized = normalize_features(hidden)
    assert torch.allclose(normalized.norm(dim=-1), torch.ones(2, 2))


def test_hook_capture_removes_hooks_after_context():
    module = nn.Linear(2, 2)
    with HookCapture({"linear": module}) as record:
        module(torch.ones(1, 2))
        assert "linear" in record.tensors
    assert len(module._forward_hooks) == 0
