"""Small, model-agnostic primitives for comparing causal interventions.

A circuit claim requires an intervention supplied by the caller. This module
therefore focuses on faithfully ranking supplied clean/intervened measurements
rather than silently inferring causality from attention weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterventionScore:
    """Effect of an intervention on a scalar behavior metric."""

    component: str
    clean_value: float
    intervened_value: float

    @property
    def effect(self) -> float:
        """Signed change from clean to intervened behavior."""
        return self.intervened_value - self.clean_value

    @property
    def absolute_effect(self) -> float:
        """Magnitude of the observed intervention effect."""
        return abs(self.effect)


def rank_interventions(
    clean_value: float,
    intervened_values: dict[str, float],
) -> list[InterventionScore]:
    """Rank caller-supplied interventions by absolute metric change.

    This is intentionally separate from model execution: an adapter or
    experiment runner should produce ``intervened_values`` using its own hook
    and ablation policy, then pass the measurements here for comparison.
    """
    scores = [
        InterventionScore(component=name, clean_value=clean_value, intervened_value=value)
        for name, value in intervened_values.items()
    ]
    return sorted(scores, key=lambda score: score.absolute_effect, reverse=True)
