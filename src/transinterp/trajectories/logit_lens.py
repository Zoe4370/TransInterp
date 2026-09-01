"""Project intermediate hidden states into vocabulary space.

The logit lens reads a model's intermediate residual stream through its own
output head, giving a per-layer view of what the model would predict if it
stopped early. ``transinterp.trajectories.decision`` could already turn
per-layer logits into a trajectory, but nothing produced those logits; this
module closes that gap.

Three things belong with every result this produces.

**The final norm must be applied — exactly once.** The unembedding matrix was
trained on normalized inputs, so skipping normalization yields plausible but
wrong predictions. The opposite error is just as easy: Hugging Face models
return a ``hidden_states`` tuple whose *last* entry already has the final norm
applied, so normalizing it again double-normalizes and silently shifts the
readout. :func:`logit_lens` handles that case explicitly via
``final_layer_already_normalized``.

**Verify before trusting.** Because both errors above produce output that
looks reasonable, :meth:`LogitLens.check` compares the final-layer readout
against the model's own logits. If they do not match, the lens is
misconfigured and every earlier layer is suspect too. Running this once per
model costs one forward pass and rules out a whole class of silent error.

**The lens assumes intermediate layers write in the output head's basis.**
That holds better in some models than others, and where it fails the lens
produces confidently wrong readouts — the known limitation that motivated
learned alternatives such as the tuned lens. A flat or nonsensical early-layer
trajectory is at least as likely to indicate basis mismatch as to reveal
something real about the computation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from transinterp.trajectories.decision import DecisionTrajectory

__all__ = ["LensCheck", "LogitLens", "logit_lens"]


def _unembed(hidden: torch.Tensor, unembedding: nn.Module | torch.Tensor) -> torch.Tensor:
    if isinstance(unembedding, torch.Tensor):
        return hidden @ unembedding.T
    return unembedding(hidden)


def logit_lens(
    hidden_states: Sequence[torch.Tensor],
    *,
    final_norm: nn.Module,
    unembedding: nn.Module | torch.Tensor,
    position: int = -1,
    layers: Sequence[int] | None = None,
    final_layer_already_normalized: bool = True,
) -> dict[int, torch.Tensor]:
    """Return ``layer -> logits`` by unembedding each hidden state.

    Args:
        hidden_states: Per-layer residual stream tensors shaped
            ``(batch, tokens, d_model)``. A Hugging Face ``output_hidden_states``
            tuple works directly; index 0 is the embedding output.
        final_norm: The model's final normalization layer.
        unembedding: The output head, either a module or a weight matrix
            shaped ``(vocab, d_model)``.
        position: Token position to read, defaulting to the last.
        layers: Subset of layer indices to compute. Defaults to all.
        final_layer_already_normalized: Whether the last entry of
            ``hidden_states`` has already passed through ``final_norm``. True
            matches the Hugging Face convention. Set False when supplying raw
            block outputs collected through your own hooks.

    Returns:
        Mapping from layer index to logits shaped ``(batch, vocab)``.
    """
    if not hidden_states:
        raise ValueError("hidden_states cannot be empty")

    count = len(hidden_states)
    selected = list(layers) if layers is not None else list(range(count))
    out_of_range = [layer for layer in selected if not -count <= layer < count]
    if out_of_range:
        raise IndexError(f"layer indices {out_of_range} are outside the hidden-state list")

    logits: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for layer in selected:
            index = layer % count
            hidden = hidden_states[index]
            if hidden.ndim != 3:
                raise ValueError(
                    f"hidden state for layer {layer} must be (batch, tokens, d_model), "
                    f"got {tuple(hidden.shape)}"
                )
            at_position = hidden[:, position, :]
            is_last = index == count - 1
            normalized = (
                at_position
                if (is_last and final_layer_already_normalized)
                else final_norm(at_position)
            )
            logits[index] = _unembed(normalized, unembedding)
    return logits


@dataclass(frozen=True)
class LensCheck:
    """Result of validating a lens against a model's own output."""

    ok: bool
    max_absolute_difference: float
    tolerance: float
    top_token_matches: bool

    def summary(self) -> str:
        """One-line human-readable status."""
        if self.ok:
            return (
                f"lens verified: final-layer readout matches model logits "
                f"(max |diff| {self.max_absolute_difference:.2e})"
            )
        return (
            f"lens MISCONFIGURED: final-layer readout differs from model logits by "
            f"{self.max_absolute_difference:.2e} (tolerance {self.tolerance:.0e}); "
            f"top-1 token {'agrees' if self.top_token_matches else 'disagrees'}. "
            "Check the final_layer_already_normalized setting and the unembedding."
        )


@dataclass
class LogitLens:
    """Bound logit lens for one model.

    Holding the norm and unembedding together removes the most common way to
    misuse the lens. :meth:`check` removes the second most common way.
    """

    final_norm: nn.Module
    unembedding: nn.Module | torch.Tensor
    final_layer_already_normalized: bool = True

    def __call__(
        self,
        hidden_states: Sequence[torch.Tensor],
        *,
        position: int = -1,
        layers: Sequence[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Return per-layer logits for the given hidden states."""
        return logit_lens(
            hidden_states,
            final_norm=self.final_norm,
            unembedding=self.unembedding,
            position=position,
            layers=layers,
            final_layer_already_normalized=self.final_layer_already_normalized,
        )

    def check(
        self,
        hidden_states: Sequence[torch.Tensor],
        reference_logits: torch.Tensor,
        *,
        position: int = -1,
        tolerance: float = 1e-4,
    ) -> LensCheck:
        """Compare the final-layer readout against the model's actual logits.

        Args:
            hidden_states: The same tuple passed to ``__call__``.
            reference_logits: Logits the model itself produced, shaped
                ``(batch, tokens, vocab)`` or ``(batch, vocab)``.
            position: Token position to compare.
            tolerance: Maximum allowed absolute difference.

        Returns:
            A :class:`LensCheck`. A failure means the lens is wired wrong and
            no layer's readout should be trusted.
        """
        last = len(hidden_states) - 1
        predicted = self(hidden_states, position=position, layers=[last])[last]
        expected = (
            reference_logits[:, position, :]
            if reference_logits.ndim == 3
            else reference_logits
        )
        expected = expected.to(predicted.device, predicted.dtype)
        difference = float((predicted - expected).abs().max())
        return LensCheck(
            ok=difference <= tolerance,
            max_absolute_difference=difference,
            tolerance=tolerance,
            top_token_matches=bool(
                torch.equal(predicted.argmax(dim=-1), expected.argmax(dim=-1))
            ),
        )

    def trajectory(
        self,
        hidden_states: Sequence[torch.Tensor],
        *,
        position: int = -1,
        layers: Sequence[int] | None = None,
    ) -> DecisionTrajectory:
        """Return a :class:`DecisionTrajectory` over the model's depth."""
        from transinterp.trajectories.decision import from_logits

        return from_logits(self(hidden_states, position=position, layers=layers))
