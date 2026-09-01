"""Executable activation patching and ablation.

Until now this package could only *rank* intervention measurements that a
caller produced elsewhere, which left the hardest part — actually running a
model with a modified internal state — as an exercise for the user. This
module performs the intervention.

The design keeps the same model-agnostic boundary as the capture layer: a
patch targets a named ``nn.Module``, and adapters decide which modules
correspond to which semantic roles. Nothing here imports a specific
architecture.

Three things are worth knowing before using it.

Patching writes to a module's *output*, so patching ``layer.3.mlp`` replaces
what that MLP contributed, not the residual stream downstream of it. Those are
different claims and the distinction matters when reporting a result.

Interventions compose: several specs targeting the same module are applied in
order, so a positional patch followed by a scaling patch does what it reads
like.

An effect found by patching is evidence that a component is *sufficient* or
*necessary* for a behavior under the specific corruption used. Changing the
corruption can change the answer, which is why the scan records its corruption
source in the result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

__all__ = ["ActivationPatcher", "PatchSpec", "patched_forward"]

PatchMode = Literal["replace", "zero", "mean", "scale", "add", "noise"]


@dataclass
class PatchSpec:
    """One intervention on one module's output.

    Args:
        module: Key into the module map being patched.
        mode: How to modify the activation. ``replace`` substitutes ``source``;
            ``zero`` writes zeros; ``mean`` writes the token-mean of the
            activation, which is usually a better ablation baseline than zero
            because it stays inside the model's normal activation range;
            ``scale`` multiplies by ``factor``; ``add`` adds ``source``;
            ``noise`` adds Gaussian noise scaled by ``factor``.
        source: Replacement or addend, required for ``replace`` and ``add``.
        positions: Token positions to patch. ``None`` patches every position.
        heads: Attention head indices to patch. Requires ``n_heads`` so the
            feature axis can be split into per-head slices.
        n_heads: Number of heads the feature axis encodes.
        factor: Multiplier for ``scale`` and noise scale for ``noise``.
    """

    module: str
    mode: PatchMode = "replace"
    source: torch.Tensor | None = None
    positions: Sequence[int] | None = None
    heads: Sequence[int] | None = None
    n_heads: int | None = None
    factor: float = 1.0

    def __post_init__(self) -> None:
        if self.mode in {"replace", "add"} and self.source is None:
            raise ValueError(f"mode={self.mode!r} requires a source tensor")
        if self.heads is not None and self.n_heads is None:
            raise ValueError("patching specific heads requires n_heads")

    def apply(self, activation: torch.Tensor) -> torch.Tensor:
        """Return a modified copy of ``activation``."""
        patched = activation.clone()
        target = self._region(patched)
        replacement = self._value(activation, target)
        self._write(patched, replacement)
        return patched

    # -- internals -------------------------------------------------------

    def _position_index(self, activation: torch.Tensor) -> Any:
        if self.positions is None:
            return slice(None)
        length = activation.shape[-2] if activation.ndim >= 2 else activation.shape[0]
        resolved = [position % length for position in self.positions]
        return torch.tensor(resolved, dtype=torch.long)

    def _head_slice(self, activation: torch.Tensor) -> tuple[int, int] | None:
        if self.heads is None or self.n_heads is None:
            return None
        features = activation.shape[-1]
        if features % self.n_heads:
            raise ValueError(
                f"feature dimension {features} is not divisible by n_heads={self.n_heads}"
            )
        return features // self.n_heads, features

    def _region(self, activation: torch.Tensor) -> torch.Tensor:
        index = self._position_index(activation)
        if activation.ndim >= 3:
            return activation[:, index, :]
        return activation[index]

    def _value(self, activation: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        if self.mode == "replace":
            source = self._align(self.source, region)
            return source
        if self.mode == "zero":
            return torch.zeros_like(region)
        if self.mode == "mean":
            # Mean over the token axis of the full activation, broadcast back.
            axis = -2 if activation.ndim >= 2 else 0
            mean = activation.mean(dim=axis, keepdim=True)
            return mean.expand_as(region).clone()
        if self.mode == "scale":
            return region * self.factor
        if self.mode == "add":
            return region + self._align(self.source, region)
        if self.mode == "noise":
            generator_free_noise = torch.randn_like(region)
            return region + generator_free_noise * self.factor
        raise ValueError(f"unknown patch mode {self.mode!r}")

    @staticmethod
    def _align(source: torch.Tensor | None, region: torch.Tensor) -> torch.Tensor:
        if source is None:  # pragma: no cover - guarded in __post_init__
            raise ValueError("source tensor is required")
        aligned = source.to(device=region.device, dtype=region.dtype)
        if aligned.shape == region.shape:
            return aligned
        try:
            return aligned.expand_as(region).clone()
        except RuntimeError as error:
            raise ValueError(
                f"source shaped {tuple(aligned.shape)} cannot be broadcast to the patched "
                f"region shaped {tuple(region.shape)}"
            ) from error

    def _write(self, activation: torch.Tensor, value: torch.Tensor) -> None:
        index = self._position_index(activation)
        head_geometry = self._head_slice(activation)

        if head_geometry is None:
            if activation.ndim >= 3:
                activation[:, index, :] = value
            else:
                activation[index] = value
            return

        head_dim, _ = head_geometry
        for head in self.heads or ():
            start, stop = head * head_dim, (head + 1) * head_dim
            if activation.ndim >= 3:
                activation[:, index, start:stop] = value[..., start:stop]
            else:
                activation[index, start:stop] = value[..., start:stop]


def _extract(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"cannot extract a tensor from module output of type {type(output)!r}")


def _repack(output: Any, tensor: torch.Tensor) -> Any:
    """Put a patched tensor back into whatever container the module returned.

    Transformer blocks commonly return ``(hidden_states, present, ...)``; the
    trailing entries must survive untouched or the forward pass breaks.
    """
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor,) + tuple(output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError(f"cannot repack module output of type {type(output)!r}")


@contextmanager
def patched_forward(
    modules: Mapping[str, nn.Module], specs: Sequence[PatchSpec]
) -> Iterator[None]:
    """Install patches for the duration of a block.

    Example:
        >>> with patched_forward(module_map, [PatchSpec("layer.2.mlp", mode="zero")]):
        ...     logits = model(**batch).logits
    """
    grouped: dict[str, list[PatchSpec]] = {}
    for spec in specs:
        if spec.module not in modules:
            raise KeyError(
                f"{spec.module!r} is not in the module map; available: {sorted(modules)}"
            )
        grouped.setdefault(spec.module, []).append(spec)

    handles: list[Any] = []

    def make_hook(module_specs: list[PatchSpec]):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            tensor = _extract(output)
            for spec in module_specs:
                tensor = spec.apply(tensor)
            return _repack(output, tensor)

        return hook

    try:
        for name, module_specs in grouped.items():
            handles.append(modules[name].register_forward_hook(make_hook(module_specs)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class PatchResult:
    """Measured effect of patching one component."""

    component: str
    mode: str
    clean_metric: float
    corrupted_metric: float
    patched_metric: float
    normalized: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effect(self) -> float:
        """Raw change relative to the corrupted baseline."""
        return self.patched_metric - self.corrupted_metric

    @property
    def absolute_effect(self) -> float:
        """Magnitude of the raw change."""
        return abs(self.effect)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for artifact manifests."""
        return {
            "component": self.component,
            "mode": self.mode,
            "clean_metric": self.clean_metric,
            "corrupted_metric": self.corrupted_metric,
            "patched_metric": self.patched_metric,
            "normalized": self.normalized,
            "effect": self.effect,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class ActivationPatcher:
    """Cache activations from one run and patch them into another.

    This implements the standard denoising protocol: take a corrupted input on
    which the model behaves incorrectly, restore one component's activation
    from the clean run, and measure how much correct behavior returns. A large
    restoration means that component carried the information the corruption
    destroyed.

    The class holds no model-specific knowledge — it works on any mapping of
    names to ``nn.Module`` objects, so the same code serves a Hugging Face
    causal LM, a vision encoder, or a toy model in a unit test.
    """

    def __init__(self, modules: Mapping[str, nn.Module]) -> None:
        if not modules:
            raise ValueError("modules cannot be empty")
        self.modules = dict(modules)

    def cache(self, run: Callable[[], Any], *, to_cpu: bool = False) -> dict[str, torch.Tensor]:
        """Run a forward pass and return each target module's output tensor.

        Defaults to leaving tensors on their original device, since cached
        activations are usually patched straight back into another forward
        pass on the same device; moving to CPU and back would be wasted work.
        """
        captured: dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                tensor = _extract(output).detach()
                captured[name] = tensor.cpu() if to_cpu else tensor

            return hook

        handles = [
            module.register_forward_hook(make_hook(name))
            for name, module in self.modules.items()
        ]
        try:
            run()
        finally:
            for handle in handles:
                handle.remove()
        return captured

    def run_with_patches(self, run: Callable[[], Any], specs: Sequence[PatchSpec]) -> Any:
        """Execute ``run`` with the given patches installed."""
        with patched_forward(self.modules, specs):
            return run()

    def scan(
        self,
        *,
        clean_run: Callable[[], Any],
        corrupted_run: Callable[[], Any],
        metric: Callable[[Any], torch.Tensor | float],
        targets: Sequence[str] | None = None,
        mode: PatchMode = "replace",
        positions: Sequence[int] | None = None,
        heads: Sequence[int] | None = None,
        n_heads: int | None = None,
        factor: float = 1.0,
    ) -> list[PatchResult]:
        """Patch each target component in turn and measure the effect.

        Args:
            clean_run: Callable returning model output on the clean input.
            corrupted_run: Callable returning output on the corrupted input.
            metric: Maps model output to a scalar; higher should mean "more
                like clean behavior" for the normalization to read naturally.
            targets: Components to scan. Defaults to every module in the map.
            mode: ``replace`` restores the clean activation (denoising).
                ``zero`` and ``mean`` ablate during the clean run instead
                (noising), which answers the complementary question of whether
                a component is necessary.
            positions: Restrict the patch to specific token positions.
            heads, n_heads: Restrict the patch to specific attention heads.
            factor: Multiplier for ``scale`` and ``noise`` modes.

        Returns:
            Results sorted by descending absolute effect.
        """
        selected = list(targets) if targets is not None else list(self.modules)
        unknown = [name for name in selected if name not in self.modules]
        if unknown:
            raise KeyError(f"unknown components: {unknown}; available: {sorted(self.modules)}")

        clean_metric = float(_reduce(metric(clean_run())))
        corrupted_metric = float(_reduce(metric(corrupted_run())))

        # Denoising patches clean activations into the corrupted run; ablation
        # modes act on the clean run, because zeroing a component during an
        # already-corrupted pass conflates two interventions.
        denoising = mode in {"replace", "add"}
        cache = self.cache(clean_run) if denoising else {}
        base_run = corrupted_run if denoising else clean_run

        results: list[PatchResult] = []
        for name in selected:
            spec = PatchSpec(
                module=name,
                mode=mode,
                source=cache.get(name) if denoising else None,
                positions=positions,
                heads=heads,
                n_heads=n_heads,
                factor=factor,
            )
            patched_metric = float(_reduce(metric(self.run_with_patches(base_run, [spec]))))
            results.append(
                PatchResult(
                    component=name,
                    mode=mode,
                    clean_metric=clean_metric,
                    corrupted_metric=corrupted_metric,
                    patched_metric=patched_metric,
                    normalized=_normalized(patched_metric, clean_metric, corrupted_metric),
                    metadata={
                        "protocol": "denoising" if denoising else "noising",
                        "positions": list(positions) if positions else None,
                        "heads": list(heads) if heads else None,
                    },
                )
            )

        return sorted(results, key=lambda result: result.absolute_effect, reverse=True)


def _reduce(value: torch.Tensor | float) -> float:
    """Collapse a metric to a scalar, averaging over the batch if needed."""
    if isinstance(value, torch.Tensor):
        return float(value.mean())
    return float(value)


def _normalized(patched: float, clean: float, corrupted: float) -> float:
    from transinterp.interventions.metrics import normalized_effect

    return normalized_effect(patched, clean, corrupted)
