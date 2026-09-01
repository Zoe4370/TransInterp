"""Seed and determinism control for reproducible experiments.

``ExperimentConfig`` has always carried a ``seed`` field, but nothing consumed
it. This module makes the seed operative and, more importantly, makes the
*degree* of determinism explicit and recordable.

Full bitwise determinism is not free: it disables nondeterministic CUDA
kernels and can slow a run down substantially. Rather than silently choosing
for the researcher, :func:`set_seed` seeds every relevant generator and
returns a description of exactly what was done, which the artifact layer
stores in the manifest. A replayed run can then be compared against the
determinism level of the original instead of assuming they match.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

__all__ = ["DeterminismState", "deterministic", "set_seed"]


@dataclass(frozen=True)
class DeterminismState:
    """A record of the determinism controls applied to a run."""

    seed: int
    strict: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    torch_deterministic_algorithms: bool
    cublas_workspace_config: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view for the artifact manifest."""
        return asdict(self)


def set_seed(seed: int, *, strict: bool = False) -> DeterminismState:
    """Seed Python, NumPy, and torch RNGs; optionally force deterministic kernels.

    Args:
        seed: Value applied to every generator.
        strict: When ``True``, also request deterministic algorithms from
            torch and cuDNN. This can raise at forward time if a model uses an
            operation with no deterministic implementation, which is
            deliberate: a silent fallback to a nondeterministic kernel would
            make the resulting artifact unreproducible without any warning.

    Returns:
        A :class:`DeterminismState` describing what was actually applied.
    """
    notes: list[str] = []

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        notes.append("CUDA unavailable; only CPU generators were seeded.")

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        notes.append("NumPy not importable; its global RNG was not seeded.")

    cublas_config: str | None = None
    if strict:
        # CUBLAS needs this set before the first CUDA context is created to
        # make matmul reductions reproducible; setting it afterwards is a
        # no-op, so flag that case rather than implying a guarantee.
        cublas_config = ":4096:8"
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            notes.append(
                "CUDA was already initialized; CUBLAS_WORKSPACE_CONFIG may not take effect "
                "for this process."
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        notes.append(
            "Non-strict mode: nondeterministic kernels remain enabled, so bitwise "
            "reproduction across machines is not guaranteed."
        )

    return DeterminismState(
        seed=seed,
        strict=strict,
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        torch_deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        cublas_workspace_config=cublas_config,
        notes=notes,
    )


@contextmanager
def deterministic(seed: int, *, strict: bool = True) -> Iterator[DeterminismState]:
    """Apply determinism settings for a block, then restore the previous ones.

    Useful when only one section of a larger program should be pinned, for
    example an evaluation run inside a training script that must keep using
    fast nondeterministic kernels elsewhere.
    """
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    previous_random_state = random.getstate()
    previous_torch_state = torch.get_rng_state()

    try:
        yield set_seed(seed, strict=strict)
    finally:
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.use_deterministic_algorithms(previous_algorithms, warn_only=False)
        if previous_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas
        random.setstate(previous_random_state)
        torch.set_rng_state(previous_torch_state)
