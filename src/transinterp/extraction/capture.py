"""Model hooks and normalized activation containers.

The capture layer intentionally stores tensors without imposing a particular
Transformer implementation. Adapters can register modules by semantic role,
which keeps experiments portable across Hugging Face architectures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import torch
from torch import nn


@dataclass
class ActivationRecord:
    """One forward-pass snapshot, keyed by semantic module name."""

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(
        self, name: str, value: torch.Tensor, *, detach: bool = True, to_cpu: bool = True
    ) -> None:
        """Store a tensor using explicit memory-management policy."""
        if detach:
            value = value.detach()
        if to_cpu:
            value = value.cpu()
        self.tensors[name] = value


class HookCapture:
    """Context manager that captures outputs from selected ``nn.Module`` objects."""

    def __init__(
        self,
        modules: dict[str, nn.Module],
        *,
        detach: bool = True,
        to_cpu: bool = True,
        transform: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
        self.modules = modules
        self.detach = detach
        self.to_cpu = to_cpu
        self.transform = transform or self._default_transform
        self.record = ActivationRecord()
        self._handles: list[Any] = []

    @staticmethod
    def _default_transform(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(f"Cannot extract a tensor from module output of type {type(output)!r}")

    def __enter__(self) -> ActivationRecord:
        def make_hook(name: str):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                self.record.add(
                    name, self.transform(output), detach=self.detach, to_cpu=self.to_cpu
                )

            return hook

        self._handles = [
            module.register_forward_hook(make_hook(name)) for name, module in self.modules.items()
        ]
        return self.record

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
