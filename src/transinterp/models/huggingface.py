"""A Hugging Face adapter for causal language-model inspection.

The adapter's job is to translate one architecture's module names into the
semantic roles the rest of the package uses (``layer.N.residual``,
``layer.N.mlp``, ``layer.N.attn``), so analysis code never imports a model
class. Discovery is explicit and inspectable rather than magical: the resolved
mapping is available as :attr:`module_map` and recorded in artifacts, and an
unrecognized architecture raises with the paths that were tried instead of
silently returning an empty map.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from transinterp.extraction.capture import ActivationRecord
from transinterp.interventions.patching import ActivationPatcher
from transinterp.trajectories.logit_lens import LogitLens

__all__ = ["HuggingFaceCausalLM"]

# Ordered candidates, tried in turn. Covers the GPT-2/GPT-Neo family, the
# Llama/Mistral/Qwen family, and the OPT/BLOOM layouts.
_BLOCK_PATHS: tuple[tuple[str, ...], ...] = (
    ("transformer", "h"),
    ("model", "layers"),
    ("model", "decoder", "layers"),
    ("gpt_neox", "layers"),
    ("transformer", "blocks"),
)
_NORM_PATHS: tuple[tuple[str, ...], ...] = (
    ("transformer", "ln_f"),
    ("model", "norm"),
    ("model", "decoder", "final_layer_norm"),
    ("gpt_neox", "final_layer_norm"),
    ("transformer", "norm_f"),
)
_UNEMBED_PATHS: tuple[tuple[str, ...], ...] = (
    ("lm_head",),
    ("embed_out",),
)
_ATTENTION_NAMES = ("attn", "self_attn", "attention", "self_attention")
_MLP_NAMES = ("mlp", "feed_forward", "ffn")


def _resolve(root: Any, path: Sequence[str]) -> Any | None:
    current = root
    for attribute in path:
        current = getattr(current, attribute, None)
        if current is None:
            return None
    return current


def _first_match(root: Any, paths: Iterable[tuple[str, ...]]) -> Any | None:
    for path in paths:
        found = _resolve(root, path)
        if found is not None:
            return found
    return None


def _child(block: nn.Module, names: Sequence[str]) -> nn.Module | None:
    for name in names:
        found = getattr(block, name, None)
        if isinstance(found, nn.Module):
            return found
    return None


@dataclass
class HuggingFaceCausalLM:
    """Run a causal LM and normalize its introspection outputs.

    Supports models implementing the standard Hugging Face causal
    language-model interface. Architectures whose block list lives somewhere
    unusual fail loudly at :attr:`module_map` rather than producing an empty
    analysis.
    """

    model: Any
    tokenizer: Any
    device: torch.device
    model_id: str | None = None
    revision: str | None = None
    _module_map: dict[str, nn.Module] | None = field(default=None, repr=False, init=False)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str = "eager",
    ) -> HuggingFaceCausalLM:
        """Load a tokenizer and causal LM using standard Transformers APIs.

        ``attn_implementation`` defaults to ``eager`` because fused kernels
        such as SDPA and FlashAttention never materialize the attention
        matrix, so every attention metric in this package would silently
        receive nothing. Override it only if attention weights are not needed.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, revision=revision, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            revision=revision,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        resolved = cls._resolve_device(device)
        model.to(resolved).eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved,
            model_id=name_or_path,
            revision=revision,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # structure
    # ------------------------------------------------------------------

    @property
    def blocks(self) -> nn.ModuleList:
        """The model's transformer block list."""
        found = _first_match(self.model, _BLOCK_PATHS)
        if found is None:
            tried = ", ".join(".".join(path) for path in _BLOCK_PATHS)
            raise AttributeError(
                f"could not locate transformer blocks on {type(self.model).__name__}; "
                f"tried: {tried}. Pass an explicit module map to the analysis functions."
            )
        return found

    @property
    def final_norm(self) -> nn.Module:
        """The final normalization layer, required for the logit lens."""
        found = _first_match(self.model, _NORM_PATHS)
        if found is None:
            tried = ", ".join(".".join(path) for path in _NORM_PATHS)
            raise AttributeError(
                f"could not locate a final norm on {type(self.model).__name__}; tried: {tried}"
            )
        return found

    @property
    def unembedding(self) -> nn.Module:
        """The output head projecting hidden states to vocabulary logits."""
        found = _first_match(self.model, _UNEMBED_PATHS)
        if found is None:
            tried = ", ".join(".".join(path) for path in _UNEMBED_PATHS)
            raise AttributeError(
                f"could not locate an unembedding on {type(self.model).__name__}; tried: {tried}"
            )
        return found

    @property
    def module_map(self) -> dict[str, nn.Module]:
        """Semantic role names mapped to concrete modules.

        Produces ``layer.N.residual`` for each block plus ``layer.N.attn`` and
        ``layer.N.mlp`` where those submodules are discoverable. Sublayers a
        given architecture does not expose are simply absent, so callers should
        read the keys rather than assume a fixed set.
        """
        if self._module_map is None:
            mapping: dict[str, nn.Module] = {}
            for index, block in enumerate(self.blocks):
                mapping[f"layer.{index}.residual"] = block
                attention = _child(block, _ATTENTION_NAMES)
                if attention is not None:
                    mapping[f"layer.{index}.attn"] = attention
                mlp = _child(block, _MLP_NAMES)
                if mlp is not None:
                    mapping[f"layer.{index}.mlp"] = mlp
            self._module_map = mapping
        return dict(self._module_map)

    @property
    def n_layers(self) -> int:
        """Number of transformer blocks."""
        return len(self.blocks)

    @property
    def n_heads(self) -> int | None:
        """Attention heads per layer, when the config reports it."""
        config = getattr(self.model, "config", None)
        for attribute in ("num_attention_heads", "n_head", "num_heads"):
            value = getattr(config, attribute, None)
            if isinstance(value, int):
                return value
        return None

    def patcher(self) -> ActivationPatcher:
        """Return an :class:`ActivationPatcher` bound to this model's modules."""
        return ActivationPatcher(self.module_map)

    def lens(self) -> LogitLens:
        """Return a :class:`LogitLens` bound to this model's norm and head."""
        return LogitLens(final_norm=self.final_norm, unembedding=self.unembedding)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def tokenize(self, text: str | Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenize text or a batch of texts onto the adapter device.

        Batched input is padded, which requires a pad token. Many causal LMs
        ship without one, so the EOS token is used as a stand-in when needed —
        the same convention Transformers' own generation utilities apply.
        """
        if not isinstance(text, str) and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            padding=not isinstance(text, str),
        )
        return {name: value.to(self.device) for name, value in batch.items()}

    def token_strings(self, input_ids: torch.Tensor) -> list[str]:
        """Convert one sequence of ids into display tokens for plotting."""
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        return self.tokenizer.convert_ids_to_tokens(ids.tolist())

    def provenance(self) -> dict[str, Any]:
        """Model identity details for the artifact manifest.

        Records the resolved commit hash where Transformers exposes one, which
        is what actually pins the weights; a branch name like ``main`` can
        point at different weights on different days.

        Structural fields degrade to ``None`` when the architecture cannot be
        introspected. Recording a partial environment is strictly better than
        failing a run that was otherwise fine, and analyses that genuinely need
        the block list will raise on their own with a clearer message.
        """
        config = getattr(self.model, "config", None)

        try:
            n_layers: int | None = self.n_layers
        except AttributeError:
            n_layers = None

        try:
            dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        except (StopIteration, AttributeError):
            dtype = None

        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "resolved_commit": getattr(config, "_commit_hash", None),
            "model_class": type(self.model).__name__,
            "architectures": getattr(config, "architectures", None),
            "dtype": dtype,
            "device": str(self.device),
            "n_layers": n_layers,
            "n_heads": self.n_heads,
            "vocab_size": getattr(config, "vocab_size", None),
            "tokenizer_class": type(self.tokenizer).__name__,
        }

    @torch.no_grad()
    def run(
        self,
        text: str | Sequence[str],
        *,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
    ) -> tuple[torch.Tensor, ActivationRecord]:
        """Return logits and a normalized record of hidden states and attention.

        The record's metadata carries the model identity, the input text, and
        the tokenization, so a saved record stays interpretable without the
        script that produced it.
        """
        batch = self.tokenize(text)
        outputs = self.model(
            **batch,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        record = ActivationRecord(
            metadata={
                **self.provenance(),
                "input_text": text if isinstance(text, str) else list(text),
                "sequence_length": int(batch["input_ids"].shape[-1]),
                "batch_size": int(batch["input_ids"].shape[0]),
                "input_ids": batch["input_ids"].detach().cpu().tolist(),
                "tokens": self.token_strings(batch["input_ids"]),
            }
        )
        record.add("logits", outputs.logits)
        for layer, hidden in enumerate(outputs.hidden_states or ()):
            record.add(f"layer.{layer}.hidden", hidden)
        for layer, attention in enumerate(outputs.attentions or ()):
            record.add(f"layer.{layer}.attention", attention)

        if output_attentions and not outputs.attentions:
            record.metadata["warnings"] = [
                (
                    "The model returned no attention weights. This usually means a fused "
                    "attention kernel is active; reload with attn_implementation='eager'."
                )
            ]

        return outputs.logits.detach().cpu(), record

    @torch.no_grad()
    def logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Run a pre-tokenized batch and return logits only."""
        return self.model(**batch, return_dict=True).logits
