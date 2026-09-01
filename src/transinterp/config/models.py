"""Typed, serializable experiment configuration.

Configuration is the unit of reproduction in this project: a bundle stores the
exact config that produced it, and ``transinterp replay`` re-executes from
that stored config rather than from a notebook the reader does not have.

``extra="forbid"`` is set on every model deliberately. A silently ignored typo
in a YAML key is a reproducibility bug, because the run that gets recorded is
not the run the author believed they were describing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Base):
    """Model-loading configuration."""

    name_or_path: str
    revision: str | None = None
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    trust_remote_code: bool = False
    attn_implementation: str = "eager"


class CaptureConfig(_Base):
    """Controls which internal tensors are captured from a forward pass."""

    layers: list[int] = Field(default_factory=list)
    capture_residual: bool = True
    capture_mlp: bool = True
    capture_attention: bool = True
    capture_logits: bool = True
    detach: bool = True
    to_cpu: bool = True


class InputConfig(_Base):
    """The prompts an experiment runs on."""

    prompts: list[str] = Field(default_factory=list)
    corrupted_prompts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_alignment(self) -> InputConfig:
        if self.corrupted_prompts and len(self.corrupted_prompts) != len(self.prompts):
            raise ValueError(
                "corrupted_prompts must be empty or the same length as prompts; "
                "patching compares matched pairs"
            )
        return self


class PatchingConfig(_Base):
    """Settings for a causal patching scan."""

    enabled: bool = False
    mode: Literal["replace", "zero", "mean", "scale", "add", "noise"] = "replace"
    targets: list[str] = Field(default_factory=list)
    correct_token: str | int | None = None
    incorrect_token: str | int | None = None
    position: int = -1
    factor: float = 1.0

    @model_validator(mode="after")
    def _check_metric(self) -> PatchingConfig:
        if self.enabled and (self.correct_token is None or self.incorrect_token is None):
            raise ValueError(
                "patching requires correct_token and incorrect_token to define a logit-difference "
                "metric; without a named contrast there is nothing to measure"
            )
        return self


class AnalysisConfig(_Base):
    """Which analyses to run and record."""

    logit_lens: bool = True
    attention_metrics: bool = True
    induction_scores: bool = False
    patching: PatchingConfig = Field(default_factory=PatchingConfig)


class ExperimentConfig(_Base):
    """Top-level, serializable experiment configuration."""

    name: str = "experiment"
    hypothesis: str | None = Field(
        default=None,
        description=(
            "What this experiment would show, and what observation would count against it. "
            "Recorded verbatim in the artifact so a reader can tell whether the analysis "
            "was specified before or after the result was seen."
        ),
    )
    model: ModelConfig
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    output_dir: Path = Path("artifacts")
    seed: int = 0
    strict_determinism: bool = False

    @model_validator(mode="after")
    def _check_patching_inputs(self) -> ExperimentConfig:
        patching = self.analysis.patching
        needs_corrupted = patching.enabled and patching.mode in {"replace", "add"}
        if needs_corrupted and not self.input.corrupted_prompts:
            raise ValueError(
                f"patching mode {patching.mode!r} restores clean activations into a corrupted "
                "run, so input.corrupted_prompts is required"
            )
        return self
