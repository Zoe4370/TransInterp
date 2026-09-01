"""Run an experiment from a config and emit a verifiable artifact bundle.

This is the layer that makes the project's premise operational. A researcher
writes a config, runs it once, and gets a directory containing the activations,
the measurements, the environment, and the config itself — enough for someone
else to run :func:`replay_experiment` and find out whether they get the same
numbers, and if not, what differed.

The runner deliberately records failures alongside successes. If the logit
lens fails its self-check, that check result is written into the bundle rather
than raising, because a recorded negative result is more useful to a later
reader than a crashed run that produced nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from transinterp.artifacts.bundle import ArtifactBundle
from transinterp.attention.metrics import attention_entropy
from transinterp.config.models import ExperimentConfig
from transinterp.determinism import set_seed
from transinterp.interventions.metrics import logit_difference
from transinterp.interventions.patching import PatchResult

__all__ = ["ReplayReport", "replay_experiment", "run_experiment"]

_DTYPES: dict[str, torch.dtype | None] = {
    "auto": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_token(adapter: Any, token: str | int) -> int:
    """Turn a config token spec into a vocabulary id.

    String tokens are encoded and required to be a single token. A silent
    multi-token split would make the resulting logit difference measure
    something other than what the config says it measures.
    """
    if isinstance(token, int):
        return token
    ids = adapter.tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"token {token!r} encodes to {len(ids)} tokens ({ids}); logit difference needs a "
            "single-token contrast. Try a leading space, or pass an integer id."
        )
    return int(ids[0])


def run_experiment(
    config: ExperimentConfig,
    *,
    output_dir: Path | str | None = None,
    overwrite: bool = False,
    adapter_factory: Callable[[], Any] | None = None,
) -> ArtifactBundle:
    """Execute ``config`` and write an artifact bundle.

    Args:
        config: The experiment to run.
        output_dir: Overrides ``config.output_dir``.
        overwrite: Replace an existing bundle directory.
        adapter_factory: Supplies the model adapter. Defaults to loading via
            :class:`~transinterp.models.huggingface.HuggingFaceCausalLM`;
            injecting it keeps this function testable without downloading
            weights.

    Returns:
        The written :class:`ArtifactBundle`.
    """
    if not config.input.prompts:
        raise ValueError("config.input.prompts is empty; there is nothing to run")

    determinism = set_seed(config.seed, strict=config.strict_determinism)

    if adapter_factory is None:

        def adapter_factory() -> Any:
            from transinterp.models.huggingface import HuggingFaceCausalLM

            return HuggingFaceCausalLM.from_pretrained(
                config.model.name_or_path,
                revision=config.model.revision,
                device=config.model.device,
                dtype=_DTYPES[config.model.dtype],
                trust_remote_code=config.model.trust_remote_code,
                attn_implementation=config.model.attn_implementation,
            )

    adapter = adapter_factory()

    root = Path(output_dir) if output_dir is not None else Path(config.output_dir) / config.name
    bundle = ArtifactBundle.create(root, overwrite=overwrite)
    bundle.add_config(config)
    bundle.add_determinism(determinism)

    if config.hypothesis:
        bundle.add_note(f"Hypothesis: {config.hypothesis}")
    else:
        bundle.add_note(
            "No hypothesis was recorded for this run. Results are exploratory and any "
            "pattern found here should be restated as a prediction and retested."
        )

    prompt = config.input.prompts[0]
    logits, record = adapter.run(
        prompt,
        output_attentions=config.capture.capture_attention,
        output_hidden_states=config.capture.capture_residual,
    )
    bundle.add_record(record)

    metrics: dict[str, Any] = {"n_prompts": len(config.input.prompts)}

    if config.analysis.logit_lens and config.capture.capture_residual:
        hidden = [
            record.tensors[key]
            for key in sorted(
                (k for k in record.tensors if k.endswith(".hidden")),
                key=lambda name: int(name.split(".")[1]),
            )
        ]
        if hidden:
            lens = adapter.lens()
            check = lens.check(hidden, logits)
            metrics["logit_lens_max_abs_diff"] = check.max_absolute_difference
            metrics["logit_lens_verified"] = check.ok
            if not check.ok:
                bundle.add_note(check.summary())
            trajectory = lens.trajectory(hidden)
            bundle.add_tensor("trajectory.probabilities", trajectory.probabilities)
            metrics["trajectory_predicted_class"] = trajectory.predicted_class.tolist()

    if config.analysis.attention_metrics:
        entropies = {
            name: attention_entropy(tensor).mean().item()
            for name, tensor in record.tensors.items()
            if name.endswith(".attention")
        }
        if entropies:
            metrics["attention_entropy_mean"] = entropies

    patching = config.analysis.patching
    if patching.enabled:
        results = _run_patching(adapter, config, prompt)
        metrics["patching"] = [result.to_dict() for result in results]
        bundle.add_note(
            "Patching effects describe behavior under one corruption. A component that "
            "matters here may not matter under a different corrupted input."
        )

    bundle.add_metrics(metrics)
    bundle.write()
    return bundle


def _run_patching(adapter: Any, config: ExperimentConfig, prompt: str) -> list[PatchResult]:
    patching = config.analysis.patching
    corrupted = config.input.corrupted_prompts[0] if config.input.corrupted_prompts else prompt

    correct = _resolve_token(adapter, patching.correct_token)  # type: ignore[arg-type]
    incorrect = _resolve_token(adapter, patching.incorrect_token)  # type: ignore[arg-type]

    clean_batch = adapter.tokenize(prompt)
    corrupted_batch = adapter.tokenize(corrupted)

    def metric(outputs: Any) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logit_difference(logits, correct, incorrect, position=patching.position)

    patcher = adapter.patcher()
    targets = patching.targets or None
    return patcher.scan(
        clean_run=lambda: adapter.model(**clean_batch, return_dict=True),
        corrupted_run=lambda: adapter.model(**corrupted_batch, return_dict=True),
        metric=metric,
        targets=targets,
        mode=patching.mode,
        factor=patching.factor,
    )


class ReplayReport:
    """Outcome of re-running a stored experiment."""

    def __init__(self, original: ArtifactBundle, replayed: ArtifactBundle) -> None:
        self.original = original
        self.replayed = replayed
        self.comparison = original.compare(replayed)

    @property
    def reproduced(self) -> bool:
        """True when both runs produced byte-identical data files."""
        return bool(self.comparison["identical"])

    def summary(self) -> str:
        """Human-readable verdict, naming the likely cause when it fails."""
        if self.reproduced:
            return f"reproduced exactly (fingerprint {self.original.fingerprint[:12]})"

        lines = ["did NOT reproduce"]
        differing = self.comparison["differing_files"]
        if differing:
            lines.append(f"  {len(differing)} file(s) differ, e.g. {differing[:3]}")
        for name in self.comparison["only_in_first"][:3]:
            lines.append(f"  missing from replay: {name}")
        for name in self.comparison["only_in_second"][:3]:
            lines.append(f"  new in replay: {name}")
        environment = self.comparison["environment_differences"]
        if environment:
            lines.append("  environment differences (likely cause):")
            for key, (before, after) in list(environment.items())[:6]:
                lines.append(f"    {key}: {before} -> {after}")
        else:
            lines.append(
                "  environments match, so the difference is nondeterminism in the run itself; "
                "try strict_determinism: true"
            )
        return "\n".join(lines)


def replay_experiment(
    bundle_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    overwrite: bool = True,
    adapter_factory: Callable[[], Any] | None = None,
) -> ReplayReport:
    """Re-run a stored experiment from its bundle and compare the results."""
    original = ArtifactBundle.load(bundle_path)
    if original.config is None:
        raise ValueError(
            f"{bundle_path} has no stored config, so it cannot be replayed automatically"
        )

    config = ExperimentConfig.model_validate(original.config)
    destination = Path(output_dir) if output_dir else Path(f"{bundle_path}-replay")
    replayed = run_experiment(
        config,
        output_dir=destination,
        overwrite=overwrite,
        adapter_factory=adapter_factory,
    )
    return ReplayReport(original, ArtifactBundle.load(replayed.root))
