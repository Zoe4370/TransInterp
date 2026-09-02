"""Is the "induction head" an induction head, or a ruler?

The fixed-offset induction task — ``[random N tokens][the same N tokens]`` —
has a confound that its own metric cannot see. Because the repeat always
begins exactly N positions after the original, a head that attends to
"position q minus N" and nothing else scores perfectly on in-context copying
and highly on ``induction_head_score``, while implementing no induction: it
has learned a ruler, not a rule about content.

This script is the control experiment. It trains the same architecture on a
task where the repeated block starts at a varying position and has a varying
length, so no fixed offset solves it, and then asks the same questions of both
models:

1. Does the variable-offset model learn in-context copying at all?
2. Does it transfer to repeat offsets it never saw in training? A positional
   strategy cannot; a content-based one should.
3. Which heads carry the behavior — by score, and by zero-ablating heads
   singly, in pairs, and a whole attention sublayer at a time?
4. Do the answers hold across seeds, with confidence intervals rather than one
   run's point estimate?

Two baselines are reported alongside every accuracy, because "above chance" is
much too weak a bar here. Uniform guessing over the vocabulary is one of them.
The other is the *best content-blind copier*: the single fixed offset ``d``
that maximizes "predict the token ``d`` positions back" on the evaluation set.
On the fixed-offset task that strategy is perfect. On a variable-offset set it
is bounded by how concentrated the offset distribution is over the scored
positions — which is not flat, because longer repeated blocks contribute both
more scored positions and a wider choice of offsets. It is measured rather
than assumed, and lands near 10%: fifty times chance. A model that beats
chance but not this baseline has learned a ruler with error bars.

Run::

    python examples/induction_controls.py                  # full run, 5 seeds
    python examples/induction_controls.py --quick          # smoke test

Results are written to a TransInterp ``ArtifactBundle`` (replayable, digest
verified) and figures to ``assets/``.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from transinterp.artifacts import ArtifactBundle
from transinterp.attention.induction import induction_head_score
from transinterp.determinism import set_seed
from transinterp.interventions.patching import PatchSpec, patched_forward

# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CopyTask:
    """A sequence distribution containing one repeated block.

    Every sequence is ``seq_len`` uniform random tokens, with one block of
    length ``k`` starting at ``s1`` copied to start at ``s2 = s1 + d``. The
    scored positions are ``s2 + 1 .. s2 + k - 1``: predicting the *first*
    token of a repeat is not an induction problem, because nothing in the
    prefix announces that a repeat has begun.

    ``fixed=True`` reproduces the confounded task — ``s1`` is always 0 and
    ``d`` is always ``seq_len // 2``, so the sequence is literally
    ``[random N][the same N]`` and the offset never moves.
    """

    name: str
    fixed: bool
    offset_range: tuple[int, int]
    block_range: tuple[int, int] = (6, 16)
    seq_len: int = 64
    vocab_size: int = 512

    def sample(
        self, rng: np.random.Generator, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
        """Return ``(tokens, scored_mask, metadata)``.

        ``scored_mask`` marks the positions whose *target* is a copied token,
        aligned to the token axis: position ``p`` is scored when the model
        must predict ``tokens[p]`` from ``tokens[:p]``.
        """
        length, vocabulary = self.seq_len, self.vocab_size
        tokens = torch.from_numpy(
            rng.integers(0, vocabulary, size=(batch, length)).astype(np.int64)
        )
        mask = torch.zeros(batch, length, dtype=torch.bool)
        offsets = np.zeros(batch, dtype=np.int64)
        lengths = np.zeros(batch, dtype=np.int64)
        starts = np.zeros(batch, dtype=np.int64)

        low, high = self.offset_range
        for row in range(batch):
            if self.fixed:
                offset = block = length // 2
                first = 0
            else:
                offset = int(rng.integers(low, high + 1))
                widest = min(self.block_range[1], offset - 1, length - offset)
                block = int(rng.integers(self.block_range[0], widest + 1))
                first = int(rng.integers(0, length - offset - block + 1))
            second = first + offset
            tokens[row, second : second + block] = tokens[row, first : first + block]
            mask[row, second + 1 : second + block] = True
            offsets[row], lengths[row], starts[row] = offset, block, second

        return tokens, mask, {"offset": offsets, "block": lengths, "start": starts}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable description, for the artifact manifest."""
        return asdict(self)


# The confounded task, and the control. Offsets are disjoint between training
# and the held-out evaluation, so a model that interpolates over the training
# offsets gets no credit for the held-out ones.
TRAIN_OFFSETS = (8, 20)
HELD_OUT_OFFSETS = (26, 40)

FIXED_TASK = CopyTask("fixed-offset", fixed=True, offset_range=(32, 32))
VARIABLE_TASK = CopyTask("variable-offset", fixed=False, offset_range=TRAIN_OFFSETS)
HELD_OUT_TASK = CopyTask("held-out-offset", fixed=False, offset_range=HELD_OUT_OFFSETS)


def content_blind_ceiling(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    """Accuracy of the best "copy the token d positions back" rule.

    This is the strongest strategy available to a head that routes by position
    alone: it may read any single fixed offset, but may not choose that offset
    from the content. Reported as the bar a genuinely content-based model has
    to clear.
    """
    length = tokens.shape[1]
    best_accuracy, best_offset = 0.0, 0
    for offset in range(1, length):
        predicted = tokens[:, :-offset]
        actual = tokens[:, offset:]
        window = mask[:, offset:]
        if not window.any():
            continue
        accuracy = float((predicted[window] == actual[window]).float().mean())
        if accuracy > best_accuracy:
            best_accuracy, best_offset = accuracy, offset
    return best_accuracy, best_offset


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class HeadOutputs(nn.Identity):
    """A no-op marker exposing the head-concatenated attention output.

    Head-level ablation needs a tensor whose feature axis is partitioned by
    head. GPT-2's attention module returns its output *after* the ``c_proj``
    mixing, where head boundaries no longer exist, so slicing that tensor into
    ``n_heads`` pieces would ablate an arbitrary subspace rather than a head.

    Inserting an identity in front of ``c_proj`` gives ``patched_forward`` a
    module whose output really is ``[head_0 | head_1 | ...]``. Zeroing one
    slice there removes exactly that head's contribution and leaves ``c_proj``'s
    bias intact, which is what "ablate a head" should mean.
    """


def build_model(seed: int, task: CopyTask, *, n_layer: int = 2, n_head: int = 4,
                n_embd: int = 64) -> Any:
    """A randomly initialized two-layer GPT-2, wired for head-level patching."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=task.vocab_size,
        n_positions=task.seq_len,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=0,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config)
    for block in model.transformer.h:
        block.attn.c_proj = nn.Sequential(HeadOutputs(), block.attn.c_proj)
    return model


def module_map(model: Any) -> dict[str, nn.Module]:
    """Patch targets: per-head outputs and whole attention sublayers."""
    mapping: dict[str, nn.Module] = {}
    for index, block in enumerate(model.transformer.h):
        mapping[f"layer.{index}.attn.heads"] = block.attn.c_proj[0]
        mapping[f"layer.{index}.attn"] = block.attn
    return mapping


def copy_loss(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy restricted to the positions inside a repeated block.

    Everything outside a repeat is uniform noise by construction, so including
    it would spend most of the gradient teaching the model to output a uniform
    distribution. Restricting the loss changes what is being learned, not
    whether the offset confound exists.
    """
    predicted = logits[:, :-1][mask[:, 1:]]
    target = tokens[:, 1:][mask[:, 1:]]
    return nn.functional.cross_entropy(predicted, target)


def copy_accuracy(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> tuple[int, int]:
    """Return ``(correct, total)`` over the scored positions."""
    predicted = logits[:, :-1][mask[:, 1:]].argmax(dim=-1)
    target = tokens[:, 1:][mask[:, 1:]]
    return int((predicted == target).sum()), int(target.numel())


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------


EvalBatch = tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]


def make_eval_set(
    task: CopyTask, *, seed: int, sequences: int, batch: int, device: torch.device
) -> list[EvalBatch]:
    """A frozen evaluation set, identical for every model and every ablation.

    The generator's metadata travels with each batch: the head analysis needs
    to know where each repeated block starts in order to look at the attention
    weights that matter.
    """
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(max(1, sequences // batch)):
        tokens, mask, meta = task.sample(rng, batch)
        batches.append((tokens.to(device), mask.to(device), meta))
    return batches


@torch.no_grad()
def evaluate(
    model: Any,
    batches: Sequence[EvalBatch],
    *,
    specs: Sequence[PatchSpec] = (),
    modules: dict[str, nn.Module] | None = None,
) -> float:
    """Copy accuracy over an evaluation set, optionally under ablation."""
    model.eval()
    correct = total = 0
    for tokens, mask, _ in batches:
        if specs:
            with patched_forward(modules or {}, list(specs)):
                logits = model(input_ids=tokens).logits
        else:
            logits = model(input_ids=tokens).logits
        hits, count = copy_accuracy(logits, tokens, mask)
        correct += hits
        total += count
    return correct / max(total, 1)


def train(
    model: Any,
    task: CopyTask,
    *,
    seed: int,
    steps: int,
    batch: int,
    learning_rate: float,
    device: torch.device,
    eval_every: int,
    eval_batches: Sequence[EvalBatch],
    log: bool = True,
) -> list[dict[str, float]]:
    """Train to convergence on ``task`` and return the learning curve."""
    model.to(device).train()
    rng = np.random.default_rng(seed + 10_000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    warmup = max(1, steps // 50)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup
        if step < warmup
        else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))),
    )

    curve: list[dict[str, float]] = []
    started = time.time()
    for step in range(steps):
        tokens, mask, _ = task.sample(rng, batch)
        tokens, mask = tokens.to(device), mask.to(device)
        loss = copy_loss(model(input_ids=tokens).logits, tokens, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)

        if step % eval_every == 0 or step == steps - 1:
            accuracy = evaluate(model, eval_batches)
            model.train()
            value = float(loss.detach())
            curve.append({"step": step, "loss": value, "accuracy": accuracy})
            if log:
                print(
                    f"    step {step:>6}  loss {value:6.3f}  copy-acc {accuracy:6.2%}"
                    f"  ({time.time() - started:5.0f}s)",
                    flush=True,
                )
    return curve


# --------------------------------------------------------------------------
# Head analysis
# --------------------------------------------------------------------------


@torch.no_grad()
def _attentions(model: Any, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
    model.eval()
    outputs = model(input_ids=tokens, output_attentions=True)
    if not outputs.attentions:
        raise RuntimeError(
            "the model returned no attention weights; build it with "
            "attn_implementation='eager'"
        )
    return tuple(attention.float().cpu() for attention in outputs.attentions)


@torch.no_grad()
def head_induction_scores(
    model: Any, batches: Sequence[EvalBatch], *, sequences: int = 32
) -> np.ndarray:
    """The library's ``induction_head_score`` per head, shaped ``(layers, heads)``."""
    tokens = torch.cat([batch[0] for batch in batches], dim=0)[:sequences]
    ids = tokens.cpu()
    return np.stack(
        [
            induction_head_score(attention, ids).mean(dim=0).numpy()
            for attention in _attentions(model, tokens)
        ]
    )


@torch.no_grad()
def span_attention_scores(
    model: Any, batches: Sequence[EvalBatch], *, sequences: int = 32
) -> dict[str, np.ndarray]:
    """Where each head looks from inside a repeated block, per head.

    ``induction_head_score`` averages over every query whose token appeared
    earlier — which, on these sequences, is mostly random filler that happens
    to collide with an earlier random token. That dilution is proportional to
    how much of the sequence is a repeat, so it is much heavier on the
    variable-offset task (one short block) than on the fixed-offset task (half
    the sequence). Comparing the two models on the library metric alone would
    compare task structure as much as head behavior.

    This uses the generator's ground truth instead. The query that must
    predict the copied token at ``s2 + j`` sits at ``s2 + j - 1`` and carries
    the same token as ``s1 + j - 1``. Two source positions are then worth
    telling apart:

    ``induction_target`` (``s1 + j``)
        the token that *followed* the earlier occurrence — what an induction
        head attends to, and what ``induction_head_score`` measures.
    ``earlier_occurrence`` (``s1 + j - 1``)
        the earlier occurrence of the query's own token. A head that matches
        on content but does not shift by one lands here. It scores zero as an
        induction head while still doing the content matching that in-context
        copying requires, so a low induction score does not by itself mean a
        head is ignoring content.
    """
    totals = {"induction_target": None, "earlier_occurrence": None}
    positions = 0
    counted = 0
    for tokens, _, meta in batches:
        if counted >= sequences:
            break
        take = min(tokens.shape[0], sequences - counted)
        stacked = torch.stack(_attentions(model, tokens[:take]))
        shape = (stacked.shape[0], stacked.shape[2])
        for name in totals:
            if totals[name] is None:
                totals[name] = np.zeros(shape)
        for row in range(take):
            start = int(meta["start"][row])
            block = int(meta["block"][row])
            first = start - int(meta["offset"][row])
            for step in range(1, block):
                query = start + step - 1
                totals["induction_target"] += stacked[:, row, :, query, first + step].numpy()
                totals["earlier_occurrence"] += (
                    stacked[:, row, :, query, first + step - 1].numpy()
                )
                positions += 1
        counted += take
    return {
        name: (grid / max(positions, 1)) if grid is not None else np.zeros((1, 1))
        for name, grid in totals.items()
    }


def head_specs(layer: int, heads: Sequence[int], n_head: int) -> list[PatchSpec]:
    """Zero-ablate ``heads`` in one layer, at the head-partitioned tensor."""
    return [
        PatchSpec(
            module=f"layer.{layer}.attn.heads", mode="zero", heads=list(heads), n_heads=n_head
        )
    ]


def ablation_conditions(n_layer: int, n_head: int) -> dict[str, list[tuple[int, tuple[int, ...]]]]:
    """Every ablation condition, as ``name -> [(layer, heads), ...]``.

    Individual heads, all unordered pairs of heads (within and across layers),
    and each whole attention sublayer.
    """
    heads = [(layer, head) for layer in range(n_layer) for head in range(n_head)]
    conditions: dict[str, list[tuple[int, tuple[int, ...]]]] = {}

    for layer, head in heads:
        conditions[f"head L{layer}H{head}"] = [(layer, (head,))]

    for first in range(len(heads)):
        for second in range(first + 1, len(heads)):
            (layer_a, head_a), (layer_b, head_b) = heads[first], heads[second]
            name = f"pair L{layer_a}H{head_a}+L{layer_b}H{head_b}"
            if layer_a == layer_b:
                conditions[name] = [(layer_a, (head_a, head_b))]
            else:
                conditions[name] = [(layer_a, (head_a,)), (layer_b, (head_b,))]

    for layer in range(n_layer):
        conditions[f"sublayer L{layer} (all heads)"] = [(layer, tuple(range(n_head)))]

    return conditions


def run_ablations(
    model: Any,
    batches: Sequence[EvalBatch],
    *,
    n_layer: int,
    n_head: int,
) -> dict[str, float]:
    """Accuracy under every ablation condition, plus the unablated baseline."""
    modules = module_map(model)
    results = {"none (baseline)": evaluate(model, batches)}

    for name, groups in ablation_conditions(n_layer, n_head).items():
        specs = [spec for layer, heads in groups for spec in head_specs(layer, heads, n_head)]
        results[name] = evaluate(model, batches, specs=specs, modules=modules)

    # The whole attention sublayer output, bias included. Zeroing every head
    # leaves c_proj's bias in place; this removes the sublayer entirely, and
    # the two agreeing is a check that the head decomposition is complete.
    for layer in range(n_layer):
        specs = [PatchSpec(module=f"layer.{layer}.attn", mode="zero")]
        results[f"sublayer L{layer} (whole output)"] = evaluate(
            model, batches, specs=specs, modules=modules
        )
    return results


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def bootstrap_ci(
    values: Iterable[float], *, resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Mean and percentile bootstrap confidence interval over seeds.

    With five seeds the interval is coarse — the resampling distribution has
    only 5**5 distinct draws — so it is reported as what it is: a spread over
    a handful of training runs, not a population estimate.
    """
    sample = np.asarray(list(values), dtype=float)
    if sample.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    if sample.size == 1:
        value = float(sample[0])
        return {"mean": value, "low": value, "high": value, "n": 1}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sample.size, size=(resamples, sample.size))
    means = sample[draws].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(sample.mean()),
        "low": float(low),
        "high": float(high),
        "n": int(sample.size),
        "values": [float(v) for v in sample],
    }


def summarize(
    per_seed: Sequence[dict[str, float]], keys: Iterable[str]
) -> dict[str, dict[str, float]]:
    """Bootstrap every named key across the per-seed dictionaries."""
    return {key: bootstrap_ci([run[key] for run in per_seed]) for key in keys}


# --------------------------------------------------------------------------
# One seed, end to end
# --------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Everything one training seed produced."""

    seed: int
    curves: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    accuracy: dict[str, float] = field(default_factory=dict)
    induction: dict[str, list[list[float]]] = field(default_factory=dict)
    span_induction: dict[str, list[list[float]]] = field(default_factory=dict)
    span_match: dict[str, list[list[float]]] = field(default_factory=dict)
    ablations: dict[str, dict[str, float]] = field(default_factory=dict)
    offset_profile: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return asdict(self)


def run_seed(seed: int, args: argparse.Namespace, device: torch.device) -> SeedResult:
    """Train both models on one seed and run the full analysis on each."""
    result = SeedResult(seed=seed)
    evaluation = {
        "fixed": make_eval_set(
            FIXED_TASK, seed=99_001, sequences=args.eval_sequences, batch=args.batch,
            device=device,
        ),
        "train-offsets": make_eval_set(
            VARIABLE_TASK, seed=99_002, sequences=args.eval_sequences, batch=args.batch,
            device=device,
        ),
        "held-out-offsets": make_eval_set(
            HELD_OUT_TASK, seed=99_003, sequences=args.eval_sequences, batch=args.batch,
            device=device,
        ),
    }

    plan = [
        ("fixed-offset", FIXED_TASK, args.steps_fixed, "fixed"),
        ("variable-offset", VARIABLE_TASK, args.steps_variable, "train-offsets"),
    ]

    for label, task, steps, native in plan:
        print(f"  [seed {seed}] training {label} model for {steps} steps", flush=True)
        model = build_model(
            seed, task, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd
        )
        result.curves[label] = train(
            model,
            task,
            seed=seed,
            steps=steps,
            batch=args.batch,
            learning_rate=args.learning_rate,
            device=device,
            eval_every=max(1, steps // args.curve_points),
            eval_batches=evaluation[native],
        )

        for name, batches in evaluation.items():
            result.accuracy[f"{label} on {name}"] = evaluate(model, batches)

        result.induction[label] = head_induction_scores(model, evaluation[native]).tolist()
        span = span_attention_scores(model, evaluation[native])
        result.span_induction[label] = span["induction_target"].tolist()
        result.span_match[label] = span["earlier_occurrence"].tolist()
        result.ablations[label] = run_ablations(
            model, evaluation[native], n_layer=args.n_layer, n_head=args.n_head
        )
        result.offset_profile[label] = offset_profile(model, args, device)
        del model

    return result


def offset_profile(model: Any, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    """Copy accuracy as a function of the repeat offset, one offset at a time."""
    profile: dict[str, float] = {}
    for offset in range(8, 45, 2):
        task = CopyTask(f"offset-{offset}", fixed=False, offset_range=(offset, offset))
        batches = make_eval_set(
            task, seed=98_000 + offset, sequences=args.profile_sequences,
            batch=args.batch, device=device,
        )
        profile[str(offset)] = evaluate(model, batches)
    return profile


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def baselines(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Chance and best-content-blind-copier accuracy for each evaluation set."""
    sets = {
        "fixed": FIXED_TASK,
        "train-offsets": VARIABLE_TASK,
        "held-out-offsets": HELD_OUT_TASK,
    }
    seeds = {"fixed": 99_001, "train-offsets": 99_002, "held-out-offsets": 99_003}
    computed: dict[str, dict[str, float]] = {}
    for name, task in sets.items():
        batches = make_eval_set(
            task, seed=seeds[name], sequences=args.eval_sequences, batch=args.batch,
            device=torch.device("cpu"),
        )
        tokens = torch.cat([b[0] for b in batches], dim=0)
        mask = torch.cat([b[1] for b in batches], dim=0)
        accuracy, offset = content_blind_ceiling(tokens, mask)
        computed[name] = {
            "chance": 1.0 / task.vocab_size,
            "content_blind_ceiling": accuracy,
            "content_blind_best_offset": float(offset),
        }
    return computed


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2df"
SERIES = {"fixed-offset": "#2a78d6", "variable-offset": "#eb6834"}


def _style(axis: Any) -> None:
    axis.set_facecolor("#fcfcfb")
    axis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)


def make_figures(
    seeds: Sequence[SeedResult], base: dict[str, dict[str, float]], assets: Path
) -> list[Path]:
    """Write every figure to ``assets`` and return the paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "figure.facecolor": "#fcfcfb", "axes.titlesize": 11})
    assets.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. Learning curves ---------------------------------------------------
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    _style(axis)
    for label in ("fixed-offset", "variable-offset"):
        curves = [run.curves[label] for run in seeds]
        steps = [point["step"] for point in curves[0]]
        values = np.array([[point["accuracy"] for point in curve] for curve in curves]) * 100
        stats = [bootstrap_ci(values[:, i]) for i in range(values.shape[1])]
        axis.plot(steps, [s["mean"] for s in stats], color=SERIES[label], linewidth=2,
                  label=f"{label} model", zorder=3)
        axis.fill_between(steps, [s["low"] for s in stats], [s["high"] for s in stats],
                          color=SERIES[label], alpha=0.18, linewidth=0, zorder=2)
    ceiling = base["train-offsets"]["content_blind_ceiling"] * 100
    axis.axhline(ceiling, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    axis.annotate(f"best content-blind copier on varying offsets  {ceiling:.1f}%",
                  xy=(0.99, ceiling), xycoords=("axes fraction", "data"),
                  ha="right", va="bottom", color=MUTED, fontsize=8.5)
    axis.set_xlabel("training step", color=MUTED)
    axis.set_ylabel("copy accuracy on the repeated span (%)", color=MUTED)
    axis.set_title("Both tasks are learned; the variable-offset task takes longer",
                   color=INK, loc="left")
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=9)
    axis.set_ylim(-3, 103)
    figure.tight_layout()
    path = assets / "induction_controls_learning_curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path)

    # 2. Generalization ----------------------------------------------------
    sets = ["fixed", "train-offsets", "held-out-offsets"]
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    _style(axis)
    width = 0.36
    positions = np.arange(len(sets))
    for index, label in enumerate(("fixed-offset", "variable-offset")):
        stats = [
            bootstrap_ci([run.accuracy[f"{label} on {name}"] * 100 for run in seeds])
            for name in sets
        ]
        offsets = positions + (index - 0.5) * width
        means = [s["mean"] for s in stats]
        errors = np.array([[s["mean"] - s["low"] for s in stats],
                           [s["high"] - s["mean"] for s in stats]])
        axis.bar(offsets, means, width * 0.92, color=SERIES[label], label=f"{label} model",
                 zorder=3, edgecolor="#fcfcfb", linewidth=2)
        axis.errorbar(offsets, means, yerr=np.clip(errors, 0, None), fmt="none",
                      ecolor=INK, elinewidth=1.2, capsize=3, zorder=4)
        for x, value in zip(offsets, means, strict=True):
            axis.text(x, value + 2.5, f"{value:.1f}", ha="center", va="bottom",
                      color=MUTED, fontsize=8.5)
    for index, name in enumerate(sets):
        ceiling = base[name]["content_blind_ceiling"] * 100
        axis.plot([index - 0.5, index + 0.5], [ceiling, ceiling], color=MUTED,
                  linestyle="--", linewidth=1.2, zorder=5)
    axis.plot([], [], color=MUTED, linestyle="--", linewidth=1.2,
              label="best content-blind copier")
    axis.set_xticks(positions)
    axis.set_xticklabels(["fixed offset 32\n(training task A)",
                          "offsets 8-20\n(training task B)",
                          "offsets 26-40\n(held out)"], color=MUTED)
    axis.set_ylabel("copy accuracy (%)", color=MUTED)
    axis.set_title("Transfer to unseen repeat offsets", color=INK, loc="left")
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=9)
    axis.set_ylim(0, 112)
    figure.tight_layout()
    path = assets / "induction_controls_generalization.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path)

    # 3. Induction scores --------------------------------------------------
    metrics = [
        ("induction_head_score\n(library, all queries)", "induction"),
        ("induction target\n(inside a real repeat)", "span_induction"),
        ("earlier occurrence\nof the query's own token", "span_match"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(8.2, 7.8))
    grids = {
        (attribute, label): np.mean(
            [np.array(getattr(run, attribute)[label]) for run in seeds], axis=0
        )
        for _, attribute in metrics
        for label in SERIES
    }
    top = max(float(grid.max()) for grid in grids.values())
    for row, (title, attribute) in enumerate(metrics):
        for column, label in enumerate(SERIES):
            axis = axes[row][column]
            grid = grids[(attribute, label)]
            image = axis.imshow(grid, cmap="Blues", vmin=0, vmax=max(top, 1e-6), aspect="auto")
            for layer in range(grid.shape[0]):
                for head in range(grid.shape[1]):
                    value = grid[layer, head]
                    axis.text(head, layer, f"{value:.2f}", ha="center", va="center",
                              fontsize=9, color="#ffffff" if value > 0.6 * top else INK)
            axis.set_xticks(range(grid.shape[1]))
            axis.set_xticklabels([f"H{h}" for h in range(grid.shape[1])], color=MUTED)
            axis.set_yticks(range(grid.shape[0]))
            axis.set_yticklabels([f"L{layer}" for layer in range(grid.shape[0])], color=MUTED)
            axis.tick_params(colors=MUTED, labelsize=9)
            if row == 0:
                axis.set_title(f"{label} model", color=INK, fontsize=10)
            if column == 0:
                axis.set_ylabel(title, color=MUTED, fontsize=8.5)
    figure.colorbar(image, ax=axes, shrink=0.7, label="attention mass on the induction target")
    figure.suptitle(
        "A head can score as an induction head without doing induction",
        color=INK, x=0.02, ha="left", fontsize=11,
    )
    path = assets / "induction_controls_head_scores.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    # 4. Ablations ---------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for axis, label in zip(axes, SERIES, strict=True):
        _style(axis)
        names = list(seeds[0].ablations[label])
        singles = [n for n in names if n.startswith("head ")]
        sublayers = [n for n in names if n.startswith("sublayer") and "all heads" in n]
        pairs = [n for n in names if n.startswith("pair ")]
        pair_means = {
            n: float(np.mean([run.ablations[label][n] for run in seeds])) for n in pairs
        }
        worst_pairs = sorted(pair_means, key=pair_means.get)[:4]
        chosen = ["none (baseline)", *singles, *worst_pairs, *sublayers]
        stats = [
            bootstrap_ci([run.ablations[label][n] * 100 for run in seeds]) for n in chosen
        ]
        colors = [MUTED] + [SERIES[label]] * len(singles) + ["#1baf7a"] * len(worst_pairs) + [
            "#4a3aa7"
        ] * len(sublayers)
        positions = np.arange(len(chosen))
        means = [s["mean"] for s in stats]
        errors = np.array([[s["mean"] - s["low"] for s in stats],
                           [s["high"] - s["mean"] for s in stats]])
        axis.barh(positions, means, 0.72, color=colors, zorder=3,
                  edgecolor="#fcfcfb", linewidth=1.5)
        axis.errorbar(means, positions, xerr=np.clip(errors, 0, None), fmt="none",
                      ecolor=INK, elinewidth=1.1, capsize=2.5, zorder=4)
        axis.set_yticks(positions)
        axis.set_yticklabels([n.replace("sublayer ", "sublayer ") for n in chosen],
                             fontsize=8.5, color=MUTED)
        axis.set_xlabel("copy accuracy under ablation (%)", color=MUTED)
        axis.set_title(f"{label} model", color=INK, loc="left")
    # One call only: the axes share y, so inverting per-subplot cancels out.
    axes[0].invert_yaxis()
    figure.suptitle("Zero-ablation: single heads, worst pairs, whole sublayers",
                    color=INK, x=0.01, ha="left", fontsize=11)
    figure.tight_layout()
    path = assets / "induction_controls_ablations.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path)

    # 5. Offset profile ----------------------------------------------------
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    _style(axis)
    axis.axvspan(TRAIN_OFFSETS[0], TRAIN_OFFSETS[1], color="#eb6834", alpha=0.10,
                 zorder=1, linewidth=0)
    axis.axvspan(HELD_OUT_OFFSETS[0], HELD_OUT_OFFSETS[1], color="#2a78d6", alpha=0.08,
                 zorder=1, linewidth=0)
    for label in SERIES:
        offsets = sorted(int(k) for k in seeds[0].offset_profile[label])
        stats = [
            bootstrap_ci([run.offset_profile[label][str(o)] * 100 for run in seeds])
            for o in offsets
        ]
        axis.plot(offsets, [s["mean"] for s in stats], color=SERIES[label], linewidth=2,
                  marker="o", markersize=4, label=f"{label} model", zorder=3)
        axis.fill_between(offsets, [s["low"] for s in stats], [s["high"] for s in stats],
                          color=SERIES[label], alpha=0.18, linewidth=0, zorder=2)
    axis.axvline(32, color=MUTED, linestyle=":", linewidth=1.2, zorder=2)
    axis.annotate("offset 32\n(fixed task)", xy=(32, 96), color=MUTED, fontsize=8.5,
                  ha="center", va="top")
    axis.annotate("trained offsets\n(variable task)", xy=(14, 4), color="#b8460f",
                  fontsize=8.5, ha="center")
    axis.set_xlabel("repeat offset (positions between the two copies)", color=MUTED)
    axis.set_ylabel("copy accuracy (%)", color=MUTED)
    axis.set_title("Where each model works, as a function of repeat offset",
                   color=INK, loc="left")
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=9, loc="center right")
    axis.set_ylim(-3, 103)
    figure.tight_layout()
    path = assets / "induction_controls_offset_profile.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path)

    return written


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def verdict(
    seeds: Sequence[SeedResult], base: dict[str, dict[str, float]]
) -> tuple[list[str], dict[str, Any]]:
    """Turn the numbers into a plain-language reading, and the facts behind it.

    Every claim below is a comparison the caller can check against the metrics
    in the bundle. Nothing is asserted that the confidence intervals do not
    support.
    """
    def accuracy(label: str, name: str) -> dict[str, float]:
        return bootstrap_ci([run.accuracy[f"{label} on {name}"] for run in seeds])

    facts: dict[str, Any] = {
        "fixed_model": {name: accuracy("fixed-offset", name)
                        for name in ("fixed", "train-offsets", "held-out-offsets")},
        "variable_model": {name: accuracy("variable-offset", name)
                           for name in ("fixed", "train-offsets", "held-out-offsets")},
        "baselines": base,
    }

    for label in SERIES:
        names = list(seeds[0].ablations[label])
        drops = {}
        for name in names:
            if name == "none (baseline)":
                continue
            drops[name] = bootstrap_ci(
                [run.ablations[label]["none (baseline)"] - run.ablations[label][name]
                 for run in seeds]
            )
        singles = {n: d for n, d in drops.items() if n.startswith("head ")}
        pairs = {n: d for n, d in drops.items() if n.startswith("pair ")}
        sublayers = {n: d for n, d in drops.items() if n.startswith("sublayer")}
        layers = sorted({int(name.split("L")[1][0]) for name in singles})
        per_layer = {
            layer: {
                "worst_single_head": max(
                    ((n, d) for n, d in singles.items() if n.startswith(f"head L{layer}")),
                    key=lambda kv: kv[1]["mean"],
                ),
                "sublayer": sublayers[f"sublayer L{layer} (whole output)"],
            }
            for layer in layers
        }
        facts[f"{label}_ablation"] = {
            "baseline": bootstrap_ci([run.ablations[label]["none (baseline)"] for run in seeds]),
            "worst_single_head": max(singles.items(), key=lambda kv: kv[1]["mean"]),
            "worst_pair": max(pairs.items(), key=lambda kv: kv[1]["mean"]),
            "sublayers": sublayers,
            "per_layer": per_layer,
        }

    lines: list[str] = []
    variable_train = facts["variable_model"]["train-offsets"]
    variable_held = facts["variable_model"]["held-out-offsets"]
    fixed_own = facts["fixed_model"]["fixed"]
    fixed_held = facts["fixed_model"]["held-out-offsets"]
    ceiling_train = base["train-offsets"]["content_blind_ceiling"]
    ceiling_held = base["held-out-offsets"]["content_blind_ceiling"]

    # 1. Did the control task get learned at all?
    if variable_train["low"] > ceiling_train:
        lines.append(
            f"1. The variable-offset task was learned: {variable_train['mean']:.1%} "
            f"[{variable_train['low']:.1%}, {variable_train['high']:.1%}] on the trained "
            f"offset range, against {ceiling_train:.1%} for the best content-blind copier "
            f"and {base['train-offsets']['chance']:.2%} for chance. The whole interval is "
            "above the content-blind bar, so this is not a positional strategy with luck."
        )
    else:
        lines.append(
            f"1. The variable-offset task was NOT convincingly learned: "
            f"{variable_train['mean']:.1%} [{variable_train['low']:.1%}, "
            f"{variable_train['high']:.1%}] against a content-blind ceiling of "
            f"{ceiling_train:.1%}. Nothing downstream of this can settle the confound; "
            "the model needs more training before the head analysis means anything."
        )

    # 2. Does it transfer to offsets it never saw?
    if variable_held["low"] > ceiling_held:
        lines.append(
            f"2. It transfers to offsets held out of training: {variable_held['mean']:.1%} "
            f"[{variable_held['low']:.1%}, {variable_held['high']:.1%}] on offsets "
            f"{HELD_OUT_OFFSETS[0]}-{HELD_OUT_OFFSETS[1]}, which it never saw, against a "
            f"content-blind ceiling of {ceiling_held:.1%}. A head that routes by position "
            "alone cannot do this. The positional confound is ruled out FOR THIS MODEL."
        )
    else:
        lines.append(
            f"2. It does NOT transfer to held-out offsets: {variable_held['mean']:.1%} "
            f"[{variable_held['low']:.1%}, {variable_held['high']:.1%}] on offsets "
            f"{HELD_OUT_OFFSETS[0]}-{HELD_OUT_OFFSETS[1]} against a content-blind ceiling "
            f"of {ceiling_held:.1%}. Varying the offset during training was not enough to "
            "force a content-based solution; the model appears to have learned a mixture "
            "over the training offsets. The positional confound is NOT ruled out."
        )

    # 3. What the fixed-offset task was actually measuring.
    lines.append(
        f"3. The fixed-offset model scores {fixed_own['mean']:.1%} "
        f"[{fixed_own['low']:.1%}, {fixed_own['high']:.1%}] on its own task and "
        f"{fixed_held['mean']:.1%} [{fixed_held['low']:.1%}, {fixed_held['high']:.1%}] "
        f"when the repeat offset moves. "
        + (
            "That collapse is the confound made visible: its score on the fixed task does "
            "not license any claim about content-based induction."
            if fixed_held["high"] < ceiling_held
            else "It retains some accuracy off its training offset, so its solution is not "
            "purely positional."
        )
    )

    # 4. Where the behavior lives, per model and per layer.
    for index, label in enumerate(SERIES):
        ablation = facts[f"{label}_ablation"]
        baseline = ablation["baseline"]["mean"]
        pair_name, pair = ablation["worst_pair"]
        clauses = []
        for layer in sorted(ablation["per_layer"]):
            layer_facts = ablation["per_layer"][layer]
            head_name, head = layer_facts["worst_single_head"]
            whole = layer_facts["sublayer"]["mean"]
            if whole < 0.02:
                verdictum = "the layer is dispensable"
            elif head["mean"] > 0.6 * whole:
                verdictum = f"concentrated in {head_name.replace('head ', '')}"
            else:
                verdictum = "redundant: no single head matters, the sublayer does"
            clauses.append(
                f"L{layer}: worst head -{head['mean']:.1%}, whole sublayer -{whole:.1%} "
                f"({verdictum})"
            )
        lines.append(
            f"4{'ab'[index]}. {label} model, zero-ablation from a {baseline:.1%} baseline. "
            + "; ".join(clauses)
            + f". Worst pair overall: {pair_name} at -{pair['mean']:.1%}."
        )

    # 5. What the induction metric says about all of this.
    for index, label in enumerate(SERIES):
        library = np.array([run.induction[label] for run in seeds]).mean(axis=0)
        span = np.array([run.span_induction[label] for run in seeds]).mean(axis=0)
        match = np.array([run.span_match[label] for run in seeds]).mean(axis=0)
        best = np.unravel_index(int(library.argmax()), library.shape)
        per_layer = ", ".join(
            f"L{layer} {library[layer].max():.2f}" for layer in range(library.shape[0])
        )
        lines.append(
            f"5{'ab'[index]}. {label} model, induction_head_score: highest head is "
            f"L{best[0]}H{best[1]} at {library[best]:.2f} (best per layer: {per_layer}). "
            "Inside a real repeated block, the best head per layer puts "
            + ", ".join(f"L{layer} {span[layer].max():.2f}" for layer in range(span.shape[0]))
            + " on the induction target and "
            + ", ".join(f"L{layer} {match[layer].max():.2f}" for layer in range(match.shape[0]))
            + " on the earlier occurrence of the query's own token."
        )

    return lines, facts


# --------------------------------------------------------------------------
# Bundle and entry point
# --------------------------------------------------------------------------


def write_bundle(
    root: Path,
    *,
    args: argparse.Namespace,
    seeds: Sequence[SeedResult],
    base: dict[str, dict[str, float]],
    lines: Sequence[str],
    facts: dict[str, Any],
    figures: Sequence[Path],
    determinism: Any,
) -> ArtifactBundle:
    """Save the whole run as a verifiable, replayable artifact bundle."""
    bundle = ArtifactBundle.create(root, overwrite=True)
    bundle.add_config(
        {
            "script": "examples/induction_controls.py",
            "args": vars(args),
            "tasks": {
                "fixed": FIXED_TASK.to_dict(),
                "variable_train": VARIABLE_TASK.to_dict(),
                "variable_held_out": HELD_OUT_TASK.to_dict(),
            },
        }
    )
    bundle.add_determinism(determinism)

    condition_names = list(seeds[0].ablations["fixed-offset"])
    offsets = sorted(int(k) for k in seeds[0].offset_profile["fixed-offset"])
    for label in SERIES:
        key = label.replace("-", "_")
        bundle.add_tensor(
            f"{key}.learning_curve",
            torch.tensor([[p["accuracy"] for p in run.curves[label]] for run in seeds]),
        )
        bundle.add_tensor(
            f"{key}.learning_curve_steps",
            torch.tensor([p["step"] for p in seeds[0].curves[label]], dtype=torch.int64),
        )
        bundle.add_tensor(
            f"{key}.induction_scores",
            torch.tensor([run.induction[label] for run in seeds]),
        )
        bundle.add_tensor(
            f"{key}.span_induction_scores",
            torch.tensor([run.span_induction[label] for run in seeds]),
        )
        bundle.add_tensor(
            f"{key}.span_match_scores",
            torch.tensor([run.span_match[label] for run in seeds]),
        )
        bundle.add_tensor(
            f"{key}.ablation_accuracy",
            torch.tensor([[run.ablations[label][n] for n in condition_names] for run in seeds]),
        )
        bundle.add_tensor(
            f"{key}.offset_profile",
            torch.tensor([[run.offset_profile[label][str(o)] for o in offsets] for run in seeds]),
        )

    accuracy_keys = sorted(seeds[0].accuracy)
    bundle.add_metrics(
        {
            "seeds": [run.seed for run in seeds],
            "baselines": base,
            "accuracy": summarize([run.accuracy for run in seeds], accuracy_keys),
            "ablation_conditions": condition_names,
            "offset_profile_offsets": offsets,
            "ablation_accuracy": {
                label: summarize([run.ablations[label] for run in seeds], condition_names)
                for label in SERIES
            },
            "verdict_facts": facts,
            "figures": [str(path) for path in figures],
        }
    )

    for line in lines:
        bundle.add_note(line)
    bundle.add_note(
        "Scope: a 2-layer, 4-head GPT-2 trained from scratch on a synthetic copy task. "
        "These results are about that model, not about induction heads in pretrained "
        "language models."
    )
    bundle.add_note(
        f"Confidence intervals are percentile bootstrap over {len(seeds)} training seeds. "
        "With that few seeds the interval is coarse and should be read as a spread across "
        "runs, not a population estimate."
    )
    return bundle.write()


def _worker_setup(threads: int) -> None:
    """Keep each worker single-threaded; the parallelism is across seeds."""
    torch.set_num_threads(threads)


def _run_one(job: tuple[int, argparse.Namespace, str]) -> SeedResult:
    """Module-level entry point so the pool can pickle it."""
    seed, args, device = job
    return run_seed(seed, args, torch.device(device))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps-fixed", type=int, default=3000)
    parser.add_argument("--steps-variable", type=int, default=24000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--eval-sequences", type=int, default=1024)
    parser.add_argument("--profile-sequences", type=int, default=512)
    parser.add_argument("--curve-points", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; seeds are 0..n-1.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/induction-controls"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--threads", type=int, default=0, help="0 leaves torch's default.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Train this many seeds concurrently, in separate processes. These models "
        "are small enough that one thread per seed parallelizes far better than many "
        "threads on one seed. Ignored on CUDA, where the seeds share one device.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Tiny budget, for checking the plumbing."
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.steps_fixed, args.steps_variable = 2, 120, 200
        args.eval_sequences, args.profile_sequences, args.curve_points = 256, 256, 4
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    determinism = set_seed(args.seed)

    print(f"device: {device}   seeds: {args.seeds}   "
          f"steps: {args.steps_fixed} fixed / {args.steps_variable} variable", flush=True)

    base = baselines(args)
    for name, values in base.items():
        print(f"  baseline [{name:>16}]  chance {values['chance']:.2%}   "
              f"best content-blind copier {values['content_blind_ceiling']:.2%} "
              f"(offset {int(values['content_blind_best_offset'])})", flush=True)

    started = time.time()
    numbers = list(range(args.seed, args.seed + args.seeds))
    workers = 1 if device.type == "cuda" else max(1, min(args.workers, len(numbers)))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_setup, initargs=(1,)
        ) as pool:
            seeds = list(pool.map(_run_one, [(seed, args, str(device)) for seed in numbers]))
    else:
        seeds = [run_seed(seed, args, device) for seed in numbers]
    print(f"\ntraining and analysis finished in {(time.time() - started) / 60:.1f} min",
          flush=True)

    lines, facts = verdict(seeds, base)
    figures = make_figures(seeds, base, args.assets)
    bundle = write_bundle(
        args.output, args=args, seeds=seeds, base=base, lines=lines, facts=facts,
        figures=figures, determinism=determinism,
    )

    print("\n" + "=" * 78)
    print("POSITIONAL CONFOUND: PLAIN-LANGUAGE SUMMARY")
    print("=" * 78)
    for line in lines:
        print("\n" + line)
    print("\n" + "-" * 78)
    print(f"bundle:      {bundle.root}  (fingerprint {bundle.fingerprint[:12]})")
    print(f"verified:    {bundle.verify().summary()}")
    for path in figures:
        print(f"figure:      {path}")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
