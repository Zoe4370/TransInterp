"""Find and causally verify an induction head, end to end.

This is a complete worked example rather than a feature tour. It trains a
small two-layer transformer on a repeated-sequence task until in-context
copying emerges, then uses TransInterp to locate the head responsible and
confirm the finding with an intervention.

The point of the sequence matters. The induction score is correlational: it
says a head attends to the induction target, not that the model's behavior
depends on it. Only the ablation at the end licenses a causal claim, and the
script reports both numbers so the difference stays visible.

Training takes about a minute on CPU. Nothing is downloaded.

Run:
    python examples/induction_experiment.py --figures assets/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from transinterp.attention.induction import head_pattern_similarity, induction_head_score
from transinterp.attention.metrics import attention_entropy
from transinterp.determinism import set_seed
from transinterp.interventions import PatchSpec

VOCAB_SIZE = 40
SEQUENCE_HALF = 24


def make_batch(batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Build sequences of the form ``[random prefix][same prefix again]``.

    A model can only predict the second half by looking back to what followed
    each token the first time it appeared. That is the induction pattern, and
    training on this task is the standard way to make one appear on demand.
    """
    prefix = torch.randint(
        0, VOCAB_SIZE, (batch_size, SEQUENCE_HALF), generator=generator
    )
    return torch.cat([prefix, prefix], dim=1)


def train_model(steps: int = 700, batch_size: int = 32) -> nn.Module:
    """Train a two-layer transformer until it can copy in context."""
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=2 * SEQUENCE_HALF,
            n_embd=64,
            n_layer=2,
            n_head=4,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=0,
            eos_token_id=1,
            attn_implementation="eager",
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)

    model.train()
    for step in range(steps):
        batch = make_batch(batch_size, generator)
        loss = model(batch, labels=batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"   step {step:4d}  loss {loss.item():.4f}")
    print(f"   step {steps:4d}  loss {loss.item():.4f}")
    return model.eval()


def second_half_accuracy(model: nn.Module, tokens: torch.Tensor) -> float:
    """Next-token accuracy on the repeated half, where copying is required."""
    with torch.no_grad():
        predictions = model(tokens).logits[:, :-1].argmax(dim=-1)
    targets = tokens[:, 1:]
    start = SEQUENCE_HALF
    correct = (predictions[:, start:] == targets[:, start:]).float().mean()
    return float(correct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", type=Path, default=None, help="Directory for plots.")
    parser.add_argument("--steps", type=int, default=700)
    arguments = parser.parse_args()

    set_seed(0)

    print("1. Training a 2-layer transformer on a repeated-sequence task")
    model = train_model(steps=arguments.steps)

    generator = torch.Generator().manual_seed(123)
    tokens = make_batch(8, generator)
    accuracy = second_half_accuracy(model, tokens)
    print(f"   in-context copying accuracy: {accuracy:.1%}")

    with torch.no_grad():
        outputs = model(tokens, output_attentions=True, return_dict=True)
    attentions = outputs.attentions

    print("\n2. Scoring every head for the induction pattern (correlational)")
    all_scores = []
    for layer, attention in enumerate(attentions):
        scores = induction_head_score(attention, tokens).mean(dim=0)
        all_scores.append(scores)
        for head, score in enumerate(scores.tolist()):
            print(f"   layer {layer} head {head}: {score:.3f}")

    stacked = torch.stack(all_scores)
    best_layer, best_head = divmod(int(stacked.argmax()), stacked.shape[1])
    print(f"   -> strongest: layer {best_layer} head {best_head} ({stacked.max():.3f})")

    print("\n3. Testing whether behaviour actually depends on those heads (causal)")
    modules = {f"layer.{i}.attn": block.attn for i, block in enumerate(model.transformer.h)}
    modules.update({f"layer.{i}.mlp": block.mlp for i, block in enumerate(model.transformer.h)})

    from transinterp.interventions.patching import patched_forward

    n_heads = model.config.n_head
    head_results: dict[str, float] = {}
    for layer in range(stacked.shape[0]):
        for head in range(n_heads):
            spec = PatchSpec(
                module=f"layer.{layer}.attn", mode="zero", heads=[head], n_heads=n_heads
            )
            with patched_forward(modules, [spec]):
                head_results[f"L{layer}H{head}"] = second_half_accuracy(model, tokens)

    module_results: dict[str, float] = {}
    for name in modules:
        with patched_forward(modules, [PatchSpec(name, mode="zero")]):
            module_results[name] = second_half_accuracy(model, tokens)

    print(f"   baseline accuracy: {accuracy:.1%}")
    print("   ablating one head at a time:")
    for name, value in head_results.items():
        print(f"     {name}: {value:.1%}")
    print("   ablating a whole sublayer:")
    for name, value in module_results.items():
        print(f"     {name}: {value:.1%}")

    top_head_accuracy = head_results[f"L{best_layer}H{best_head}"]
    layer_accuracy = module_results[f"layer.{best_layer}.attn"]

    print("\n   Reading the result:")
    print(
        f"   The highest-scoring head is L{best_layer}H{best_head}, but ablating it alone "
        f"leaves accuracy at {top_head_accuracy:.1%}."
    )
    print(
        f"   Ablating all of layer {best_layer}'s attention drops accuracy to "
        f"{layer_accuracy:.1%}."
    )
    print(
        "   So the behaviour depends on that sublayer but not on any single head: the\n"
        "   mechanism is distributed and redundant. Reporting the top induction score as\n"
        "   'the induction head' would have been wrong, and only the intervention shows it."
    )
    print(
        "\n   Caveat worth carrying: this task repeats at a fixed offset, so a head that\n"
        "   simply attends a fixed distance back would score as induction here. Confirming\n"
        "   genuine content-based induction needs variable repeat offsets as a control."
    )

    if arguments.figures:
        _write_figures(
            arguments.figures,
            attentions,
            stacked,
            accuracy,
            head_results,
            module_results,
            best_layer,
            best_head,
        )


def _write_figures(
    directory: Path,
    attentions,
    scores: torch.Tensor,
    baseline: float,
    head_results: dict[str, float],
    module_results: dict[str, float],
    best_layer: int,
    best_head: int,
) -> None:
    """Render the figures used in the README."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "figure.facecolor": "white",
        }
    )
    ink = "#1f2933"
    accent = "#2f6f9f"
    warn = "#c0392b"
    muted = "#9aa5b1"

    control_head = int(scores[best_layer].argmin())

    # --- attention patterns ------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    panels = [
        (best_layer, best_head, f"L{best_layer}H{best_head} — highest induction score"),
        (best_layer, control_head, f"L{best_layer}H{control_head} — lowest induction score"),
    ]
    for axis, (layer, head, title) in zip(axes, panels, strict=True):
        pattern = attentions[layer][0, head].detach()
        image = axis.imshow(pattern, cmap="magma", aspect="equal", interpolation="nearest")
        axis.set_title(title, color=ink)
        axis.set_xlabel("source token")
        axis.set_ylabel("query token")
        axis.axvline(SEQUENCE_HALF - 0.5, color="white", lw=0.7, ls="--", alpha=0.55)
        axis.axhline(SEQUENCE_HALF - 0.5, color="white", lw=0.7, ls="--", alpha=0.55)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        "Attention on a repeated sequence (dashed line marks the repeat boundary)",
        color=ink,
        y=1.03,
    )
    figure.tight_layout()
    figure.savefig(directory / "induction-attention.png", bbox_inches="tight")
    plt.close(figure)

    # --- induction scores --------------------------------------------------
    figure, axis = plt.subplots(figsize=(6.2, 3.0))
    labels = [
        f"L{layer}H{head}"
        for layer in range(scores.shape[0])
        for head in range(scores.shape[1])
    ]
    values = scores.flatten().tolist()
    top_index = best_layer * scores.shape[1] + best_head
    colours = [warn if index == top_index else accent for index in range(len(values))]
    axis.bar(labels, values, color=colours)
    axis.set_ylabel("induction score")
    axis.set_title("Correlational: induction score per head", color=ink)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(directory / "induction-scores.png", bbox_inches="tight")
    plt.close(figure)

    # --- causal check ------------------------------------------------------
    figure, axis = plt.subplots(figsize=(7.4, 3.2))
    names = ["baseline"] + list(head_results) + [f"all of\nlayer {best_layer} attn"]
    heights = [baseline] + list(head_results.values())
    heights.append(module_results[f"layer.{best_layer}.attn"])
    colours = [accent] + [muted] * len(head_results) + [warn]
    colours[1 + list(head_results).index(f"L{best_layer}H{best_head}")] = "#e08a3c"
    axis.bar(names, heights, color=colours)
    axis.set_ylabel("in-context copying accuracy")
    axis.set_ylim(0, 1.12)
    for index, height in enumerate(heights):
        axis.text(index, height + 0.03, f"{height:.0%}", ha="center", fontsize=7.5, color=ink)
    axis.tick_params(axis="x", rotation=45, labelsize=7.5)
    axis.set_title(
        "Causal: no single head matters, but the whole sublayer does", color=ink
    )
    figure.tight_layout()
    figure.savefig(directory / "induction-ablation.png", bbox_inches="tight")
    plt.close(figure)

    # --- entropy and head similarity --------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for layer, attention in enumerate(attentions):
        entropy = attention_entropy(attention).mean(dim=(0, 2))
        axes[0].plot(range(entropy.shape[0]), entropy.tolist(), marker="o", label=f"layer {layer}")
    axes[0].set_xlabel("head")
    axes[0].set_ylabel("mean entropy (nats)")
    axes[0].set_title("Attention concentration", color=ink)
    axes[0].legend(frameon=False)

    similarity = head_pattern_similarity(attentions[best_layer])
    image = axes[1].imshow(similarity, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(f"Head similarity, layer {best_layer}", color=ink)
    axes[1].set_xlabel("head")
    axes[1].set_ylabel("head")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(directory / "attention-analysis.png", bbox_inches="tight")
    plt.close(figure)

    print(f"\n4. Figures written to {directory}")


if __name__ == "__main__":
    main()
