"""End-to-end reproducibility demo. Downloads nothing; runs in seconds.

Builds a tiny randomly-initialized model, runs a patching experiment against
it, writes an artifact bundle, verifies the bundle, tampers with it to show
verification failing, and replays the experiment to show it reproducing.

The model is random, so the *numbers* mean nothing. What the example
demonstrates is the machinery: that a result can be sealed, checked, and
re-run by someone who has only the artifact.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from transinterp.artifacts import ArtifactBundle
from transinterp.config import ExperimentConfig
from transinterp.experiment import replay_experiment, run_experiment

VOCAB = [
    "<unk>", "<eos>", "the", "cat", "sat", "on", "mat",
    "dog", "ran", "to", "park", "john", "mary", "gave", "book",
]


def build_tiny_model(directory: Path) -> Path:
    """Create a small GPT-2 and word-level tokenizer on disk."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<eos>",
        pad_token="<eos>",
    ).save_pretrained(str(directory))

    torch.manual_seed(0)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(VOCAB), n_positions=64, n_embd=32, n_layer=3, n_head=4,
            bos_token_id=1, eos_token_id=1, attn_implementation="eager",
        )
    ).eval().save_pretrained(str(directory))
    return directory


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="transinterp-demo-"))
    model_dir = build_tiny_model(workspace / "model")

    config = ExperimentConfig.model_validate(
        {
            "name": "demo",
            "hypothesis": (
                "Layer 1 attention carries the subject-token information. "
                "Falsified if patching it restores under 20% of the clean logit difference."
            ),
            "model": {"name_or_path": str(model_dir), "device": "cpu"},
            "input": {
                "prompts": ["the cat sat on the"],
                "corrupted_prompts": ["the dog sat on the"],
            },
            "analysis": {
                "patching": {
                    "enabled": True,
                    "correct_token": "mat",
                    "incorrect_token": "park",
                }
            },
            "output_dir": str(workspace / "artifacts"),
        }
    )

    print("1. Running experiment")
    bundle = run_experiment(config, overwrite=True)
    print(f"   wrote {bundle.root}")
    print(f"   fingerprint {bundle.fingerprint[:16]}")
    print(f"   logit lens verified: {bundle.metrics['logit_lens_verified']}")

    print("\n2. Top patching effects (random weights, so values are meaningless)")
    for result in bundle.metrics["patching"][:5]:
        print(
            f"   {result['component']:22} "
            f"effect={result['effect']:+.4f}  restored={result['normalized']:+.1%}"
        )

    print("\n3. Verifying integrity")
    print(f"   {ArtifactBundle.load(bundle.root).verify().summary()}")

    print("\n4. Tampering with metrics.json")
    metrics_path = bundle.root / "metrics.json"
    payload = json.loads(metrics_path.read_text())
    payload["n_prompts"] = 999
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    result = ArtifactBundle.load(bundle.root).verify()
    print(f"   {result.summary()}")
    print(f"   flagged: {result.corrupted}")

    print("\n5. Restoring and replaying")
    bundle.write()
    report = replay_experiment(bundle.root, output_dir=workspace / "replay")
    print(f"   {report.summary()}")

    print("\n6. What a reader learns from the bundle alone")
    reloaded = ArtifactBundle.load(bundle.root)
    provenance = reloaded.provenance
    print(f"   torch {provenance.packages['torch']}, python {provenance.python_version}")
    print(f"   stored config re-validates: {ExperimentConfig.model_validate(reloaded.config).name}")
    for note in reloaded.notes:
        print(f"   note: {note}")

    print(f"\nBundle left at {bundle.root}")


if __name__ == "__main__":
    main()
