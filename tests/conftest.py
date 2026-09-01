"""Shared fixtures.

Every fixture here builds a model from scratch on disk. No test downloads
weights, so the suite runs offline, in CI, and on a plane. The models are
randomly initialized, which is fine because these tests check that the
machinery is correct, not that any particular model behaves a particular way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

TINY_VOCAB = [
    "<unk>",
    "<eos>",
    "the",
    "cat",
    "sat",
    "on",
    "mat",
    "dog",
    "ran",
    "to",
    "park",
    "john",
    "mary",
    "gave",
    "book",
]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal GPT-2 and word-level tokenizer saved to disk."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    directory = tmp_path_factory.mktemp("tiny-lm")

    vocab = {word: index for index, word in enumerate(TINY_VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<eos>",
        pad_token="<eos>",
    )
    tokenizer.save_pretrained(str(directory))

    config = GPT2Config(
        vocab_size=len(TINY_VOCAB),
        n_positions=64,
        n_embd=32,
        n_layer=3,
        n_head=4,
        bos_token_id=1,
        eos_token_id=1,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    GPT2LMHeadModel(config).eval().save_pretrained(str(directory))

    (directory / "vocab_list.json").write_text(json.dumps(TINY_VOCAB))
    return directory


@pytest.fixture
def adapter(tiny_model_dir: Path):
    """A loaded adapter over the tiny model."""
    from transinterp.models import HuggingFaceCausalLM

    return HuggingFaceCausalLM.from_pretrained(str(tiny_model_dir), device="cpu")


@pytest.fixture
def tiny_transformer() -> torch.nn.Module:
    """A bare GPT-2 with no tokenizer, for module-level tests."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).eval()
