from types import SimpleNamespace

import torch

from transinterp.models import HuggingFaceCausalLM


class FakeTokenizer:
    """A stand-in that accepts the same call signature as a real tokenizer.

    The adapter now passes ``padding`` so batched prompts work, so the fake
    has to tolerate it. Accepting ``**kwargs`` keeps this test from breaking
    again the next time the adapter forwards a new tokenizer option.
    """

    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, return_tensors="pt", **kwargs):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}

    def convert_ids_to_tokens(self, ids):
        return [f"tok{i}" for i in ids]


class FakeModel:
    config = SimpleNamespace(
        num_attention_heads=2, vocab_size=5, architectures=["Fake"], _commit_hash=None
    )

    def parameters(self):
        yield torch.zeros(1)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        assert kwargs["output_hidden_states"] is True
        assert kwargs["output_attentions"] is True
        return SimpleNamespace(
            logits=torch.zeros(1, 3, 5),
            hidden_states=(torch.ones(1, 3, 4), torch.ones(1, 3, 4) * 2),
            attentions=(torch.ones(1, 2, 3, 3) / 3,),
        )


def test_huggingface_adapter_normalizes_common_outputs():
    adapter = HuggingFaceCausalLM(FakeModel(), FakeTokenizer(), torch.device("cpu"))
    logits, record = adapter.run("hello")
    assert logits.shape == (1, 3, 5)
    assert record.tensors["layer.0.hidden"].shape == (1, 3, 4)
    assert record.tensors["layer.1.hidden"].shape == (1, 3, 4)
    assert record.tensors["layer.0.attention"].shape == (1, 2, 3, 3)
    assert record.metadata["sequence_length"] == 3
    assert record.metadata["input_text"] == "hello"
    assert record.metadata["tokens"] == ["tok1", "tok2", "tok3"]


def test_provenance_degrades_when_architecture_is_unknown():
    """A run must not fail just because block discovery did not work.

    Regression guard: ``provenance()`` used to call ``n_layers`` unguarded,
    so any model whose blocks were not at a known path raised mid-run.
    """
    adapter = HuggingFaceCausalLM(FakeModel(), FakeTokenizer(), torch.device("cpu"))
    provenance = adapter.provenance()
    assert provenance["n_layers"] is None
    assert provenance["model_class"] == "FakeModel"
    assert provenance["n_heads"] == 2
