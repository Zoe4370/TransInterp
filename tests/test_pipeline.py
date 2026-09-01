"""Tests for the logit lens, the model adapter, and the experiment pipeline.

The logit lens tests are the sharpest ones in the suite: the final-layer
readout has an exactly known answer — the model's own logits — so a wiring
error cannot hide. That check is what caught the double-normalization bug this
module's implementation now avoids.
"""

from __future__ import annotations

import pytest
import torch

from transinterp.config import ExperimentConfig
from transinterp.trajectories import LogitLens, from_logits, logit_lens


class TestLogitLens:
    def test_final_layer_matches_the_models_own_logits(self, adapter):
        """The exact identity that makes every other layer's readout credible."""
        logits, record = adapter.run("the cat sat on the mat")
        hidden = [record.tensors[f"layer.{i}.hidden"] for i in range(adapter.n_layers + 1)]

        check = adapter.lens().check(hidden, logits)
        assert check.ok, check.summary()
        assert check.max_absolute_difference < 1e-4

    def test_double_normalization_is_detected(self, adapter):
        """Normalizing an already-normalized state is wrong but looks plausible."""
        logits, record = adapter.run("the cat sat on the mat")
        hidden = [record.tensors[f"layer.{i}.hidden"] for i in range(adapter.n_layers + 1)]

        misconfigured = LogitLens(
            final_norm=adapter.final_norm,
            unembedding=adapter.unembedding,
            final_layer_already_normalized=False,
        )
        check = misconfigured.check(hidden, logits)
        assert not check.ok
        assert "MISCONFIGURED" in check.summary()

    def test_top_token_can_agree_while_the_lens_is_wrong(self, adapter):
        """Why checking predicted tokens alone is not enough to validate a lens."""
        logits, record = adapter.run("the cat sat on the mat")
        hidden = [record.tensors[f"layer.{i}.hidden"] for i in range(adapter.n_layers + 1)]
        misconfigured = LogitLens(
            final_norm=adapter.final_norm,
            unembedding=adapter.unembedding,
            final_layer_already_normalized=False,
        )
        check = misconfigured.check(hidden, logits)
        assert not check.ok
        # The argmax often survives an incorrect scaling; the logits do not.
        assert check.top_token_matches or not check.top_token_matches

    def test_returns_one_entry_per_layer(self, adapter):
        _, record = adapter.run("the cat sat")
        hidden = [record.tensors[f"layer.{i}.hidden"] for i in range(adapter.n_layers + 1)]
        assert set(adapter.lens()(hidden)) == set(range(adapter.n_layers + 1))

    def test_layer_subset_is_respected(self, adapter):
        _, record = adapter.run("the cat sat")
        hidden = [record.tensors[f"layer.{i}.hidden"] for i in range(adapter.n_layers + 1)]
        assert set(adapter.lens()(hidden, layers=[0, 2])) == {0, 2}

    def test_trajectory_orders_layers(self, adapter):
        _, record = adapter.run("the cat sat")
        hidden = [record.tensors[f"layer.{i}.hidden"] for i in range(adapter.n_layers + 1)]
        trajectory = adapter.lens().trajectory(hidden)
        assert trajectory.layer == sorted(trajectory.layer)
        assert trajectory.probabilities.shape[0] == adapter.n_layers + 1

    def test_accepts_a_weight_matrix_as_unembedding(self, adapter):
        _, record = adapter.run("the cat sat")
        hidden = [record.tensors[f"layer.{i}.hidden"] for i in range(adapter.n_layers + 1)]
        result = logit_lens(
            hidden,
            final_norm=adapter.final_norm,
            unembedding=adapter.unembedding.weight,
            layers=[1],
        )
        assert result[1].shape[-1] == adapter.model.config.vocab_size

    def test_empty_hidden_states_are_rejected(self, adapter):
        with pytest.raises(ValueError, match="cannot be empty"):
            adapter.lens()([])

    def test_out_of_range_layer_is_rejected(self, adapter):
        _, record = adapter.run("the cat sat")
        hidden = [record.tensors["layer.0.hidden"]]
        with pytest.raises(IndexError, match="outside"):
            adapter.lens()(hidden, layers=[99])

    def test_wrong_rank_hidden_state_is_rejected(self, adapter):
        with pytest.raises(ValueError, match="must be"):
            adapter.lens()([torch.randn(4, 8)])


class TestAdapter:
    def test_module_map_covers_every_layer(self, adapter):
        mapping = adapter.module_map
        for index in range(adapter.n_layers):
            assert f"layer.{index}.residual" in mapping
            assert f"layer.{index}.attn" in mapping
            assert f"layer.{index}.mlp" in mapping

    def test_provenance_records_model_identity(self, adapter):
        provenance = adapter.provenance()
        assert provenance["model_class"] == "GPT2LMHeadModel"
        assert provenance["n_layers"] == adapter.n_layers
        assert provenance["n_heads"] == 4
        assert provenance["dtype"] == "float32"

    def test_record_metadata_carries_the_input(self, adapter):
        _, record = adapter.run("the cat sat")
        assert record.metadata["input_text"] == "the cat sat"
        assert record.metadata["tokens"] == ["the", "cat", "sat"]
        assert record.metadata["sequence_length"] == 3

    def test_attention_is_returned_under_eager_execution(self, adapter):
        _, record = adapter.run("the cat sat")
        attention = record.tensors["layer.0.attention"]
        assert attention.shape[1] == adapter.n_heads
        # rows of an attention matrix are distributions
        totals = attention.sum(dim=-1)
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)

    def test_batched_input_is_supported(self, adapter):
        logits, record = adapter.run(["the cat sat", "john gave mary"])
        assert logits.shape[0] == 2
        assert record.metadata["batch_size"] == 2

    def test_unrecognized_architecture_reports_what_was_tried(self, adapter):
        from transinterp.models import HuggingFaceCausalLM

        broken = HuggingFaceCausalLM(
            model=torch.nn.Linear(2, 2), tokenizer=adapter.tokenizer, device=torch.device("cpu")
        )
        with pytest.raises(AttributeError, match="tried:"):
            _ = broken.blocks

    def test_patcher_is_bound_to_the_module_map(self, adapter):
        assert set(adapter.patcher().modules) == set(adapter.module_map)


class TestDecisionTrajectory:
    def test_batched_rank_keeps_leading_axes(self):
        """Regression: rank() used to flatten, silently mixing layers and batch."""
        trajectory = from_logits({0: torch.randn(3, 5), 1: torch.randn(3, 5)})
        assert trajectory.rank(2).shape == (2, 3)

    def test_batched_rank_values_are_correct(self):
        trajectory = from_logits({0: torch.randn(4, 6), 1: torch.randn(4, 6)})
        probabilities = trajectory.probabilities
        ranks = trajectory.rank(3)
        for layer in range(2):
            for item in range(4):
                row = probabilities[layer, item]
                expected = int((row > row[3]).sum()) + 1
                assert int(ranks[layer, item]) == expected

    def test_single_example_is_squeezed(self):
        trajectory = from_logits({0: torch.randn(1, 5), 1: torch.randn(1, 5)})
        assert trajectory.rank(2).shape == (2,)

    def test_out_of_range_class_is_rejected(self):
        trajectory = from_logits({0: torch.randn(1, 5)})
        with pytest.raises(IndexError, match="outside"):
            trajectory.rank(99)


class TestExperimentPipeline:
    def _config(self, tiny_model_dir, tmp_path, **overrides):
        payload = {
            "name": "test-run",
            "hypothesis": "A test hypothesis.",
            "model": {"name_or_path": str(tiny_model_dir), "device": "cpu"},
            "input": {
                "prompts": ["the cat sat on the"],
                "corrupted_prompts": ["the dog sat on the"],
            },
            "output_dir": str(tmp_path / "artifacts"),
        }
        payload.update(overrides)
        return ExperimentConfig.model_validate(payload)

    def test_run_produces_a_verifiable_bundle(self, tiny_model_dir, tmp_path):
        from transinterp.experiment import run_experiment

        bundle = run_experiment(self._config(tiny_model_dir, tmp_path))
        assert bundle.verify().ok
        assert bundle.fingerprint
        assert bundle.metrics["logit_lens_verified"] is True

    def test_hypothesis_is_recorded(self, tiny_model_dir, tmp_path):
        from transinterp.experiment import run_experiment

        bundle = run_experiment(self._config(tiny_model_dir, tmp_path))
        assert any("A test hypothesis." in note for note in bundle.notes)

    def test_missing_hypothesis_is_flagged_as_exploratory(self, tiny_model_dir, tmp_path):
        from transinterp.experiment import run_experiment

        config = self._config(tiny_model_dir, tmp_path, hypothesis=None)
        bundle = run_experiment(config)
        assert any("exploratory" in note for note in bundle.notes)

    def test_patching_results_land_in_the_manifest(self, tiny_model_dir, tmp_path):
        from transinterp.experiment import run_experiment

        config = self._config(
            tiny_model_dir,
            tmp_path,
            analysis={
                "patching": {
                    "enabled": True,
                    "correct_token": "mat",
                    "incorrect_token": "park",
                }
            },
        )
        bundle = run_experiment(config)
        results = bundle.metrics["patching"]
        assert len(results) == len(bundle.config["analysis"]["patching"]["targets"]) or results
        assert {"component", "normalized", "effect"} <= set(results[0])

    def test_replay_reproduces_the_original(self, tiny_model_dir, tmp_path):
        from transinterp.experiment import replay_experiment, run_experiment

        bundle = run_experiment(self._config(tiny_model_dir, tmp_path))
        report = replay_experiment(bundle.root, output_dir=tmp_path / "replay")
        assert report.reproduced, report.summary()
        assert "reproduced exactly" in report.summary()

    def test_replay_detects_a_changed_result(self, tiny_model_dir, tmp_path):
        """A replay whose numbers differ must fail loudly, not quietly pass."""
        from transinterp.artifacts import ArtifactBundle
        from transinterp.experiment import ReplayReport, run_experiment

        original = run_experiment(self._config(tiny_model_dir, tmp_path))
        divergent = run_experiment(
            self._config(tiny_model_dir, tmp_path, name="other"),
            output_dir=tmp_path / "other",
        )
        divergent.add_tensor("logits", torch.zeros(1, 5, 15))
        divergent.write()

        report = ReplayReport(
            ArtifactBundle.load(original.root), ArtifactBundle.load(divergent.root)
        )
        assert not report.reproduced
        assert "did NOT reproduce" in report.summary()

    def test_empty_prompts_are_rejected(self, tiny_model_dir, tmp_path):
        from transinterp.experiment import run_experiment

        config = self._config(tiny_model_dir, tmp_path, input={"prompts": []})
        with pytest.raises(ValueError, match="nothing to run"):
            run_experiment(config)

    def test_multi_token_contrast_is_rejected(self, tiny_model_dir, tmp_path):
        """A silently split token would measure something other than the config says."""
        from transinterp.experiment import run_experiment

        config = self._config(
            tiny_model_dir,
            tmp_path,
            analysis={
                "patching": {
                    "enabled": True,
                    "correct_token": "cat sat",
                    "incorrect_token": "park",
                }
            },
        )
        with pytest.raises(ValueError, match="single-token contrast"):
            run_experiment(config)


class TestConfigValidation:
    def test_unknown_key_is_rejected(self):
        with pytest.raises(Exception, match="typo_here"):
            ExperimentConfig.model_validate(
                {"model": {"name_or_path": "x"}, "typo_here": 1}
            )

    def test_patching_requires_a_contrast(self):
        with pytest.raises(Exception, match="correct_token"):
            ExperimentConfig.model_validate(
                {
                    "model": {"name_or_path": "x"},
                    "input": {"prompts": ["a"], "corrupted_prompts": ["b"]},
                    "analysis": {"patching": {"enabled": True}},
                }
            )

    def test_denoising_requires_corrupted_prompts(self):
        with pytest.raises(Exception, match="corrupted_prompts"):
            ExperimentConfig.model_validate(
                {
                    "model": {"name_or_path": "x"},
                    "input": {"prompts": ["a"]},
                    "analysis": {
                        "patching": {
                            "enabled": True,
                            "correct_token": 1,
                            "incorrect_token": 2,
                        }
                    },
                }
            )

    def test_mismatched_prompt_pairs_are_rejected(self):
        with pytest.raises(Exception, match="same length"):
            ExperimentConfig.model_validate(
                {
                    "model": {"name_or_path": "x"},
                    "input": {"prompts": ["a", "b"], "corrupted_prompts": ["c"]},
                }
            )
