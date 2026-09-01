"""Tests for causal interventions.

The important tests here are the ones with a known correct answer. Patching a
component's own cached output back into the same run must be a no-op; patching
the final block must fully restore clean behavior. Both are exact identities,
so they catch hook-plumbing errors that a "the number moved" assertion would
miss entirely.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from transinterp.interventions import (
    ActivationPatcher,
    PatchSpec,
    kl_divergence,
    logit_difference,
    normalized_effect,
    patched_forward,
    target_probability,
)


def module_map(model: nn.Module) -> dict[str, nn.Module]:
    mapping: dict[str, nn.Module] = {}
    for index, block in enumerate(model.transformer.h):
        mapping[f"layer.{index}.residual"] = block
        mapping[f"layer.{index}.attn"] = block.attn
        mapping[f"layer.{index}.mlp"] = block.mlp
    return mapping


class TestPatchSpec:
    def test_zero_mode_writes_zeros(self):
        spec = PatchSpec("m", mode="zero")
        assert torch.equal(spec.apply(torch.ones(1, 4, 8)), torch.zeros(1, 4, 8))

    def test_replace_requires_a_source(self):
        with pytest.raises(ValueError, match="requires a source"):
            PatchSpec("m", mode="replace")

    def test_heads_require_head_count(self):
        with pytest.raises(ValueError, match="requires n_heads"):
            PatchSpec("m", mode="zero", heads=[0])

    def test_positions_restrict_the_patch(self):
        activation = torch.ones(1, 4, 8)
        patched = PatchSpec("m", mode="zero", positions=[1]).apply(activation)
        assert patched[0, 1].abs().sum() == 0
        assert patched[0, 0].sum() == 8
        assert patched[0, 2].sum() == 8

    def test_negative_positions_index_from_the_end(self):
        patched = PatchSpec("m", mode="zero", positions=[-1]).apply(torch.ones(1, 4, 8))
        assert patched[0, 3].abs().sum() == 0
        assert patched[0, 0].sum() == 8

    def test_heads_restrict_the_feature_axis(self):
        activation = torch.ones(1, 2, 8)
        patched = PatchSpec("m", mode="zero", heads=[1], n_heads=4).apply(activation)
        # head 1 occupies features 2:4 when 8 features are split across 4 heads
        assert patched[0, 0, 2:4].abs().sum() == 0
        assert patched[0, 0, 0:2].sum() == 2
        assert patched[0, 0, 4:8].sum() == 4

    def test_scale_multiplies(self):
        patched = PatchSpec("m", mode="scale", factor=0.5).apply(torch.ones(1, 2, 4))
        assert torch.allclose(patched, torch.full((1, 2, 4), 0.5))

    def test_mean_stays_in_the_activation_range(self):
        activation = torch.tensor([[[0.0, 10.0], [2.0, 20.0]]])
        patched = PatchSpec("m", mode="mean").apply(activation)
        assert torch.allclose(patched, torch.tensor([[[1.0, 15.0], [1.0, 15.0]]]))

    def test_incompatible_source_shape_is_reported(self):
        spec = PatchSpec("m", mode="replace", source=torch.ones(3, 3, 3))
        with pytest.raises(ValueError, match="cannot be broadcast"):
            spec.apply(torch.ones(1, 4, 8))

    def test_unknown_module_name_is_rejected(self, tiny_transformer):
        modules = module_map(tiny_transformer)
        with (
            pytest.raises(KeyError, match="not in the module map"),
            patched_forward(modules, [PatchSpec("nope", mode="zero")]),
        ):
            pass


class TestActivationPatcher:
    def test_cache_returns_one_tensor_per_module(self, tiny_transformer):
        ids = torch.randint(0, 32, (1, 6))
        patcher = ActivationPatcher(module_map(tiny_transformer))
        cache = patcher.cache(lambda: tiny_transformer(ids))
        assert set(cache) == set(patcher.modules)
        assert all(tensor.ndim >= 2 for tensor in cache.values())

    def test_patching_a_module_with_its_own_output_is_a_no_op(self, tiny_transformer):
        """Self-patching must not change the result. If it does, hooks are wrong."""
        ids = torch.randint(0, 32, (1, 6))
        patcher = ActivationPatcher(module_map(tiny_transformer))
        baseline = tiny_transformer(ids).logits
        cache = patcher.cache(lambda: tiny_transformer(ids))

        for name, activation in cache.items():
            spec = PatchSpec(name, mode="replace", source=activation)
            patched = patcher.run_with_patches(lambda: tiny_transformer(ids), [spec]).logits
            assert torch.allclose(baseline, patched, atol=1e-5), f"self-patch changed {name}"

    def test_patching_final_block_fully_restores_clean_behavior(self, tiny_transformer):
        """Everything downstream of the last block is determined by its output."""
        clean_ids = torch.randint(0, 32, (1, 6))
        corrupted_ids = clean_ids.clone()
        corrupted_ids[0, 2] = (corrupted_ids[0, 2] + 5) % 32

        patcher = ActivationPatcher(module_map(tiny_transformer))
        results = patcher.scan(
            clean_run=lambda: tiny_transformer(clean_ids),
            corrupted_run=lambda: tiny_transformer(corrupted_ids),
            metric=lambda out: logit_difference(out.logits, 3, 7),
            targets=["layer.1.residual"],
        )
        assert results[0].normalized == pytest.approx(1.0, abs=1e-3)

    def test_zeroing_a_component_changes_behavior(self, tiny_transformer):
        ids = torch.randint(0, 32, (1, 6))
        patcher = ActivationPatcher(module_map(tiny_transformer))
        results = patcher.scan(
            clean_run=lambda: tiny_transformer(ids),
            corrupted_run=lambda: tiny_transformer(ids),
            metric=lambda out: logit_difference(out.logits, 3, 7),
            targets=["layer.0.mlp"],
            mode="zero",
        )
        assert results[0].metadata["protocol"] == "noising"
        assert results[0].absolute_effect > 0

    def test_hooks_are_removed_after_patching(self, tiny_transformer):
        ids = torch.randint(0, 32, (1, 6))
        modules = module_map(tiny_transformer)
        patcher = ActivationPatcher(modules)
        patcher.run_with_patches(
            lambda: tiny_transformer(ids), [PatchSpec("layer.0.mlp", mode="zero")]
        )
        assert all(not module._forward_hooks for module in modules.values())

    def test_hooks_are_removed_even_when_the_run_raises(self, tiny_transformer):
        modules = module_map(tiny_transformer)
        patcher = ActivationPatcher(modules)

        def explode():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            patcher.run_with_patches(explode, [PatchSpec("layer.0.mlp", mode="zero")])
        assert all(not module._forward_hooks for module in modules.values())

    def test_results_are_sorted_by_absolute_effect(self, tiny_transformer):
        clean_ids = torch.randint(0, 32, (1, 6))
        corrupted_ids = clean_ids.clone()
        corrupted_ids[0, 1] = (corrupted_ids[0, 1] + 3) % 32
        patcher = ActivationPatcher(module_map(tiny_transformer))
        results = patcher.scan(
            clean_run=lambda: tiny_transformer(clean_ids),
            corrupted_run=lambda: tiny_transformer(corrupted_ids),
            metric=lambda out: logit_difference(out.logits, 3, 7),
        )
        effects = [result.absolute_effect for result in results]
        assert effects == sorted(effects, reverse=True)

    def test_unknown_target_is_rejected(self, tiny_transformer):
        patcher = ActivationPatcher(module_map(tiny_transformer))
        with pytest.raises(KeyError, match="unknown components"):
            patcher.scan(
                clean_run=lambda: None,
                corrupted_run=lambda: None,
                metric=lambda _: 0.0,
                targets=["layer.99.mlp"],
            )

    def test_empty_module_map_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ActivationPatcher({})


class TestMetrics:
    def test_logit_difference_subtracts_the_named_tokens(self):
        logits = torch.tensor([[[0.0, 5.0, 2.0]]])
        assert logit_difference(logits, 1, 2).item() == pytest.approx(3.0)

    def test_logit_difference_accepts_two_dimensional_logits(self):
        logits = torch.tensor([[0.0, 5.0, 2.0]])
        assert logit_difference(logits, 1, 0).item() == pytest.approx(5.0)

    def test_target_probability_is_a_softmax_entry(self):
        logits = torch.zeros(1, 1, 4)
        assert target_probability(logits, 2).item() == pytest.approx(0.25)

    def test_kl_divergence_is_zero_for_identical_distributions(self):
        logits = torch.randn(2, 3, 8)
        assert kl_divergence(logits, logits).abs().max().item() < 1e-6

    def test_kl_divergence_is_positive_for_different_distributions(self):
        assert kl_divergence(torch.zeros(1, 4), torch.tensor([[10.0, 0, 0, 0]])).item() > 0

    def test_normalized_effect_spans_the_clean_corrupted_interval(self):
        assert normalized_effect(0.0, 1.0, 0.0) == pytest.approx(0.0)
        assert normalized_effect(1.0, 1.0, 0.0) == pytest.approx(1.0)
        assert normalized_effect(0.5, 1.0, 0.0) == pytest.approx(0.5)

    def test_normalized_effect_is_not_clipped(self):
        """Overshoot and backfire are real findings, so they must survive."""
        assert normalized_effect(1.5, 1.0, 0.0) == pytest.approx(1.5)
        assert normalized_effect(-0.5, 1.0, 0.0) == pytest.approx(-0.5)

    def test_normalized_effect_guards_a_collapsed_denominator(self):
        assert normalized_effect(5.0, 1.0, 1.0) == 0.0

    def test_bad_logit_shape_is_rejected(self):
        with pytest.raises(ValueError, match="expected logits"):
            logit_difference(torch.zeros(2, 2, 2, 2), 0, 1)
