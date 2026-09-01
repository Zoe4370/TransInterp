"""Tests for the reproducibility layer.

These cover the properties the artifact format is supposed to guarantee:
tensors survive a round trip exactly, tampering is detected, identical data
yields an identical fingerprint regardless of when it was written, and a
failed comparison names what differed.
"""

from __future__ import annotations

import json

import pytest
import torch

from transinterp.artifacts import SCHEMA_VERSION, ArtifactBundle
from transinterp.determinism import deterministic, set_seed
from transinterp.extraction.capture import ActivationRecord
from transinterp.provenance import capture_provenance, git_state


class TestRoundTrip:
    def test_tensors_survive_exactly(self, tmp_path):
        original = torch.randn(3, 5, 7)
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_tensor("layer.0.hidden", original)
        bundle.write()

        restored = ArtifactBundle.load(tmp_path / "b").load_tensor("layer.0.hidden")
        assert torch.equal(original, restored)

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.int64, torch.int32, torch.bfloat16]
    )
    def test_dtypes_are_preserved(self, tmp_path, dtype):
        original = (torch.randn(4, 4) * 10).to(dtype)
        bundle = ArtifactBundle.create(tmp_path / f"b-{dtype}")
        bundle.add_tensor("t", original)
        bundle.write()

        restored = ArtifactBundle.load(tmp_path / f"b-{dtype}").load_tensor("t")
        assert restored.dtype == dtype
        assert torch.equal(original, restored)

    def test_record_round_trips_with_metadata(self, tmp_path):
        record = ActivationRecord(metadata={"model_id": "tiny", "tokens": ["a", "b"]})
        record.add("logits", torch.randn(1, 2, 4))
        record.add("layer.0.attention", torch.rand(1, 2, 2, 2))

        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_record(record)
        bundle.write()

        restored = ArtifactBundle.load(tmp_path / "b").to_record()
        assert restored.metadata["model_id"] == "tiny"
        assert set(restored.tensors) == set(record.tensors)
        assert torch.equal(restored.tensors["logits"], record.tensors["logits"])

    def test_dotted_names_survive_the_filesystem(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_tensor("layer.0.attn/head.3", torch.ones(2, 2))
        bundle.write()
        assert "layer.0.attn/head.3" in ArtifactBundle.load(tmp_path / "b").tensor_names()

    def test_missing_tensor_name_lists_alternatives(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_tensor("present", torch.ones(2))
        bundle.write()
        with pytest.raises(KeyError, match="present"):
            ArtifactBundle.load(tmp_path / "b").load_tensor("absent")


class TestIntegrity:
    def test_a_clean_bundle_verifies(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_tensor("t", torch.ones(2, 2))
        bundle.add_metrics({"score": 1.0})
        bundle.write()
        assert ArtifactBundle.load(tmp_path / "b").verify().ok

    def test_modified_file_is_detected(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_metrics({"score": 1.0})
        bundle.write()

        (tmp_path / "b" / "metrics.json").write_text('{"score": 999.0}')
        result = ArtifactBundle.load(tmp_path / "b").verify()
        assert not result.ok
        assert "metrics.json" in result.corrupted

    def test_deleted_file_is_detected(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_tensor("t", torch.ones(2))
        bundle.write()

        (tmp_path / "b" / "tensors" / "t.bin").unlink()
        result = ArtifactBundle.load(tmp_path / "b").verify()
        assert not result.ok
        assert result.missing == ["tensors/t.bin"]

    def test_smuggled_file_is_detected(self, tmp_path):
        """An unlisted file means the directory is not what the manifest describes."""
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_tensor("t", torch.ones(2))
        bundle.write()

        (tmp_path / "b" / "extra.txt").write_text("snuck in")
        result = ArtifactBundle.load(tmp_path / "b").verify()
        assert not result.ok
        assert "extra.txt" in result.unexpected

    def test_load_rejects_a_corrupted_tensor(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_tensor("t", torch.ones(4))
        bundle.write()

        (tmp_path / "b" / "tensors" / "t.bin").write_bytes(b"\x00" * 16)
        with pytest.raises(ValueError, match="digest mismatch"):
            ArtifactBundle.load(tmp_path / "b").load_tensor("t")

    def test_unknown_schema_version_is_refused(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.write()
        manifest_path = tmp_path / "b" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = SCHEMA_VERSION + 99
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="schema version"):
            ArtifactBundle.load(tmp_path / "b")

    def test_creating_over_an_existing_bundle_needs_permission(self, tmp_path):
        ArtifactBundle.create(tmp_path / "b").write()
        with pytest.raises(FileExistsError, match="overwrite=True"):
            ArtifactBundle.create(tmp_path / "b")
        assert ArtifactBundle.create(tmp_path / "b", overwrite=True) is not None


class TestFingerprint:
    def test_identical_data_yields_identical_fingerprints(self, tmp_path):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        first = ArtifactBundle.create(tmp_path / "one")
        first.add_tensor("t", tensor)
        first.write()

        second = ArtifactBundle.create(tmp_path / "two")
        second.add_tensor("t", tensor.clone())
        second.write()

        assert first.fingerprint == second.fingerprint

    def test_different_data_yields_different_fingerprints(self, tmp_path):
        first = ArtifactBundle.create(tmp_path / "one")
        first.add_tensor("t", torch.zeros(2, 2))
        first.write()

        second = ArtifactBundle.create(tmp_path / "two")
        second.add_tensor("t", torch.ones(2, 2))
        second.write()

        assert first.fingerprint != second.fingerprint

    def test_compare_names_the_differing_file(self, tmp_path):
        first = ArtifactBundle.create(tmp_path / "one")
        first.add_tensor("shared", torch.zeros(2))
        first.add_tensor("only_first", torch.zeros(2))
        first.write()

        second = ArtifactBundle.create(tmp_path / "two")
        second.add_tensor("shared", torch.ones(2))
        second.write()

        result = ArtifactBundle.load(tmp_path / "one").compare(
            ArtifactBundle.load(tmp_path / "two")
        )
        assert not result["identical"]
        assert "tensors/shared.bin" in result["differing_files"]
        assert "tensors/only_first.bin" in result["only_in_first"]


class TestProvenanceAndDeterminism:
    def test_provenance_records_the_environment(self):
        provenance = capture_provenance()
        assert provenance.packages["torch"] is not None
        assert provenance.python_version
        assert provenance.created_at.endswith("+00:00")

    def test_provenance_survives_a_round_trip(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.write()
        loaded = ArtifactBundle.load(tmp_path / "b")
        assert loaded.provenance is not None
        assert loaded.provenance.packages["torch"] == capture_provenance().packages["torch"]

    def test_provenance_compare_reports_version_drift(self):
        first = capture_provenance()
        second = capture_provenance(extra={})
        object.__setattr__(second, "packages", {**second.packages, "torch": "0.0.1"})
        differences = first.compare(second)
        assert "packages.torch" in differences

    def test_git_state_is_safe_outside_a_repository(self, tmp_path):
        state = git_state(tmp_path)
        assert state.commit is None
        assert state.is_reproducible_checkout is False

    def test_set_seed_makes_sampling_repeatable(self):
        set_seed(123)
        first = torch.randn(5)
        set_seed(123)
        assert torch.equal(first, torch.randn(5))

    def test_set_seed_reports_what_it_did(self):
        state = set_seed(7, strict=False)
        assert state.seed == 7
        assert state.strict is False
        assert any("Non-strict" in note for note in state.notes)

    def test_deterministic_context_restores_previous_settings(self):
        before = torch.are_deterministic_algorithms_enabled()
        with deterministic(11, strict=True) as state:
            assert state.strict is True
        assert torch.are_deterministic_algorithms_enabled() == before

    def test_determinism_state_is_recorded_in_the_manifest(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_determinism(set_seed(5))
        bundle.write()
        manifest = json.loads((tmp_path / "b" / "manifest.json").read_text())
        assert manifest["determinism"]["seed"] == 5


class TestNotesAndConfig:
    def test_notes_travel_with_the_result(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_note("Correlational only; no intervention was run.")
        bundle.write()
        assert "Correlational only" in ArtifactBundle.load(tmp_path / "b").notes[0]

    def test_config_is_stored_and_reloadable(self, tmp_path):
        from transinterp.config import ExperimentConfig

        config = ExperimentConfig(model={"name_or_path": "tiny"}, input={"prompts": ["hi"]})
        bundle = ArtifactBundle.create(tmp_path / "b")
        bundle.add_config(config)
        bundle.write()

        stored = ArtifactBundle.load(tmp_path / "b").config
        assert stored["model"]["name_or_path"] == "tiny"
        assert ExperimentConfig.model_validate(stored).input.prompts == ["hi"]

    def test_non_tensor_input_is_rejected(self, tmp_path):
        bundle = ArtifactBundle.create(tmp_path / "b")
        with pytest.raises(TypeError, match="not a torch.Tensor"):
            bundle.add_tensor("t", [1, 2, 3])
