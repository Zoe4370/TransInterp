"""Content-addressed artifact bundles.

An experiment result is only checkable if a reader can tell, mechanically,
that the files in front of them are the files that were written. A bundle is
a directory whose ``manifest.json`` lists every data file with its SHA-256
digest, plus the provenance and determinism records for the run that produced
them.

Three properties are load-bearing:

*Tensors round-trip exactly.* Values are stored as raw little-endian buffers
with dtype and shape in the manifest, so a float64 stays a float64 and a
bfloat16 stays a bfloat16. Nothing is cast on the way in or out.

*Tampering is detected in both directions.* ``verify()`` reports files whose
contents changed, files the manifest lists that are gone, and files present in
the directory that the manifest does not list. The third case matters: a
directory with an extra file is not the directory the manifest describes, even
though every listed digest still matches.

*The fingerprint depends on data, not on timing.* It is computed over the
sorted (path, digest) pairs of the data files and excludes the manifest, whose
``created_at`` would otherwise make two identical runs look different. That is
what lets ``replay`` answer "same numbers?" with a single comparison.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch

from transinterp.determinism import DeterminismState
from transinterp.extraction.capture import ActivationRecord
from transinterp.provenance import Provenance, capture_provenance

__all__ = ["SCHEMA_VERSION", "ArtifactBundle", "VerificationResult"]

SCHEMA_VERSION = 1

MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.json"
CONFIG_NAME = "config.json"
TENSOR_DIR = "tensors"


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to a raw buffer without changing its dtype.

    ``numpy`` has no bfloat16, so the round trip goes through a byte view of
    the tensor's own storage instead of the array protocol.
    """
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    return bytes(flat.view(torch.uint8).numpy())


def _tensor_from_bytes(payload: bytes, dtype: str, shape: list[int]) -> torch.Tensor:
    torch_dtype = getattr(torch, dtype)
    if not payload:
        return torch.empty(tuple(shape), dtype=torch_dtype)
    flat = torch.frombuffer(bytearray(payload), dtype=torch_dtype)
    return flat.reshape(tuple(shape)).clone()


def _relative_paths(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


@dataclass
class VerificationResult:
    """Outcome of checking a bundle directory against its manifest."""

    root: Path
    checked: int = 0
    corrupted: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only when the directory is exactly what the manifest describes."""
        return not (self.corrupted or self.missing or self.unexpected)

    def summary(self) -> str:
        """One-line verdict naming the counts of each failure kind."""
        if self.ok:
            return f"verified {self.checked} file(s); all digests match"
        parts = []
        if self.corrupted:
            parts.append(f"{len(self.corrupted)} digest mismatch")
        if self.missing:
            parts.append(f"{len(self.missing)} missing file(s)")
        if self.unexpected:
            parts.append(f"{len(self.unexpected)} unlisted file(s)")
        return f"verification failed: {', '.join(parts)}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "ok": self.ok,
            "checked": self.checked,
            "corrupted": list(self.corrupted),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
        }


class ArtifactBundle:
    """A directory of tensors, metrics, and provenance with per-file digests.

    Create one, add content, and call :meth:`write`. Nothing touches the
    filesystem's data files until then, so a run that fails partway leaves an
    empty directory rather than a bundle that claims to describe a result it
    does not have.
    """

    def __init__(self, root: Path | str, *, manifest: dict[str, Any] | None = None) -> None:
        self.root = Path(root)
        self._manifest: dict[str, Any] = manifest if manifest is not None else {}
        self._tensors: dict[str, torch.Tensor] = {}
        self._metrics: dict[str, Any] = dict(self._manifest.get("metrics_inline") or {})
        self._notes: list[str] = list(self._manifest.get("notes") or [])
        self._config: dict[str, Any] | None = None
        self._record_metadata: dict[str, Any] = dict(self._manifest.get("record_metadata") or {})
        self._determinism: dict[str, Any] | None = self._manifest.get("determinism")
        self._provenance: Provenance | None = None

    # -- construction ----------------------------------------------------

    @classmethod
    def create(cls, root: Path | str, *, overwrite: bool = False) -> ArtifactBundle:
        """Start a new bundle at ``root``.

        Refuses to write over an existing directory unless ``overwrite`` is
        set, because silently replacing a previous result is the kind of
        accident that makes a record untrustworthy.
        """
        path = Path(root)
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists; pass overwrite=True to replace it"
                )
            shutil.rmtree(path)
        path.mkdir(parents=True)

        bundle = cls(path)
        bundle._provenance = capture_provenance(repo_path=Path.cwd())
        return bundle

    @classmethod
    def load(cls, root: Path | str) -> ArtifactBundle:
        """Open an existing bundle and read its manifest."""
        path = Path(root)
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"{path} is not a bundle: no {MANIFEST_NAME}")

        manifest = json.loads(manifest_path.read_text())
        stored_version = manifest.get("schema_version")
        if stored_version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written with schema version {stored_version}; this build reads "
                f"schema version {SCHEMA_VERSION}"
            )

        bundle = cls(path, manifest=manifest)
        provenance = manifest.get("provenance")
        if provenance:
            bundle._provenance = Provenance.from_dict(provenance)

        metrics_path = path / METRICS_NAME
        if metrics_path.exists():
            bundle._metrics = json.loads(metrics_path.read_text())
        config_path = path / CONFIG_NAME
        if config_path.exists():
            bundle._config = json.loads(config_path.read_text())
        return bundle

    # -- content ---------------------------------------------------------

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Stage one tensor under ``name``."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name!r} is not a torch.Tensor (got {type(tensor).__name__})")
        self._tensors[name] = tensor.detach().cpu()

    def add_record(self, record: ActivationRecord) -> None:
        """Stage every tensor in an activation record, plus its metadata."""
        for name, tensor in record.tensors.items():
            self.add_tensor(name, tensor)
        self._record_metadata.update(record.metadata)

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge scalar or JSON-serializable measurements into the bundle."""
        self._metrics.update(metrics)

    def add_note(self, note: str) -> None:
        """Attach a free-text caveat that travels with the result."""
        self._notes.append(note)

    def add_config(self, config: Any) -> None:
        """Store the config that produced this run, so it can be re-executed."""
        if hasattr(config, "model_dump"):
            self._config = config.model_dump(mode="json")
        elif isinstance(config, dict):
            self._config = json.loads(json.dumps(config, default=str))
        else:
            raise TypeError(f"cannot store a config of type {type(config).__name__}")

    def add_determinism(self, state: DeterminismState) -> None:
        """Record which determinism controls were applied to the run."""
        self._determinism = state.to_dict()

    def add_provenance(self, provenance: Provenance) -> None:
        """Override the environment snapshot captured at creation time."""
        self._provenance = provenance

    # -- writing ---------------------------------------------------------

    def write(self) -> ArtifactBundle:
        """Write staged content and the manifest describing it."""
        self.root.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict[str, Any]] = {}
        tensor_index: dict[str, dict[str, Any]] = {}

        if self._tensors:
            (self.root / TENSOR_DIR).mkdir(exist_ok=True)
        for name, tensor in sorted(self._tensors.items()):
            relative = f"{TENSOR_DIR}/{quote(name, safe='')}.bin"
            payload = _tensor_bytes(tensor)
            (self.root / relative).write_bytes(payload)
            files[relative] = {"sha256": _digest_bytes(payload), "bytes": len(payload)}
            tensor_index[name] = {
                "path": relative,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
            }

        metrics_payload = json.dumps(self._metrics, indent=2, sort_keys=True, default=str)
        (self.root / METRICS_NAME).write_text(metrics_payload)
        files[METRICS_NAME] = {
            "sha256": _digest_bytes(metrics_payload.encode()),
            "bytes": len(metrics_payload.encode()),
        }

        if self._config is not None:
            config_payload = json.dumps(self._config, indent=2, sort_keys=True, default=str)
            (self.root / CONFIG_NAME).write_text(config_payload)
            files[CONFIG_NAME] = {
                "sha256": _digest_bytes(config_payload.encode()),
                "bytes": len(config_payload.encode()),
            }

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": _fingerprint(files),
            "files": files,
            "tensors": tensor_index,
            "notes": self._notes,
            "record_metadata": self._record_metadata,
            "determinism": self._determinism,
            "provenance": self._provenance.to_dict() if self._provenance else None,
        }
        (self.root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        self._manifest = manifest
        return self

    # -- reading ---------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """Digest over the data files, independent of when they were written."""
        stored = self._manifest.get("fingerprint")
        if stored:
            return str(stored)
        return _fingerprint(self._manifest.get("files") or {})

    @property
    def provenance(self) -> Provenance | None:
        """The environment snapshot for this run."""
        return self._provenance

    @property
    def metrics(self) -> dict[str, Any]:
        """Measurements recorded for this run."""
        return self._metrics

    @property
    def notes(self) -> list[str]:
        """Free-text caveats recorded with the result."""
        return self._notes

    @property
    def config(self) -> dict[str, Any] | None:
        """The stored config, or ``None`` when the run had none."""
        return self._config

    @property
    def determinism(self) -> dict[str, Any] | None:
        """The determinism controls applied to the run."""
        return self._determinism

    def tensor_names(self) -> list[str]:
        """Names of every tensor in the bundle, staged or stored."""
        stored = self._manifest.get("tensors") or {}
        return sorted(set(stored) | set(self._tensors))

    def load_tensor(self, name: str) -> torch.Tensor:
        """Read one tensor back, verifying its digest first."""
        if name in self._tensors:
            return self._tensors[name]

        index = self._manifest.get("tensors") or {}
        if name not in index:
            raise KeyError(
                f"{name!r} is not in this bundle; available tensors: {sorted(index)}"
            )

        entry = index[name]
        path = self.root / entry["path"]
        if not path.exists():
            raise FileNotFoundError(f"{path} is listed in the manifest but missing from disk")

        payload = path.read_bytes()
        expected = (self._manifest.get("files") or {}).get(entry["path"], {}).get("sha256")
        if expected is not None and _digest_bytes(payload) != expected:
            raise ValueError(
                f"digest mismatch for {entry['path']}: the file on disk is not the file the "
                "manifest describes"
            )
        return _tensor_from_bytes(payload, entry["dtype"], entry["shape"])

    def to_record(self) -> ActivationRecord:
        """Rebuild an :class:`ActivationRecord` from the stored tensors."""
        record = ActivationRecord(metadata=dict(self._manifest.get("record_metadata") or {}))
        for name in self.tensor_names():
            record.tensors[name] = self.load_tensor(name)
        return record

    # -- integrity -------------------------------------------------------

    def verify(self) -> VerificationResult:
        """Check the directory against the manifest, in both directions."""
        listed = self._manifest.get("files") or {}
        on_disk = set(_relative_paths(self.root)) - {MANIFEST_NAME}

        corrupted: list[str] = []
        missing: list[str] = []
        for relative, entry in sorted(listed.items()):
            path = self.root / relative
            if not path.exists():
                missing.append(relative)
                continue
            if _digest_file(path) != entry["sha256"]:
                corrupted.append(relative)

        unexpected = sorted(on_disk - set(listed))
        return VerificationResult(
            root=self.root,
            checked=len(listed),
            corrupted=corrupted,
            missing=missing,
            unexpected=unexpected,
        )

    def compare(self, other: ArtifactBundle) -> dict[str, Any]:
        """Diff two bundles by digest, and report environment drift."""
        mine = self._manifest.get("files") or {}
        theirs = other._manifest.get("files") or {}

        shared = sorted(set(mine) & set(theirs))
        differing = [name for name in shared if mine[name]["sha256"] != theirs[name]["sha256"]]
        only_first = sorted(set(mine) - set(theirs))
        only_second = sorted(set(theirs) - set(mine))

        environment: dict[str, tuple[Any, Any]] = {}
        if self.provenance is not None and other.provenance is not None:
            environment = self.provenance.compare(other.provenance)

        return {
            "identical": not (differing or only_first or only_second),
            "fingerprints": (self.fingerprint, other.fingerprint),
            "differing_files": differing,
            "only_in_first": only_first,
            "only_in_second": only_second,
            "environment_differences": environment,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ArtifactBundle(root={str(self.root)!r}, fingerprint={self.fingerprint[:12]!r})"


def _fingerprint(files: dict[str, dict[str, Any]]) -> str:
    """Hash the sorted (path, digest) pairs of a bundle's data files."""
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(files[relative]["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()
