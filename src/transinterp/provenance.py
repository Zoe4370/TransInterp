"""Environment capture for auditable interpretability artifacts.

An interpretability result is only checkable if a reader can tell what
produced it. The failure mode this module targets is concrete and documented:
two studies reached contradictory conclusions about the same mechanism and the
disagreement was only resolved once someone reconstructed the differences in
their experimental setups. Recording the setup at run time is far cheaper than
reconstructing it later.

Everything here is deliberately local and offline. No network calls are made
and no user identity is collected, because provenance that quietly exfiltrates
information is not something a researcher can safely enable by default.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = ["GitState", "Provenance", "capture_provenance", "git_state"]

_TRACKED_PACKAGES = (
    "torch",
    "numpy",
    "transformers",
    "pydantic",
    "safetensors",
    "matplotlib",
)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class GitState:
    """Repository state at run time, when the code lives in a git checkout."""

    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    remote: str | None = None

    @property
    def is_reproducible_checkout(self) -> bool:
        """True only when a commit is known and the tree has no local edits.

        A dirty tree means the recorded commit does not describe the code that
        actually ran, so the artifact cannot be reproduced from the commit
        alone. Callers should surface this rather than hide it.
        """
        return bool(self.commit) and self.dirty is False


def git_state(path: Path | None = None) -> GitState:
    """Return the git commit, branch, and dirtiness of ``path``.

    Returns an empty :class:`GitState` when git is unavailable or the path is
    not a repository, which is the common case for pip-installed usage.
    """
    cwd = Path(path or Path.cwd())

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return GitState()

    status = run("status", "--porcelain")
    return GitState(
        commit=commit,
        branch=run("rev-parse", "--abbrev-ref", "HEAD"),
        dirty=bool(status) if status is not None else None,
        remote=run("config", "--get", "remote.origin.url"),
    )


@dataclass(frozen=True)
class Provenance:
    """A snapshot of the software environment that produced an artifact."""

    created_at: str
    transinterp_version: str | None
    python_version: str
    platform: str
    machine: str
    packages: dict[str, str | None] = field(default_factory=dict)
    git: GitState = field(default_factory=GitState)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Provenance:
        """Rebuild a :class:`Provenance` from a manifest payload."""
        data = dict(payload)
        data["git"] = GitState(**data.get("git", {}))
        return cls(**data)

    def compare(self, other: Provenance) -> dict[str, tuple[Any, Any]]:
        """Return the environment differences between two runs.

        This is what makes a replay informative rather than merely pass/fail.
        When a replayed result diverges, the first question is what changed;
        this answers it directly instead of leaving the researcher to guess.
        """
        differences: dict[str, tuple[Any, Any]] = {}
        for name in sorted(set(self.packages) | set(other.packages)):
            mine, theirs = self.packages.get(name), other.packages.get(name)
            if mine != theirs:
                differences[f"packages.{name}"] = (mine, theirs)
        for attribute in ("python_version", "platform", "machine", "transinterp_version"):
            mine, theirs = getattr(self, attribute), getattr(other, attribute)
            if mine != theirs:
                differences[attribute] = (mine, theirs)
        if self.git.commit != other.git.commit:
            differences["git.commit"] = (self.git.commit, other.git.commit)
        return differences


def capture_provenance(
    *,
    repo_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Provenance:
    """Collect versions, platform details, and git state for the current run."""
    return Provenance(
        created_at=datetime.now(timezone.utc).isoformat(),
        transinterp_version=_package_version("transinterp"),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        packages={name: _package_version(name) for name in _TRACKED_PACKAGES},
        git=git_state(repo_path),
        extra=dict(extra or {}),
    )
