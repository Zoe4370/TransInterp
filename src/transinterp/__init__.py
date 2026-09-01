"""TransInterp: auditable tooling for Transformer interpretability."""

from transinterp.artifacts.bundle import ArtifactBundle
from transinterp.config.models import ExperimentConfig
from transinterp.determinism import set_seed
from transinterp.extraction.capture import ActivationRecord, HookCapture
from transinterp.interventions.patching import ActivationPatcher, PatchSpec
from transinterp.models import HuggingFaceCausalLM
from transinterp.provenance import capture_provenance
from transinterp.trajectories.decision import DecisionTrajectory
from transinterp.trajectories.logit_lens import LogitLens

__all__ = [
    "ActivationPatcher",
    "ActivationRecord",
    "ArtifactBundle",
    "DecisionTrajectory",
    "ExperimentConfig",
    "HookCapture",
    "HuggingFaceCausalLM",
    "LogitLens",
    "PatchSpec",
    "capture_provenance",
    "set_seed",
]
__version__ = "0.4.0"
