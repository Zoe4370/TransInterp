# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-09-02

The release that makes the project's stated premise real. Prior versions
described reproducible research artifacts but had no code to produce them.

### Added

- **Artifact bundles** (`transinterp.artifacts`). Content-addressed experiment
  directories with per-file SHA-256 digests and a single fingerprint over the
  data. Tensors are stored as raw buffers rather than through `torch.save`,
  which writes timestamps and would give two identical runs different hashes.
  Verification detects modified, deleted, and added files.
- **Executable activation patching** (`transinterp.interventions`). Hook-based
  interventions with `replace`, `zero`, `mean`, `scale`, `add`, and `noise`
  modes; targeting by token position and by attention head; denoising and
  noising protocols; and logit-difference, KL, and normalized-effect metrics.
  Previously the package could only rank measurements a caller computed
  elsewhere.
- **Logit lens** (`transinterp.trajectories.logit_lens`), including
  `LogitLens.check`, which validates the final-layer readout against the
  model's own logits. The README had advertised logit-lens support since 0.1;
  no implementation existed.
- **Provenance capture** (`transinterp.provenance`). Package versions,
  platform, Python version, git commit and dirty flag, with `compare()` to
  diff two runs' environments. No network calls; no identity collected.
- **Determinism controls** (`transinterp.determinism`). `set_seed` and the
  `deterministic` context manager, both reporting what was actually applied.
  The `seed` config field existed since 0.1 but nothing consumed it.
- **Experiment runner and replay** (`transinterp.experiment`).
  `run_experiment` turns a config into a bundle; `replay_experiment` re-runs
  from the stored config and reports whether results match, naming environment
  differences when they do not.
- **Working CLI**: `run`, `replay`, `verify`, `inspect`, and `compare`, in
  addition to the existing `config validate`.
- Config schema extended with `hypothesis`, `input`, and `analysis` sections.
  Unknown keys are now rejected, since a silently ignored YAML typo means the
  recorded run is not the run the author intended.
- Adapter gains `module_map`, `patcher()`, `lens()`, batched input, token
  strings, and model-identity provenance including the resolved commit hash.
- `examples/reproducible_experiment.py`, a full run/verify/tamper/replay demo
  that downloads nothing.

### Fixed

- **`DecisionTrajectory.rank` silently corrupted batched input.** It collected
  match positions with `nonzero`, which flattens: for a batch of 3 across 2
  layers it returned 6 values in one dimension instead of a `(2, 3)` grid.
  Wrong, but shaped plausibly enough to go unnoticed.
- **Logit lens double-normalization.** Hugging Face already applies the final
  norm to the last hidden state; normalizing again shifts every readout while
  leaving the top-1 predicted token unchanged. Handled explicitly via
  `final_layer_already_normalized`, and detectable through `check()`.
- `provenance()` raised on models whose block list is not at a known path,
  failing runs that were otherwise fine. Structural fields now degrade to
  `None`.
- `rank` now validates `class_index` instead of returning empty results.

### Changed

- README rewritten to describe implemented behavior only, with an explicit
  "What this does not do" section and a comparison against TransformerLens,
  nnsight, and SAELens.
- Adapter documents why `attn_implementation` defaults to `eager`: fused
  kernels never materialize the attention matrix, so attention metrics would
  silently receive nothing.
- Test suite grew from 19 to 111 tests. Tests build tiny transformers on disk
  and download no weights.

## [0.3.0] - 2026-09-01

### Added

- PCA and sparse-autoencoder feature extraction, induction-head and
  head-similarity detectors, JSON attention-graph export, and a minimal
  Hugging Face causal-LM adapter.
