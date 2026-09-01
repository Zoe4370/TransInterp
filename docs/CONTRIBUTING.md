# Contributing to TransInterp

Thank you for helping make interpretability research more reproducible. Before proposing a feature, please open an issue that explains the research question, the expected API, and the smallest example that demonstrates the value. For bug fixes, include a failing case whenever possible.

## Development setup

Install the development extras with `pip install -e '.[dev,notebooks]'`. Run `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest --cov=transinterp --cov-report=term-missing` before submitting a pull request — this mirrors the GitHub Actions CI workflow in `.github/workflows/ci.yml`, which runs the same checks on Python 3.10–3.12. Documentation-only changes should still be checked for broken links and executable examples.

## Pull requests

Keep pull requests focused. A good change includes a short rationale, tests for behavior rather than implementation details, and documentation for any public interface. New metrics must specify tensor axes, numerical assumptions, and at least one edge case. New visualizations should include a deterministic fixture or a small example artifact.

Please do not commit model weights, private datasets, API keys, raw user prompts containing personal information, or generated files that can be recreated. Large experiments belong in an external artifact store with a stable reference in the accompanying report.

## Research standards

Interpretability claims should distinguish observation, association, and causal evidence. Report negative results and known failure modes. If an experiment uses a benchmark or pretrained model, cite the original source and preserve the exact revision. Avoid presenting a single saliency map or attention pattern as a definitive explanation.

## Commit style

Use imperative, scoped messages such as `feat(attention): add head entropy metric` or `docs(paper): clarify intervention protocol`. Squash fixups before merge unless a maintainer requests otherwise.

## Review checklist

Reviewers look for API clarity, numerical correctness, shape-safe behavior, tests, reproducibility metadata, reasonable memory use, and language that does not overstate what the evidence supports.
