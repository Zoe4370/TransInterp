# TransInterp setup guide

This guide takes a fresh machine from checkout to a verified local installation. The shortest path uses Python's built-in `venv` and runs the included synthetic examples without downloading model weights.

## 1. Check prerequisites

Use Python 3.10 or newer and Git. Confirm both are available:

```bash
python3 --version
git --version
```

For real model inspection, install a PyTorch build appropriate for your operating system and accelerator. The examples and tests in this repository do not require a model download.

## 2. Clone the repository

```bash
git clone https://github.com/Zoe4370/TransInterp.git
cd TransInterp
```

If the repository is still private, authenticate with GitHub before cloning. Once it becomes public, the same command works without repository access.

## 3. Create an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use `.venv\\Scripts\\Activate.ps1` instead. Upgrade packaging tools before installing the project:

```bash
python -m pip install --upgrade pip
```

## 4. Install TransInterp

Install the package with development and notebook extras:

```bash
pip install -e '.[dev,notebooks]'
```

For a smaller runtime-only installation, use `pip install -e .`. The development extras provide testing, linting, formatting, and type-checking tools.

## 5. Verify the installation

Run the complete local quality gate:

```bash
pytest -q
ruff check .
ruff format --check .
```

A healthy checkout should report all tests passing and no Ruff errors.

## 6. Run the offline example

The offline example uses synthetic tensors, so it is safe to run before configuring model access:

```bash
python examples/basic_usage.py
python examples/feature_extraction_demo.py
```

These scripts demonstrate attention metrics, decision trajectories, PCA, sparse-autoencoder features, induction-head scoring, and graph export.

## 7. Inspect a real causal language model

The first real-model path uses the Hugging Face adapter:

```python
from transinterp.models import HuggingFaceCausalLM

adapter = HuggingFaceCausalLM.from_pretrained("distilgpt2", device="cpu")
logits, record = adapter.run("The capital of France is")

print(logits.shape)
print(record.metadata)
print(sorted(record.tensors)[:5])
```

The first run downloads the tokenizer and model weights into the normal Hugging Face cache. Keep `trust_remote_code=False` unless you have reviewed and intentionally trust a model repository's custom code.

## 8. Use a GPU when available

Install the PyTorch wheel matching your CUDA or ROCm environment, then pass `device="cuda"` or the appropriate device string to `from_pretrained`. Do not assume that a model fits in memory. Start with a small model, capture only the layers needed for the hypothesis, and keep `detach=True` and `to_cpu=True` when storing artifacts.

## 9. Validate an experiment configuration

The included CLI validates YAML configuration and prints the normalized form:

```bash
transinterp config validate configs/example.yaml
```

Use an explicit model revision and seed for experiments intended to be shared or published.

## 10. Troubleshooting

If installation fails while building PyTorch, install PyTorch separately using the official wheel selector and rerun `pip install -e .`. If a gated Hugging Face model cannot be downloaded, authenticate with Hugging Face and verify that your account has accepted the model's license. If memory usage is high, shorten the sequence, reduce captured layers, use CPU offload, or run the synthetic examples first.

## 11. Before opening a pull request

Run `pytest -q`, `ruff check .`, and `ruff format --check .`. Add a regression test for any changed tensor contract. Include the model revision, tokenizer, seed, device, dtype, and data slice in research-facing changes. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the review standard.
