# Examples

Each example states what it demonstrates and whether it downloads anything.

| Example | Downloads weights? | Demonstrates |
|---|---|---|
| `basic_usage.py` | no | Hook capture, attention metrics, decision trajectories on a synthetic model |
| `feature_extraction_demo.py` | no | PCA and sparse-autoencoder decomposition of synthetic activations |
| `reproducible_experiment.py` | no | The full loop: run, verify, tamper-detect, replay |

Start with `reproducible_experiment.py`. It builds a tiny transformer on disk,
runs a patching experiment, writes an artifact bundle, shows verification
catching a tampered file, and replays the run to confirm it reproduces:

```bash
python examples/reproducible_experiment.py
```

```
1. Running experiment
   fingerprint b9aa5dae1a27fa7f
   logit lens verified: True
3. Verifying integrity
   verified 11 file(s); all digests match
4. Tampering with metrics.json
   verification failed: 1 digest mismatch
5. Restoring and replaying
   reproduced exactly (fingerprint b9aa5dae1a27)
```

The models in these examples are randomly initialized, so the numbers carry no
interpretive meaning. They exercise the machinery, not any claim about a real
model. To study real weights, point a config at a Hugging Face model and use
`transinterp run` — see `configs/example.yaml` and [../docs/SETUP.md](../docs/SETUP.md).
