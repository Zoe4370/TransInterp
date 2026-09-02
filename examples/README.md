# Examples

Each example states what it demonstrates and whether it downloads anything.

| Example | Downloads weights? | Demonstrates |
|---|---|---|
| `basic_usage.py` | no | Hook capture, attention metrics, decision trajectories on a synthetic model |
| `feature_extraction_demo.py` | no | PCA and sparse-autoencoder decomposition of synthetic activations |
| `reproducible_experiment.py` | no | The full loop: run, verify, tamper-detect, replay |
| `induction_controls.py` | no | A control experiment: separating induction heads from positional heads |

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

## `induction_controls.py`

The others exercise machinery. This one runs an experiment with a result, and
is the longest-running example here — it trains ten small transformers from
scratch, so budget roughly an hour on a GPU and a few hours on CPU:

```bash
python examples/induction_controls.py            # 5 seeds, full budget
python examples/induction_controls.py --quick    # smoke test, ~1 minute
```

It exists because the standard induction-head task has a confound. Training on
`[random N tokens][the same N tokens]` puts the repeat at a fixed offset, so a
head that attends exactly N positions back scores highly on in-context copying
*and* on `induction_head_score` while implementing no induction at all. The
script trains a matched pair of models — one on the fixed-offset task, one on a
task where the repeat's position and length vary — and puts both through the
same battery: transfer to offsets held out of training, per-head and per-pair
and per-sublayer zero-ablation, and induction scoring. Five seeds, bootstrap
confidence intervals, results written to an artifact bundle and figures to
`assets/`.

Unlike the other examples, these models are trained, so the numbers do mean
something — about these models on this synthetic task, and about nothing else.

The models in the other examples are randomly initialized, so their numbers carry no
interpretive meaning. They exercise the machinery, not any claim about a real
model. To study real weights, point a config at a Hugging Face model and use
`transinterp run` — see `configs/example.yaml` and [../docs/SETUP.md](../docs/SETUP.md).
