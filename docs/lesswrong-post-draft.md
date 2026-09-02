# When an induction score points at the wrong head

*A reproducible toy experiment on correlational scores, causal ablation, and why the intervention has to be the test*

Mechanistic interpretability has a recurring failure mode: a measurement identifies a component with an appealing pattern, and the write-up quietly upgrades that pattern into a causal explanation. I built a small repeated-sequence experiment to make that failure mode concrete.

The task is deliberately simple. A two-layer transformer is trained on repeated sequences until it copies the token that followed an earlier occurrence. One attention head develops the familiar induction-like stripe: after the sequence repeats, it attends to the token that followed the earlier copy. An induction-head score ranks that head highest.

The tempting conclusion is: **that head is the induction head, so ablating it should break the behavior.**

It does not.

Ablating the top-scoring head changes nothing measurable. Neither does ablating any other individual head. When the entire first attention sublayer is ablated, accuracy collapses from 100% to 2%. The behavior depends causally on the sublayer, but no single head is individually necessary. The mechanism is redundant: several heads cover for one another.

This is not evidence that the score is useless. The attention pattern was real, and the score correctly identified a region where the mechanism lives. It is evidence that the score answers a different question from the one the causal claim requires. A correlational detector can find a plausible participant without establishing that the participant is necessary, sufficient, or uniquely responsible.

There is a second caveat. The experiment repeats at a fixed offset, so a head that simply attends a fixed distance backward can score as induction without using the repeated content. Variable repeat offsets would be a necessary control before calling this genuine content-based induction.

The code and complete result record are in [TransInterp](https://github.com/Zoe4370/TransInterp). The point of the repository is not to replace circuit-analysis libraries. It is to make the experiment record survive inspection: the configuration, captured tensors, metrics, environment, file digests, and replay result are stored together. The same bundle can be verified for tampering and replayed to test whether the numbers return.

The practical lesson is modest but useful: **treat attention patterns and detector scores as hypotheses; use interventions to test the causal story.** In this toy setting, the interesting result was not “we found the induction head.” It was that a plausible head-level story was wrong at the level of necessity, and the sublayer-level intervention exposed why.

## Questions I would like feedback on

1. Which controls should be considered minimally necessary before interpreting an induction score on a fixed-offset task?
2. What is the cleanest way to distinguish redundant head groups from a genuinely distributed sublayer mechanism?
3. Which intervention designs best test sufficiency without turning a small toy experiment into an unmanageable search?

## Reproduction

```bash
git clone https://github.com/Zoe4370/TransInterp.git
cd TransInterp
python -m pip install -e '.[dev]'
python examples/induction_experiment.py --figures assets/
```

The experiment downloads no model weights and writes its generated figures to the requested directory.
