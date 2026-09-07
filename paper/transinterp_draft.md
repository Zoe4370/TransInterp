# Probe-Distribution Confounding in Induction-Head Identification

**Zoe Faith Gumise**

Independent researcher
[TransInterp repository](https://github.com/Zoe4370/TransInterp)

## Abstract

Prefix-matching scores are widely used to identify induction heads by measuring attention mass on the token that followed the current token's earlier occurrence. A standard probe repeats a random token block at a constant period. Under that construction, the induction target and a fixed positional offset coincide, so the score cannot distinguish content matching from position counting. We give a constructive demonstration that this ambiguity is consequential rather than merely theoretical. Two-layer, four-head transformers trained on a fixed-period copy task obtain a mean induction score of 0.88 in layer 0 and 0.94 when scoring is restricted to genuine repeats, while placing 0.00 attention on the earlier occurrence itself. These models do not implement the intended two-layer induction circuit: removing their entire second attention sublayer changes accuracy by 0.0 percentage points, and copying fails outside the trained period, reaching 1.0% on offsets 8–20 where a content-blind copier reaches 11.2%. Models trained with variable periods generalize to held-out periods in all five seeds, yet score only 0.07 on the same metric. Across this synthetic task family, the score is therefore inversely ordered with the mechanism it is intended to identify. We recommend period randomization as a required control and report a diagnostic that separates the two attention patterns. The scope is limited to randomly initialized transformers trained on synthetic copy tasks; whether the effect transfers to pretrained language models remains untested.

## 1. Introduction

Induction heads are commonly described as attention heads that attend from a token to the position following an earlier occurrence of that token, thereby increasing the logit of the earlier continuation [4, 12]. A standard prefix-matching score estimates this behavior by running a model on a sequence containing a repeat and averaging the attention weight placed on the designated induction target.

The probe construction has a structural weakness. If the repeated block has a fixed length, then the induction target is always at the same distance behind the query. A head that attends to that distance without reading token identity receives the same score as a head that performs content-based retrieval. The probe supplies no case in which the positional and content-based rules disagree.

This confound is especially relevant when the training distribution itself uses a fixed repeat period. Prior work shows that data diversity can steer shallow transformers toward either a generalizable induction algorithm or a positional shortcut [8]. That work establishes a behavioral and mechanistic distinction between the two solutions. We ask a downstream measurement question: **What does the standard induction score report when it is applied to a model that learned the positional shortcut?**

Our result is a caution about interpretability metrology. The fixed-period model displays the canonical attention signature associated with induction, including high attention on the target and near-zero attention on the earlier occurrence of the query token. Yet it fails to generalize, its second attention sublayer is causally inert, and its copying behavior is explained by a fixed positional rule. The metric does not merely become uninformative; within this task family, it ranks the non-generalizing model above the model that generalizes.

We make three contributions:

1. We show constructively that a fixed-period probe makes positional counting and content matching arithmetically indistinguishable.
2. We combine generalization, a content-blind reference, and zero-ablation to establish the negative case causally rather than from attention patterns alone.
3. We propose two inexpensive controls: randomized repeat periods and a paired diagnostic measuring attention on both the induction target and the earlier occurrence of the query token.

## 2. Related work

### 2.1 Induction heads and their measurement

Elhage et al. [4] identify induction-head circuits in small attention-only transformers and describe the composition of a previous-token head with a matching head. Olsson et al. [12] connect induction-head formation to a training-loss phase change and introduce prefix-matching and copying statistics. Later work studies semantic and fuzzy variants of induction [3, 5, 14]. These extensions change what counts as a match, but they do not by themselves remove a fixed-period positional confound.

### 2.2 Positional shortcuts in copy tasks

Kawata et al. [8] provide the closest prior result. They show that the diversity of trigger-to-trigger distances influences whether a shallow transformer learns a generalizable induction algorithm or a positional shortcut. Their analysis is focused on algorithm selection, generalization, and training dynamics. Our contribution is complementary: we measure how a standard induction statistic behaves when applied to the shortcut solution.

### 2.3 Attention and causal importance

Attention weights are not sufficient evidence of causal responsibility [7, 15]. Interpretability illusions can arise when a diagnostic tracks a correlated feature rather than the computation that determines the output [1, 11]. Induction-head studies also report redundancy and small effects under individual-head ablation [13, 15b]. We therefore treat attention scores as hypotheses and use interventions and out-of-distribution evaluation to test the implied mechanism.

### 2.4 Shortcut learning as a measurement problem

Shortcut learning describes models exploiting statistical regularities that satisfy a training objective without implementing the intended competence [6]. We apply the same framing to interpretability measurement. A synthetic probe can contain a shortcut, and a metric computed on that probe can measure the shortcut rather than the mechanism named by the metric.

## 3. Method

### 3.1 Task family

Sequences have length 64 and contain a random block appearing twice. The repeat offset is the distance between the two copies. We compare two training distributions that share sequence length and vocabulary:

| Condition | Training repeat offsets | Evaluation use |
|---|---|---|
| Fixed-period | Offset 32 only | Fixed offset 32 and offsets 8–20 and 26–40 |
| Variable-period | Uniformly sampled offsets 8–20 | Training-range offsets 8–20 and held-out offsets 26–40 |

Chance accuracy on the repeated span is 0.20%. The fixed-period condition makes a content-blind rule sufficient: attending to a single fixed distance can solve the training distribution whenever that distance matches the repeat offset.

### 3.2 Content-blind reference

Accuracy relative to chance can overstate what a model has learned because a fixed-distance strategy succeeds whenever the evaluation offset matches its preferred distance. We therefore compute a content-blind reference for each evaluation set: the best single fixed-offset copier, selected using no token identity. This reference reaches 100% on the fixed-period set by construction, 11.2% on offsets 8–20, and 8.7% on held-out offsets 26–40.

A model claiming content-based retrieval should exceed the corresponding reference. This comparison is stronger than asking only whether the model exceeds chance.

### 3.3 Models and training

Both conditions use two-layer, four-head GPT-2-style transformers with learned positional embeddings and eager attention, trained from random initialization on next-token prediction. Five random seeds are used per condition. We report means with 95% bootstrap confidence intervals over seeds. Attention weights are materialized rather than produced by fused kernels because fused kernels do not expose the attention matrix required by the measurements.

**Reproducibility status of the supplied record.** The current archival materials specify the architecture family, sequence length, task distributions, seed count, and attention implementation, but they do not preserve the exact vocabulary size, optimizer, learning rate, or total training-step configuration used to generate the reported PDF. Those values are intentionally not reconstructed from plausibility. The quantitative claims below are transcribed from the supplied paper record; the experiment should not be described as fully independently reproducible until the original training configuration or a content-addressed bundle containing it is recovered.

### 3.4 Metrics

For each head, we compute three quantities:

1. **Standard induction score:** average attention mass on the token that followed the query token's earlier occurrence, over queries whose token appears earlier in the sequence.
2. **Genuine-repeat induction score:** the same quantity restricted to queries inside an actual repeated block, reducing accidental token-collision effects.
3. **Previous-token mass:** attention on the earlier occurrence of the query token itself, which is the previous-token position rather than the induction target.

The second and third quantities make the measurement more diagnostic. A content-matching head that shifts by one position, a content-matching head that does not shift, and a content-blind positional head can otherwise produce similar target scores under a fixed-period probe.

### 3.5 Interventions

We zero target components during the clean forward pass and measure the resulting change in copy accuracy. Targets include each individual head, each within-layer pair, and each complete attention sublayer. These interventions measure necessity, not sufficiency. Mean ablation, resampling, and alternative patching directions may produce different conclusions.

### 3.6 Reproducibility infrastructure

The experiments are intended to be run through TransInterp's configuration-driven runner. A run records activations, configuration, metrics, package versions, git state, seed, and determinism settings in a content-addressed artifact bundle with a SHA-256 digest per file and a fingerprint over the bundle. The supplied paper reports two independent executions of the five-seed protocol with identical printed results and fingerprint `d61961e37b17`, covering 16 files and 14 tensors. Because the underlying training configuration is not present in the current archival materials, that fingerprint is reported as provenance for the supplied result record, not as a substitute for the missing configuration fields.

## 4. Results

### 4.1 Fixed-period training produces a positional ruler

The fixed-period model succeeds at offset 32 and fails at other offsets. It reaches 100.0% at the trained offset and approximately 0% near offsets 30 and 34. The variable-period model instead exhibits a broader accuracy profile across its training range and partial transfer into the held-out range.

| Evaluation set | Fixed-period model | Variable-period model | Content-blind reference |
|---|---:|---:|---:|
| Fixed offset 32 | 100.0 [100.0, 100.0] | 41.4 [23.8, 63.1] | 100.0 |
| Offsets 8–20 | 1.0 [0.9, 1.1] | 86.7 [84.1, 89.2] | 11.2 |
| Offsets 26–40 | 7.2 [7.2, 7.3] | 47.0 [30.9, 65.5] | 8.7 |

*Table 1. Copy accuracy on the repeated span, reported as mean with 95% bootstrap intervals over five seeds. Values are transcribed from the supplied paper record.*

![Copy accuracy against repeat offset](../assets/induction-ablation.png)

The fixed-period model scores 1.0% on offsets 8–20, below the 11.2% content-blind reference. The variable-period model exceeds the held-out reference in all five seeds, although its transfer is partial and seed-dependent; the supplied record reports a held-out interval from 30.9% to 65.5% for the variable-period condition.

### 4.2 The induction metric is inverted across the two conditions

The fixed-period model records a mean standard induction score of 0.88 across layer-0 heads and 0.94 when scoring is restricted to genuine repeats. Attention mass on the earlier occurrence of the query token is 0.00. This is the textbook attention signature commonly associated with induction: the head skips the matched token and lands on its successor.

The variable-period model records at most 0.07 on the same metric, while it generalizes to held-out periods. A researcher ranking these two models by the standard score would select the model that cannot copy at untrained periods. The metric is therefore anti-correlated with the intended generalizing mechanism within this task family.

![Per-head induction diagnostics](../assets/induction-scores.png)

The result does not show that the metric is universally invalid. It shows that the probe distribution is part of the measurement definition. Fixed periodicity makes the positional and content-based targets coincide, so the score cannot identify which rule produced the attention.

### 4.3 Ablation separates the models causally

Zero-ablation reveals a distinction that attention scores alone miss.

| Ablation target | Fixed-period model | Variable-period model |
|---|---:|---:|
| Worst single head, layer 0 | −0.0 points | −77.0 points |
| Worst single head, layer 1 | −0.0 points | −2.8 points |
| Worst within-layer pair | −0.2 points | −85.5 points |
| Entire layer-0 attention | −99.8 points | −86.4 points |
| Entire layer-1 attention | −0.0 points | −86.2 points |

*Table 2. Change in copy accuracy under zero-ablation, relative to each model's own unablated accuracy. Values are transcribed from the supplied paper record.*

Removing the entire second attention sublayer from the fixed-period model costs 0.0 percentage points. Removing its layer-0 attention costs 99.8 points. The fixed-period model therefore depends on the first attention sublayer but not on the two-layer composition usually invoked for induction. Individual heads and pairs are dispensable, indicating redundancy within the essential sublayer.

The variable-period model has a different profile. Each layer-0 head is individually important, while no individual layer-1 head is highly necessary even though the full layer-1 sublayer is. Redundancy is therefore a property of the learned computation and training distribution, not simply of the architecture.

![Accuracy under zero-ablation](../assets/induction-ablation.png)

### 4.4 Learning dynamics

The supplied record reports that the fixed-period task is solved in under 100 optimizer steps. The variable-period task remains near the content-blind reference for approximately 1,500 steps and rises to roughly 87% between steps 2,000 and 2,700. The additional training time is associated with a transition into the more generalizing solution rather than with a smooth incremental improvement.

These values are useful as reported observations, but the missing optimizer and learning-rate configuration prevents exact reproduction from the current archive. The training curve should therefore be treated as a result requiring provenance recovery, not as a fully specified benchmark.

## 5. Discussion

### 5.1 What the metric measures under fixed periodicity

The prefix-matching score is a conditional statement about attention geometry: given a query, how much mass lands on one designated position? When periodicity is fixed, that position can be identified either by token identity or by an index. Since the probe contains no disagreement cases, the score cannot distinguish the rules.

Period randomization supplies those disagreement cases without requiring a change to the scoring code. It changes only the sequence-generation distribution and makes a fixed-distance ruler fail while a content-based rule can continue to succeed.

### 5.2 A paired diagnostic

Attention on the induction target and attention on the earlier occurrence of the query token provide a useful diagnostic pair. Together they distinguish a head that matches content and shifts, a head that matches content without shifting, and a head that ignores content. However, the pair is informative only when evaluated across varied repeat periods. Under fixed periodicity, a ruler can reproduce the same apparent signature.

### 5.3 An unresolved observation

The variable-period model copies and generalizes to held-out periods, yet no head concentrates substantial attention on a single source position. The supplied record reports 0.04 attention on the earlier occurrence, 0.08 on the induction target, and less than 0.01 in layer 1. Deleting any layer-0 head still costs at least 66 percentage points. The mechanism is therefore causally real but not well described by a single-position prefix-matching score.

Possible explanations include distributed retrieval and non-induction in-context mechanisms. The present study does not distinguish between them. We report the observation as a reason to expand the metric family rather than force the model into a familiar label.

## 6. Limitations and research integrity statement

The models are two-layer, four-head transformers trained from random initialization on synthetic sequences. The demonstration establishes that fixed-period probes can fail to identify induction heads in models trained on fixed-period data. It does not establish that induction scores reported for pretrained language models are wrong. Pretrained models were not trained on this probe and may not learn a ruler matched to it.

The variable-period transfer is seed-dependent. The supported claim is that period diversity removes the specific fixed-position confound, not that the resulting model generalizes perfectly. Only zero-ablation in one direction is reported. The copying score from the original induction-head work is not included, and a full replication of that two-part criterion would strengthen the analysis.

Most importantly, the current source record does not preserve the exact vocabulary size, optimizer, learning rate, or total training-step configuration. We do not fill those fields with plausible values. Before submission as a fully reproducible empirical paper, the original training script or content-addressed experiment bundle should be recovered and the numerical tables regenerated from it. If that recovery is not possible, the manuscript should retain this limitation and be submitted only with claims calibrated to the supplied result record.

## 7. Conclusion

A metric is only as trustworthy as the distribution on which it is computed. A fixed-period induction probe makes position counting and content matching arithmetically identical. A transformer trained on that same fixed-period distribution can therefore obtain a high induction score while implementing a positional shortcut and no two-layer induction circuit.

Period randomization is a low-cost control that removes the ambiguity. Attention on the earlier occurrence of the query token provides an additional diagnostic, but it cannot replace distributional variation. The general methodological lesson is broader than induction heads: descriptive patterns should generate causal hypotheses, not settle them. Out-of-distribution evaluation and intervention are the tests that determine whether the named mechanism is actually present.

## Reproducibility statement

Code and figure-generating examples are available in the [TransInterp repository](https://github.com/Zoe4370/TransInterp). The repository contains configuration-driven experiment infrastructure, executable interventions, content-addressed artifacts, provenance capture, and verification tools. The supplied paper record reports fingerprint `d61961e37b17` for two identical executions of the five-seed protocol. Exact training hyperparameters are not present in the current materials and must be restored before claiming complete independent reproduction.

## References

[1]: https://arxiv.org/abs/2104.07143 "An interpretability illusion for BERT"

[2]: https://arxiv.org/abs/2306.00802 "Birth of a transformer: a memory viewpoint"

[3]: https://arxiv.org/abs/2407.07011 "Induction heads as an essential mechanism for pattern matching in in-context learning"

[4]: https://transformer-circuits.pub/2021/framework/index.html "A Mathematical Framework for Transformer Circuits"

[5]: https://arxiv.org/abs/2504.03022 "The dual-route model of induction"

[6]: https://doi.org/10.1038/s42256-020-00257-z "Shortcut learning in deep neural networks"

[7]: https://aclanthology.org/N19-1357/ "Attention is not Explanation"

[8]: https://arxiv.org/abs/2512.18634 "Ryotaro Kawata, Yujin Song, Alberto Bietti, Naoki Nishikawa, Taiji Suzuki, Samuel Vaiter, and Denny Wu. From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers. Advances in Neural Information Processing Systems 38 (NeurIPS 2025). arXiv:2512.18634. DOI: 10.52202/085713-2337."

[9]: https://arxiv.org/abs/2307.15771 "The Hydra Effect: Emergent Self-repair in Language Model Computations"

[10]: https://github.com/TransformerLensOrg/TransformerLens "TransformerLens: a library for mechanistic interpretability of generative language models"

[11]: https://arxiv.org/abs/2311.17030 "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching"

[12]: https://arxiv.org/abs/2209.11895 "In-context Learning and Induction Heads"

[13]: https://arxiv.org/abs/2402.15390 "Explorations of Self-Repair in Language Models"

[14]: https://arxiv.org/abs/2402.13055 "Identifying Semantic Induction Heads to Understand In-context Learning"

[15]: https://aclanthology.org/P19-1580/ "Is Attention Interpretable?"

[15b]: https://arxiv.org/abs/2404.07129 "What Needs to Go Right for an Induction Head?"

[16]: https://aclanthology.org/P19-1580/ "Analyzing Multi-Head Self-Attention"

[17]: https://arxiv.org/abs/2211.00593 "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small"

[18]: https://arxiv.org/abs/2502.14010 "Which Attention Heads Matter for In-Context Learning?"

[19]: https://arxiv.org/abs/2309.16042 "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods"
