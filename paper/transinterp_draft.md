# TransInterp: A Reproducible Toolkit for Inspecting Transformer Computations

**Status:** Research draft and implementation proposal  
**Author:** Zoe Faith Gumise  
**Contact:** gumisezoe@gmail.com

## Abstract

Mechanistic interpretability seeks explanations of neural network behavior in terms of internal representations and computational mechanisms. Yet analysis code is frequently coupled to a single model family, a single notebook, or an unrecoverable sequence of tensor transformations. We present TransInterp, an open-source toolkit organized around normalized activation records, attention-level measurements, and layer-wise decision trajectories. The system separates model adapters from analysis functions and treats visualizations as reproducible views over saved artifacts. We describe the design, an evaluation protocol for faithfulness and stability, and limitations concerning identifiability, attention interpretation, feature superposition, and model-specific instrumentation. The project is intended as infrastructure for comparative research rather than as a claim that any one visualization constitutes an explanation.

## 1. Introduction

Transformer models perform useful computations through distributed representations, residual streams, attention, and feed-forward transformations. Mechanistic studies have shown that targeted circuit analysis can connect internal computation to behavior, while feature-learning approaches have investigated whether learned directions can be decomposed into more interpretable components [1] [2] [3]. These results motivate tooling that can preserve the link between raw internal states, analysis decisions, and the final claim.

The practical problem is reproducibility. A researcher may need to compare several layers, inspect head-level attention, project hidden states into a feature basis, and explain when a target prediction becomes likely. These operations have different tensor contracts and memory requirements. If each experiment implements them ad hoc, small discrepancies in tokenization, masking, indexing, or dtype can be mistaken for scientific findings.

TransInterp addresses this problem with three design commitments: semantic module names at the adapter boundary; explicit tensor and metadata contracts; and artifacts that are sufficient to regenerate figures. The central research question is: **Can a small, model-agnostic analysis layer make mechanistic interpretability experiments easier to audit without encouraging stronger claims than the evidence supports?**

## 2. Related work

The toolkit is informed by circuit-level analyses of Transformer behavior [3], dictionary-learning approaches to latent features [1] [2], and work studying sparse autoencoder features [4]. Attention analysis is included as one diagnostic among several rather than a privileged explanation method. Attribution and probing methods can be useful comparison baselines, but TransInterp emphasizes internal state trajectories and intervention-ready records.

## 3. System design

### 3.1 Normalized records

An `ActivationRecord` maps semantic names to tensors and stores metadata separately. The record does not assume a model's internal class names. A model adapter may expose `block.05.residual`, `block.05.mlp`, and `block.05.attention` even when the underlying framework uses a different module hierarchy.

### 3.2 Latent feature extraction

Feature extraction operates on hidden states shaped `(batch, tokens, d_model)`. Two extraction strategies are implemented. A PCA basis (`fit_pca`) gives a dense, orthonormal decomposition that is fast, deterministic, and useful as a first pass, at the cost of components that need not align with individual semantic concepts. A sparse autoencoder (`SparseAutoencoder`) follows the standard dictionary-learning recipe from prior work on decomposing language-model activations: a linear encoder with a non-negative activation, an L1 sparsity penalty on the code, and a linear decoder trained to reconstruct the input from that code. Both interfaces return feature scores that a researcher can pass to `top_activating_examples` for qualitative inspection. Feature discovery is not equivalent to semantic labeling; a coherent top-k list for a direction is evidence, not proof, and labels must be tested against counterexamples and, where possible, an intervention.

### 3.3 Attention dissection

Attention utilities expose entropy, target-token mass, and top-k token edges for tensors shaped `(batch, heads, query_tokens, source_tokens)`. Two additional detectors target well-characterized head behaviors: an induction-head score, which measures how much attention mass a head places on the token that previously followed a repeated token, and a head-similarity matrix, which reports pairwise cosine similarity between heads' flattened attention patterns as a hint of redundant or complementary roles. All of these measurements can identify focused, diffuse, or repeated patterns, but they do not alone establish that a head caused a prediction. Future releases will pair edge summaries with activation patching and head ablation protocols, and the current caller-supplied intervention ranking in `attention.circuits` is the intended bridge from a descriptive score to a causal claim.

### 3.4 Decision trajectories

A decision trajectory stacks logits or probabilities at selected layers. It can answer descriptive questions such as when a target class overtakes alternatives, whether a prediction is stable, and which competing classes exchange rank. In causal experiments, trajectories can be computed for clean and corrupted runs and compared under an intervention.

### 3.5 Visualization

Plots are generated from records and configurations, not hidden notebook state. The implemented static plots cover decision trajectories, single-head attention heatmaps, and two-dimensional feature-space scatter plots over a fitted basis. A separate graph builder converts a head's attention weights into a plain, JSON-serializable node/edge structure intended for external interactive viewers rather than for a bundled renderer, keeping the core package's plotting dependencies small. Every figure should carry the model revision, prompt identifier, tokenization summary, and analysis parameters in a sidecar JSON file. Interactive graph views are useful for exploration, while static plots remain the archival representation.

## 4. Evaluation protocol

We propose four evaluation dimensions. **Numerical correctness** checks metric outputs against hand-computed fixtures and reference implementations. **Stability** measures whether conclusions persist under seeds, equivalent tokenization batches, and small perturbations. **Faithfulness** compares an interpretability claim with interventions such as activation patching, feature ablation, or head masking. **Usability** records whether an independent researcher can regenerate a figure from the artifact bundle and configuration.

| Dimension | Example measure | Minimum report |
|---|---|---|
| Numerical correctness | Absolute error on synthetic tensors | Fixture, tolerance, dtype |
| Stability | Rank correlation across seeds | Number of runs and variance |
| Faithfulness | Behavioral change after intervention | Control and effect size |
| Reproducibility | Independent artifact-to-figure rerun | Environment and revision |

A benchmark suite should include synthetic attention matrices, a small public Transformer, and at least one task where the predicted class changes during the forward computation. The suite should report memory use because capture policies can change which models are feasible to inspect.

## 5. Limitations and responsible use

Internal representations are often distributed and context-dependent. A feature direction that looks coherent on a small prompt set may fail under distribution shift. Attention weights are conditional routing coefficients, not automatically explanations. Logit-lens projections depend on the chosen unembedding and normalization conventions. A successful intervention can show causal relevance without providing a complete human-readable account of the mechanism. These limitations should appear in experiment reports, not only in documentation.

Researchers should avoid uploading private prompts or sensitive datasets to shared artifact stores. Model and dataset licenses must be checked independently. Published figures should identify the model revision, data slice, and any filtering that could affect interpretation.

## 6. Conclusion

TransInterp proposes a modest infrastructure contribution: a stable boundary between model instrumentation, tensor analysis, and research communication. Its value should be judged by whether it helps independent researchers reproduce, challenge, and extend interpretability claims. PCA and sparse-autoencoder feature extraction, induction-head and head-similarity detection, and graph export are implemented as a first version; the next milestones are model adapters, activation-patching intervention execution, distributed artifact writing, and a benchmarked evaluation suite covering the four dimensions in Section 4.

## References

[1]: https://arxiv.org/abs/2211.00593 "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
[2]: https://arxiv.org/abs/2305.06324 "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
[3]: https://arxiv.org/abs/2309.16042 "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small"
[4]: https://arxiv.org/abs/2404.14082 "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
