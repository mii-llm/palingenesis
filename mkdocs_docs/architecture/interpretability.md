# Interpretability Research

*The literature and open-source code behind [Reading Model Internals](../guides/internals.md), kept to what changed a
design decision.*

---

## Decisions at a glance

| Decision | Why | Sources |
|---|---|---|
| Hooks on decoder-layer outputs, never `output_hidden_states` | transformers 5 returns the final norm's output as the last `hidden_states` entry and keeps every `[B, T, d]` | transformers `utils/output_capturing.py` (verified in tests) |
| Residual r = n is pre-norm; lenses apply the model's own norm and head | a lens on `hidden_states` normalises the last layer twice; Qwen3.5 / Gemma norms compute x̂(1 + w) | logit lens; TransformerLens `apply_ln_to_stack`; nnsight logit-lens notes |
| Readout identity check before any lens number | wrong norm, missing softcap or wrong submodule give plausible garbage | TransformerLens `_require_readout_identity`; nnsight docs |
| Gather positions inside the hook, stop the forward after the deepest layer | memory bounded by selected rows; a mid-depth probe costs part of a prefill | baukit `Trace(stop=True)`; SAELens `stop_at_layer`; TransformerLens copies before slicing |
| Right padding | causal attention and linear-attention state never see padding at a real position | palingenesis.margins; checked on the hybrid in tests |
| Grouped splits, cluster-bootstrap CIs | rows of one question / entity / translation are not independent | Orgad et al. 2024 (bootstrap); probing-practice surveys |
| Layer chosen on validation, not test | max-over-layers on test is optimistic | standard model selection |
| Logistic probe by default, diff-of-means always reported | LR has the best recall at low FPR; diff-of-means transfers as well or better | Goldowsky-Dill et al. 2025; Paulo et al. 2024; Marks & Tegmark 2023 |
| Intervene with the diff-of-means, never the classifier weights | the mean difference is the causally used direction | Marks & Tegmark 2023; Li et al. 2023 (ITI); Im & Li 2025; Arditi et al. 2024 |
| Group-level control task, selectivity, embedding layer and length baselines | high probe accuracy alone says nothing | Hewitt & Liang 2019; Belinkov 2022 |
| Platt calibration on validation, per domain; thresholds at fixed FPR | discrimination transfers out of distribution, calibration does not; gates need low-FPR operating points | Kadavath et al. 2022; Goldowsky-Dill et al. 2025 |
| Debiased linear CKA, always next to probe transfer | biased CKA inflates when n is not ≫ d; CKA misses low-variance directions probes use | Kornblith et al. 2019; Nguyen et al. 2021; Murphy et al. 2024; Davari et al. 2022; Ding et al. 2021 |
| Steering at every position, prompt included; dose in units of μ⁺ − μ⁻ | hybrid models carry decisions in the recurrent state from the prefill; absolute coefficients do not transfer across layers | Yap 2026; Panickssery et al. 2024 (CAA); nnsight steering notes |
| Causal check against random directions of the same norm, plus directional ablation | any large push moves outputs; ablation tests necessity | Arditi et al. 2024; Elazar et al. 2021 |
| Patching between minimal twins, logit-difference metric, single layers, denoise and noise | Gaussian-noise corruption creates spurious effects; probabilities saturate; windows inflate | Zhang & Nanda 2023; Heimersheim & Nanda 2024 |
| Probe as a reward only through a frozen scorer, with warnings; labels and gates preferred | RL against a detector learns to evade it | Bailey et al. 2024; Cundy & Gleave 2025; Taufeeque et al. 2026 |
| Tuned lens: residual affine translators at zero init, KL to final, chat-distribution data | identity start, weight decay toward the logit lens; post-trained models are used in chat format | Belrose et al. 2023; tuned-lens code |
| No SAEs, no CCS, no attribution patching in v1 | see [Out of scope](#out-of-scope) | |

---

## Probing and its pitfalls

**Hewitt & Liang 2019**, *Designing and Interpreting Probes with Control Tasks*, EMNLP
([arXiv:1909.03368](https://arxiv.org/abs/1909.03368)).
- A control task gives each word type a fixed random label. Only a probe that memorises can solve it.
- Selectivity = task accuracy − control accuracy.
- The most accurate layer is not necessarily the most selective. Linear, regularised probes are the most selective.
- We use the group (question, entity) as the "type": a random label per group, drawn at the base rate. With grouped
  splits the control must stay at chance on test. When it does not, groups leak.

**Belinkov 2022**, *Probing Classifiers: Promises, Shortcomings, and Advances*, Computational Linguistics
([arXiv:2102.12452](https://arxiv.org/abs/2102.12452)).
- Probes need baselines: input embeddings, controls, and dataset shortcuts such as length or template.
- Decodable is not the same as used.
- Effect on the design: the embedding layer is in every sweep, there is a length-only AUROC, and there is a separate
  causal module.

**Ravichander et al. 2021**, *Probing the Probing Paradigm* ([arXiv:2005.00719](https://arxiv.org/abs/2005.00719)), and
**Elazar et al. 2021**, *Amnesic Probing* ([arXiv:2006.00995](https://arxiv.org/abs/2006.00995)).
- Properties are encoded whether or not the task needs them.
- Removing a decodable property may barely change behaviour.
- Effect on the design: every probe gets an ablation test against random directions.

## Self-knowledge and truthfulness probes

**Kadavath et al. 2022**, *Language Models (Mostly) Know What They Know*
([arXiv:2207.05221](https://arxiv.org/abs/2207.05221)).
- P(IK) is read at the last token of the question.
- Labels are the share correct among 30 samples at T = 1, so soft labels are supported.
- Out of distribution, AUROC partly transfers but Brier scores degrade badly. Calibration is therefore per domain, and
  AUROC and Brier are reported separately.

**Kossen et al. 2024**, *Semantic Entropy Probes* ([arXiv:2406.15927](https://arxiv.org/abs/2406.15927)).
- The "token before generation" (our `last_prompt`) works with one forward pass and no generation, which allows
  abstaining before answering. It is slightly weaker than reading after the answer.
- The probe is logistic regression.
- Semantic-entropy labels generalise better across tasks than accuracy labels. The `label` field accepts any such
  target.

**Orgad et al. 2024**, *LLMs Know More Than They Show* ([arXiv:2410.02707](https://arxiv.org/abs/2410.02707)).
- The exact-answer tokens carry the strongest signal.
- Cross-task transfer is weak and only works between tasks that share a skill.
- The test set is bootstrapped.
- Effect on the design: every sweep reports transfer to other test sets, answer-span positions are available, and
  baselines from the model's own logits should be compared on the same split.

**Marks & Tegmark 2023**, *The Geometry of Truth* (COLM 2024, [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)).
- The mass-mean probe is θ = μ⁺ − μ⁻. Its covariance-adjusted variant Σ⁻¹θ (our `lda`) is for classification.
- Adding the mass-mean direction moves outputs far more than adding the logistic direction, in 7 of 8 conditions.
- The source of the split between the classify and intervene directions.

**Burns et al. 2022** (CCS, [arXiv:2212.03827](https://arxiv.org/abs/2212.03827)) and **Farquhar et al. 2023**,
*Challenges with Unsupervised LLM Knowledge Discovery* ([arXiv:2312.10029](https://arxiv.org/abs/2312.10029)).
- Arbitrary binary features satisfy CCS's consistency structure.
- Unsupervised methods find the most prominent feature, not necessarily knowledge.
- Not implemented.

## Steering and ablation

**Turner et al. 2023**, *Steering Language Models With Activation Engineering* (ActAdd,
[arXiv:2308.10248](https://arxiv.org/abs/2308.10248)).
- The steering vector is a difference of activations, added at one layer.
- Adding at only the last position breaks syntax, and single-concept vectors do little.
- Our default adds at all positions.

**Panickssery (Rimsky) et al. 2024**, *Steering Llama 2 via Contrastive Activation Addition* (ACL 2024,
[arXiv:2312.06681](https://arxiv.org/abs/2312.06681)).
- The vector is a mean difference, added at all positions after the prompt.
- The best layers are at about 40 % of depth, with multipliers of ±1 to 2.
- Residual norms grow with depth, so a coefficient does not transfer across layers. We express doses in units of
  |μ⁺ − μ⁻| at the probe's own layer.

**Arditi et al. 2024**, *Refusal in Language Models Is Mediated by a Single Direction* (NeurIPS 2024,
[arXiv:2406.11717](https://arxiv.org/abs/2406.11717)).
- Candidate directions are differences of means at post-instruction positions and layers. They are selected by causal
  filters: ablation must bypass the behaviour, addition must induce it, KL on unrelated prompts must stay below 0.1,
  and the layer must be below 80 % of depth.
- Directional ablation is x ← x − r̂r̂ᵀx at every layer and position.
- Selection uses the log-odds of the behaviour's first tokens. Our `direction_effect` reads the same metric through
  palingenesis.margins.

**Li et al. 2023**, *Inference-Time Intervention* (NeurIPS 2023, [arXiv:2306.03341](https://arxiv.org/abs/2306.03341)).
- The mass-mean shift beats the probe-weight direction and random directions.
- Strength is scaled by the activation spread along the direction.

**Im & Li 2025**, *A Unified Understanding and Evaluation of Steering Methods*
([arXiv:2502.02716](https://arxiv.org/abs/2502.02716)).
- The mean of differences is the optimal steering vector in their model, and it beats PCA- and classifier-based
  vectors on most tasks.

**Zou et al. 2023**, *Representation Engineering* ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405)).
- Reading vectors from PCA of differences. Superseded for our purposes by the mean difference (Im & Li).

## Lenses

**nostalgebraist 2020**, *interpreting GPT: the logit lens*
([LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)).
- Apply the final norm and the unembedding to an intermediate residual.
- It fails on some models: early layers decode the input token.

**Belrose et al. 2023**, *Eliciting Latent Predictions from Transformers with the Tuned Lens*
([arXiv:2303.08112](https://arxiv.org/abs/2303.08112)) and the
[tuned-lens code](https://github.com/AlignmentResearch/tuned-lens).
- One affine translator per layer, applied residually (h + Ah + b) with zero initialisation, so it starts as the
  logit lens and weight decay pulls toward it.
- The objective is KL(final ‖ lens), and no translator is needed for the last layer.
- The code trains with SGD-Nesterov or Adam (lr 1e-3).
- Rogue high-variance dimensions and drift across layers explain logit-lens failures.

**Paulo et al. 2024**, *Does Transformer Interpretability Transfer to RNNs?* (AAAI 2025,
[arXiv:2404.05971](https://arxiv.org/abs/2404.05971)).
- On Mamba and RWKV, CAA steering works with middle layers best, and the tuned lens beats the logit lens at every
  layer.
- With tied embeddings the early logit lens decodes the input.
- In their transfer probes, diff-of-means was best or tied.
- They also show that steering the recurrent state works, which we have not implemented.

**Wendler et al. 2024**, *Do Llamas Work in English?* ([arXiv:2402.10588](https://arxiv.org/abs/2402.10588)).
- Through the logit lens, non-English prompts pass through middle layers that decode the concept in English, then
  switch to the input language late.
- This motivates the decision-token readout per layer for language switching, and the expectation that cross-lingual
  probe transfer peaks in middle layers.

## Patching

**Zhang & Nanda 2023**, *Towards Best Practices of Activation Patching* (ICLR 2024,
[arXiv:2309.16042](https://arxiv.org/abs/2309.16042)).
- Prefer symmetric token replacement to Gaussian noise, which pushes activations off distribution.
- Prefer logit differences to probabilities.
- Sliding windows of layers inflate effects, so start with single layers.

**Heimersheim & Nanda 2024**, *How to use and interpret activation patching*
([arXiv:2404.15255](https://arxiv.org/abs/2404.15255)).
- Denoising (clean into corrupt) tests sufficiency; noising tests necessity. They disagree on AND / OR structures.
- Watch for effects above 1 and backup behaviour.
- Hence `--mode denoise|noise` and the normalised effect.

**Attribution patching**: Nanda 2023
([blog](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)); Syed et al. 2023
([arXiv:2310.10348](https://arxiv.org/abs/2310.10348)); Kramár et al. 2024, AtP\*
([arXiv:2403.00745](https://arxiv.org/abs/2403.00745)).
- A first-order approximation for screening many components, with known false negatives.
- Not implemented: we patch whole residuals at a few positions, where exact patching is affordable.

## Representation similarity

**Kornblith et al. 2019**, *Similarity of Neural Network Representations Revisited* (ICML 2019,
[arXiv:1905.00414](https://arxiv.org/abs/1905.00414)).
- Linear CKA is invariant to orthogonal transforms and isotropic scaling, but not to arbitrary invertible maps.

**Nguyen et al. 2021**, *Do Wide and Deep Networks Learn the Same Things?* (ICLR 2021,
[arXiv:2010.15327](https://arxiv.org/abs/2010.15327)).
- Uses the unbiased HSIC estimator of Song et al. 2012. We compute it from d × d products without an n × n Gram matrix.

**Murphy et al. 2024**, *Correcting Biased Centered Kernel Alignment Measures*
([arXiv:2405.01012](https://arxiv.org/abs/2405.01012)).
- Biased CKA can reach its maximum on independent data when n ≪ d, so debiased CKA is the default. A test checks
  both behaviours.

**Davari et al. 2022**, *Reliability of CKA as a Similarity Measure in Deep Learning*
([arXiv:2210.16156](https://arxiv.org/abs/2210.16156)), and **Ding et al. 2021**, *Grounding Representation Similarity
Through Statistical Testing* ([arXiv:2108.01661](https://arxiv.org/abs/2108.01661)).
- CKA is sensitive to outliers and can be manipulated without changing function.
- It is dominated by the top principal components and insensitive to low-variance directions that probes rely on.
- Hence probe transfer and direction cosines next to CKA.

## Probes inside training: the evasion problem

**Bailey et al. 2024**, *Obfuscated Activations Bypass LLM Latent-Space Defenses*
([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)).
- Fine-tuning against latent monitors (logistic, MLP and diff-of-means probes, SAE probes) drives recall to near 0
  while the behaviour is kept.
- Obfuscating many layers at once is costlier, which argues for multi-layer monitors.

**Cundy & Gleave 2025**, *Preference Learning with Lie Detectors can Induce Honesty or Evasion* (NeurIPS 2025,
[arXiv:2505.13787](https://arxiv.org/abs/2505.13787)).
- With a detector-labelled reward, GRPO learned to evade unless the detector's TPR was very high and the KL to the
  reference strong.
- Off-policy DPO stayed mostly honest.
- Hence "labels and gates first" and a held-out monitor.

**Taufeeque et al. 2026**, *The Obfuscation Atlas* ([arXiv:2602.15515](https://arxiv.org/abs/2602.15515)).
- In RLVR, representations drift enough to break frozen probes even with the detector's coefficient at 0.
- Never backpropagate through the probe.
- Hence the scorer's warning when weights differ, and `compare`'s transfer columns before a probe is reused.

**Goldowsky-Dill et al. 2025**, *Detecting Strategic Deception Using Linear Probes*
([arXiv:2502.03407](https://arxiv.org/abs/2502.03407)).
- Logistic regression on standardised activations.
- Thresholds are set at 1 % FPR on a control distribution.
- Recall at low FPR is fragile across layers, even where AUROC holds. Hence the stored `fpr_*` thresholds and their
  TPR / FPR on every test set.

**Fine-tuning and mechanisms**:
- Prakash et al. 2024 ([arXiv:2402.14811](https://arxiv.org/abs/2402.14811)): fine-tuning enhances existing
  mechanisms.
- Kissane et al. 2024, *SAEs (usually) transfer between base and chat models* (Alignment Forum).
- Lee et al. 2024 ([arXiv:2401.01967](https://arxiv.org/abs/2401.01967)): DPO bypasses rather than removes.

Probes usually survive mild post-training. Measure it rather than assume it.

## Hybrid and recurrent models

- **Sharma et al. 2024**, *Locating and Editing Factual Associations in Mamba*
  ([arXiv:2404.03646](https://arxiv.org/abs/2404.03646)): causal tracing finds a transformer-like layout of facts.
- **Ali et al. 2024**, *The Hidden Attention of Mamba Models* ([arXiv:2403.01590](https://arxiv.org/abs/2403.01590)):
  selective state-space layers as implicit attention.
- **Yap 2026**, *Behavioral Steering in a 35B MoE Language Model via SAE-Decoded Probe Vectors*
  ([arXiv:2603.16335](https://arxiv.org/abs/2603.16335)). A single-author preprint on a Gated DeltaNet hybrid:
  steering only during decoding had no effect, while steering that included the prefill did. This is the reason our
  steering default covers the prompt.

We found no peer-reviewed work probing or steering Gated DeltaNet models specifically. Architecturally, these models
are pre-norm, and the residual stream between blocks is the same object after linear-attention and attention layers.
Reports therefore tag each residual with its mixer.

## Code we read

| Library | What we took |
|---|---|
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | hook context with cleanup in `finally`; recompute the final norm per layer for the lens; the readout-identity check; no weight folding for RMSNorm models |
| [nnsight](https://github.com/ndif-team/nnsight) | decoder layers return tensors (`output[0]` silently takes batch row 0: our hooks assert `[B, T, d]`); steering must persist past the prefill; collect under `no_grad` |
| [pyvene](https://github.com/stanfordnlp/pyvene) | position gather / scatter per example; a call counter tells prefill from generated tokens |
| [baukit](https://github.com/davidbau/baukit) | `StopForward` to cut the forward; roll back partial hook registration |
| [tuned-lens](https://github.com/AlignmentResearch/tuned-lens) | translator architecture and init, KL objective, per-layer backward |
| [repeng](https://github.com/vgel/repeng), [steering-vectors](https://github.com/steering-vectors/steering-vectors) | all-position steering; slicing positions relative to the current chunk breaks after prefill; plain PCA on differences removes the mean difference |
| [refusal_direction](https://github.com/andyrdt/refusal_direction) | float64 log-odds selection metric; ablation at every layer; depth and KL filters |
| [geometry-of-truth](https://github.com/saprmarks/geometry-of-truth), [deception-detection](https://github.com/ApolloResearch/deception-detection), [semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes) | probe defaults: standardised logistic regression, the covariance-adjusted mass-mean, thresholds at fixed FPR on a control set |
| [SAELens](https://github.com/jbloomAus/SAELens) | fixed-size shards with a manifest and an explicit storage dtype |

## Out of scope

- **Sparse autoencoders**: Bricken et al. 2023; Cunningham et al. 2023
  ([arXiv:2309.08600](https://arxiv.org/abs/2309.08600)); Templeton et al. 2024; Gemma Scope
  ([arXiv:2408.05147](https://arxiv.org/abs/2408.05147)).
  - They are useful for unsupervised discovery, but expensive per layer and must be revalidated after each stage.
  - On supervised concepts, linear probes and the difference of means match or beat SAE-based probes and steering.
- **CCS and other unsupervised probes**: see Farquhar et al. above.
- **Attribution patching**: a screening tool for many components. Our patching unit is a whole residual.
- **Recurrent-state probes and steering for Gated DeltaNet layers**: Paulo et al. show state steering works on RNNs.
  This is a natural extension of the hook layer.
