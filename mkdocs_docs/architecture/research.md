# Research

*Every decision in palingenesis traces to a specific paper. This page documents the lineage.*

---

## The reading process

We scanned 352 papers from arXiv published between 2024 and mid-2026, focusing on:

- Optimizers and learning rate schedules
- Memory efficiency techniques
- Loss functions and token weighting
- Data curation and curriculum
- Training stability
- Distributed systems
- SFT-to-RL transitions

Each paper was evaluated on: (1) does it compose with our existing stack? (2) is the claimed improvement plausible at our target scale (0.8B-35B)? (3) does it add complexity proportional to its benefit?

Implemented does not mean reproduced: unless a page says otherwise, the gains below are the papers' own claims, mostly from pretraining or other settings. Each technique is opt-in or documented as such, and checked against its paper's definition in the test suite.

---

## Implemented techniques

### Optimizers

| Technique | Paper | Key insight | Our implementation |
|-----------|-------|-------------|-------------------|
| Hyperball | Wen et al., Stanford, Jun 2026 | Scale-invariant layers only care about weight *direction*. Fix the norm, optimize the angle. | `optim.Hyperball` — wraps any base optimizer; projection after each step (Algorithm 1) |
| MONA | Li et al., Meituan, May 2026 | EMA of gradient differences ≈ Hessian·Δθ. Augments gradients with curvature before orthogonalization. | `optim.MONAAcceleration` — bf16 buffers, streaming computation |
| Muon | Jordan et al., 2024 | Newton-Schulz polar decomposition on momentum = steepest descent under spectral norm. | PyTorch native `torch.optim.Muon` |
| Lion 8-bit | Chen et al., 2023 + bitsandbytes | Sign-based update, one momentum buffer. 4 bytes/param. | `bitsandbytes.optim.Lion8bit` |
| AdamC | Defazio, 2025 | Weight decay × (γ_t/γ_max) prevents gradient explosion at end of training. | `optim.AdamCCorrection` |
| SAGE | Lee & Kim, Apr 2026 | O(d) adaptive damper for embeddings. Bounded ≤ 1.0. Fixes Lion/Muon on sparse gradients. | `optim.SAGE` |

### Schedulers

| Technique | Paper | Key insight |
|-----------|-------|-------------|
| Power-decay | Li et al., Peking, Feb 2026 | η(z) = η_peak·(1-z/N)^γ with γ≈4, derived as better than cosine in a functional-scaling-law model of pretraining (capacity β > 3). |
| WSD | Hu et al., 2024 | Maintain peak LR for 80% of training, decay only at end. Optimal for hard tasks. |

### Loss

| Technique | Paper | Key insight |
|-----------|-------|-------------|
| DEFT | Wang et al., Feb 2026 (arXiv:2602.11424) | CE gated by `p^α`, α = Σp² (Rényi-2 collision probability), stop-gradient. Parameter-free; NLL and DFT are its α→0 and α=1 ends. Reports math-reasoning gains from base models; not reproduced here. |
| DFT | Wu et al., 2025 | CE weighted by the (stop-gradient) target probability `p`. |
| InfoSFT | Sabbaghi et al., May 2025 (arXiv:2605.14967) | Weight `q·[logit(p̄) − logit(q)]₊`, stop-gradient, p̄ = 0.93 (palingenesis also normalises it to mean 1 per batch). |
| Chunked CE | Aligned with torchtitan | Split [B,S,V] logit computation into N chunks. FSDP-aware reshard management. |
| Cut Cross-Entropy | Apple, ICLR 2025 | Computes CE without materializing logits. O(1) memory. Triton kernel. |

### Memory

| Technique | Paper | Key insight |
|-----------|-------|-------------|
| Gradient release | FORGE, Jun 2026 | Fuse optimizer step into backward hook. Peak grad memory = one tensor. |
| Selective differentiation | Apr 2024 | Frozen layers skip activation saving entirely. |
| FSDP2 per-layer sharding | Meta (torchtitan) | Each layer is an independent FSDP unit → communication overlaps with compute. |

### Stability

| Technique | Paper | Key insight |
|-----------|-------|-------------|
| AdaGC | ICML 2026 | Per-tensor EMA of gradient norms. Clips relative to each tensor's own history. |
| Spike detection | ZClip-inspired, 2025 | Z-score on grad norms. Skip update if anomalous. Only updates stats with non-spike values. |
| EMA | TMLR 2024 | Shadow weights on CPU. Better generalization, noise robustness. |
| Base model SLERP | SFA, Jan 2025 | Periodic merge-back toward pretrained weights. Spherical interpolation preserves norms. |

### Data

| Technique | Paper | Key insight |
|-----------|-------|-------------|
| TFP packing | Dong et al., Aug 2024 | Greedy TSP ordering with threshold filtering, so related samples share a packed sequence. Needs packed documents to see each other, which palingenesis prevents: available as a standalone utility only. |
| ECHO | ICML 2026 | Train on tool/observation tokens too. Model becomes world model. |
| J-shaped difficulty | Heuristic, loosely after 2605.12906 and 2502.02797 | 20% easy + 50% medium + 25% hard + 5% very hard. Neither paper tests a difficulty *mixture*: 2605.12906 trains base math models on single-difficulty subsets and finds the best difficulty moves harder as data grows; 2502.02797 (FLOW) re-weights the loss toward low-loss samples. |
| HES scoring | Li et al., May 2026 (arXiv:2605.22389) | Sum of the top 0.5% token entropies of each *response*; top-20% selection ≈ full-set SFT on math/STEM, worse on code. palingenesis computes it over the whole rendered chat and no selection strategy uses it yet. |
| mSFT | Koh et al., Mar 2026 (arXiv:2603.21606) | Exclude the earliest-overfitting sub-dataset and roll back to its best checkpoint; +1.8 points on 10 small tasks (base models). `msft.AdaptiveSourceTracker` (a soft-decay variant) exists but is **not wired into training**: `data.msft_tracking` has no effect. |

### SFT → RL

| Finding | Paper | Implication |
|---------|-------|-------------|
| Entropy collapse kills GRPO | Aphale & Liu, Jun 2026 (arXiv:2606.18487) | Over-trained SFT lowers entropy and later GRPO peaks; their threshold (0.18 nats mean next-token entropy, Qwen2.5-Coder-3B) is "not universal". |
| Excessive SFT destroys plasticity | Liu et al., Jun 2026 (arXiv:2606.09932) | A 32-epoch SFT checkpoint trains worse under RL than a 2-epoch one; the paper gives no general epoch limit. |
| SFT and RL data must be disjoint | Apr 2026 | Overlap causes interference patterns. |
| Data quality > RL algorithm | CacheRL, Jun 2026 (arXiv:2606.14179) | In one tool-calling study (Qwen3-4B-Thinking, GRPO on cached environments) RL added stability but little accuracy over strong SFT. |

---

## Papers we read but didn't implement (and why)

| Paper | Why not |
|-------|---------|
| AdEMAMix (dual EMA) | Needs thousands of steps for slow EMA to fill. SFT runs are too short. |
| AdamS (momentum as normalizer) | Reported on par with AdamW; no reason to add it. |
| GaLore (low-rank gradient) | Lion8bit + gradient release already solves memory. GaLore adds complexity for marginal gain. |
| RASFT (rollout-adaptive SFT) | Requires on-policy rollout generation. Fundamentally changes the training paradigm. |
| PACT (privileged traces) | RL infrastructure needed. Out of scope for an SFT library. |
| ScheduleFree+ | Not validated with Muon. Available as option but not default. |
| MegaTrain (100B on 1 GPU) | CPU-streamed layer-by-layer. Too slow for practical SFT iteration. |

---

## How to read the source

Every optimization in the codebase has a comment block citing its paper. Example from `optim.py`:

```python
# ==============================================================================
# HYPERBALL: NORM-CONSTRAINED OPTIMIZER WRAPPER
# Paper: "Fantastic Pretraining Optimizers II: Hyperball" (Stanford, 2026)
# arxiv:2606.16899
#
# Key insight: For scale-invariant layers (after LayerNorm/RMSNorm), only the
# DIRECTION of the weight matrix matters. Weight decay indirectly controls the
# angular learning rate via equilibrium norm. Hyperball makes this explicit.
# ==============================================================================
```

If you want to understand why something is the way it is, the arxiv link is always there.
