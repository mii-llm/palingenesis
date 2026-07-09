# ★ Paper Analysis — Integration Candidates for palingenesis

This document evaluates all papers marked ★ in NEXT_STEPS.md, with deep analysis of integration value, implementation complexity, and priority ordering for this boutique-quality library.

---

## Tier 1: High-Value, Should Integrate

### 1. Hyperball (2606.16899) — Stanford/Princeton

**What it does:** A simple optimizer *wrapper* that constrains weight matrices to a hypersphere of fixed radius. Instead of letting weight decay indirectly control the angular learning rate via equilibrium norm dynamics, Hyperball makes this explicit: normalize the update, take a step of size η·R, project back to the sphere.

**Key results:**
- MuonH: 20-30% token-equivalent speedup over MuonW at 1.2B scale (vs only ~10% for raw Muon)
- LR transfer: optimal LR drift across depth/width is ~1.4× (vs 2-4× for weight decay baselines)
- Works with any base optimizer (Adam, Muon, KL-SOAP all tested)
- 5-line wrapper on top of any existing optimizer

**Integration value: VERY HIGH**
- We already have Muon. Hyperball is literally 5 lines of code wrapping it.
- The 20-30% speedup at scale is massive — it fixes the known problem of Muon gains shrinking.
- Zero memory overhead, negligible compute overhead.
- The theoretical motivation is clean: scale-invariant layers (everything after RMSNorm in a Transformer) only care about the *direction* of W, not the norm. Hyperball makes this explicit.

**Implementation plan:**
- Apply Hyperball to attention + MLP weight matrices
- Keep Adam for embeddings, norms, biases (their norm carries semantic meaning)
- R = ‖W₀‖_F (set once at init)
- Update: W_{t+1} = R · Normalize(W_t - η_t · R · Normalize(u_t))

**Compatibility notes:**
- Composes trivially with gradient_release (element-wise optimizer requirement unaffected)
- Composes with Lion/Muon/AdamW — it wraps *any* base optimizer
- Replaces weight decay for constrained matrices (WD becomes meaningless when norm is fixed)
- For FSDP: normalize locally within each shard (Frobenius norm decomposes)

**Risk:** Low. Worst case it performs the same as weight decay. The theory is well-motivated.

---

### 2. MONA (2605.26842) — Meituan

**What it does:** Adds a curvature-aware acceleration term to Muon. Before orthogonalization, the raw gradient is augmented with an EMA of gradient *differences*: `G̃_k = G_k + α·A_k` where `A_k = β_a·A_{k-1} + (1-β_a)·(G_k - G_{k-1})`.

**Key results:**
- Outperforms both Muon and AdamW across 1B, 6B, 68B MoE models on 1T tokens
- Average benchmark score: 0.4557 (MONA) vs 0.4478 (Muon) vs 0.4382 (AdamW) at 68B/700B tokens
- MONA-Lite (bf16 buffers + streaming gradient): 75% memory overhead reduction, identical performance
- Negligible wall-clock overhead (<1% at iteration level)

**Integration value: HIGH**
- Direct upgrade to our existing Muon implementation
- The acceleration term adds curvature awareness: it preferentially escapes sharp minima
- MONA-Lite makes the overhead minimal (store A_k in bf16, stream G_{k-1} in-place)
- Validated at massive scale (68B, 1T tokens) — this isn't a toy result

**Implementation plan:**
- Add acceleration buffer A_k (bf16) to Muon's state
- Store previous gradient G_{k-1} via streaming (compute diff in-place, overwrite)
- Default hyperparams: β_a = 0.975 (scale-dependent), α = -1/(2(1-β_a))
- Apply acceleration *before* momentum and Newton-Schulz (Algorithm 1 in paper)

**Compatibility notes:**
- Composable with Hyperball (acceleration → momentum → Newton-Schulz → Hyperball projection)
- Extra memory: ~0.5 gradient buffers in bf16 (after MONA-Lite optimization)
- For our gradient_release setup: needs GA=1 (already required), streaming computation is natural

**Risk:** Low. Well-validated at scale. The only tuning is β_a which scales predictably with model size.

---

### 3. TFP — Threshold Filtering Packing (2408.09327)

**What it does:** Instead of random packing, uses semantic embeddings to group *related but diverse* samples within the same pack. Uses a TSP-inspired greedy algorithm on embedding distances with a threshold filter to prevent overly-similar samples from co-occurring.

**Key results:**
- +7% on GSM8K, +4% on HumanEval over random packing
- +15% on bias benchmarks (fairness improvement)
- Related context within a pack provides useful few-shot signal across document boundaries
- Minimal overhead (one-time embedding + ordering during data preparation)

**Integration value: HIGH**
- We already have sorted-length packing with greedy bin-packing. TFP is a *better* ordering strategy that replaces the "sort by length" step.
- The insight: random packing wastes the cross-document context window; related-but-diverse samples create implicit few-shot learning.
- Implementation is in the data preparation pipeline, not the training loop — zero runtime cost.

**Implementation plan:**
- During data prep: embed all samples with `sentence-transformers/all-MiniLM-L6-v2` (~22M params, fast)
- Build a nearest-neighbor graph with cosine similarity
- Apply greedy TSP traversal with threshold filtering (drop edges below sim_threshold)
- Feed the resulting ordering into our existing greedy bin-packing
- Threshold ≈ 0.3-0.5 (prevents too-similar samples, allows related context)

**Compatibility notes:**
- Orthogonal to all training-loop optimizations
- Compatible with our FA2 document masking (samples within a pack still get position_id resets)
- One-time preprocessing cost per dataset (cached)
- Needs `sentence-transformers` as optional dependency for data prep only

**Risk:** Very low. Even if semantic ordering doesn't help, it can't hurt vs random. The bin-packing efficiency is unchanged.

---

## Tier 2: Solid Value, Worth Integrating

### 4. ScheduleFree+ (2605.19095) — Meta FAIR

**What it does:** A learning-rate-free and schedule-free optimizer for LLM training. Replaces the learning rate schedule with aggressive iterate averaging. Key innovations: inner momentum for large-batch stability, inverse-gradient-norm step sizing, Polyak step size for automatic LR, and increasing outer-β for long training.

**Key results:**
- 31% training time reduction vs WSD schedules at 1000 tokens/param
- Automatic LR (Polyak step size) matches tuned grid search
- Anytime stopping (no schedule to finish)
- Works with any base optimizer (AdamW, Muon)

**Integration value: MEDIUM-HIGH**
- Excellent for our "autopilot" mode where users want zero hyperparameter tuning
- The 31% improvement is specifically for *long* training runs (high tokens-per-parameter)
- Anytime stopping is very practical for SFT where you don't know the optimal duration
- Eliminates the need to specify warmup ratio, min_lr_ratio, scheduler type

**Implementation plan:**
- Implement as alternative scheduler in `build_scheduler()`
- Maintain z (fast iterate), x (averaged iterate), y (eval point)
- Use Polyak step size with fully-decoupled AdamC for weight decay
- β annealing: 0.8 → 0.965 over training duration
- Return x for evaluation, compute gradients at y

**Compatibility notes:**
- Mutually exclusive with our current scheduler types (replaces them)
- Requires fully-decoupled weight decay (γ² scaling) — use AdamC
- The x vs z distinction requires careful checkpoint handling
- Not yet validated with Muon (paper uses AdamW) — needs testing

**Risk:** Medium. The Polyak step size hasn't been validated with Muon or Lion. The β annealing schedule needs tuning for SFT (paper focuses on pretraining). Good as an *option*, not as default.

---

### 5. SAGE (2604.07663) — Embedding Optimizer

**What it does:** Solves the "embedding layer dilemma" for sign-based optimizers (Lion, Muon). Embeddings have sparse, high-variance gradients that break stateless/sign-based methods. SAGE uses an O(d) adaptive damper based on per-dimension L1 gradient norms, bounded by 1.0.

**Key results:**
- SAGE-Hybrid (SinkGD for dense + SAGE for embeddings): 24.33 PPL at 1.3B (vs 27.81 AdamW, 28.37 Lion)
- 50% optimizer memory reduction vs AdamW for embeddings
- Enables higher learning rates than Lion (safe damping prevents instability)
- Scales better at larger models (gap widens with size)

**Integration value: MEDIUM-HIGH for Qwen3.5 (262K vocab)**
- Qwen3.5 has 262K vocabulary. The embedding layer is *massive*.
- Currently we use Lion8bit for everything including embeddings. SAGE could improve embedding optimization specifically.
- The O(d) state vs O(V·d) for Adam's second moment: significant memory savings
- Our `freeze_non_attention: true` mode means embeddings are frozen anyway for Qwen3.5, but for other models (Llama, Gemma) this matters.

**Implementation plan:**
- Implement SAGE as standalone optimizer class
- Use in hybrid mode: SAGE for embeddings + Muon/Lion for weight matrices
- Adaptive scale: σ_rms / (Ŝ_j + ε), clipped to [0, 1.0]
- EMA state St for mean absolute gradient per embedding dimension

**Compatibility notes:**
- Pairs naturally with our `_HybridOptimizer` pattern
- For Qwen3.5 with freeze_non_attention: only relevant if we train embeddings (rare)
- For Llama/Gemma: direct replacement of AdamW on the embedding layer
- Compatible with gradient_release (element-wise, GA=1 requirement met)

**Risk:** Low for the hybrid use case. The paper validates at 1.3B — needs testing at 4B+.

---

### 6. AdEMAMix (2409.03137) — EPFL/Apple

**What it does:** Adds a *second* momentum EMA (β₃ = 0.9999) to Adam, combining fast response (β₁ = 0.9) with very long memory (~7000 steps). The update becomes (m̂₁ + α·m₂) / √v̂.

**Key results:**
- 1.3B LLM trained on 101B tokens performs like AdamW on 197B tokens (+95% data efficiency)
- Significantly slows forgetting during training
- Mid-training switch from AdamW → AdEMAMix works (no restart needed)
- Mamba (non-Transformer) also benefits

**Integration value: MEDIUM**
- The +95% data efficiency is extraordinary, but:
  - Only validated with cosine schedule, fixed seq length (1024), RedPajama
  - Not validated with Muon or Lion (uses AdamW as base)
  - Requires long training runs to benefit (β₃ = 0.9999 needs thousands of steps to fill)
  - Extra memory: one full-size buffer for m₂ (unless β₁=0)
- For our SFT use case: training runs are typically 1-3 epochs on small-to-medium datasets. The slow momentum might not have time to be useful.
- Better suited for pretraining or very large-scale SFT (50K+ samples, multiple epochs)

**Implementation plan:**
- Add as optimizer variant "ademamix" in build_optimizer
- Schedule β₃ and α warmup (prevents early instability)
- β₁=0 variant eliminates extra buffer (same memory as AdamW)
- Compose with LLRD layer groups

**Compatibility notes:**
- Not trivially compatible with Muon (Muon uses Newton-Schulz, not m/√v)
- Compatible with gradient_release if using element-wise variant
- The forgetting reduction aligns with our pretraining replay strategy

**Risk:** Medium. Unclear benefit for short SFT. Worth offering as an option for long training.

---

## Tier 3: Informational / Future Work

### 7. AdamS (2505.16363) — Peking University

**What it does:** Replaces Adam's second-moment EMA with `ν_t = β₂·m²_{t-1} + (1-β₂)·g²_t` — the momentum squared serves as the normalizer. Same memory as SGD+momentum.

**Assessment:** Interesting but marginal over what we have.
- We already have Lion8bit (4 bytes/param) and gradient_release.
- AdamS matches AdamW but doesn't exceed it.
- The memory savings vs AdamW are real but we've already solved this with Lion/gradient_release.
- **Verdict: Skip.** Not enough delta over existing solutions.

---

### 8. RASFT (2606.07006) — Rollout-Adaptive SFT

**What it does:** Calibrates expert supervision based on model's on-policy rollout success rate. Hard problems get more expert weight; easy problems use self-generated trajectories.

**Assessment:** Conceptually beautiful but architecturally complex.
- Requires on-policy rollout generation during training (K rollouts per sample)
- Requires a verifier (final-answer matching for math, test execution for code)
- The solvability-adaptive weighting is smart but adds massive infrastructure
- +2-3 points over DFT on math benchmarks (which we already implement)
- **Verdict: Track, don't implement.** This is a training *paradigm* not a module. Would require restructuring the entire training loop. Better suited for RL pipelines.

---

### 9. PACT (2606.16215) — Privileged Trace Co-Training

**What it does:** Uses expert traces only during *optimization* (not during rollout). Combines trace-conditioned RL with component-aware SFT that anneals reasoning supervision.

**Assessment:** Excellent for agentic RL training, but:
- Requires RL infrastructure (GRPO, rollouts, rewards)
- Our library is SFT-focused. PACT bridges SFT↔RL.
- The component-aware SFT with annealing is the most portable idea: supervise tool-call tokens fully, anneal reasoning prefix gradually.
- **Verdict: Extract the component-aware SFT annealing idea.** Could be added to our loss plugins as a "progressive_supervision" mode. But the full PACT is out of scope for an SFT library.

---

### 10. CacheRL (2606.14179) — Accenture

**What it does:** Trains small agent models via cached rollouts + hybrid reward. Key finding: "data quality > RL algorithm."

**Assessment:** The finding validates our approach (focus on high-quality SFT data, use DEFT/DFT for token weighting). No direct implementation opportunity.
- **Verdict: Informational only.** Confirms our design choices are correct.

---

### 11. SFT Overtraining → Rank Inversion (2606.18487) and Plasticity Loss (2606.09932)

**What they do:** Both papers show that excessive SFT kills plasticity for subsequent RL. Entropy collapse predicts GRPO failure.

**Assessment:** Critical for users who plan SFT→RL pipelines.
- Our health monitoring already tracks entropy (spike detection, GNS)
- Could add an "RL readiness" diagnostic: measure output entropy, warn if collapsed
- The "Rejuvenation" method (base-anchored fusion + neuron reset) is interesting but only relevant post-SFT

**Actionable items:**
1. Add entropy monitoring to health.py (warn if entropy drops below threshold)
2. Document in training_guide.md: "if planning RL after SFT, stop before entropy collapses"
3. The SFT-GRPO disjoint data finding (2604.13515) reinforces: keep SFT and RL data separate

**Verdict: Add monitoring, document guidance. Don't implement Rejuvenation (out of scope).**

---

### 12. Dataset Decomposition (2405.13226) — Apple

**What it does:** Variable sequence length training with short→long curriculum. Decompose dataset into power-of-2 length buckets, sample with curriculum during training.

**Assessment:** We already have sorted-length packing which achieves similar benefits.
- DD's advantage: quadratic attention savings from short sequences early in training
- Our packing fills sequences to max_seq_len, avoiding the short-sequence waste
- The curriculum aspect (start short, grow long) could complement our existing approach
- **Verdict: Consider as enhancement to data.py.** Add optional `seq_len_curriculum` that ramps max_seq_len during training. Simple to implement, orthogonal to packing.

---

### 13. Optimal LR Schedules (2602.06797) — Peking University

**What it does:** Derives optimal LR schedules from functional scaling laws. Key finding: power-decay `η*(z) = η_peak · (1-z/N)^(2β-1)` is optimal for easy tasks; WSD is optimal for hard tasks.

**Assessment:** Theoretically elegant, practically we already support power/cosine/WSD.
- The main actionable insight: the capacity saturation phenomenon shows cosine (γ=2) is suboptimal when β > 3.
- Power-decay with γ=2β-1 ≈ 4-5 for typical LLMs would be better.
- **Verdict: Add "power_decay" scheduler variant with configurable γ.** Low-effort, could improve over cosine.

---

### 14. DIVE (2603.11076) — Scaling Diversity in Agentic Tasks

**What it does:** Evidence-first synthesis of diverse agentic tasks. Execute tools first, derive tasks from traces. +22 points on OOD benchmarks.

**Assessment:** Data synthesis methodology, not a training technique.
- Validates that diversity >> quantity for tool-use generalization
- Relevant for our data preparation pipeline, not the trainer itself
- **Verdict: Informational. Document the finding: diverse tool combinations matter more than dataset size.**

---

### 15. Other Starred Papers (AgenticQwen, SFT-GRPO disjoint, etc.)

Quick assessments:
- **AgenticQwen (2604.21590):** Alibaba's recipe. Validates dual-flywheel (SFT + self-play). Informational.
- **SFT-GRPO disjoint (2604.13515):** Keep SFT and RL data separate. Document as guidance.
- **Scaling laws for optimizers (2602.07712):** How Muon scales vs AdamW. Validates Hyperball's fix.
- **Complexity-aware fine-tuning (2506.21220):** Entropy-based split. Could enhance our data filtering.
- **FlashOptim (2602.23349):** Tighter optimizer quantization. Marginal over bitsandbytes.
- **GaLore (2403.03507):** Low-rank gradient. Alternative to 8-bit. Interesting but we have Lion8bit.
- **Quartet FP4 (2505.14669):** Blackwell-only. Future hardware.
- **Power Scheduler (2408.13359):** Batch/token agnostic LR. Subsumed by our autopilot logic.
- **DeltaNet parallelization (2406.06484):** Relevant for Qwen3.5 long-context. Complex.
- **Unveiling SFT recipe (2412.13337):** Practical SFT guide. Informational.
- **Data difficulty vs generalization (2605.12906):** Validates curriculum approach.
- **On-Policy SFT DDT (2602.12222):** Distribution discriminant theory. Complex, unclear benefit.

---

## Priority Implementation Order

Based on: value/effort ratio, composability with existing code, risk level.

| Priority | Paper | Effort | Value | Risk |
|----------|-------|--------|-------|------|
| 1 | **Hyperball wrapper** | ~50 lines | Very High (20-30% speedup) | Very Low |
| 2 | **MONA acceleration** | ~100 lines | High (consistent gains at scale) | Low |
| 3 | **TFP semantic packing** | ~200 lines (data prep) | High (+7% quality) | Very Low |
| 4 | **ScheduleFree+** (as option) | ~300 lines | Medium-High (autopilot mode) | Medium |
| 5 | **SAGE embedding optimizer** | ~150 lines | Medium-High (for non-Qwen3.5) | Low |
| 6 | **Power-decay scheduler** | ~30 lines | Medium (better than cosine) | Very Low |
| 7 | **Entropy monitoring** | ~50 lines | Medium (RL readiness) | None |
| 8 | **AdEMAMix** (as option) | ~150 lines | Medium (long training only) | Medium |
| 9 | **Seq-len curriculum** | ~80 lines | Low-Medium | Low |

---

## Composition Strategy: The Full Stack

The most impactful combination for our primary use case (Qwen3.5-4B agentic SFT on single A100-80GB):

```
Optimizer: Muon + MONA acceleration + Hyperball projection
  - For attention/MLP weight matrices (the 25% attention blocks we can train)
  - Everything else frozen (freeze_non_attention: true)
  
Memory: gradient_release + selective_diff + Lion8bit fallback
  - Combined: ~15GB for Qwen3.5-4B (fits on 40GB consumer GPU)
  
Loss: Chunked DEFT (parameter-free, memory-efficient)
  
Data: TFP semantic packing + smart truncation + ECHO observation loss
  - Related samples provide implicit few-shot context
  - FA2 document masking prevents cross-contamination
  
Stability: AdaGC per-tensor clipping + spike detection
  
Monitoring: entropy tracking for RL-readiness + health metrics
```

For the Hyperball+MONA combination: these compose cleanly because Hyperball wraps the *output* of any optimizer (including MONA-enhanced Muon). The pipeline is:

1. Compute gradient
2. MONA: augment gradient with acceleration term
3. Muon: momentum → Newton-Schulz orthogonalization
4. Hyperball: normalize update direction, take step of size η·R, project to sphere

---

## Conclusion

The top 3 integrations (Hyperball, MONA, TFP) are all low-risk, well-validated, and compose cleanly with our existing architecture. Together they could deliver:
- 20-30% convergence speedup (Hyperball)
- Additional quality improvement from curvature-aware optimization (MONA)
- +7% benchmark quality from semantic packing (TFP)

All with minimal memory overhead and no architectural changes to the training loop.
