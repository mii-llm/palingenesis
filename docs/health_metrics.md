# Health Metrics & Training Diagnostics

## Philosophy

Standard training frameworks track loss and learning rate. That's like monitoring a patient's temperature only — you know they're sick when it's already 40C, but you can't predict or prevent it.

We track **leading indicators** — metrics that change BEFORE catastrophic failures happen. An AI agent monitoring these can intervene (adjust LR, stop training, flag data issues) before the run is wasted.

## Tiered Metric Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: EVERY STEP (0 ms overhead)                          │
│ • Loss mean/std/CV over sliding window                      │
│ • Token efficiency (% of batch tokens with loss)            │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: EVERY 10 STEPS (~10 ms)                             │
│ • Gradient cosine similarity (consecutive updates)          │
│ • CUDA peak memory / utilization %                          │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: EVERY 100 STEPS (~200 ms)                           │
│ • Per-layer weight norms (detect imbalance)                 │
│ • Stable rank of key matrices (collapse early warning)      │
│ • Weight drift from initialization (forgetting detection)   │
│ • Health warning count                                      │
└─────────────────────────────────────────────────────────────┘
```

## Metric Reference

### Tier 1 — Convergence & Data Quality

#### `health/loss_cv` — Loss Coefficient of Variation

**Formula**: `std(loss_window) / mean(loss_window)`

**What it measures**: How noisy training is relative to the loss magnitude.

**Interpretation**:
| Value | Meaning | Action |
|-------|---------|--------|
| < 0.1 | Very stable | Normal |
| 0.1 - 0.5 | Normal noise | OK |
| 0.5 - 1.0 | High noise | Consider larger batch |
| > 1.0 | Extremely noisy | Increase grad_accum, check data |

**Why it matters**: High CV means the effective batch size is too small for the loss landscape curvature. The optimizer is seeing inconsistent gradient signals and may oscillate instead of converge.

#### `health/token_efficiency` — Token Utilization

**Formula**: `count(labels != -100) / total_tokens_in_batch`

**What it measures**: What fraction of tokens you're actually training on.

**Interpretation**:
| Value | Meaning | Action |
|-------|---------|--------|
| > 0.5 | High utilization | Normal for short-turn chat |
| 0.2 - 0.5 | Moderate | Normal for multi-turn with system prompts |
| 0.05 - 0.2 | Low | Expected for long agentic traces |
| < 0.05 | Very low | Check masking, consider packing |

**Why it matters**: Low efficiency means you're paying GPU compute for tokens that don't contribute to learning. If it drops over time, your data pipeline might be degrading (longer system prompts, more tool outputs, etc.).

---

### Tier 2 — Optimization Dynamics

#### `health/grad_cosine_sim` — Gradient Direction Stability

**Formula**: `cosine_similarity(grad_t, grad_{t-1})`

**What it measures**: Are consecutive optimizer steps pushing the model in the same direction?

**Interpretation**:
| Value | Meaning | Action |
|-------|---------|--------|
| > 0.8 | Very stable (nearly deterministic) | Normal for large batch |
| 0.3 - 0.8 | Healthy exploration | Ideal range |
| 0.0 - 0.3 | Noisy but progressing | OK if loss is decreasing |
| < 0.0 | **Oscillating** — updates cancel each other | **Reduce LR by 2-5x** |
| < -0.3 | **Severe oscillation** | **Reduce LR by 10x or stop** |

**Why it matters**: Negative cosine similarity means you're taking step A, then step -A, then step A again. You're spending compute and going nowhere. This is the clearest signal that learning rate is too high for the current loss landscape.

**Research basis**: This is related to the "gradient noise scale" (McCandlish et al., 2018) which determines the optimal batch size. Low cosine sim = you need a larger batch or smaller LR.

#### `health/cuda_utilization_pct` — GPU Memory Pressure

**Formula**: `allocated_memory / total_gpu_memory * 100`

**Why it matters**: If utilization is >90%, you have no headroom for memory spikes (which happen during gradient checkpointing recomputation, collectives, or occasional longer sequences). An OOM crash at step 4999 of 5000 wastes your entire run.

---

### Tier 3 — Structural Health (The Expensive Ones)

#### `health/stable_rank_*` — Matrix Effective Rank

**Formula**: `||W||_F^2 / sigma_max(W)^2`

Where `||W||_F` is the Frobenius norm and `sigma_max` is the largest singular value.

**What it measures**: How many "effective dimensions" a weight matrix uses. A matrix with stable rank 1 is essentially rank-1 — all its energy is in one direction. A matrix with stable rank 100 uses 100 independent directions.

**Interpretation**:
| Value | Meaning | Action |
|-------|---------|--------|
| > 50 | High-rank, diverse representations | Healthy |
| 20 - 50 | Moderate rank | Normal for trained models |
| 5 - 20 | Low-ish rank | Monitor trend |
| < 5 | **Collapsing** | **Reduce LR, add weight decay, consider stopping** |
| < 2 | **Collapsed** | **Training is producing degenerate outputs** |

**Why it matters**: Research (arxiv:2602.01734) shows that stable rank decline PRECEDES training collapse by hundreds of steps. If you see stable rank dropping, you have time to intervene before the loss explodes.

**How we compute it efficiently**: We don't run full SVD on the entire weight matrix (that would be O(min(m,n)^2 * max(m,n))). Instead:
1. Subsample the matrix to 512x512
2. Compute `svdvals()` (singular values only, no vectors)
3. Take the first singular value for spectral norm
4. This gives an approximation that's within 5% of the true stable rank

**Cost**: ~30ms per sampled layer, we sample 5-6 layers → ~200ms per tier-3 check.

#### `health/weight_norm_ratio` — Layer Norm Imbalance

**Formula**: `max(layer_norms) / min(layer_norms)`

**What it measures**: Whether all layers have roughly the same scale of weights.

**Interpretation**:
| Value | Meaning | Action |
|-------|---------|--------|
| 1 - 5x | Uniform | Healthy |
| 5 - 20x | Some imbalance | Monitor |
| 20 - 100x | Significant imbalance | Some layers may be undertrained |
| > 100x | **Extreme** | **Layer-wise LR or check initialization** |

**Why it matters**: If early layers have 100x larger norms than late layers, gradient flow is severely imbalanced. Early layers dominate the loss landscape and late layers barely update. This happens when LR is too high for some layers or weight decay is insufficient.

#### `health/weight_drift_mean` / `health/weight_drift_max` — Catastrophic Forgetting

**Formula**: `|current_norm - init_norm| / init_norm` (per-parameter, aggregated)

**What it measures**: How far the model has moved from its pretrained initialization.

**Interpretation** (for SFT from an instruct model):
| Value | Meaning | Action |
|-------|---------|--------|
| < 5% | Minimal change | May be undertrained |
| 5 - 15% | Moderate | Ideal for SFT |
| 15 - 30% | Significant | Check if general capabilities are preserved |
| > 30% | **High drift** | **Likely catastrophic forgetting** |
| > 50% | **Extreme** | **Model is probably broken for general use** |

**Why it matters**: SFT should teach the model new behaviors while preserving existing capabilities. If weights drift too far, the model "forgets" how to do basic language modeling. This is especially important for agentic models that need to maintain reasoning, tool use, AND follow instructions.

---

## Alert Thresholds

The health monitor emits warnings (logged + counted in `health/warnings`) when:

| Condition | Severity | Likely cause |
|-----------|----------|--------------|
| stable_rank < 3 | Critical | LR too high, training collapsing |
| weight_norm_ratio > 100 | Warning | Layer imbalance, possibly dead layers |
| weight_drift_max > 50% | Warning | Catastrophic forgetting |
| grad_cosine_sim < -0.1 | Warning | LR too high, oscillating |
| loss_cv > 1.0 | Warning | Batch too small or data issues |
| cuda_utilization > 95% | Warning | OOM risk |

## Using Health Metrics for Hyperparameter Tuning

An agent monitoring these metrics can make informed decisions:

```
IF loss_cv > 0.8 AND loss is not decreasing:
    → Increase gradient_accumulation_steps (effectively larger batch)

IF grad_cosine_sim < 0 for 3 consecutive checks:
    → Reduce learning_rate by 3x

IF stable_rank is declining (compare to 200 steps ago):
    → Reduce learning_rate by 2x
    → Increase weight_decay

IF weight_drift_max > 0.3:
    → Reduce learning_rate
    → Reduce max_steps (stop earlier)
    → Consider smaller dataset

IF token_efficiency < 0.05:
    → Enable data.packing = true
    → Or filter dataset for samples with more assistant content

IF cuda_utilization > 90%:
    → Increase memory.loss_num_chunks
    → Or reduce data.max_seq_length
```

## Core Training Metrics (`train/` and `eval/`)

Logged every `logging_steps` optimizer steps:

| Metric | Meaning |
|--------|---------|
| `train/loss` | Per-token loss, averaged over the accumulation window |
| `train/ppl` | `exp(train/loss)` — human-readable scale |
| `train/lr` | Current learning rate |
| `train/grad_norm` | Global gradient norm (post clipping / AdaGC) |
| `train/tokens_per_sec` | Per-GPU throughput on loss-bearing tokens |
| `train/tokens_per_sec_global` | Throughput × world size |
| `train/tokens_total` | Cumulative trained tokens (this process) |
| `train/step_time_s` | Wall time per optimizer step |
| `train/spikes_skipped` | Cumulative steps skipped by spike detection (>5% → filter your data) |
| `train/adagc_clips` | Cumulative per-tensor AdaGC clips (if `adagc: true`) |
| `eval/loss` | Validation loss (every `data.eval_every` steps) |
| `eval/ppl` | `exp(eval/loss)` |
| `eval/gap` | `eval/loss − train/loss` — rising gap = overfitting |

With `logging.rl_readiness: true`, `health/output_entropy` is recorded every logging step — including under chunked/CCE loss where full logits are never materialized (a sample of valid positions is projected through the LM head instead).

**Note on gradient metrics**: `health/grad_cosine_sim` and `health/gw_ratio/*` inspect raw gradients after the optimizer step (gradient zeroing is deferred until after the health check). They are unavailable under `memory.gradient_release`, where gradients are freed inside the backward pass.

## Visualization in wandb/trackio

All metrics are logged with hierarchical prefixes for clean dashboard organization:

```
train/          — Core training metrics (loss, ppl, lr, grad_norm, tok/s, tokens_total)
eval/           — Validation metrics (loss, ppl, generalization gap)
health/         — All health diagnostics
health/wnorm/   — Per-layer weight norms
health/srank/   — Per-layer stable ranks
```

All metrics share `train/global_step` as x-axis (`wandb.define_metric`), so panels stay aligned across restarts — and metrics are logged without an explicit wandb step, so wandb's step-monotonicity rule can never silently drop rows. The wandb run id is persisted to `{output_dir}/tracker_run_id.json` — crash + `train.resume_from: auto` continues the SAME run instead of scattering one training over several dashboards, while a fresh start in the same output dir mints a new run id. Tracker init/log failures never kill training (they degrade to rate-limited warnings).

Recommended wandb panel layout:
1. **Overview**: loss, ppl, lr, grad_norm, tokens_per_sec
2. **Generalization**: eval/loss, eval/ppl, eval/gap
3. **Stability**: grad_cosine_sim, loss_cv, spikes_skipped, adagc_clips
4. **Model Health**: stable_rank_min, weight_norm_ratio, weight_drift, output_entropy
5. **Resources**: cuda_peak_gb, cuda_utilization_pct, step_time_s
