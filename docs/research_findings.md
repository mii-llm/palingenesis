# Research Findings & Training Recipes

Insights distilled from 348 papers (2024-2026) that inform how to use this library optimally.

## Key Findings

### 1. Data Repetition Beats Data Scaling (arxiv:2602.11149)

**Counterintuitive result**: For long-CoT reasoning SFT, training for 128 epochs on 400 samples outperforms 1 epoch on 51,200 samples by 12-26 percentage points.

**Practical implications**:
- For reasoning/agentic SFT: use fewer, higher-quality samples and train for many epochs
- Stopping criterion: train token accuracy saturation (when model memorizes training data, stop)
- No additional catastrophic forgetting from multi-epoch (measured vs single-epoch)
- 8x less compute than single-epoch on full dataset

**For our configs**: Consider `epochs: 20-50` on curated 500-2000 sample datasets for reasoning tasks.

### 2. Weight Decay Must Scale with Epochs (arxiv:2509.14786)

When training multi-epoch on limited data, the optimal weight decay is **30x larger** than standard (0.1 → 3.0).

**Why**: Multi-epoch training overfits harder. Stronger regularization is needed.

**For our configs**: When `epochs > 5` on small datasets, increase `weight_decay` significantly.

### 3. End-of-Training Gradient Explosion (arxiv:2506.02285)

With cosine LR decay, weight decay forces gradient norms to follow `||g||/||w|| = sqrt(2λ/γ_t)`. As `γ_t → 0` at end of training, this → ∞.

**Fix implemented**: `train.adamc: true` — scales weight decay by `(current_lr / peak_lr)` for each step, keeping the ratio constant.

### 4. DFT Drifts on Knowledge Tasks (arxiv:2509.23753, ICLR 2026)

DFT/DEFT improve math reasoning by +70% (per original papers) but HURT medical/knowledge tasks by -4 points due to distributional drift (KL from base model grows unbounded).

**Fix**: Add lightweight KL anchoring via `pre_rl` with `entropy_coeff: 0, kl_coeff: 0.05`.

### 5. Only 12% of Tokens Matter for Reasoning (arxiv:2510.10974)

Critical Token Fine-Tuning shows that counterfactual-identified "critical" tokens (those whose replacement causes incorrect answers) are only 12% of the data. Training only on these outperforms full SFT.

**Our approximation**: DEFT naturally down-weights non-critical tokens via its confidence-based trust gate.

### 6. Batch Size Ramp = 36% Faster Training (arxiv:2510.14717)

"Seesaw": When the scheduler would halve LR, instead multiply LR by 1/√2 and double batch size. Same loss dynamics, 36% fewer serial steps.

**For autopilot**: Implement Seesaw-style batch ramp in the sweep/profile phases.

### 7. GNS from Norm Layers (arxiv:2411.00999, NeurIPS 2024)

Gradient Noise Scale (critical batch size proxy) can be estimated from LayerNorm gradient norms alone with zero overhead. Total GNS ≈ 1.4 × LayerNorm GNS.

**Implemented**: `health/gns` metric estimates GNS from per-micro-batch loss variance.

### 8. Per-Tensor Gradient Clipping Eliminates All Spikes (arxiv:2502.11034, ICML 2026)

AdaGC: per-tensor EMA-based clipping reduces spike scores to ZERO on Llama-2 7B, Mixtral, ERNIE. Global clipping is fundamentally flawed (temporal + spatial mismatch).

**Implemented**: `train.adagc: true` with `adagc_lambda: 1.5`, `adagc_beta: 0.95`.

### 9. EMA Improves Everything (arxiv:2411.18704, TMLR 2024)

Exponential Moving Average of weights improves: generalization, robustness to noisy labels, calibration, and transfer learning. Simple plug-in with minimal overhead.

**Implemented**: `train.ema: true`, `train.ema_decay: 0.999`.

### 10. Cut Cross-Entropy: Zero-Memory Loss (arxiv:2411.09009, ICLR 2025)

Apple's CCE computes CE without materializing the logit tensor. For Gemma 4 (262K vocab): saves 8+ GB per batch. Uses custom Triton kernels.

**Implemented**: Automatic when `cut-cross-entropy` is installed and no token-weighting plugin is active. Falls back to chunked CE otherwise.

---

## Architecture-Specific Insights

### Hybrid Models (Qwen3.5 — DeltaNet + Attention)

From arxiv:2604.22127: Adapting DeltaNet layers is DESTRUCTIVE (-14.8pp). Only attention layers should be trained.

**Config**: `train.freeze_non_attention: true`

### Gemma 4 (PLE Architecture)

- PLE params are like a second embedding — fragile, need aggressive LLRD
- 262K vocab makes logit memory the bottleneck (CCE critical here)
- `llrd_decay: 0.88` for E2B, `0.9` for E4B/12B

### Small Models (≤3B params)

From arxiv:2506.07712: Small models exhibit "Long CoT Degradation" — they degrade with long chain-of-thought data that helps larger models.

**Recommendation**: For models ≤3B, use shorter sequences or curriculum (short → long).

---

## Training Recipe Quick Reference

| Scenario | Loss | Regularization | Optimizer | Notes |
|----------|------|---------------|-----------|-------|
| Agentic tool-use (format learning) | DEFT | SymNoise α=5 | AdamW + AdamC | Standard recipe |
| Math reasoning (distillation) | DEFT | SymNoise α=5 | AdamW + AdaGC | High epochs on small data |
| Knowledge/Medical fine-tuning | DEFT + pre_rl (kl=0.05) | SymNoise α=7 | AdamW + AdamC | Anchor against drift |
| Long context (>8K) | Standard CE + chunked | SymNoise α=5 | AdamW + FSDP + CP | Memory constraints |
| Pre-GRPO warm start | Pre-RL mode | SymNoise α=5 | AdamW | Preserve diversity |
| Heterogeneous dataset | CADFT | SymNoise α=5 | AdamW + AdaGC + EMA | Variance reduction |
| Edge model (E2B) | DEFT | SymNoise α=7 | AdamW + EMA | Conservative, regularize |

---

## Composability Matrix

All techniques compose independently:

```
Embedding:    SymNoise ──────────────────────────────────────────────────┐
                                                                         │
Loss:         DEFT / DFT / CADFT / InfoSFT / Pre-RL / Standard CE ─────┤
                                                                         │
Gradient:     AdaGC (per-tensor clipping) ──────────────────────────────┤
                                                                         │
Optimizer:    AdamW + AdamC (corrected WD) / Schedule-Free ─────────────┤
                                                                         │
Post-step:    EMA (shadow weights) ─────────────────────────────────────┤
                                                                         │
Monitoring:   GNS / Health Tiers / Spike Detection ─────────────────────┘
```

Any combination is valid. The only constraints:
- DEFT, DFT, CADFT, InfoSFT, pre_rl are mutually exclusive (pick one loss)
- AdaGC and spike_detection can both be active (AdaGC clips, spike detector is last-resort skip)
- AdamC requires a scheduler (does nothing with schedule_free)
- CCE is only used when no token-weighting plugin is active (plugins need full logits)
