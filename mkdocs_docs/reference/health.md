# Health Monitoring

*Tiered diagnostics that detect problems before they ruin your run.*

---

## What gets logged every step

Core training dynamics, logged under `train/` and `eval/` on every `logging_steps` interval. All metrics share `train/global_step` as their x-axis, so wandb/trackio charts stay aligned across restarts.

| Metric | Meaning | What to watch for |
|--------|---------|-------------------|
| `train/loss` | Per-token **objective** value, averaged over the accumulation window. With DEFT/DFT-family losses this is the gated loss — numerically much smaller than a cross-entropy (0.05–0.2 is normal) | Smooth decrease. Spikes → data or LR issues |
| `train/ce_loss` | Unweighted cross-entropy on the same batches (logged under chunked DEFT, computed for free). This is the number comparable to `eval/loss` | Same scale as eval loss |
| `train/deft_gate` | Mean DEFT trust gate `p^α` ∈ (0,1]. Rising gate = model growing confident on its targets | Slow rise over training |
| `train/ppl` | `exp` of the true CE (`train/ce_loss` when available). Omitted when the objective is not a CE — `exp(DEFT)` would be meaningless | A ppl of 2-6 is typical mid-SFT |
| `train/lr` | Current learning rate from the scheduler | Matches your configured schedule |
| `train/grad_norm` | Global gradient norm (after clipping / AdaGC) | Stable band. Growth → instability brewing |
| `train/tokens_per_sec` | Per-GPU throughput on loss-bearing tokens | Sudden drops → data loading bottleneck |
| `train/tokens_per_sec_global` | Throughput × world size | Cluster-level throughput |
| `train/tokens_total` | Cumulative trained tokens (this process) | Compare runs by tokens, not steps |
| `train/step_time_s` | Wall time per optimizer step | Spikes → GC, checkpointing, or dataloader stalls |
| `train/spikes_skipped` | Cumulative steps skipped by spike detection | > 5% of steps → run `pgs prepare` to filter data |
| `train/adagc_clips` | Cumulative per-tensor clips by AdaGC (if enabled) | Steady growth is fine; explosive growth → reduce LR |
| `train/epoch` | Current epoch | — |
| `eval/loss` | Validation loss (every `data.eval_every` steps) | The number that matters. Drives best-model tracking |
| `eval/ppl` | `exp(eval/loss)` | Human-readable validation quality |
| `eval/gap` | `eval/loss − train/ce_loss` — the generalization gap, CE vs CE (falls back to `train/loss` only when the objective is a plain CE) | Rising gap = overfitting. Stop or regularize |

With `logging.rl_readiness: true`, `health/output_entropy` is also recorded every logging step — including under chunked and CCE loss, where full logits are never materialized (a sample of positions is projected through the LM head instead).

---

## How it works

The `HealthMonitor` runs at three frequencies, each with increasing cost:

| Tier | Frequency | Cost | What it measures |
|------|-----------|------|------------------|
| 1 | Every logging step | < 0.1 ms | Loss statistics, token efficiency, output entropy |
| 2 | Every 10 steps (`logging.health_tier2_every`) | ~ 10 ms | Gradient direction stability, GNS, CUDA memory |
| 3 | Every 100 steps (`logging.health_tier3_every`) | ~ 200 ms | Weight norms, stable rank, model drift from init |

All metrics are logged to wandb/trackio automatically.

!!! tip "Short test runs"
    Tier 3 first fires at `health_tier3_every` (default step 100). For smoke tests with `max_steps: 50`, lower it (e.g. `health_tier3_every: 25`) or you'll get no stable-rank/drift data at all. Both cadences should be multiples of `train.logging_steps` — health metrics are only collected on logging steps.

---

## Key signals

### Gradient cosine similarity (`health/grad_cosine_sim`)

How consistent is the gradient direction between consecutive steps?

- \> 0.5: very stable, consistent learning
- 0.1 – 0.5: normal noise level
- < 0.1: oscillating, LR may be too high
- < 0: fighting itself, definitely reduce LR

### Gradient Noise Scale (`health/gns`)

Estimates the critical batch size — the batch size where adding more samples stops helping. From McCandlish et al. (OpenAI, 2018).

- GNS < batch_size: batch is oversized (wasting compute)
- GNS ≈ batch_size: optimal trade-off
- GNS > batch_size: gradient is noisy, would benefit from larger batch

### Stable rank (`health/stable_rank_min`)

`stable_rank = ‖W‖²_F / σ_max(W)²`. Measures effective dimensionality of weight matrices.

Declining stable rank = the matrix is collapsing toward rank-1 = representation collapse. If any key matrix drops below 3.0, a warning is logged.

### Output entropy (`health/output_entropy`)

Only active when `logging.rl_readiness: true`. Measures the average entropy of the model's output distribution on valid tokens.

Declining entropy = model becoming overconfident = bad for subsequent RL. See [SFT → RL guide](../guides/sft-to-rl.md).

### Weight drift (`health/weight_drift_max`)

How far have the weights moved from initialization (as fraction of initial norm)?

- 5-20%: normal for SFT
- \> 50%: catastrophic forgetting territory

---

## Automatic warnings

The health monitor logs warnings when it detects:

- Stable rank below 3.0 on any key matrix
- Weight norm imbalance > 100× between layers
- Weight drift > 50% from initialization
- Output entropy below `rl_entropy_floor` (if RL-readiness enabled)

These appear as standard Python `WARNING` log messages.

---

## Usage

Health monitoring is always active. No configuration needed. Metrics appear in your wandb/trackio dashboard under the `health/` prefix.

To see them in console, check tier 2 and 3 metrics which log periodically:

```
step=100 ... health/grad_cosine_sim=0.42 health/cuda_peak_gb=14.2
step=200 ... health/stable_rank_min=12.3 health/weight_drift_max=0.08
```

!!! note "Gradient metrics and `gradient_release`"
    `health/grad_cosine_sim` and `health/gw_ratio/*` inspect raw gradients after the optimizer step (gradient zeroing is deferred until after the health check). Under `memory.gradient_release`, gradients are freed inside the backward pass, so these two metrics are unavailable — everything else still works.

---

## Tracker robustness

Metrics logging is designed so it can never cost you a run:

- **Crash-resume continues the same run.** The wandb run id is persisted to `{output_dir}/tracker_run_id.json`. Restarting with `train.resume_from: auto` appends to the existing wandb run; a fresh start in the same `output_dir` gets a new run id instead (reattaching would make wandb silently drop rows below the old history step). trackio resumes by project + run name. One training = one dashboard, no matter how many restarts.
- **Backend failures don't kill training.** wandb/trackio init and log calls are wrapped; a network hiccup logs a warning (rate-limited) and training continues.
- **One x-axis for everything.** `wandb.define_metric` pins all metrics to `train/global_step`, so `train/*`, `eval/*`, and `health/*` panels line up exactly — across restarts too.


---

*For entropy monitoring in the context of RL preparation, see [SFT → RL Transition](../guides/sft-to-rl.md).*


---

## Agent tooling commands

Palingenesis includes diagnostic CLI tools for debugging training before, during, and after a run. These wrap the `agent_tooling/` module.

### Pre-flight: diagnose

Run all sanity checks before training starts:

```bash
pgs diagnose --config configs/qwen35_4b/a100_80gb.yaml --mode pre --json
```

Checks: memory estimation (will it fit?), masking validation (are labels correct?), config sanity (any obvious misconfigurations?). Exit code 0 = healthy, 1 = issues found.

### Visualize masking: inspect

See exactly what the model will be trained on — tokenized text with loss-receiving tokens highlighted:

```bash
pgs inspect --config configs/qwen35_4b/a100_80gb.yaml --num_samples 5
```

Shows each token, its ID, and whether it has loss or is masked. Essential for debugging "loss not decreasing" issues (often a masking bug).

### Bulk masking validation: validate

Check masking correctness across many samples:

```bash
pgs validate --config configs/qwen35_4b/a100_80gb.yaml --num_samples 200
```

Reports: fraction of tokens with loss, any samples with zero valid tokens (would produce NaN), token efficiency statistics.

### Memory estimation: profile

Estimate peak GPU memory before committing to a training run:

```bash
pgs profile --config configs/qwen35_4b/a100_80gb.yaml --gpu_memory_gb 80
```

Reports: model memory, optimizer memory, activation memory, loss peak, total. Tells you whether it'll fit and how much headroom you have.

### Running job status: monitor

Check on a training run in progress from its log file:

```bash
pgs monitor --log_file outputs/train.log --brief
```

Reports: current step, loss trend, throughput, ETA, any warnings detected.

### Post-hoc loss analysis: loss

Analyze a completed training run's loss curve:

```bash
pgs loss --log_file outputs/train.log
```

Reports: convergence rate, spike count, plateau detection, recommended next steps.

---

## Training dynamics troubleshooting

*Every pathology, its symptoms, how to diagnose it, and the config fix.*

| Problem | Symptoms | Diagnosis | Fix |
|---------|----------|-----------|-----|
| **LR too high** | Loss NaN/spikes, `grad_cosine_sim < 0` | `train/grad_norm` exploding | `learning_rate` ÷ 3 |
| **LR too low** | Loss flat, slow convergence | eval not improving for 500+ steps | `learning_rate` × 2 |
| **Overfitting** | eval↑ while train↓ | `eval/loss` diverging from `train/loss` | `epochs: 2`, `base_merge: true` |
| **Catastrophic forgetting** | Eval on general tasks degrades | `health/weight_drift_max > 0.5` | `base_merge: true`, `pretrain_replay_dataset` |
| **Layer imbalance** | One layer learns 10× faster than others | `health/gw_ratio/layer_X` >> others | `llrd_decay: 0.9` |
| **Representation collapse** | Model outputs become repetitive | `health/stable_rank_min < 3` | Reduce LR, enable `sym_noise` |
| **Entropy collapse (RL)** | Output diversity disappearing | `health/output_entropy < 1.5` | `pre_rl: true`, stop early |
| **Data quality issues** | Loss noisy, frequent spikes | `SPIKE SKIPPED` > 5% of steps | `pgs prepare` to filter |
| **Packing cross-contamination** | Loss suspiciously low | `pgs inspect` shows wrong masking | `attn_implementation: flash_attention_2` |
| **Memory pressure** | OOM or CUDA malloc fails | `health/cuda_utilization_pct > 95%` | Reduce `per_device_batch_size`, increase `loss_num_chunks` |
| **Slow throughput** | tok/s below expected | `health/cuda_utilization_pct` low | `model.compile: true`, increase batch, `num_workers` |
| **Source overfitting (multi-dataset)** | One source's eval degrades | Per-source eval in multi-eval | `msft_tracking: true` |

---

*For the full monitoring metric reference, see the top of this page. For entropy-specific guidance, see [SFT → RL Transition](../guides/sft-to-rl.md).*
