# Autopilot

*The learning rate is the most important hyperparameter in deep learning. Autopilot finds it for you.*

---

## The problem with learning rates

Too high: the model diverges (loss goes to infinity, weights become NaN). Too low: training is stable but barely moves — you waste hours of compute to achieve what a better LR would do in minutes. The optimal LR depends on your model size, batch size, data distribution, sequence length, and training duration. It is not knowable in advance.

The standard approach: pick something from a paper (2e-5? 5e-5?), train for a while, check if it's working, maybe try again. This wastes time and GPU hours.

Autopilot solves this systematically.

---

## What it does

```bash
pgs autopilot \
    --model Qwen/Qwen3.5-4B \
    --dataset your_data.jsonl \
    --output ./autopilot-output \
    --max_steps 5000
```

Four phases, fully automatic:

### Phase 1: Profile (5 seconds)

Reads your GPU's memory, compute capability, and available bandwidth. From this plus the model size, it determines:

- Maximum batch size that fits
- Whether to use gradient release or FSDP
- Optimal number of loss chunks
- Whether FP8 is available (H100+)

### Phase 2: Adaptive LR Sweep (5-10 minutes)

A two-stage search informed by scaling-law research (arxiv:2409.19913, 2503.04715, 2405.14578):

**Coarse sweep (5 candidates):** Logarithmically-spaced LRs centered on a scaling-law estimate. The center accounts for model size (`N^{-0.5}`) and effective batch tokens using the **surge phenomenon** — Adam's optimal LR peaks near B_crit ≈ 100K tokens/step and falls on both sides.

**Quadratic refinement (3 candidates):** A parabola is fit in (log₁₀(LR), loss) space to the coarse results. The parabola's minimizer gives an analytical LR estimate. Refinement candidates are placed tightly around it.

The winner is selected by final loss, penalized for curvature (risk of early plateau).

Then: **adaptive horizon correction**. A sweep at 100 steps overestimates the optimal LR for 5000 steps. The correction factor is `(sweep_steps / full_steps)^α`:
- Base α = 0.088 (from arxiv:2409.19913, Table 5)
- Attenuated ×0.7 for SFT (pretrained model → weaker horizon dependence, arxiv:2606.05610)
- Adjusted by loss curvature: flattening curve → raise α; still-dropping → lower α
- Effective range: α ∈ [0.04, 0.20]

Bad trials terminate early (NaN, increasing loss, no progress), saving time for refinement.

### Phase 3: Full training (the main event)

Trains with the corrected LR and all optimizations:

- DEFT loss
- Power-decay scheduler
- Hyperball + EMA + base merge
- AdaGC spike protection
- Best-model tracking

Checkpoints are saved for auto-resume. If this phase is interrupted and you re-run the same command, it picks up from the last checkpoint.

### Phase 4: Report

Writes `autopilot_report.json` with every decision made:

```json
{
  "best_lr": 1.47e-5,
  "sweep_results": [...],
  "hardware": {"gpu_name": "NVIDIA A100-SXM4-80GB", "memory_gb": 80},
  "recommended_config": {...}
}
```

---

## The LR estimation model

Even before the sweep runs, autopilot has an initial estimate from scaling laws:

```
center_lr = C × model_size^(-0.5) × surge_factor(batch_tokens)
```

Where `surge_factor` models the Adam-specific non-monotone batch-size relationship:
- Below B_crit (100K tokens): noisy gradients → factor < 1 → lower center
- At B_crit: factor = 1.0 (peak, reference point)
- Above B_crit: saturation → factor < 1 → lower center

This centers the coarse sweep around the most probable optimal region, avoiding wasted trials.

---

## When autopilot is wrong

Autopilot can fail in specific scenarios:

| Scenario | What happens | Fix |
|----------|-------------|-----|
| All sweep trials diverge | Falls back to LR = 2e-5 | Data may be corrupted. Run `pgs validate`. |
| Loss plateaus in Phase 3 | LR overcorrected (too low) | Re-run with `--lr_sweep_steps 200` (more curvature data) |
| OOM during Phase 3 | Profile overestimated memory | Reduce `per_device_batch_size` from report and re-run manually |
| Very small batch (<4K tok) | Surge factor aggressively lowers center | The sweep still covers ±1.2 decades — it'll find the right range |

For all these: the `autopilot_report.json` tells you exactly what was decided. You can take that information, create a manual config, and iterate.

---

## Resumable

Every phase writes to `autopilot_state.json`. Re-running the same command skips completed phases:

```bash
# First run: profiles, sweeps, starts training → interrupted at step 2000
pgs autopilot --model ... --dataset ... --output ./out

# Second run: skips profile+sweep, resumes training from step 2000
pgs autopilot --model ... --dataset ... --output ./out
```

---

## When to use manual instead

| Situation | Recommendation |
|-----------|:---:|
| First time with a new model or dataset | **Autopilot** |
| You already know the optimal LR | Manual (faster, no sweep overhead) |
| Multi-node training | Manual (autopilot is single-node only) |
| Rapid data iteration (trying 10 datasets) | Manual (sweep per dataset is wasteful) |
| Production deployment needing reproducibility | Manual (explicit config is auditable) |

Autopilot's real value: it converts unknowns into knowns. Once you have a known-good LR for your model/data combination, switch to manual configs.

---

## Output

```
autopilot-output/
├── autopilot_state.json    # Which phases completed, resume state
├── autopilot_report.json   # All decisions + results
├── best/                   # Lowest eval loss checkpoint
│   └── model/
├── final/                  # Last training step
│   └── model/
└── step-*/                 # Periodic snapshots (auto-purged)
```
