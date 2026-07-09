# Autopilot Mode

Fully autonomous training: profile, sweep LR, train with optimal settings, early-stop.

## Usage

```bash
pgs autopilot --model meta-llama/Llama-3.1-8B-Instruct \
                      --dataset your-org/agentic-traces \
                      --val_dataset your-org/val \
                      --output ./best-model \
                      --max_steps 5000
```

## What It Does

| Phase | What | Time |
|-------|------|------|
| 1. Profile | Detect GPU, estimate model size, auto-size batch/chunks | ~5s |
| 2. Adaptive LR Sweep | Coarse grid → quadratic fit → refinement → horizon correction | ~5-15min |
| 3. Full Training | Best LR, all optimizations, resume-aware | Varies |
| 4. Report | JSON output with everything tried | Instant |

## Resumability

Progress is saved after EACH phase to `autopilot_state.json`. If interrupted:
- Re-run the same command
- Completed phases are skipped automatically
- Phase 3 (training) resumes from last checkpoint via `resume_from: auto`

## LR Selection (Adaptive Sweep)

The sweep is informed by scaling-law research (arxiv:2409.19913, 2503.04715, 2405.14578):

### Phase 1: Coarse Sweep (5 candidates)

Candidates are centered using a scaling-law anchor:
```
center_lr = C × model_params^{-0.5} × batch_factor(effective_tokens)
```

The `batch_factor` models the **surge phenomenon** (arxiv:2405.14578): for Adam optimizers, optimal LR first rises then falls as batch size increases, peaking at B_crit ≈ 100K tokens/step.

| Effective tokens/step | Regime | Factor | Behavior |
|----------------------|--------|--------|----------|
| 4K (1 GPU, bs=1, no GA) | Below B_crit | 0.11 | LR center lowered (noisy gradients) |
| 100K (sweet spot) | At B_crit | 1.00 | Peak, reference LR |
| 524K (8 GPU, GA=4) | Above B_crit | 0.95 | Slightly past peak |
| 4M (large cluster) | Well above | 0.43 | LR center lowered (saturation) |

### Phase 2: Quadratic Refinement

Following arxiv:2409.19913 Section 3.1, a **parabola is fit in (log₁₀(LR), loss) space** to the coarse trial results. The minimizer of this parabola gives an analytical LR estimate (R² > 0.99 in the paper). Three refinement candidates are placed around this estimate.

If the parabola fit fails (too few points, bad curvature), falls back to bracketing between the top-2 trials.

### Phase 3: Adaptive Horizon Correction

The sweep-found LR is corrected for the full training horizon:
```
corrected_lr = best_lr × (sweep_steps / full_steps)^α
```

The exponent α is **not hardcoded**. It's estimated from the loss curve curvature:
- **Literature baseline**: α = 0.088 (arxiv:2409.19913, median for 350M–2.7B models)
- **SFT attenuation**: ×0.7 (continued training from pretrained checkpoint needs weaker correction)
- **Curvature adjustment**: flattening curve → raise α; still-dropping → lower α

Effective range: α ∈ [0.04, 0.20]. For typical SFT: α ≈ 0.06.

### Early Stopping of Trials

Bad trials terminate early to save time for refinement:
- NaN/Inf loss → immediate stop
- Loss increasing for >30% of steps → stop
- Already worse than best known trial at 50% mark → stop
- No progress (<1% improvement) after 25% of steps → stop

## Auto-Configuration

From hardware profiling + model size, autopilot determines:
- `per_device_batch_size`: largest that fits in memory
- `gradient_accumulation_steps`: to hit effective batch of ~32 sequences
- `loss_num_chunks`: based on seq_len × vocab_size
- `context_parallel`: enabled if seq_len > 16k and GPUs >= 4
- `float8_training`: enabled on H100+ for 3B+ models
- `gradient_release`: enabled when GA=1 (freed memory compensates)

## Config Validation

Before training starts, `config.validate()` checks for incompatible settings:
- **Hard errors** (raises): gradient_release + GA>1, packing + CP, multiple losses, etc.
- **Warnings** (logs): untested feature combinations, redundant options

See `docs/configuration.md` for the full compatibility matrix.

## Output Structure

```
output_dir/
  autopilot_state.json    # Phase progress (for resume)
  autopilot_report.json   # Final results (hardware, sweep, best LR)
  final/                  # Best model (HF safetensors format)
  step-500/              # Checkpoints
  step-1000/
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | Llama-3.1-8B-Instruct | HF model name |
| `--dataset` | (required) | Training dataset |
| `--dataset_split` | train_sft | Split name |
| `--val_dataset` | None | Validation dataset (uses train split if None) |
| `--val_split` | test_sft | Validation split |
| `--output` | ./autopilot-output | Output directory |
| `--max_steps` | 5000 | Max training steps |
| `--seq_length` | 4096 | Max sequence length |
| `--lr_sweep_steps` | 100 | Steps per sweep trial |
