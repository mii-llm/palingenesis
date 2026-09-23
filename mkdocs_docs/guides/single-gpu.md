# Single GPU

*How a 4-billion parameter model fine-tunes in about 16 GiB — and what each technique contributes.*

---

## The problem

A 4B parameter model in bf16 is 8 GB. AdamW stores two fp32 states per parameter: that's 32 GB for the optimizer alone. Add gradients (8 GB) and activations (12 GB), and you need 60 GB just to begin a training step.

That's why most people reach for LoRA. But LoRA is a compromise — you're training a low-rank shadow of the model, not the model itself. The representations it can learn are fundamentally limited.

Palingenesis takes a different path: eliminate the waste.

---

## Where the memory goes (and doesn't)

| Component | Naive | Palingenesis | How |
|-----------|:-----:|:------------:|-----|
| Weights | 8 GB | 8 GB | No change (bf16 is already minimal) |
| Optimizer | 32 GB | **1.4 GiB** | Lion 8-bit (1 byte/param), on the 36% of weights that train (`freeze_non_attention`) |
| Gradients | 8 GB | **0 GB** | Gradient release: each grad freed after use |
| Activations | 12 GB | **~2.5 GiB** | Selective AC: keep attention outputs and every other matmul, recompute the rest |
| **Total** | **60 GB** | **~16 GiB** | `pgs profile --config configs/qwen35_4b/a100_40gb.yaml` |

`pgs profile --measure` runs two real optimizer steps and reports the measured peak for your config.

---

## The optimization stack

### Gradient release (FORGE, June 2026)

The standard training loop:

1. Forward: compute loss
2. Backward: compute ALL gradients (all live simultaneously in memory)
3. Optimizer step: read all gradients, update all weights
4. Zero gradients

Step 2 is the waste. Why store all gradients at once when the optimizer processes them one at a time?

Gradient release registers a hook on each parameter. The moment a gradient is computed during backward, the hook fires: it runs the optimizer step for that parameter, then frees the gradient. By the time backward finishes, all weights are updated and all gradients are gone.

Peak gradient memory: one tensor (the largest single parameter, typically ~500 MB) instead of all tensors (8 GB for 4B).

```yaml
memory:
  gradient_release: true
train:
  gradient_accumulation_steps: 1  # Required — can't accumulate freed grads
  per_device_batch_size: 4        # Use the freed memory for bigger batch
```

!!! note "The trade-off"
    Gradient release requires `gradient_accumulation_steps: 1`. But the memory it frees lets you increase the real batch size — so effective batch stays the same or increases.

### Lion 8-bit

AdamW needs two fp32 buffers (momentum + variance): 16 bytes per parameter. Lion uses one buffer (momentum only, sign-based update): 4 bytes. With bitsandbytes 8-bit quantization: even less.

The catch: every Lion update element has magnitude 1, so at the same learning rate Lion moves the weights more than AdamW. The [Lion paper](https://arxiv.org/abs/2302.06675) recommends a learning rate 3–10× *smaller* than AdamW's. `learning_rate` is passed to the optimizer as is.

```yaml
train:
  optimizer: lion8bit
  learning_rate: 3.0e-6   # vs ~1e-5 to 2e-5 for AdamW fine-tuning
```

### Hyperball (Stanford, June 2026)

Here's a subtle insight: in a Transformer, every weight matrix that sits between two normalization layers is *scale-invariant* — the loss doesn't care about its magnitude, only its direction. Weight decay's real job isn't regularization; it's indirectly controlling the *angular* learning rate.

Hyperball makes this explicit. Each attention/MLP matrix is kept at its initial Frobenius norm R. The base optimizer only supplies a *direction* u, and every step moves the matrix by a fixed fraction of its norm:

```
W ← R · normalize(W − η · R · normalize(u))
```

η is an **angular** step. The optimizer's scale and the gradient scale drop out, and weight decay is replaced by the constraint. See the [optimizer reference](../reference/optimizers.md#hyperball) for details.

The paper reports a 20–30% token-equivalent speedup at 1B+ scale, measured in pretraining with Muon (arxiv:2606.16899). It is experimental and not independently reproduced. It also reports better LR transfer across scales. Memory cost: one snapshot bucket (≤1 GiB).

```yaml
train:
  hyperball: true
  hyperball_lr: 0.0    # 0 = calibrate each matrix to the base optimizer's first step
```

### Power-decay scheduler

The theory (Li et al., February 2026) derives LR schedules from functional scaling laws. In its model of pretraining, for capacity exponent β > 3, cosine saturates and power decay with γ ≈ 2β-1 ≈ 4 does better. It has not been benchmarked here for fine-tuning; cosine is the library default and the baseline to compare against.

```yaml
train:
  lr_scheduler: power_decay
```

---

## Which config for which GPU

| GPU | VRAM | Config | Batch × sequence |
|-----|------|--------|-------------|
| RTX 3090/4090 | 24 GB | `qwen35_4b/a100_40gb.yaml` | 2 × 2048 |
| A100-40GB | 40 GB | `qwen35_4b/a100_40gb.yaml` | 2 × 2048 |
| A100-80GB | 80 GB | `qwen35_4b/a100_80gb.yaml` | 4 × 4096 |
| H100-80GB | 80 GB | `qwen35_4b/h100_80gb.yaml` | 8 × 8192, FP8 |
| B200 | 192 GB | `qwen35_4b/b200.yaml` | 32 × 8192 |

Check any of them on your GPU with `pgs profile --config <config> --measure` before a long run.

---

## Running

```bash
./run.sh configs/qwen35_4b/a100_80gb.yaml
```

The `run.sh` script auto-detects your GPU count and launches appropriately. For single GPU, it's equivalent to:

```bash
torchrun --standalone --nproc_per_node=1 -m palingenesis.train --config configs/qwen35_4b/a100_80gb.yaml
```

!!! note "First step is slow (30-60 seconds)"
    `torch.compile` traces the computation graph on the first forward pass (and once more for each new batch shape early on). Don't cancel because step 1 looks frozen — it's compiling.

---

## What to expect

Console output:

```
step=50  loss=2.34 lr=4.5e-05 tok/s=6102 grad_norm=0.41 dt=1.2s
step=100 loss=1.89 lr=4.5e-05 tok/s=6234 grad_norm=0.33 dt=1.2s eval=1.92
step=150 loss=1.67 lr=4.4e-05 tok/s=6180 grad_norm=0.29 dt=1.2s
```

- `tok/s` — training throughput (tokens processed per second)
- `grad_norm` — gradient norm before clipping. Its scale depends on the model and the loss (5–30 at the start of a fine-tune is common); judge it against the run's own recent values. `pgs monitor` does that.
- `eval` — validation loss (appears every `eval_every` steps)
- `dt` — wall-clock per step. Should be stable; spikes indicate GC stalls.

Healthy training: loss decreases smoothly, grad_norm is stable, tok/s is constant.

---

## When things go wrong

| What you see | What's happening | What to do |
|------|------|------|
| Loss=NaN on step 1 | LR way too high | Divide `learning_rate` by 10 |
| Loss decreases then spikes | Bad batch hit | Normal (<1% of steps). If >5%, filter data. |
| Loss flat for 100+ steps early on | LR too low or all data is easy | Increase LR 2× or run `pgs prepare` (a slow decrease after the initial drop is normal) |
| OOM crash | Batch too big or seq too long | Reduce `per_device_batch_size` by 1 |
| Very slow (~1K tok/s) | Compile disabled | Set `model.compile: true` |
| "SPIKE SKIPPED" in logs | Anomalous gradient detected | Palingenesis handled it. Investigate if frequent. |

!!! tip "The best debugging tool"
    ```bash
    pgs inspect --config your_config.yaml --num_samples 3
    ```
    This shows you exactly what the model sees: tokenized text, which tokens have loss, how masking works. Most "loss not decreasing" issues are data masking bugs.
