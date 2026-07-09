# Single GPU

*How a 4-billion parameter model fits in 15 GB — and why that matters.*

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
| Optimizer | 32 GB | **1 GB** | Lion 8-bit: sign-based updates, 4 bytes/param |
| Gradients | 8 GB | **0 GB** | Gradient release: each grad freed after use |
| Activations | 12 GB | **4 GB** | Selective AC: save only attention outputs |
| **Total** | **60 GB** | **~15 GB** | — |

This isn't theoretical. Run it and watch `nvidia-smi`.

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

The catch: Lion's learning rate needs to be ~3× higher than AdamW for the same convergence speed. Palingenesis configs handle this automatically.

```yaml
train:
  optimizer: lion8bit
  learning_rate: 4.5e-5   # Already 3× scaled
```

### Hyperball (Stanford, June 2026)

Here's a subtle insight: in a Transformer, every weight matrix that sits between two normalization layers is *scale-invariant* — the loss doesn't care about its magnitude, only its direction. Weight decay's real job isn't regularization; it's indirectly controlling the *angular* learning rate.

Hyperball makes this explicit: after each optimizer step, it normalizes the weight matrix back to its initial Frobenius norm. The optimizer only changes the direction, never the scale.

Result: 20-30% token-equivalent speedup at 1B+ scale per the original paper (arxiv:2606.16899; experimental, not independently reproduced). Zero memory cost (it's a 5-line projection after each step). Better LR transfer (optimal LR varies only 1.4× across model scales, vs 3-4× with weight decay).

```yaml
train:
  hyperball: true
```

### Power-decay scheduler

Cosine annealing is the default everywhere. It's wrong.

The theory (Li et al., February 2026) derives optimal LR schedules from functional scaling laws. Result: for capacity exponent β > 3 (always true for modern LLMs), cosine *saturates* — it can't exploit the full model capacity. Power-decay with γ ≈ 2β-1 ≈ 4 is provably optimal.

In practice: same code complexity as cosine, consistently better final loss.

```yaml
train:
  lr_scheduler: power_decay
```

---

## Which config for which GPU

| GPU | VRAM | Config | What you get |
|-----|------|--------|-------------|
| RTX 3090/4090 | 24 GB | `qwen35_4b/a100_40gb.yaml` | batch=2, seq=2048, ~4K tok/s |
| A100-40GB | 40 GB | `qwen35_4b/a100_40gb.yaml` | batch=2-3, seq=2048, ~5K tok/s |
| A100-80GB | 80 GB | `qwen35_4b/a100_80gb.yaml` | batch=4, seq=4096, ~6K tok/s |
| H100-80GB | 80 GB | `qwen35_4b/h100_80gb.yaml` | batch=8, seq=8192, FP8, ~12K tok/s |
| B200 | 192 GB | `qwen35_4b/b200.yaml` | batch=32, seq=8192, ridiculous headroom |

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
    `torch.compile` traces the computation graph on the first forward pass. This is a one-time cost per session. Steps 2+ run at full speed (~6,000 tok/s on A100-80GB). Don't cancel because step 1 looks frozen — it's compiling.

---

## What to expect

Console output:

```
step=50  loss=2.34 lr=4.5e-05 tok/s=6102 grad_norm=0.41 dt=1.2s
step=100 loss=1.89 lr=4.5e-05 tok/s=6234 grad_norm=0.33 dt=1.2s eval=1.92
step=150 loss=1.67 lr=4.4e-05 tok/s=6180 grad_norm=0.29 dt=1.2s
```

- `tok/s` — training throughput (tokens processed per second)
- `grad_norm` — should be stable (0.1-1.0). If spiking: data quality issue.
- `eval` — validation loss (appears every `eval_every` steps)
- `dt` — wall-clock per step. Should be stable; spikes indicate GC stalls.

Healthy training: loss decreases smoothly, grad_norm is stable, tok/s is constant.

---

## When things go wrong

| What you see | What's happening | What to do |
|------|------|------|
| Loss=NaN on step 1 | LR way too high | Divide `learning_rate` by 10 |
| Loss decreases then spikes | Bad batch hit | Normal (<1% of steps). If >5%, filter data. |
| Loss flat for 100+ steps | LR too low or all data is easy | Increase LR 2× or run `pgs prepare` |
| OOM crash | Batch too big or seq too long | Reduce `per_device_batch_size` by 1 |
| Very slow (~1K tok/s) | Compile disabled | Set `model.compile: true` |
| "SPIKE SKIPPED" in logs | Anomalous gradient detected | Palingenesis handled it. Investigate if frequent. |

!!! tip "The best debugging tool"
    ```bash
    pgs inspect --config your_config.yaml --num_samples 3
    ```
    This shows you exactly what the model sees: tokenized text, which tokens have loss, how masking works. Most "loss not decreasing" issues are data masking bugs.
