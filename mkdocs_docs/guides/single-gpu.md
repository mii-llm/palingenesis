# Single GPU

*How a 4-billion parameter model fine-tunes in about 16 GiB — and what each technique contributes.*

!!! abstract "Use this guide when"
    You train on one GPU and want to know what fits, which memory setting is the fastest that still fits, and what each memory technique costs. For sequences of 32k tokens and more, read [Long Sequences (SeCO)](long-sequences.md) next; for several GPUs, [Multi-GPU](multi-gpu.md).

## In short

```bash
pgs profile --config my.yaml --measure     # will it fit? two real optimizer steps, measured peak
pgs fit --config my.yaml                   # the fastest memory setting that fits this GPU
pgs train --config my.yaml                 # train
```

A config that already fits needs nothing else. The rest of this page explains where the memory goes and what to change when it does not fit.

---

## The problem

A 4B parameter model in bf16 is 8 GB. AdamW stores two fp32 states per parameter: that's 32 GB for the optimizer alone. Add gradients (8 GB) and activations (12 GB), and you need 60 GB just to begin a training step.

That's why most people reach for LoRA. But LoRA is a compromise — you're training a low-rank shadow of the model, not the model itself. The representations it can learn are fundamentally limited. (When adapters are the right tool, see the [LoRA guide](lora.md).)

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

### Trading speed for memory: `pgs fit`

Activation memory grows with the sequence, and every setting that lowers it costs speed: activation checkpointing recomputes part (`selective`) or all (`full`) of each layer in the backward, and SeCO (`memory.seco`) runs the sequence chunk by chunk. Which one is the fastest that still fits depends on the model, the sequence length and the GPU, so measure it:

```bash
pgs fit --config my.yaml            # budget: this GPU
pgs fit --config my.yaml --gpu 24   # budget: 24 GiB
```

Each setting runs the real trainer for a few steps on worst-case rows (every row at `data.max_seq_length`) in its own process; the report gives the peak and the throughput of each, and recommends the fastest one under the budget with the options to set. The settings it measures:

- the config as it is, then the three activation-checkpointing modes, `none`, `selective` and `full`;
- the same three modes with their saved activations offloaded to host memory (`memory.activation_offload`, [below](#activation-offload));
- SeCO at chunk sizes of 8192, 4096, 2048 and 1024 tokens (those shorter than `data.max_seq_length`), then the smallest chunk with its K/V cache offloaded (`memory.seco_kv_offload`);
- full checkpointing with parameters, gradients and optimizer state in CPU memory (`parallel.cpu_offload`, [below](#models-larger-than-the-gpu-cpu-offload)), the most frugal and the slowest. Settings the config cannot take (SeCO with DPO, for example) are listed as skipped.

| Option | Default | Meaning |
|---|---|---|
| `--gpu` | this GPU's memory | budget in GiB |
| `--headroom` | `0.92` | share of the budget the measured peak may use (allocator fragmentation and everything else on the GPU take the rest) |
| `--steps` | `3` | optimizer steps per setting (the last one is timed) |
| `--first-fit` | off | stop at the first setting that fits |
| `--json` | off | print the report as JSON |
| `--section.option value` | | config overrides, as for `pgs train` |

The report ends with the recommendation, as command-line options and as config lines (`Recommended: ... Add to the command line: ...`). When nothing fits, `pgs fit` exits with code 1 and suggests what to lower next: `data.max_seq_length` or `train.per_device_batch_size` (raise `train.gradient_accumulation_steps` to keep the effective batch), freezing part of the model (`train.freeze_non_attention` on hybrids), or splitting sequences over GPUs (`parallel.context_parallel`).

Qwen3.5-0.8B, one 16,384-token row per micro-batch, on an H100:

| Setting | Peak | Tokens/s |
|---|:---:|:---:|
| `gradient_checkpointing: none` | 34.9 GiB | 19,900 |
| `selective` (default) | 19.4 GiB | 18,100 |
| `full` | 11.5 GiB | 16,600 |
| SeCO, 8192-token chunks | 15.9 GiB | 13,000 |
| SeCO, 4096-token chunks | 14.0 GiB | 10,500 |
| SeCO, 1024-token chunks | 10.3 GiB | 2,800 |

With memory to spare, `none` is the fastest; `full` gives up 17% of the speed for a third of the memory. On this model and length SeCO is slower than full checkpointing at a higher peak: it pays off on sequences that full checkpointing cannot hold.

!!! note "Throughput is measured on worst-case rows"
    Every row is at `data.max_seq_length`, so the peak is an upper bound for the run; shorter rows train faster in every setting.

### Activation offload

Activation checkpointing trades memory for recompute; activation offload trades it for PCIe traffic. The tensors the transformer layers save for the backward wait in pinned host memory instead of on the GPU: they are copied out during the forward and back during the backward on a side CUDA stream, while other layers compute.

```yaml
memory:
  activation_offload: true
train:
  gradient_checkpointing: full   # what moves depends on the checkpointing mode (table below)
```

What moves, per `train.gradient_checkpointing`:

| Checkpointing | Moved to host memory | Left on the GPU |
|---|---|---|
| `none` | every activation the layers save (above `activation_offload_min_bytes`) | the last layer's, the layers in flight or prefetched |
| `selective` | each layer's input, plus the outputs the selective policy keeps (attention, every other matmul) on torch versions whose checkpoint routes them through saved-tensor hooks (`respect_saved_tensors_hooks`); older torch: the layer inputs only (logged) | the recomputed rest |
| `full` | each layer's input, the only thing full checkpointing saves | one layer's recompute |

The schedule works per transformer layer, as Megatron-LM's offloading does: in the forward the copies of a layer may still run while the next `activation_offload_in_flight` layers compute (each of those keeps its activations on the GPU until its copies finish); in the backward, reaching layer *i* fetches the saved tensors of layers *i-1* ... *i-*`activation_offload_prefetch` while layer *i*'s backward runs. The last layer is never offloaded (its backward comes first), nor are tensors saved outside the layers (embeddings, final norm, loss), parameters and their casts, or tensors below `activation_offload_min_bytes` (1 MiB). A tensor saved by several ops is copied once, and a dense permuted tensor keeps its strides, so the copies are plain `memcpy`s and the backward sees the same layout. The values come back bit-identical: the gradients are those of a run without offload (`tests/test_activation_offload.py`: equal gradients on tiny Qwen3 and Qwen3.5 models under every checkpointing mode, packed rows, SeCO chunks, gradient accumulation, `retain_graph` and compiled layers; the trainer's loss equal with and without).

| Option | Default | Meaning |
|---|---|---|
| `memory.activation_offload` | `false` | turn it on |
| `memory.activation_offload_min_bytes` | `1048576` | smaller saved tensors stay on the GPU |
| `memory.activation_offload_in_flight` | `1` | layers whose device-to-host copies may overlap later layers' forward (>= 1) |
| `memory.activation_offload_prefetch` | `1` | layers fetched ahead of the backward (0: when their backward starts) |
| `memory.activation_offload_boundaries_only` | `false` | move only each checkpointed layer's inputs; what selective checkpointing keeps stays (needs `selective` or `full`) |

**Host memory.** The offloaded tensors live in pinned memory carved from 256 MiB chunks that every step reuses (torch's caching host allocator would round each tensor up to a power of two). The arena is bounded by 40% of the machine's RAM, split between the GPUs of the node; past that the step stops with an error naming the options that move less (checkpointing, `boundaries_only`, a shorter sequence). `pgs profile` reports the estimate (`Offloaded activations: ... GiB [host memory]`), and the log line has `memory/offload_gb` (moved per micro-batch) and `memory/offload_host_gb` (the arena).

**Composes with** (tested, CPU) the three checkpointing modes, packing, the chunked loss (it runs outside the layers), gradient accumulation, SeCO (each chunk's forward and backward are one offload pass; with `seco_kv_offload` both stream over PCIe, and the config warns) and compiled layers. With `model.compile` the layer-boundary hooks run eagerly: a graph break before and after each compiled layer, the layer body stays one graph; `scripts/gpu_checks/offload_fused_optim.sh` measures the step time and gradients with and without. Nothing in the design is specific to one process (each FSDP2 rank offloads its own activations, the host budget is split between a node's ranks), but FSDP2 on several GPUs, context parallel, gradient release, DPO and off-policy distillation have not been run with it yet.

**When it pays.** Offload costs PCIe bandwidth (about 25 GB/s each way on PCIe 4.0): it is free when a layer's copies take less time than the next layer's compute, which favours long sequences and large hidden sizes, and it is most useful where recompute is the expensive alternative. `pgs fit` measures it next to the checkpointing modes and SeCO and picks the fastest setting that fits; the GPU check script above reports peak memory and step time against no checkpointing and selective checkpointing on Qwen3.5-2B at 4k and 16k tokens.

After torchtune's `OffloadActivations` (BSD-3) and the layer-wise CPU offload of Megatron-LM / Transformer Engine (Apache-2.0); re-implemented, no code copied.

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

Hyperball makes this explicit. Each attention/MLP matrix is kept at its initial Frobenius norm $R$. The base optimizer only supplies a *direction* $u$, and every step moves the matrix by a fixed fraction of its norm:

$$
W \leftarrow R \cdot \operatorname{normalize}\!\big(W - \eta R \operatorname{normalize}(u)\big), \qquad \operatorname{normalize}(X) = \frac{X}{\lVert X \rVert_F}
$$

where $R$ is the matrix's norm at the first step, $u$ the base optimizer's update direction and $\eta$ the angular step (`train.hyperball_lr`, scaled by the LR schedule).
{: .pg-where }

$\eta$ is an **angular** step. The optimizer's scale and the gradient scale drop out, and weight decay is replaced by the constraint. See the [optimizer reference](../reference/optimizers.md#hyperball) for details.

The paper reports a 20–30% token-equivalent speedup at 1B+ scale, measured in pretraining with Muon (arxiv:2606.16899). It is experimental and not independently reproduced. It also reports better LR transfer across scales. Memory cost: one snapshot bucket (≤1 GiB).

```yaml
train:
  hyperball: true
  hyperball_lr: 0.0    # 0 = calibrate each matrix to the base optimizer's first step
```

### Power-decay scheduler

The theory (Li et al., February 2026) derives LR schedules from functional scaling laws. In its model of pretraining, for capacity exponent $\beta > 3$, cosine saturates and power decay with $\gamma \approx 2\beta - 1 \approx 4$ does better (the schedule is in the [optimizer reference](../reference/optimizers.md#power_decay)). It has not been benchmarked here for fine-tuning; cosine is the library default and the baseline to compare against.

```yaml
train:
  lr_scheduler: power_decay
```

### Models larger than the GPU: CPU offload

When the weights, gradients and optimizer state do not fit the GPU at all (an 8B model with fp32 master weights needs about 128 GB for them), keep them in host memory:

```yaml
parallel:
  cpu_offload: true          # parameters, gradients, optimizer state in pinned CPU memory
train:
  gradient_checkpointing: full
  optimizer: adamw_bf16      # with model.torch_dtype: bfloat16: halves the host memory (8 B/param)
```

On one GPU this runs FSDP2 over a process group of one rank (started for you, no `torchrun` needed), as `pgs rl` does with `train.cpu_offload`: each layer's weights are copied to the GPU for its forward and backward, gradients go back to the CPU, and the optimizer steps there. The GPU holds one layer's weights and the activations. Budget host RAM of about 16 B/param with fp32 weights and `adamw`, 8 B/param with bf16 weights and `adamw_bf16`, plus pinned staging. Each step moves the weights over PCIe twice and the gradients once, so throughput is far below an in-GPU run: use it when nothing else fits (`pgs fit` tries it last). A run on one process with offload trains to the same weights as without (`tests/test_single_gpu_offload.py`, CPU). Measure it on your machine with `pgs fit --config my.yaml`, which reports its peak and tokens/s next to the other settings.

### Attention kernels on Hopper and Blackwell

With `attn_implementation: sdpa` (the default) every attention call, packed variable-length rows and SeCO chunks included, runs torch's built-in FlashAttention-2 kernel. On Hopper (H100, H200) FlashAttention-3 and on Blackwell (B200, B300) FlashAttention-4 are faster at long sequences (their papers report 1.5-2x the forward of FlashAttention-2 on H100 for FA3, and the Halo framework reports 1.2-3x for FA4 over FA2 on B300 at 32k-128k tokens under context parallelism; not measured here yet). torch can put either behind the same ops, so nothing else in the trainer changes:

```yaml
model:
  attn_implementation: sdpa
  flash_kernel: auto    # FA3 on Hopper, FA4 on Blackwell, when installed; otherwise the built-in kernel
```

At startup the chosen kernel runs a forward and backward of dense and variable-length attention on the model's head shape next to the built-in kernel; `auto` keeps the built-in kernel when a package is missing or the check fails, an explicit `fa3` / `fa4` stops with a config error instead. The log line `attention: ... flash kernel: fa3` says which one trains. Measure the gain on your GPU and model before a long run:

```bash
python scripts/bench_attention.py --model <model> --seqs 4096,16384,65536   # attention alone, fwd + bwd
python scripts/bench_attention.py --model <model> --e2e --seq 16384          # the whole model
```

`flash_attention_3` and `flash_attention_4` are also accepted as `attn_implementation` (transformers' own flash path, like `flash_attention_2`).

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
- `eval` — validation loss (appears every `eval_every` steps, and once as `step=0 eval=... (before training)`)
- `eval_sources=name:loss,...` — with per-source eval sets (`data.eval_sources`), each source's loss follows the composite `eval`, so a regression on one source shows in the plain log ([per-capability eval](data.md#per-capability-eval-eval_sources))
- `dt` — wall-clock per step. Should be stable; spikes indicate GC stalls.

Healthy training: loss decreases smoothly, grad_norm is stable, tok/s is constant.

---

## When things go wrong

| What you see | What's happening | What to do |
|------|------|------|
| Loss=NaN on step 1 | LR way too high | Divide `learning_rate` by 10 |
| Loss decreases then spikes | Bad batch hit | Normal (<1% of steps). If >5%, filter data. |
| Loss flat for 100+ steps early on | LR too low or all data is easy | Increase LR 2× (a slow decrease after the initial drop is normal) |
| OOM crash | Batch too big or seq too long | Reduce `per_device_batch_size` by 1, or run `pgs fit` for the fastest setting that fits |
| Very slow (~1K tok/s) | Compile disabled | Set `model.compile: true` |
| "SPIKE SKIPPED" in logs | Anomalous gradient detected | Palingenesis handled it. Investigate if frequent. |

!!! tip "The best debugging tool"
    ```bash
    pgs inspect --config your_config.yaml --num_samples 3
    ```
    This shows you exactly what the model sees: tokenized text, which tokens have loss, how masking works. Most "loss not decreasing" issues are data masking bugs.
