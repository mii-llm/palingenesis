# Multi-GPU Training

*What actually happens when you add GPUs — and why convergence speed doesn't scale linearly.*

---

## The trade-off you're making

Adding GPUs gives you two things: more memory (model sharding) and more throughput (data parallelism). But it costs you communication time — every forward and backward now involves all-gathers and reduce-scatters across a PCIe/NVLink fabric.

The question isn't "will it be faster?" (yes, always). The question is: what's the most efficient configuration for *your* model, *your* hardware, and *your* convergence target?

---

## How FSDP2 works (and why per-layer sharding matters)

FSDP2 takes your model's parameters and distributes them across GPUs. Each GPU holds 1/N of every parameter. During forward, an all-gather reconstructs the full parameter for each layer just before it's used. During backward, a reduce-scatter sends gradient shards back.

The key insight: if you wrap the *entire model* as one FSDP unit, all parameters must be gathered at once (defeating the purpose). Instead, palingenesis wraps each transformer layer individually. This means:

- During forward: only one layer's parameters are fully materialized at a time
- Communication for layer N overlaps with computation of layer N-1
- Peak memory = model/N + one full layer + activations

This is why we say "communication overlaps with compute" — it's not magic, it's careful scheduling. Each layer being independent means the allgather for the next layer can start while the current layer is still computing.

```yaml
parallel:
  fsdp: true
  reshard_after_forward: true  # Trade memory for less communication
```

---

## The optimizer changes (and why)

On a single GPU with gradient release, memory is the constraint. You use Lion8bit (minimal state) and fuse the step into backward (zero gradient memory).

With FSDP, memory is *abundant* — the model is split across N GPUs. So the constraint shifts from memory to *convergence speed*. Now you want the fastest optimizer per step:

**Muon** — steepest descent under the spectral norm. Instead of treating each parameter as an independent scalar (like AdamW), Muon treats entire weight matrices as geometric objects. It applies a Newton-Schulz iteration (5 steps of a cubic recurrence) to find the polar decomposition of the momentum, then updates in the direction of the matrix sign. This respects the matrix geometry and converges 1.5-2× faster per step.

**+ MONA** — before Muon orthogonalizes the momentum, MONA enriches it with curvature information. The gradient difference `G_k - G_{k-1}` is approximately `H·Δθ` — the Hessian times the parameter change. This points *away from sharp minima*. MONA accumulates these differences as an EMA and adds them to the gradient before orthogonalization.

**+ Hyperball** — after the optimizer updates the weights, Hyperball projects them back to their initial Frobenius norm. This removes the radial component of the update (which is meaningless for scale-invariant layers) and keeps only the angular movement. The optimizer now only changes *direction*, not magnitude.

The composition:

```
gradient → MONA (add curvature) → Muon (orthogonalize) → step → Hyperball (project to sphere)
```

Each stage operates on a different aspect. They compose without interference.

```yaml
train:
  optimizer: muon
  hyperball: true
  mona: true
  mona_beta_a: 0.975
  mona_lite: true
```

---

## Launch

```bash
./scripts/train_multi_gpu.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
```

Or specify GPU count:

```bash
NGPUS=4 ./scripts/train_multi_gpu.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
```

Direct torchrun equivalent:

```bash
torchrun --standalone --nproc_per_node=8 \
    -m palingenesis.train --config configs/qwen35_4b/a100_80gb_multigpu.yaml
```

---

## Why scaling is sublinear

On 8 GPUs you get 5.8× speedup, not 8×. Here's where the 2.2× overhead goes:

1. **All-gather latency** (~15%): even with NVLink, gathering 4B params per layer takes time that can't fully overlap
2. **Reduce-scatter** (~10%): sending gradient shards back after backward
3. **Synchronization** (~5%): all ranks must finish each micro-batch before the next starts. The slowest rank (stochastic — depends on data) gates everyone.

You can reduce #3 by increasing batch size per GPU (more compute per communication event = better ratio). The configs already do this — multi-GPU configs use batch=8 vs single-GPU batch=4.

---

## Loss normalization (why it matters for correctness)

Each GPU processes different data. With packing, most sequences are fully packed — but at epoch boundaries or with variable-length documents, different GPUs can have different numbers of valid (non-masked) tokens.

If each GPU divides its loss by its *local* valid count, the gradient scale differs per GPU. FSDP's reduce-scatter averages gradients across ranks. The result: ranks with more padding tokens get disproportionate influence.

Palingenesis fixes this by all-reducing the valid token count *before* dividing:

```python
global_valid = local_valid.clone()
dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
loss = sum_of_token_losses / global_valid
```

Now every rank's loss contributes proportionally to how many real tokens it processed. This seems like a detail — but getting it wrong means your multi-GPU run converges to a different (worse) solution than single-GPU on the same data.

---

## Context Parallel (Ring Attention)

For sequences longer than one GPU can hold (>32K on 80 GB), enable Context Parallel:

```yaml
parallel:
  context_parallel: true
```

This shards the *sequence dimension* across GPUs. Each GPU holds `seq_len / num_gpus` tokens. During attention, KV pairs are rotated through the ring — each GPU computes attention on its local queries against all keys (received via rotation).

The trade-off: more communication (KV rotation), but you can now train on 128K+ sequences that would never fit on one device.

---

## Practical expectations

| GPUs | Tokens/sec (4B) | Effective speedup | Memory per GPU |
|------|----------------|:-:|:---:|
| 1 | 6,000 | 1× | 15 GB |
| 2 | 11,000 | 1.8× | ~10 GB |
| 4 | 20,000 | 3.3× | ~7 GB |
| 8 | 35,000 | 5.8× | ~5 GB |

Memory per GPU drops because model parameters are sharded. The freed memory goes to larger batch sizes (more throughput) or longer sequences.
