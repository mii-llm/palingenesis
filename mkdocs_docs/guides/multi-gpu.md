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

**Muon** — steepest descent under the spectral norm. Instead of treating each parameter as an independent scalar (like AdamW), Muon treats entire weight matrices as geometric objects. It applies a Newton-Schulz iteration (5 steps of a cubic recurrence) to find the polar decomposition of the momentum, then updates in the direction of the matrix sign. Reported faster than AdamW per step in pretraining; not benchmarked here for fine-tuning.

**+ MONA** — before Muon orthogonalizes the momentum, MONA enriches it with curvature information. The gradient difference `G_k - G_{k-1}` is approximately `H·Δθ` — the Hessian times the parameter change. This points *away from sharp minima*. MONA accumulates these differences as an EMA and adds them to the gradient before orthogonalization.

**+ Hyperball** — after the optimizer updates the weights, Hyperball projects them back to their initial Frobenius norm. This removes the radial component of the update (which is meaningless for scale-invariant layers) and keeps only the angular movement. The optimizer now only changes *direction*, not magnitude.

The composition:

```
gradient → MONA (add curvature) → Muon (orthogonalize) → step → Hyperball (project to sphere)
```

Each stage operates on a different aspect of the update. The combination runs, but has not been benchmarked against AdamW for fine-tuning: treat it as experimental.

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

Scaling is sublinear. Where the overhead goes:

1. **All-gather latency**: even with NVLink, gathering each layer's parameters takes time that can't fully overlap
2. **Reduce-scatter**: sending gradient shards back after backward
3. **Synchronization**: all ranks must finish each micro-batch before the next starts. The slowest rank (stochastic — depends on data) gates everyone.

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

## Context Parallel

Context parallelism helps when one sequence's activations do not fit on one GPU (more than about 32K tokens on 80 GB). It splits every sequence over a group of GPUs:

```yaml
parallel:
  fsdp: true             # required: FSDP sums the gradients of the GPUs that split a sequence
  context_parallel: true
  cp_degree: 4           # GPUs per sequence; 0 (default) = min(world_size, 8)
```

With 8 GPUs and `cp_degree: 4` the mesh is 2 (data) x 4 (context). Each group of 4 GPUs reads the same rows and holds a quarter of each. FSDP shards the parameters over all 8.

**Computation.** Embeddings, MLPs, norms and the loss are computed per token, so each GPU runs them on its own tokens. Attention needs the whole sequence:

- Every attention layer all-gathers the sequence's K/V from the group. This is the all-gather form of context parallelism, as in Llama 3 training. Grouped-query attention keeps K/V small.
- Each GPU attends its own queries to the gathered K/V, causally, using its tokens' global positions.
- The backward gathers K/V again and reduce-scatters the K/V gradients (in fp32) back to the GPUs that own those tokens. So no layer keeps the whole sequence's K/V between forward and backward.

The result is exact: loss and gradients equal those of the unsplit sequence. The tests check this with several processes, end to end through the trainer.

**Layout.** Each sequence is cut into 2 x `cp_degree` chunks, and GPU *r* holds chunks *r* and 2 x `cp_degree` - 1 - *r* (head-tail load balancing). Every GPU then does the same amount of causal attention work. Rows are right-padded to a multiple of 2 x `cp_degree`. This padding carries no label, and causal attention keeps it inert. Labels are shifted on the whole row before it is split, so the token that predicts across a chunk boundary keeps its target.

**Normalisation.** The loss is the sum over every GPU's tokens divided by the global count of trained tokens. The logged `train/loss` is summed over the context-parallel group. Evaluation is split the same way.

**Supported layers:**

| Layer | Status |
|---|---|
| Full causal attention (Llama, Qwen2/3, Mistral, Gemma 3 global layers) | Supported. Fused path: flash on CUDA for bf16 / fp16, exact math kernel otherwise. |
| Sliding-window attention (Gemma 3 local layers, Qwen3 / Mistral windows) | Supported. Only the keys within the window are attended, through SDPA with a band mask. |
| Gated DeltaNet linear attention and its causal convolution (Qwen3.5, Qwen3.5-MoE, Qwen3-Next) | Supported, by state passing (below). On flash-linear-attention and causal-conv1d when installed, transformers' torch kernels otherwise. |
| Other recurrent / linear-attention layers (Mamba, OLMo hybrid's Gated DeltaNet, other short convolutions) | Refused at startup. |
| Chunked attention (Llama 4), attention soft-capping (Gemma 2), attention sinks, ALiBi | Refused. |
| Packing, left padding, DPO, SeCO | Refused by config validation or at the first batch. |

**Linear attention.** A Gated DeltaNet layer carries a state S (per head, key dim x value dim) along the sequence: S_t = A_t S_t-1 + B_t with A_t = g_t (I - beta_t k_t k_t^T) and B_t = beta_t k_t v_t^T. Over a chunk, the final state and the outputs are therefore affine in the state entering it. Each GPU:

- runs the layer's kernel once on its two chunks, with the values extended by zero columns and the initial state [0 | I]. That one call gives the outputs from a zero state, each chunk's map S -> A S + B (its final state), and the term that carries an entering state into the outputs;
- all-gathers every chunk's (A, B) (one collective per layer, as in LASP-2, [arXiv:2502.07563](https://arxiv.org/abs/2502.07563), here with the matrix transitions of the delta rule), composes the state entering each of its chunks in sequence order, and corrects its outputs.

The backward is autograd through the composition and the kernel; the gathered states' gradients are reduce-scattered back to the GPUs that own those chunks. The short causal convolution in front of the delta rule reads the last kernel - 1 inputs of the previous chunk (a halo, also all-gathered; its gradient goes back the same way). Cost per linear-attention layer: the kernel's value width grows from the value head dim to value + key head dim, and the gathered states are 2 x `cp_degree` x heads x key dim x (key + value dim) fp32 values per row (about 67 MB per row and layer for 32 heads of 128 at `cp_degree: 8`). Unsupported under context parallelism: packed rows (`cu_seqlens`), cached states (generation runs without context parallelism), and chunks shorter than the convolution's look-back (kernel - 1 tokens).

Gated DeltaNet: [arXiv:2412.06464](https://arxiv.org/abs/2412.06464); the chunkwise delta rule: [arXiv:2406.06484](https://arxiv.org/abs/2406.06484). flash-linear-attention ships a context-parallel path of its own (`fla.ops.cp`, single varlen rows); this one uses only the public kernel, so it works with any batch size, on the torch fallback, and with the same head-tail layout as attention.

A runtime check backs up the startup checks. Under context parallelism every decoder layer must have run its sequence mixer through it in each forward (attention, or both the convolution and the delta rule of a Gated DeltaNet layer), or the step raises. A model whose layers bypass transformers' attention interface therefore fails loudly instead of silently attending within one GPU's slice.

**Cost.** Per attention layer, the forward all-gathers K/V, and the backward does one all-gather plus a reduce-scatter. Keep the context-parallel group within a node (NVLink): the default `cp_degree` caps it at 8. Communication is not overlapped with the attention kernels.

For a single GPU, use [SeCO](long-sequences.md) instead: it trains sequences of any length chunk by chunk.

---

## Practical expectations

Multi-GPU throughput has not been benchmarked for this release. Parameters, gradients and optimizer states are sharded across GPUs, so memory per GPU drops roughly with their count; the freed memory goes to larger batches or longer sequences. Measure your setup's tokens/s from the log (`tok/s`, and `train/tokens_per_sec_global` in wandb) over the first steps.
