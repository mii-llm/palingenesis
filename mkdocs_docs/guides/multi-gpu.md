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

**+ MONA** — before Muon orthogonalizes the momentum, MONA enriches it with curvature information. The gradient difference $G_k - G_{k-1}$ is approximately $H\,\Delta\theta$ — the Hessian times the parameter change. This points *away from sharp minima*. MONA accumulates these differences as an EMA and adds them to the gradient before orthogonalization.

**+ Hyperball** — after the optimizer updates the weights, Hyperball projects them back to their initial Frobenius norm. This removes the radial component of the update (which is meaningless for scale-invariant layers) and keeps only the angular movement. The optimizer now only changes *direction*, not magnitude.

The composition:

gradient → **MONA** (add curvature) → **Muon** (orthogonalize) → step → **Hyperball** (project to sphere)
{: .pg-flow }

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

**Layout.** Each sequence is cut into $2c$ chunks, where $c$ is `cp_degree`, and GPU $r$ holds chunks $r$ and $2c - 1 - r$ (head-tail load balancing). Every GPU then does the same amount of causal attention work. Rows are right-padded to a multiple of $2c$. This padding carries no label, and causal attention keeps it inert. Labels are shifted on the whole row before it is split, so the token that predicts across a chunk boundary keeps its target.

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 270" role="img" aria-labelledby="fig-cp-title fig-cp-desc">
<title id="fig-cp-title">Context parallelism with 8 GPUs and cp_degree 4: a 2 by 4 mesh and the head-tail layout of one sequence</title>
<desc id="fig-cp-desc">Left: 8 GPUs form a mesh of 2 data-parallel groups by 4 context ranks; the 4 GPUs of a group read the same rows. Right: each sequence is cut into 8 chunks, numbered 0 to 7. Context rank r holds chunks r and 7 minus r: rank 0 holds 0 and 7, rank 1 holds 1 and 6, rank 2 holds 2 and 5, rank 3 holds 3 and 4. In every attention layer each rank all-gathers the K/V of all chunks and attends its own queries causally, so every rank attends to 9 key chunks in total.</desc>
<text class="t-label" x="10" y="20">mesh · 2 data × 4 context</text>
<text class="t-muted t-small" x="10" y="78">data 0</text>
<rect class="box" x="62" y="58" width="30" height="30" rx="3"/>
<text class="t-mono" x="77" y="77" text-anchor="middle">0</text>
<rect class="box" x="98" y="58" width="30" height="30" rx="3"/>
<text class="t-mono" x="113" y="77" text-anchor="middle">1</text>
<rect class="box" x="134" y="58" width="30" height="30" rx="3"/>
<text class="t-mono" x="149" y="77" text-anchor="middle">2</text>
<rect class="box" x="170" y="58" width="30" height="30" rx="3"/>
<text class="t-mono" x="185" y="77" text-anchor="middle">3</text>
<text class="t-muted t-small" x="10" y="124">data 1</text>
<rect class="box" x="62" y="104" width="30" height="30" rx="3"/>
<text class="t-mono" x="77" y="123" text-anchor="middle">0</text>
<rect class="box" x="98" y="104" width="30" height="30" rx="3"/>
<text class="t-mono" x="113" y="123" text-anchor="middle">1</text>
<rect class="box" x="134" y="104" width="30" height="30" rx="3"/>
<text class="t-mono" x="149" y="123" text-anchor="middle">2</text>
<rect class="box" x="170" y="104" width="30" height="30" rx="3"/>
<text class="t-mono" x="185" y="123" text-anchor="middle">3</text>
<text class="t-muted t-small" x="62" y="46">context rank r</text>
<path class="line-quiet" d="M58,54 H204 V92 H58 Z"/>
<text class="t-muted t-small" x="10" y="176">One data group reads</text>
<text class="t-muted t-small" x="10" y="191">the same rows; its 4 GPUs</text>
<text class="t-muted t-small" x="10" y="206">split each sequence.</text>
<text class="t-muted t-small" x="10" y="226">FSDP shards the</text>
<text class="t-muted t-small" x="10" y="241">parameters over all 8.</text>
<line class="rule" x1="226" y1="14" x2="226" y2="256"/>
<text class="t-label" x="244" y="20">one sequence · 2 × cp_degree = 8 chunks</text>
<rect class="box" x="290" y="32" width="40" height="24" rx="2"/>
<text class="t-mono" x="310.0" y="48" text-anchor="middle">0</text>
<rect class="box" x="334" y="32" width="40" height="24" rx="2"/>
<text class="t-mono" x="354.0" y="48" text-anchor="middle">1</text>
<rect class="box" x="378" y="32" width="40" height="24" rx="2"/>
<text class="t-mono" x="398.0" y="48" text-anchor="middle">2</text>
<rect class="box" x="422" y="32" width="40" height="24" rx="2"/>
<text class="t-mono" x="442.0" y="48" text-anchor="middle">3</text>
<rect class="box" x="466" y="32" width="40" height="24" rx="2"/>
<text class="t-mono" x="486.0" y="48" text-anchor="middle">4</text>
<rect class="box" x="510" y="32" width="40" height="24" rx="2"/>
<text class="t-mono" x="530.0" y="48" text-anchor="middle">5</text>
<rect class="box" x="554" y="32" width="40" height="24" rx="2"/>
<text class="t-mono" x="574.0" y="48" text-anchor="middle">6</text>
<rect class="box" x="598" y="32" width="40" height="24" rx="2"/>
<text class="t-mono" x="618.0" y="48" text-anchor="middle">7</text>
<text class="t-muted t-small" x="290" y="74">sequence order →</text>
<text class="t-small" x="244" y="113">r = 0</text>
<rect class="box-accent" x="290" y="96" width="40" height="24" rx="2"/>
<text class="t-mono" x="310.0" y="112" text-anchor="middle">0</text>
<rect class="box-quiet" x="334" y="96" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="378" y="96" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="422" y="96" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="466" y="96" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="510" y="96" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="554" y="96" width="40" height="24" rx="2"/>
<rect class="box-accent" x="598" y="96" width="40" height="24" rx="2"/>
<text class="t-mono" x="618.0" y="112" text-anchor="middle">7</text>
<text class="t-muted t-small" x="648" y="113">1 + 8</text>
<text class="t-small" x="244" y="147">r = 1</text>
<rect class="box-quiet" x="290" y="130" width="40" height="24" rx="2"/>
<rect class="box-accent" x="334" y="130" width="40" height="24" rx="2"/>
<text class="t-mono" x="354.0" y="146" text-anchor="middle">1</text>
<rect class="box-quiet" x="378" y="130" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="422" y="130" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="466" y="130" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="510" y="130" width="40" height="24" rx="2"/>
<rect class="box-accent" x="554" y="130" width="40" height="24" rx="2"/>
<text class="t-mono" x="574.0" y="146" text-anchor="middle">6</text>
<rect class="box-quiet" x="598" y="130" width="40" height="24" rx="2"/>
<text class="t-muted t-small" x="648" y="147">2 + 7</text>
<text class="t-small" x="244" y="181">r = 2</text>
<rect class="box-quiet" x="290" y="164" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="334" y="164" width="40" height="24" rx="2"/>
<rect class="box-accent" x="378" y="164" width="40" height="24" rx="2"/>
<text class="t-mono" x="398.0" y="180" text-anchor="middle">2</text>
<rect class="box-quiet" x="422" y="164" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="466" y="164" width="40" height="24" rx="2"/>
<rect class="box-accent" x="510" y="164" width="40" height="24" rx="2"/>
<text class="t-mono" x="530.0" y="180" text-anchor="middle">5</text>
<rect class="box-quiet" x="554" y="164" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="598" y="164" width="40" height="24" rx="2"/>
<text class="t-muted t-small" x="648" y="181">3 + 6</text>
<text class="t-small" x="244" y="215">r = 3</text>
<rect class="box-quiet" x="290" y="198" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="334" y="198" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="378" y="198" width="40" height="24" rx="2"/>
<rect class="box-accent" x="422" y="198" width="40" height="24" rx="2"/>
<text class="t-mono" x="442.0" y="214" text-anchor="middle">3</text>
<rect class="box-accent" x="466" y="198" width="40" height="24" rx="2"/>
<text class="t-mono" x="486.0" y="214" text-anchor="middle">4</text>
<rect class="box-quiet" x="510" y="198" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="554" y="198" width="40" height="24" rx="2"/>
<rect class="box-quiet" x="598" y="198" width="40" height="24" rx="2"/>
<text class="t-muted t-small" x="648" y="215">4 + 5</text>
<text class="t-muted t-small" x="696" y="88" text-anchor="end">key chunks</text>
<rect class="box-accent" x="244" y="236" width="14" height="12" rx="2"/><text class="t-small" x="264" y="246">chunks the rank holds (its queries, K/V, labels)</text>
<rect class="box-quiet" x="244" y="254" width="14" height="12" rx="2"/><text class="t-small" x="264" y="264">K/V all-gathered in each attention layer, not kept for backward</text>
</svg>
</div>
<figcaption>Context parallelism with 8 GPUs and <code>cp_degree: 4</code>. Rank <em>r</em> holds chunks <em>r</em> and 7 − <em>r</em>, so with causal attention every rank attends to the same number of key chunks (right column).</figcaption>
</figure>

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

**Linear attention.** A Gated DeltaNet layer carries a state $S$ (per head, key dim $\times$ value dim) along the sequence:

$$
S_t = A_t S_{t-1} + B_t, \qquad A_t = g_t \left(I - \beta_t k_t k_t^{\top}\right), \qquad B_t = \beta_t k_t v_t^{\top}
$$

where $k_t$, $v_t$ are the token's key and value, $g_t$ its decay gate and $\beta_t$ its write strength. Over a chunk, the final state and the outputs are therefore affine in the state entering it. Each GPU:

- runs the layer's kernel once on its two chunks, with the values extended by zero columns and the initial state $[\,0 \mid I\,]$. That one call gives the outputs from a zero state, each chunk's map $S \mapsto AS + B$ (its final state), and the term that carries an entering state into the outputs;
- all-gathers every chunk's $(A, B)$ (one collective per layer, as in LASP-2, [arXiv:2502.07563](https://arxiv.org/abs/2502.07563), here with the matrix transitions of the delta rule), composes the state entering each of its chunks in sequence order, and corrects its outputs.

The backward is autograd through the composition and the kernel; the gathered states' gradients are reduce-scattered back to the GPUs that own those chunks. The short causal convolution in front of the delta rule reads the last $\text{kernel} - 1$ inputs of the previous chunk (a halo, also all-gathered; its gradient goes back the same way). Cost per linear-attention layer: the kernel's value width grows from the value head dim to value + key head dim, and the gathered states are $2c \times \text{heads} \times d_k \times (d_k + d_v)$ fp32 values per row ($d_k$, $d_v$: key and value head dims) (about 67 MB per row and layer for 32 heads of 128 at `cp_degree: 8`). Unsupported under context parallelism: packed rows (`cu_seqlens`), cached states (generation runs without context parallelism), and chunks shorter than the convolution's look-back ($\text{kernel} - 1$ tokens).

Gated DeltaNet: [arXiv:2412.06464](https://arxiv.org/abs/2412.06464); the chunkwise delta rule: [arXiv:2406.06484](https://arxiv.org/abs/2406.06484). flash-linear-attention ships a context-parallel path of its own (`fla.ops.cp`, single varlen rows); this one uses only the public kernel, so it works with any batch size, on the torch fallback, and with the same head-tail layout as attention.

A runtime check backs up the startup checks. Under context parallelism every decoder layer must have run its sequence mixer through it in each forward (attention, or both the convolution and the delta rule of a Gated DeltaNet layer), or the step raises. A model whose layers bypass transformers' attention interface therefore fails loudly instead of silently attending within one GPU's slice.

**Cost.** Per attention layer, the forward all-gathers K/V, and the backward does one all-gather plus a reduce-scatter. Keep the context-parallel group within a node (NVLink): the default `cp_degree` caps it at 8. Communication is not overlapped with the attention kernels.

For a single GPU, use [SeCO](long-sequences.md) instead: it trains sequences of any length chunk by chunk.

---

## Practical expectations

Multi-GPU throughput has not been benchmarked for this release. Parameters, gradients and optimizer states are sharded across GPUs, so memory per GPU drops roughly with their count; the freed memory goes to larger batches or longer sequences. Measure your setup's tokens/s from the log (`tok/s`, and `train/tokens_per_sec_global` in wandb) over the first steps.
