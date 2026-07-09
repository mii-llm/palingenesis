# Optimization Stack

Deep technical documentation of every optimization layer in the system.

## 1. FSDP2 (Fully Sharded Data Parallelism)

**API**: `torch.distributed.fsdp.fully_shard`  
**PyTorch version**: 2.7+  
**Memory reduction**: Parameters, gradients, and optimizer states divided by world_size

### How it works

FSDP2 uses per-parameter sharding (unlike FSDP1's FlatParameter concatenation). Each parameter is independently sharded across the data-parallel mesh via `Shard(0)` placement.

```python
fully_shard(layer, mesh=dp_mesh, mp_policy=MixedPrecisionPolicy(...))
```

During forward:
1. All-gather sharded parameters to reconstruct full parameter
2. Compute layer forward
3. Optionally reshard parameters (free memory for next layer)

During backward:
1. All-gather parameters again (if resharded)
2. Compute gradients
3. Reduce-scatter gradients (each rank gets its shard of the gradient)

### Our configuration

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,   # Compute in bf16
    reduce_dtype=torch.float32,   # Gradient reduction in fp32 (stability)
)
```

**Bottom-up sharding**: We shard each transformer layer individually, then the root model. This enables FSDP to overlap communication with computation — while layer N is computing, layer N+1's parameters are being all-gathered.

**Gradient sync control**: During gradient accumulation, we disable gradient sync for non-final microsteps via `model.set_requires_gradient_sync(False)`. This avoids unnecessary reduce-scatter operations on intermediate accumulation steps.

**Disabled automatic gradient division**: FSDP normally divides gradients by world_size. We disable this (`set_gradient_divide_factor(1.0)`) because we normalize by global valid tokens instead — this gives correct loss scaling when different ranks have different numbers of valid (non-masked) tokens.

### Memory math

For an 8B parameter model in bf16:
- Full model: 8B * 2 bytes = 16 GB
- Per-rank (8 GPUs): 16 / 8 = 2 GB params
- Optimizer (AdamW fp32): 8B * 12 bytes / 8 GPUs = 12 GB/GPU
- Gradients: 16 / 8 = 2 GB/GPU
- **Total shardable**: ~16 GB/GPU (vs 112 GB without FSDP)

---

## 2. Context Parallel (Ring Attention)

**API**: `torch.distributed.tensor.experimental.context_parallel`  
**PyTorch version**: 2.7+ (experimental/unstable)  
**Memory reduction**: Attention activations divided by CP degree

### The problem

Self-attention has O(S^2) memory for the attention score matrix. For S=65536:
- Attention scores: B * H * S * S * 2 bytes = B * 32 * 65536^2 * 2 = **256 GB per sample** (without FlashAttention)
- Even with FlashAttention (O(S) memory), the KV cache and intermediate activations scale linearly with S

### How Ring Attention works

Sequence is split across CP_degree GPUs. Each GPU holds `S / CP_degree` query tokens but needs to attend to ALL key-value tokens.

**All-gather variant** (our default, used in Llama 3 training):
1. Each GPU has local Q chunk: `[B, H, S/CP, D]`
2. All-gather K and V across CP group → each GPU reconstructs full K, V
3. Compute attention: local Q against full K, V
4. Communication overlapped with initial local attention computation

**All-to-all variant** (alternative, better for some topologies):
1. K, V shards are rotated ring-style via all-to-all collectives
2. Partial attention computed against each arriving shard
3. Results combined with online softmax normalization

### Our implementation

We shard input tensors along the sequence dimension before the model forward:

```python
# Each rank gets seq_len // cp_world_size tokens
input_ids[:, start:end]  # local sequence chunk
```

The `context_parallel()` context manager replaces `F.scaled_dot_product_attention` with Ring Attention at the dispatcher level — the model code doesn't change.

### When to use

| Sequence Length | Without CP | With CP (8 GPUs) |
|----------------|-----------|-------------------|
| 8192           | Fits 1 GPU | Overkill         |
| 32768          | Tight on 1 | 4096 per GPU     |
| 65536          | OOM       | 8192 per GPU      |
| 131072         | OOM       | 16384 per GPU     |

Rule: enable CP when `seq_len > 16384` and you have multiple GPUs.

---

## 3. Liger Kernel (Fused Triton Operations)

**Library**: `liger-kernel` (LinkedIn)  
**Throughput improvement**: ~20%  
**Memory reduction**: ~60% on fused operations

### What gets fused

| Operation | Standard HF | Liger Fused | Memory Savings |
|-----------|-------------|-------------|----------------|
| RMSNorm | Separate square, mean, rsqrt, mul | Single Triton kernel | Eliminates intermediate tensors |
| SwiGLU | gate + up + activation + mul | Fused gate-up-activation | 50% fewer intermediates |
| RoPE | Compute sin/cos, interleave, mul | Fused rotary embedding | No cos/sin materialization |
| CrossEntropy | Materialize [B,S,V] logits, softmax, NLL | Chunked tiled CE | O(B*S*chunk) instead of O(B*S*V) |

### FusedLinearCrossEntropy (the killer optimization)

The standard CE computation materializes the full logit tensor:
```
hidden [B, S, D] @ weight [V, D].T = logits [B, S, V]
```

For B=1, S=8192, V=128256 in float32: **4.0 GB** just for logits.

Liger's fused kernel computes CE loss in tiles without ever materializing the full logit tensor. It processes `chunk_size` tokens at a time, computing the local softmax normalization via online algorithms (like FlashAttention does for attention).

### How we apply it

```python
# Called BEFORE model instantiation — patches HF class definitions
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()
# Now when AutoModelForCausalLM.from_pretrained() runs, it instantiates
# patched classes with fused forward methods
```

---

## 4. Chunked Cross-Entropy Loss

**Inspiration**: torchtitan's `ChunkedCELoss`  
**Memory reduction**: Peak logit memory divided by `num_chunks`

### The problem (again)

Even with Liger's fused CE, if we use the model's built-in loss computation (which HF's `forward(labels=...)` does), it still needs the full logit tensor for the backward pass gradient.

### Our solution

We separate the model into backbone + lm_head and manage the loss computation ourselves:

```python
# 1. Run backbone only (no lm_head)
hidden = model.model(input_ids, attention_mask)  # [B, S, D]

# 2. Split hidden states into chunks
chunks = hidden.split(S // num_chunks, dim=1)  # N chunks of [B, S/N, D]

# 3. Per-chunk: project + CE + backward (immediately frees logits)
for h_chunk, label_chunk in zip(chunks, label_chunks):
    logits = lm_head(h_chunk)           # [B, S/N, V] — only this exists
    loss = CE(logits, label_chunk)      # scalar
    loss.backward()                     # frees logits immediately
    # Accumulate gradient for h_chunk

# 4. Backward through decoder with accumulated gradient
hidden.backward(accumulated_grad)
```

### Memory savings

| Config | Logit peak memory (B=1, S=8192, V=128k) |
|--------|------------------------------------------|
| Standard | 4.0 GB |
| num_chunks=4 | 1.0 GB |
| num_chunks=8 | 0.5 GB |
| num_chunks=16 | 0.25 GB |

---

## 5. Selective Activation Checkpointing

**API**: `torch.distributed.algorithms._checkpoint.checkpoint_wrapper` + `create_selective_checkpoint_contexts`  
**Memory reduction**: ~50-60% of activation memory  
**Compute overhead**: ~15-20% (recomputes cheap ops only)

### The insight

Full activation checkpointing saves nothing during forward and recomputes everything during backward — 33% compute overhead. But most of that recomputation is wasted on cheap operations (norms, element-wise activations).

Selective AC saves the outputs of EXPENSIVE ops (which are fast to compute but expensive to recompute because they're large):
- `scaled_dot_product_attention` — the attention output
- `linear` (every other one) — half the matmul outputs

And recomputes CHEAP ops:
- RMSNorm (element-wise, tiny)
- SiLU/GeLU activations (element-wise)
- The other half of linear operations (trade compute for memory)

### Policy implementation

```python
_SAVE_OPS = {
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten.linear.default,
}

def policy(ctx, func, *args, **kwargs):
    if func in _SAVE_OPS:
        if func == torch.ops.aten.linear.default:
            mm_count += 1
            if mm_count % 2 == 0:
                return CheckpointPolicy.PREFER_RECOMPUTE  # Recompute half
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE  # Recompute everything else
```

### Why "every other matmul"?

A transformer layer has ~7-8 linear operations (Q, K, V, O projections + gate, up, down in FFN). Saving ALL of them gives no memory benefit (same as no checkpointing). Saving NONE recomputes expensive operations. Saving 50% hits the sweet spot — the recomputed matmuls can overlap with the saved matmuls' gradient computation.

---

## 6. Per-Layer torch.compile

**API**: `torch.compile(layer, backend="inductor", fullgraph=True)`  
**Throughput improvement**: 10-30% depending on model and hardware

### Why per-layer, not full-model?

Full-model compilation is problematic:
1. Compilation time scales superlinearly with graph size (>30 min for 8B)
2. Breaks with dynamic control flow (FSDP hooks, gradient checkpointing)
3. Recompiles on shape changes (padding variation across batches)

Per-layer compilation:
1. Each layer is a small, repeated subgraph — compiles in seconds
2. `fullgraph=True` ensures no graph breaks within a layer
3. Inductor can fuse element-wise ops, optimize memory layout, generate efficient kernels
4. Since all layers are structurally identical, compilation is cached after the first layer

### What inductor optimizes

- Fuses chains of element-wise operations into single CUDA kernels
- Optimizes memory access patterns (layout transformations)
- Constant-folds known shapes
- Generates specialized kernels for the exact tensor sizes

---

## 7. Mixed Precision (bf16 + fp32 reduction)

### Strategy

| Component | Dtype | Rationale |
|-----------|-------|-----------|
| Model parameters | bfloat16 | 2x memory reduction, hardware acceleration |
| Forward activations | bfloat16 | Computed via autocast |
| Gradient computation | bfloat16 | Same precision as forward |
| Gradient reduction (allreduce) | float32 | Prevents precision loss in summation |
| Optimizer states | float32 | AdamW moments need precision for stability |
| Loss computation | float32 | CE with large vocab needs dynamic range |

### Why bfloat16 over float16?

bfloat16 has the same exponent range as float32 (8 bits) but reduced mantissa (7 bits vs 23). This means:
- No gradient scaling needed (unlike fp16 which has limited range)
- No loss scaling needed
- No overflow on large activations
- Only cost: slightly less precision, which is fine for training

### The fp32 reduction trick

When FSDP reduce-scatters gradients, it sums across ranks. In bf16, this summation loses precision (7-bit mantissa means catastrophic cancellation on small differences). Using fp32 for reduction preserves gradient information:

```python
MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # Stored and computed in bf16
    reduce_dtype=torch.float32,    # Reduced in fp32
)
```

---

## Composition and Interaction

These optimizations compose multiplicatively:

```
Base memory for 8B model, B=1, S=8192:
  Parameters:      16 GB
  Optimizer:       96 GB (AdamW fp32: 12 bytes/param)
  Gradients:       16 GB
  Activations:     ~40 GB (32 layers * ~1.2 GB/layer)
  CE logits:       4 GB
  TOTAL:           ~172 GB (doesn't fit anywhere)

After FSDP (8 GPUs):
  Params + Optim + Grads: 172 * (16+96+16)/172 / 8 = 16 GB/GPU
  Activations:     ~40 GB (not sharded by FSDP)
  CE logits:       4 GB
  TOTAL:           ~60 GB/GPU (tight on A100-80GB)

After Selective AC:
  Activations:     ~40 * 0.4 = 16 GB (60% reduction)
  TOTAL:           ~36 GB/GPU ✓

After Chunked CE (8 chunks):
  CE logits:       4 / 8 = 0.5 GB
  TOTAL:           ~32.5 GB/GPU ✓✓

After Liger Kernel:
  Activations:     further ~30% reduction on fused ops
  TOTAL:           ~28 GB/GPU ✓✓✓ (plenty of headroom)

With Context Parallel (additionally):
  Activations:     further / CP_degree
  Enables:         S=65536 with same memory budget
```
