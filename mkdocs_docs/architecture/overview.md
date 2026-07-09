# Architecture

*How the pieces fit together, and why each one is there.*

---

## The training step

Every optimizer step in palingenesis follows this sequence:

```
data → tokenize → pack → prefetch → forward → loss → backward → clip → step → log → checkpoint
```

Each stage is a module. They compose orthogonally — you can swap any part without touching the others. Here's the map:

| Stage | Module | What it decides |
|-------|--------|----------------|
| Data | `data.py` | Chat template masking, packing, multi-source mixing |
| Tokenize | HuggingFace tokenizer | Which tokens are system/user/assistant |
| Pack | `data.PackedDataset` | Multiple conversations per sequence, position_id resets |
| Prefetch | `perf.CUDAPrefetcher` | Overlaps PCIe transfer with previous step's compute |
| Forward | HuggingFace model + `torch.compile` | The neural network computation |
| Loss | `loss.py` | Standard CE, chunked CE, Cut CE, or DEFT |
| Backward | PyTorch autograd + `memory.GradientRelease` | Gradient computation, optional fused optimizer step |
| Clip | `perf.AdaGC` or `torch.nn.utils.clip_grad_norm_` | Per-tensor adaptive or global clipping |
| Step | Optimizer + `optim.HyperballWrapper` + `optim.MONAAcceleration` | Weight update with optional norm projection |
| Log | `health.HealthMonitor` + `logging.Tracker` | Tiered diagnostics, wandb/trackio |
| Checkpoint | `checkpoint.py` | Sharded DCP (FSDP) or safetensors (single GPU) |

---

## Memory model

The genius of the memory stack is that each optimization removes a different category of waste:

```
┌─────────────────────────────────────────────────────┐
│  TOTAL GPU MEMORY                                    │
├─────────────────────────────────────────────────────┤
│  Model weights (bf16)          │ FIXED: 2B per param │
├────────────────────────────────┼─────────────────────┤
│  Optimizer states              │ Lion8bit: 0.5B/param │
│  (AdamW would be 8B/param)    │ saved: 94%          │
├────────────────────────────────┼─────────────────────┤
│  Gradients                     │ Gradient release: 0  │
│  (normally 2B/param)           │ saved: 100%         │
├────────────────────────────────┼─────────────────────┤
│  Activations                   │ Selective AC: ~30%   │
│  (for backward recomputation) │ of naive             │
├────────────────────────────────┼─────────────────────┤
│  Loss logits [B,S,V]           │ Chunked: 1/N at a   │
│  (can be enormous: 4B×4K×262K │ time. Or CCE: zero.  │
│   = 16 GB in fp32!)            │                     │
└────────────────────────────────┴─────────────────────┘
```

These compose multiplicatively. The result: a 4B model in 15 GB.

---

## Distributed model

For multi-GPU, the architecture adds three layers:

### FSDP2 (data parallelism + sharding)

Each transformer layer is wrapped with `fully_shard()`. Parameters are sharded across GPUs; during forward/backward, all-gathers bring each layer's params to full and reduce-scatters send gradients back. Communication overlaps with compute because each layer is an independent FSDP unit.

Palingenesis adds two optimizations over vanilla FSDP2:

1. **Last-layer skip**: the final transformer layer doesn't reshard after forward, because FSDP would immediately re-gather it for backward. Saves one reshard + one all-gather per step.

2. **Weight-tying grouping**: when `tok_embeddings` and `lm_head` share a parameter (common in Llama, Qwen, Gemma), they're placed in a single FSDP unit to avoid duplicate communication.

### Context Parallel (sequence sharding)

For sequences longer than one GPU's memory can handle, the sequence dimension is split across GPUs. Attention uses Ring Attention — each GPU computes attention on its local chunk while rotating KV through the ring.

### Global valid-token normalization

Loss is `sum(token_losses) / global_valid_tokens` where `global_valid_tokens` is all-reduced across all DP ranks. This ensures correct gradient scale even when different ranks have different amounts of padding.

---

## The optimizer stack

When all features are enabled, the optimizer step is:

```python
# 1. MONA: augment gradients with curvature-aware acceleration
mona.apply()  # G̃ = G + α·A where A tracks gradient differences

# 2. Base optimizer (Muon, Lion, AdamW)
optimizer.step()  # Standard weight update using augmented gradients

# 3. Hyperball: project weights back to hypersphere
hyperball.step()  # W = R · W/‖W‖ for each constrained matrix
```

MONA adds information (curvature). The base optimizer turns that into a direction. Hyperball removes the radial component, keeping only the angular movement. The composition is clean because each stage operates on a different aspect of the update.

---

## Checkpoint design

Two modes, chosen automatically:

| Scenario | Format | Extra memory | Resume speed |
|----------|--------|:---:|:---:|
| Single GPU | HF safetensors (sharded 2 GB) | 0 | Fast (memory-mapped) |
| FSDP (multi-GPU) | Distributed Checkpoint (DCP) | 0 | Fast (each rank loads its shard) |

Final export is always HF-format (gathered to rank 0) so the output works with `from_pretrained` anywhere.

The `BestModelTracker` maintains a shadow copy at `output/best/` — updated whenever eval loss improves. Never purged. This is the checkpoint you deploy.

---

## Module dependency graph

```
train.py
├── config.py          (flat YAML → typed dataclass)
├── data.py            (ChatDataset, PackedDataset, collation)
├── loss.py            (CE, chunked CE, CCE, DEFT bridge)
├── optim.py           (build_optimizer, schedulers, Hyperball, MONA, SAGE)
├── distributed.py     (FSDP2, mesh, Context Parallel)
├── memory.py          (GradientRelease, SelectiveDiff)
├── health.py          (HealthMonitor, entropy tracking)
├── perf.py            (Prefetch, GC, AdaGC, SpikeDetector, EMA, SLERP)
├── plugins.py         (DEFT, DFT, InfoSFT, PreRL, SymNoise)
├── checkpoint.py      (save/load, DCP, auto-purge, BestModelTracker)
├── kernels.py         (Liger kernel patching, activation checkpointing)
└── logging.py         (Tracker: wandb + trackio)
```

No circular dependencies. Each module imports only from modules above it in this list (with the exception of `config.py` which everything reads from).
