# Distributed Training

## Parallelism Dimensions

The system supports three composable parallelism dimensions:

```
┌───────────────────────────────────────────────────────┐
│                    World Size = 16                     │
│                                                       │
│  ┌─────────────── DP (FSDP2) ────────────────────┐   │
│  │  Rank 0─7: shard params/grads/optim            │   │
│  │                                                │   │
│  │  ┌──── CP (Context Parallel) ────────────┐    │   │
│  │  │  Rank 0─3: seq shard (ring attention) │    │   │
│  │  │  Rank 4─7: seq shard (ring attention) │    │   │
│  │  └───────────────────────────────────────┘    │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
│  ┌─────────────── DP (FSDP2) ────────────────────┐   │
│  │  Rank 8─15: shard params/grads/optim           │   │
│  │  ...                                           │   │
│  └────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
```

### Device Mesh Construction

```python
# Without Context Parallel (pure FSDP):
mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

# With Context Parallel:
# world_size=8, cp_degree=8, dp_degree=1
mesh = init_device_mesh("cuda", (dp_degree, cp_degree), mesh_dim_names=("dp", "cp"))
```

The mesh is a multi-dimensional grid where:
- `"dp"` axis: FSDP shards parameters across these ranks
- `"cp"` axis: Context Parallel shards sequences across these ranks

### Scaling Rules

| GPUs | Recommended Config | Effective Batch | Max Sequence |
|------|-------------------|-----------------|--------------|
| 1 | No parallelism | per_device_bs * grad_accum | ~8192 (8B model) |
| 2-4 | FSDP only | bs * grad_accum * dp | ~16384 |
| 8 | FSDP + optional CP | bs * grad_accum * dp | ~65536 with CP |
| 16+ | FSDP + CP | bs * grad_accum * dp | 131072+ with CP |

## NCCL Communication Patterns

### FSDP2 Communication

Per optimizer step:
```
Forward:  all-gather × num_layers  (reconstruct params)
Backward: all-gather × num_layers  (reconstruct for grad compute)
          reduce-scatter × num_layers  (distribute gradients)
```

With `reshard_after_forward=True` (default):
- Each layer's params are freed after forward → lower peak memory
- Re-gathered in backward → extra all-gather cost
- Net: more memory-efficient, slightly more communication

With `reshard_after_forward=False`:
- Params stay gathered after forward → used directly in backward
- Lower communication but higher peak memory
- Use when memory is not the bottleneck

### Context Parallel Communication

Per attention layer:
```
All-gather based (default):
  Forward:  all-gather K, V across CP group (1 collective per layer)
  Backward: reduce-scatter dK, dV across CP group

All-to-all based (alternative):
  Forward:  N-1 all-to-all rounds (ring rotation of KV shards)
  Backward: N-1 all-to-all rounds (ring rotation of dKV shards)
```

### Communication/Computation Overlap

FSDP2 achieves overlap automatically:
1. Layer N forward starts → triggers all-gather for layer N+1
2. By the time layer N finishes → layer N+1's params are ready
3. Same for backward: reduce-scatter overlaps with next layer's backward

This is why bottom-up sharding is critical — each layer must be an independent FSDP unit for the overlap to work.

## Multi-Node Training

### Network Requirements

- Nodes connected via high-bandwidth interconnect (InfiniBand preferred)
- NCCL uses IB for inter-node, NVLink for intra-node
- Minimum: 100 Gbps inter-node for 8B models
- Recommended: 400 Gbps (HDR InfiniBand) for >13B models

### Rendezvous

We use `c10d` backend (TCP-based):
```bash
torchrun --nnodes=N --rdzv_backend=c10d --rdzv_endpoint=MASTER:PORT
```

Each node:
1. Contacts the master's rendezvous endpoint
2. Gets assigned a rank range
3. Establishes NCCL connections to all other ranks

### Fault Tolerance

Current: none (if one rank dies, all ranks abort). For production:
- Use SLURM with `--requeue` for automatic restart
- Checkpoints every N steps enable resume from last checkpoint
- Elastic training (torchrun elastic) can handle node additions/removals

## Gradient Accumulation with FSDP

Gradient accumulation interacts with FSDP's communication:

```python
for micro_step in range(grad_accum_steps):
    if micro_step < grad_accum_steps - 1:
        model.set_requires_gradient_sync(False)  # Skip reduce-scatter
    else:
        model.set_requires_gradient_sync(True)   # Do reduce-scatter on last micro

    loss = model(batch).loss
    loss.backward()

optimizer.step()
```

Without this optimization: `grad_accum_steps * reduce_scatter_per_step` collectives.
With: just `1 * reduce_scatter_per_step` collectives. Saves (N-1)/N of backward communication.

## Checkpointing with FSDP

### Saving

Uses PyTorch Distributed Checkpoint (DCP):
```python
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

options = StateDictOptions(full_state_dict=True, cpu_offload=True)
state = get_model_state_dict(model, options=options)
```

`full_state_dict=True`: gathers all shards to rank 0
`cpu_offload=True`: moves gathered state to CPU (avoids GPU OOM during save)

### Loading

For resuming training: load distributed checkpoint (each rank loads its shard)
For inference: load the full state dict saved by rank 0 into any model

### Final Model Save

After training, we save in HuggingFace format (safetensors) on rank 0:
```python
model.save_pretrained(path, safe_serialization=True)
tokenizer.save_pretrained(path)
```

This produces a standard HF model that can be loaded anywhere without distributed dependencies.
