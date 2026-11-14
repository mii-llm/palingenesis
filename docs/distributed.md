# Distributed Training

## Parallelism Dimensions

Two composable dimensions: data parallelism (FSDP2) and context parallelism (CP).

```
World size 8, parallel.cp_degree: 4  ->  mesh ("dp", "cp") of shape (2, 4)

            cp=0     cp=1     cp=2     cp=3
  dp=0   [ rank 0 | rank 1 | rank 2 | rank 3 ]   same rows, a quarter of each
  dp=1   [ rank 4 | rank 5 | rank 6 | rank 7 ]   other rows

FSDP2 shards parameters / gradients / optimizer state over all 8 (the mesh flattened).
```

### Device Mesh Construction

```python
# Without Context Parallel (pure FSDP):
mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("dp",))

# With Context Parallel (cp_degree 0 = min(world_size, 8) when it divides world_size):
mesh = init_device_mesh(device, (world_size // cp_degree, cp_degree), mesh_dim_names=("dp", "cp"))
fsdp_mesh = mesh["dp", "cp"]._flatten("dp_cp")   # distributed.fsdp_mesh
```

- `"dp"`: ranks that read different data (the data loader shards by the dp rank).
- `"cp"`: ranks that split the same sequences (palingenesis.context_parallel).
- FSDP2 shards over both: the ranks of a CP group compute gradients from different tokens, so
  their gradients are summed like data-parallel ones (sum reduction, the loss is already divided
  by the global valid-token count).

`device` is `cuda`, or `cpu` without GPUs (gloo, used by the multi-process tests).

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

### Context Parallel

Every sequence is right-padded to a multiple of 2 x cp and cut into 2 x cp chunks; CP rank r
holds chunks r and 2cp-1-r (head-tail load balancing: equal causal-attention work per rank).
Labels are shifted on the whole sequence before the split, and the model gets the tokens' global
position ids. Per attention layer:

```
Forward:  all-gather K, V over the CP group; attend this rank's two query chunks to the keys
          before them (full) and their own chunk (causal), merged by log-sum-exp
Backward: all-gather K, V again (only local K/V are saved), kernel backward,
          reduce-scatter dK, dV (fp32) to the owning ranks
Sliding-window layers: only the keys within the window, SDPA with a band mask
```

Per Gated DeltaNet layer (Qwen3.5, Qwen3.5-MoE, Qwen3-Next), by state passing:

```
Convolution: all-gather the last kernel-1 inputs of every chunk (the halo); each chunk is
             convolved after the previous chunk's halo (zeros before chunk 0)
Delta rule:  one kernel call on both local chunks with values [v | 0] from the state [0 | I]:
             outputs from a zero state, each chunk's affine map S -> A S + B and the map of
             the entering state into the outputs; all-gather every chunk's (A, B), compose the
             state entering each local chunk, correct the outputs
Backward:    autograd through the composition and the kernel; reduce-scatter of the gathered
             states' and halos' gradients to the owning ranks
```

Exact (tested with several CPU processes: loss and gradients equal the unsplit sequence's).
Other recurrent / linear-attention layers (Mamba, linear attention of another structure), chunked
attention, soft-capping and attention sinks are refused. See the multi-GPU guide for the full list.

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
- Use SLURM with `--requeue` for automatic restart, and `train.resume_from: auto`
- Checkpoints every `train.save_steps` steps enable resume from the last complete one; `pgs train`
  does not save on SIGTERM (the RL trainer does)
- torchrun's elastic restarts (`--max_restarts`) are not supported: a restarted worker group would
  start over without `resume_from: auto`

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

Resume checkpoints (`step-N/`) use PyTorch Distributed Checkpoint (DCP), sharded:
```python
from torch.distributed.checkpoint import save as dcp_save
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_optimizer_state_dict

options = StateDictOptions(full_state_dict=False)  # each rank writes its own shard: nothing is gathered
dcp_save({"model": get_model_state_dict(model, options=options),
          "optimizer": get_optimizer_state_dict(model, optimizer, options=options)}, checkpoint_id=path / "dcp")
```

Every rank also writes its own random state (`rng_state.safetensors` for rank 0,
`rng_state.rank<r>.safetensors` for the others). After a barrier, rank 0 writes the tokenizer and
`training_meta.json`, which marks the checkpoint complete. `train.output_dir` must be on a
filesystem shared by every node.

The best and final models are gathered to rank 0's CPU memory
(`StateDictOptions(full_state_dict=True, cpu_offload=True)`) and saved in Hugging Face format.
Rank 0's evaluation loss decides, for every rank, whether a new best is saved.

### Loading

For resuming training: rank 0 picks the checkpoint (`resume_from: auto`: the latest complete one), every
rank checks that it sees it, and DCP loads each rank's shard. Another number of ranks (2 or more) is
fine: DCP re-shards, and the position in the epoch is rescaled. A sharded checkpoint does not load in a
single process, nor a single-process checkpoint on several ranks (both are refused).
For inference: load `final/` (or `best/model/`) with `from_pretrained`.

### Final Model Save

After training, we save in HuggingFace format (safetensors) on rank 0:
```python
model.save_pretrained(path, safe_serialization=True)
tokenizer.save_pretrained(path)
```

This produces a standard HF model that can be loaded anywhere without distributed dependencies.
