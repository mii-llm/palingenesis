# Multi-Node & SLURM

*Crossing the node boundary changes everything about failure modes. This guide is about what breaks and how to prevent it.*

---

## When you need multiple nodes

Single-node multi-GPU (the [previous guide](multi-gpu.md)) handles models up to ~35B on 8× A100-80GB. Beyond that — or when you need more throughput than 8 GPUs provide — you cross the node boundary.

The physics change. Inside a node, GPUs communicate via NVLink (600-900 GB/s, ~5 μs latency). Between nodes, communication goes over InfiniBand (200-400 Gbps, ~1-5 μs) or worse, Ethernet (10-100 Gbps, ~50 μs). This 10-100× bandwidth difference means:

- All-gathers that were "free" (hidden behind compute) now become visible
- A straggler on one node blocks all other nodes
- A single network hiccup can timeout the entire job
- Checkpoint saving must go to a shared filesystem (which has its own latency)

Multi-node training works well when you understand these constraints.

---

## SLURM

Most HPC clusters use SLURM for job scheduling. One command:

```bash
sbatch scripts/train_slurm.sh configs/qwen35_35b_moe/a100_80gb_multigpu.yaml
```

Override node count:

```bash
sbatch --nodes=4 scripts/train_slurm.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
```

### What happens when you submit

```
sbatch → SLURM queues the job
       → SLURM allocates N nodes with exclusive access
       → srun launches 1 task per node
       → each task runs torchrun which spawns 8 worker processes
       → c10d rendezvous connects all N×8 workers via TCP to MASTER_ADDR
       → FSDP2 shards the model across all workers
       → training begins
```

The critical detail: `srun` launches torchrun, not Python directly. Torchrun handles per-node process spawning and local GPU assignment. The `--rdzv_id=$SLURM_JOB_ID` flag ensures multiple concurrent jobs on the same cluster don't interfere with each other's rendezvous.

### The SLURM script explained

```bash title="scripts/train_slurm.sh" hl_lines="5 8 13"
#SBATCH --nodes=2              # How many machines
#SBATCH --gpus-per-node=8      # All GPUs on each machine
#SBATCH --ntasks-per-node=1    # One torchrun launcher per node
#SBATCH --exclusive            # Don't share nodes (predictable performance)

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun --kill-on-bad-exit=1 \            # Kill everything if any node crashes
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=8 \
        --rdzv_id=$SLURM_JOB_ID \      # Unique per job (no collision)
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:29500 \
        -m palingenesis.train --config $CONFIG
```

---

## Bare-metal / cloud VMs

For clusters without SLURM (AWS, GCP, on-prem servers with SSH):

Run on **every node** simultaneously. They'll connect via the master address.

```bash
# Node 0 (the master — other nodes connect to this)
MASTER_ADDR=10.0.0.1 NODE_RANK=0 NNODES=4 \
    ./scripts/train_multi_node.sh configs/qwen35_4b/a100_80gb_multigpu.yaml

# Node 1
MASTER_ADDR=10.0.0.1 NODE_RANK=1 NNODES=4 \
    ./scripts/train_multi_node.sh configs/qwen35_4b/a100_80gb_multigpu.yaml

# Node 2, 3... (same pattern)
```

!!! warning "Order doesn't matter, timing does"
    All nodes must start within ~5 minutes of each other (the rendezvous timeout). If node 3 starts 10 minutes after node 0, the rendezvous will have timed out and node 0 will have crashed.

---

## What goes wrong (and how to fix it)

### Job hangs at startup

**Symptom**: all nodes print "Initializing process group" and then nothing for 5+ minutes.

**Cause**: one node can't reach `MASTER_ADDR:MASTER_PORT`. Firewall, wrong interface, DNS resolution failure.

**Fix**:
```bash
# On every node, verify connectivity:
nc -zv $MASTER_ADDR 29500
```
If this fails, check: firewall rules, `NCCL_SOCKET_IFNAME` (might need to specify the IB interface), DNS resolution of the master hostname.

### Job hangs mid-training

**Symptom**: training runs for N steps then freezes. All ranks stop logging.

**Cause**: NCCL timeout during an all-reduce. One rank hit an unusually long computation (bad batch, GC stall, or the OS scheduled something on a CPU core) and the other ranks timed out waiting.

**Fix**:
```bash
export NCCL_IB_TIMEOUT=120        # Increase from default 20
export NCCL_ASYNC_ERROR_HANDLING=1 # Fail fast instead of hanging forever
```

### Checkpoint save fails

**Symptom**: `OSError: No such file or directory` during checkpoint save.

**Cause**: `output_dir` is a local path, not on a shared filesystem. Rank 0 writes metadata, other ranks write DCP shards — all to the same directory. If the directory only exists on one machine, other machines fail.

**Fix**: use a shared filesystem path (NFS mount, Lustre, S3-backed FUSE, etc.):
```yaml
train:
  output_dir: /shared/nfs/experiments/run_001
```

### One node is 10% slower than others

**Symptom**: `dt` in logs varies between steps (1.1s, 1.3s, 1.1s, 1.8s). Throughput is lower than expected.

**Cause**: FSDP synchronizes on every all-gather. The slowest rank gates everyone. If one node has a weaker GPU, less bandwidth, or background processes stealing CPU, all nodes run at its speed.

**Fix**: ensure identical hardware across nodes. Use `--exclusive` in SLURM. Kill any background processes. Check IB link status: `ibstat | grep Rate`.

---

## NCCL environment

The launch scripts set these automatically:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1    # Required for compute/comm overlap
export NCCL_ASYNC_ERROR_HANDLING=1      # Crash immediately on NCCL error (don't hang)
export NCCL_IB_TIMEOUT=120              # 2 minutes before IB timeout (generous)
export NCCL_DEBUG=WARN                  # Print NCCL warnings (INFO is very verbose)
```

For InfiniBand clusters, you may additionally need:

```bash
export NCCL_SOCKET_IFNAME=ib0           # Use IB interface for initial TCP handshake
export NCCL_IB_GID_INDEX=3              # RoCE v2 GID (cluster-specific, ask your sysadmin)
export NCCL_NET_GDR_LEVEL=5             # Enable GPU Direct RDMA (if supported)
```

---

## Checkpointing across nodes

Intermediate checkpoints use **sharded DCP** (Distributed Checkpoint). Each rank saves only its own shard to disk. Zero gathering, zero extra memory, works at any model size.

```
output/step-500/dcp/
├── __0_0.distcp    # Rank 0's shard
├── __1_0.distcp    # Rank 1's shard
├── ...
├── __15_0.distcp   # Rank 15's shard (2 nodes × 8 GPUs)
└── .metadata       # DCP index (written by rank 0)
```

On resume, each rank loads only its own shard. No central bottleneck. Scales to any number of nodes.

The **final export** (`output/final/`) gathers the full model to rank 0 in CPU RAM and saves as standard HuggingFace safetensors. This requires 2× model size in CPU RAM on node 0 (e.g., 140 GB for a 35B model in bf16). Plan node 0's memory accordingly.

---

## Fault tolerance

- **Auto-resume**: `train.resume_from: auto` in your config. If a node dies, resubmit the same `sbatch` — palingenesis finds the last valid checkpoint and continues from there.
- **Checkpoint validity**: only directories with a `.metadata` file (meaning the DCP save completed successfully) are considered valid. Half-written checkpoints from node failures are automatically ignored.
- **Auto-purge**: keeps the last 5 checkpoints and deletes older ones. Even on multi-day runs, disk usage stays bounded.

---

## Practical example: 35B MoE on 4 nodes

```bash
sbatch --nodes=4 scripts/train_slurm.sh configs/qwen35_35b_moe/a100_80gb_multigpu.yaml
```

Total GPUs: 32. Memory per GPU:

- Model shard: 70 GB / 32 = ~2.2 GB
- Optimizer shard: ~0.3 GB
- Activations (selective AC): ~8 GB
- Headroom: ~70 GB free per GPU

This is *luxurious*. You could increase batch size significantly, or extend sequence length to 16K+, or enable Context Parallel for 128K sequences.
