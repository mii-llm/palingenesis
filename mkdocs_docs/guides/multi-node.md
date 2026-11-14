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

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 300" role="img" aria-labelledby="fig-slurm-title fig-slurm-desc">
<title id="fig-slurm-title">What happens when you submit a multi-node job</title>
<desc id="fig-slurm-desc">sbatch hands the job to SLURM, which queues it and allocates N nodes with exclusive access. On every node srun launches one task, which runs torchrun; torchrun spawns 8 worker processes, one per GPU. A c10d rendezvous connects all N times 8 workers over TCP to MASTER_ADDR. FSDP2 then shards the model across all workers and training begins.</desc>
<defs><marker id="slurm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<rect class="box" x="10" y="14" width="120" height="40" rx="4"/><text class="t-mono" x="70" y="38" text-anchor="middle">sbatch</text>
<path class="line" d="M130,34 H166" marker-end="url(#slurm-arrow)"/>
<rect class="box" x="168" y="14" width="300" height="40" rx="4"/><text x="180" y="31">SLURM queues the job</text><text class="t-muted t-small" x="180" y="46">then allocates N nodes with exclusive access</text>
<rect class="box-plain" x="10" y="78" width="330" height="128" rx="6"/>
<text class="t-label" x="22" y="96">node 0</text>
<rect class="box" x="22" y="106" width="140" height="34" rx="4"/><text x="32" y="121">srun: 1 task</text><text class="t-muted t-small" x="32" y="134">per node</text>
<path class="line" d="M162,123 H184" marker-end="url(#slurm-arrow)"/>
<rect class="box" x="186" y="106" width="142" height="34" rx="4"/><text class="t-mono" x="196" y="121">torchrun</text><text class="t-muted t-small" x="196" y="134">spawns 8 workers</text>
<rect class="box-accent" x="22" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="38.5" y="180" text-anchor="middle">0</text>
<rect class="box-accent" x="61" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="77.5" y="180" text-anchor="middle">1</text>
<rect class="box-accent" x="100" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="116.5" y="180" text-anchor="middle">2</text>
<rect class="box-accent" x="139" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="155.5" y="180" text-anchor="middle">3</text>
<rect class="box-accent" x="178" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="194.5" y="180" text-anchor="middle">4</text>
<rect class="box-accent" x="217" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="233.5" y="180" text-anchor="middle">5</text>
<rect class="box-accent" x="256" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="272.5" y="180" text-anchor="middle">6</text>
<rect class="box-accent" x="295" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="311.5" y="180" text-anchor="middle">7</text>
<path class="line" d="M257,140 V148 H38 V158" />
<path class="line" d="M257,148 H310 V158" />
<path class="line" d="M250,54 V64 H180 V76" marker-end="url(#slurm-arrow)"/>
<rect class="box-plain" x="360" y="78" width="330" height="128" rx="6"/>
<text class="t-label" x="372" y="96">node 1</text>
<rect class="box" x="372" y="106" width="140" height="34" rx="4"/><text x="382" y="121">srun: 1 task</text><text class="t-muted t-small" x="382" y="134">per node</text>
<path class="line" d="M512,123 H534" marker-end="url(#slurm-arrow)"/>
<rect class="box" x="536" y="106" width="142" height="34" rx="4"/><text class="t-mono" x="546" y="121">torchrun</text><text class="t-muted t-small" x="546" y="134">spawns 8 workers</text>
<rect class="box-accent" x="372" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="388.5" y="180" text-anchor="middle">8</text>
<rect class="box-accent" x="411" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="427.5" y="180" text-anchor="middle">9</text>
<rect class="box-accent" x="450" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="466.5" y="180" text-anchor="middle">10</text>
<rect class="box-accent" x="489" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="505.5" y="180" text-anchor="middle">11</text>
<rect class="box-accent" x="528" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="544.5" y="180" text-anchor="middle">12</text>
<rect class="box-accent" x="567" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="583.5" y="180" text-anchor="middle">13</text>
<rect class="box-accent" x="606" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="622.5" y="180" text-anchor="middle">14</text>
<rect class="box-accent" x="645" y="160" width="33" height="32" rx="3"/><text class="t-mono t-small" x="661.5" y="180" text-anchor="middle">15</text>
<path class="line" d="M607,140 V148 H388 V158" />
<path class="line" d="M607,148 H660 V158" />
<path class="line" d="M400,54 V64 H530 V76" marker-end="url(#slurm-arrow)"/>
<text class="t-muted t-small" x="345" y="148" text-anchor="middle">…</text>
<rect class="box" x="10" y="222" width="680" height="30" rx="4"/><text x="22" y="241">c10d rendezvous: all N × 8 workers connect over TCP to MASTER_ADDR</text>
<path class="line" d="M350,252 V266" marker-end="url(#slurm-arrow)"/>
<rect class="box-accent" x="10" y="268" width="680" height="28" rx="4"/><text x="22" y="287">FSDP2 shards the model across all workers, and training begins</text>
</svg>
</div>
<figcaption>Launch path of a multi-node job, two of the N nodes shown: one <code>srun</code> task per node runs <code>torchrun</code>, which starts one worker per GPU; every worker then joins one rendezvous.</figcaption>
</figure>

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
