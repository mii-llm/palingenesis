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

Two layouts keep the heavy traffic inside the node (see
[Tensor and pipeline parallelism](multi-gpu.md#tensor-and-pipeline-parallelism)): `parallel.tp_degree` up to the
GPUs per node (its per-layer all-gathers / reduce-scatters stay on NVLink, consecutive ranks are one node's), and
`parallel.pp_degree` across nodes (only one micro-batch's activations cross InfiniBand per stage boundary, instead
of FSDP's per-layer weight gathers). For example, 4 nodes x 8 GPUs, a 70B model: `tp_degree: 8`, `pp_degree: 2`,
the remaining 2 data parallel.

---

## SLURM

Most HPC clusters use SLURM for job scheduling. From the repository root:

```bash
sbatch scripts/train_slurm.sh configs/qwen35_35b_moe/a100_80gb_multigpu.yaml
```

The log is `slurm-<jobid>.out` in the directory you submit from. (If you point `#SBATCH --output` into a directory, create it first: Slurm opens the file before the script runs, and a job whose log directory is missing fails at once without a message.)

Override node count or GPUs per node (the script starts one process per allocated GPU):

```bash
sbatch --nodes=4 scripts/train_slurm.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
sbatch --nodes=4 --gpus-per-node=4 scripts/train_slurm.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
```

Arguments after the config are config overrides:

```bash
sbatch scripts/train_slurm.sh my_config.yaml --train.resume_from auto
```

### Before you submit

Every rank runs the same program and reads the same paths, so the paths must mean the same thing on every node:

- **The Python environment** on a shared filesystem, active in the shell you submit from (`sbatch` passes that shell's environment to the job): every node runs `torchrun -m palingenesis.train` from it.
- **`train.output_dir`** on a shared filesystem. Every rank writes its checkpoint shard there and reads it back on resume. The trainer checks at startup that every rank sees the checkpoint rank 0 resumes from, and stops with an error naming the ranks that do not.
- **`data.pretokenize_path`** (with `data.pretokenize: true`) on a shared filesystem. Rank 0 builds the cache, every rank reads it; ranks that cannot read it stop the run with an error.
- **The data** (`data.dataset`, `data.sources`, the eval sets) reachable from every node: a shared path or the Hub.
- **The Hugging Face cache** (`HF_HOME`) shared, with the model and datasets downloaded before the job: `hf download <model>` on a login node, then `export HF_HUB_OFFLINE=1` in the job. Otherwise every rank asks the Hub for the same files at the same second: at 64 ranks expect rate limits, and on filesystems without working file locks (some Lustre and NFS mounts), concurrent downloads into one cache.
- **CPU cores**: each rank starts `data.num_workers` loader processes. Request at least `GPUs per node × (1 + num_workers)` cores per node with `--cpus-per-task`. `srun` does not inherit `--cpus-per-task` from `sbatch` (Slurm 22.05 and later); the script exports `SRUN_CPUS_PER_TASK` so the workers are not confined to one core.
- **Identical nodes**: the same number of GPUs on every node (`--nproc_per_node` is one number), and the same GPU model.

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

The critical detail: `srun` launches torchrun, not Python directly. Torchrun handles per-node process spawning and gives every worker its `RANK`, `LOCAL_RANK` and `WORLD_SIZE`; each worker sees all of the node's GPUs and takes `cuda:<LOCAL_RANK>`. `srun python -m palingenesis.train` with several tasks does **not** work: without torchrun's variables every task trains a full copy on its own (the trainer warns). For the same reason, do not bind one GPU per worker around torchrun (`--gpus-per-task=1` with 8 tasks per node); to give each worker a single visible GPU use `torchrun --virtual-local-rank`.

The `--rdzv_id=$SLURM_JOB_ID` flag and a rendezvous port derived from the job id keep concurrent jobs on the same nodes from interfering with each other's rendezvous.

### The SLURM script explained

```bash title="scripts/train_slurm.sh (the essential lines)"
#SBATCH --nodes=2              # How many machines
#SBATCH --gpus-per-node=8      # All GPUs on each machine
#SBATCH --ntasks-per-node=1    # One torchrun launcher per node
#SBATCH --cpus-per-task=96     # The node's cores, shared by its workers and their data loaders
#SBATCH --exclusive            # Don't share nodes (predictable performance)

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))      # a port per job
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"   # srun does not inherit --cpus-per-task

# --kill-on-bad-exit: stop every node when one node's torchrun fails
# --rdzv_id: unique per job
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc_per_node="$SLURM_GPUS_ON_NODE" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        -m palingenesis.train --config "$CONFIG"
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
    Start all nodes within a few minutes of each other: the nodes that started first wait for the rest only until their rendezvous times out, then exit.

Without a scheduler nothing stops the other nodes when one node's processes fail: stop the script on every node before restarting. `train.output_dir`, `data.pretokenize_path` and the data must be on a filesystem the nodes share, as under SLURM.

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

The trainer also opens a small CPU process group (gloo, over TCP) that the ranks use to end an epoch together. It needs every node to resolve the other nodes' hostnames to a reachable address; if startup stalls right after the model is loaded and before the first step, set `GLOO_SOCKET_IFNAME` to the interface the nodes reach each other on.

### Job hangs mid-training

**Symptom**: training runs for N steps then freezes. All ranks stop logging.

**Cause**: a collective waits for a rank that will never join it. Usually a rank died (out of memory on the host or the GPU, a node failure) or lost its link, and the others keep waiting. A slow rank does not cause this: collectives have no deadline, the other ranks simply wait for it.

**Fix**: look for the first error in the per-rank logs (`--tee=3` prefixes each line with the rank) and in `dmesg` / `sacct` for an out-of-memory kill. The launch scripts already make the survivors fail instead of waiting:
```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Abort on an NCCL error instead of hanging
export NCCL_IB_TIMEOUT=22                 # InfiniBand ACK timeout exponent (4.096 us x 2^22 = 17 s per retry; valid 1-22)
```
and `srun --kill-on-bad-exit=1` ends the job on every node when one node's torchrun exits with an error.

### Checkpoint save fails

**Symptom**: the save "works" but the checkpoint cannot be resumed (`Checkpoint ... is not visible on rank(s) ...` at the next start), or a rank fails with `OSError` during the save.

**Cause**: `output_dir` is a local path, not on a shared filesystem. Rank 0 writes metadata, every rank writes its own DCP shard and random state — all to the same directory. On node-local disks each node ends up with its own partial copy.

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
export CUDA_DEVICE_MAX_CONNECTIONS=1       # Required for compute/comm overlap
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # Abort on an NCCL error (don't hang); NCCL_ASYNC_ERROR_HANDLING is the deprecated name
export NCCL_IB_TIMEOUT=22                  # InfiniBand ACK timeout exponent: 4.096 us x 2^22 = 17 s per retry (valid 1-22)
export NCCL_DEBUG=WARN                     # Print NCCL warnings (INFO is very verbose)
```

`NCCL_IB_TIMEOUT` is an exponent, not seconds: values above 22 are invalid and can make the InfiniBand connection setup fail.

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
output/step-500/
├── dcp/
│   ├── __0_0.distcp             # Rank 0's shard
│   ├── __1_0.distcp             # Rank 1's shard
│   ├── ...
│   ├── __15_0.distcp            # Rank 15's shard (2 nodes × 8 GPUs)
│   └── .metadata                # DCP index (written by rank 0)
├── rng_state.safetensors        # Rank 0's random state
├── rng_state.rank1.safetensors  # ... and every other rank's own
├── tokenizer/
└── training_meta.json           # Written last: marks the checkpoint complete
```

On resume, each rank loads only what it needs. No central bottleneck. Scales to any number of nodes.

A sharded checkpoint resumes on another number of ranks (2 or more): DCP re-shards the model and optimizer state, and the position in the epoch is rescaled. The ranks then read other shards of the data than before, and the effective batch size (`per_device_batch_size × data-parallel ranks × gradient_accumulation_steps`) changes with the rank count unless you adjust `gradient_accumulation_steps`. A sharded checkpoint does not resume in a single process, nor a single-process checkpoint on several ranks: the trainer refuses both.

The **final export** (`output/final/`) gathers the full model to rank 0 in CPU RAM and saves as standard HuggingFace safetensors. This requires 2× model size in CPU RAM on node 0 (e.g., 140 GB for a 35B model in bf16). Plan node 0's memory accordingly.

---

## Fault tolerance

- **Auto-resume**: `train.resume_from: auto` in your config (or `--train.resume_from auto` after the config on the `sbatch` line). If a node dies, resubmit the same `sbatch` — rank 0 finds the last complete checkpoint and every rank continues from it. With `#SBATCH --requeue`, Slurm resubmits a preempted job by itself.
- **Checkpoint validity**: only directories with a `training_meta.json` (written last, after every rank's shard) are considered complete. Half-written checkpoints from node failures are automatically ignored.
- **Preemption and time limits**: `pgs train` does not save on `SIGTERM`. What a preempted run loses is the work since the last `train.save_steps` checkpoint: choose `save_steps` so that this is minutes, not hours. (The RL trainer does save on `SIGUSR1` / `SIGTERM`: see `examples/rl/slurm.sbatch`.)
- **Auto-purge**: keeps the last 5 checkpoints and deletes older ones. Even on multi-day runs, disk usage stays bounded.

---

## What changes at 64+ ranks

- **Epoch length.** Every rank reads its own shard of the data, and the epoch ends for all of them when the first rank runs out (ranks must step together). With map-style data the shards are equal and the loss is a few batches. With a streaming dataset whose files differ a lot in size, a rank can hold much less than the others: the log line `Epoch N ended after M micro-batch(es) per rank (the LR schedule planned ~K)` shows it. Set `train.max_steps`, or re-shard the dataset into equal files.
- **Streaming with fewer shards than ranks.** A stream whose number of shards (files) is not a multiple of the number of data-parallel ranks is split by row: every rank reads the whole stream and keeps one row in `ranks`. Use a multiple of the rank count as the number of files, or `data.pretokenize: true`.
- **Small data.** Rows are split over `ranks × num_workers` loader shards, and each one drops its last partial batch: a dataset with fewer than `ranks × num_workers × per_device_batch_size` rows can produce no batch at all.
- **Evaluation.** Every rank evaluates the same fixed eval set (FSDP needs all ranks for a forward), so its cost does not shrink with more ranks; rank 0's result is the one logged and the one that decides the best checkpoint. Each new best gathers the full model to rank 0: on large models keep `data.eval_every` coarse.
- **Startup.** Every rank loads the model from the shared filesystem and builds its data pipeline at the same time. Expect minutes on a cold cache.
- **Context parallel.** `cp_degree: 0` picks groups of 8 when 8 divides the number of ranks, and otherwise one group over all ranks, which then spans nodes. With GPUs per node other than 8, set `cp_degree` to a divisor of the GPUs per node.

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
