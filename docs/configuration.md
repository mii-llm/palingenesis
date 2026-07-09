# Configuration Reference

## Overview

Configuration is via YAML files with CLI overrides. All fields have sensible defaults; you only need to specify what you want to change.

```bash
# Load from YAML
torchrun ... -m palingenesis.train --config configs/llama3_8b.yaml

# Override specific fields
torchrun ... -m palingenesis.train --config configs/llama3_8b.yaml \
    --train.learning_rate 1e-5 \
    --data.max_seq_length 32768
```

## Config Validation

Before training starts, call `config.validate()` to catch incompatible settings early:

```python
from palingenesis.config import Config, ConfigError

config = Config.from_yaml("configs/my_config.yaml")
try:
    warnings = config.validate()
    for w in warnings:
        print(f"⚠ {w}")
except ConfigError as e:
    print(f"✗ {e}")
    sys.exit(1)
```

### Hard Incompatibilities (raises ConfigError)

| Combination | Why |
|-------------|-----|
| `gradient_release + GA > 1` | FORGE fuses optimizer into backward; cannot accumulate |
| `gradient_release + muon` | Muon needs full gradient for orthogonalization |
| `gradient_release + ga_ramp` | Dynamic accumulation is impossible without GA |
| `packing + context_parallel` | Ring Attention requires single-document sequences |
| `mona + schedule_free` | Both replace optimizer internals |
| `hyperball + schedule_free` | Hyperball projects after step(); SF has no step() |
| Multiple token-weighting losses | Only one of dft/cadft/deft/info_sft at a time |
| `preprocess.enabled + data.sources` | Prepared output replaces the single `data.dataset`; use `prepare-multi` for multi-source |

### Soft Warnings (untested combinations)

| Combination | Concern |
|-------------|---------|
| `gradient_release + hyperball` | Both modify update path; unverified interaction |
| `ema + base_merge` | Both modify weights outside optimizer; mathematically sound but untested at scale |
| `adagc + spike_detection` | Redundant: AdaGC subsumes spike detection |
| `deft + pre_rl (kl=0)` | DEFT has drift risk; pre_rl without KL provides no anchor |

### Feature Maturity Tags

Each feature field in `config.py` has a STATUS comment:
- **`# STATUS: proven`**: Independently reproduced, widely adopted, safe default
- **`# STATUS: validated`**: Paper-backed + tested in this codebase, not externally reproduced
- **`# STATUS: experimental`**: Single paper, limited testing, use with monitoring

```bash
# Find all experimental features:
grep "STATUS: experimental" src/palingenesis/config.py
```

## Section: `model`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name_or_path` | str | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID or local path |
| `trust_remote_code` | bool | `true` | Allow custom code in model repo |
| `torch_dtype` | str | `bfloat16` | Model weight dtype: `bfloat16`, `float16`, `float32` |
| `attn_implementation` | str | `sdpa` | Attention backend: `sdpa`, `flash_attention_2`, `eager` |
| `use_liger_kernel` | bool | `true` | Apply Liger Kernel fused ops (20% throughput, 60% memory) |
| `compile` | bool | `true` | torch.compile each transformer layer |
| `compile_backend` | str | `inductor` | Compile backend (`inductor`, `aot_eager`) |

**Notes**:
- `sdpa` auto-selects the best attention kernel (FlashAttention2 if available, else math)
- `flash_attention_2` requires the `flash-attn` package installed separately
- Liger Kernel is applied BEFORE model loading via monkey-patching
- Compilation adds ~60s startup time but improves throughput by 10-30%

## Section: `data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | str | `HuggingFaceH4/ultrachat_200k` | HF dataset ID or local path |
| `dataset_split` | str | `train_sft` | Dataset split to use |
| `streaming` | bool | `true` | Stream data (no full download needed) |
| `max_seq_length` | int | `8192` | Maximum sequence length (truncation boundary) |
| `messages_field` | str | `messages` | Field name containing chat messages |
| `train_on_reasoning` | bool | `true` | Include `<think>`/`reasoning_content` traces in the loss (needed for reasoning distillation) |
| `num_workers` | int | `4` | DataLoader worker processes |
| `packing` | bool | `false` | Pack multiple sequences into fixed-length blocks |

**Notes**:
- Streaming is recommended for large datasets (>100k examples)
- `max_seq_length` must be divisible by CP degree when Context Parallel is enabled
- Packing should be disabled for long-context training (sequences already fill the block)
- `messages_field` must point to a list of `{"role": str, "content": str}` dicts

## Section: `train`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str | `./checkpoints` | Where to save checkpoints and final model |
| `epochs` | int | `1` | Number of training epochs |
| `max_steps` | int | `-1` | Max optimizer steps (-1 = unlimited, use epochs) |
| `per_device_batch_size` | int | `1` | Micro-batch size per GPU |
| `gradient_accumulation_steps` | int | `16` | Micro-batches before optimizer step |
| `learning_rate` | float | `2e-5` | Peak learning rate |
| `min_learning_rate` | float | `2e-6` | Minimum LR (end of cosine schedule) |
| `weight_decay` | float | `0.1` | AdamW weight decay (applied to 2D+ params) |
| `warmup_ratio` | float | `0.05` | Fraction of steps for linear warmup |
| `max_grad_norm` | float | `1.0` | Gradient clipping norm (0 = no clipping) |
| `lr_scheduler` | str | `cosine` | Schedule type: `cosine`, `linear`, `constant` |
| `seed` | int | `42` | Random seed (offset per rank for diversity) |
| `save_steps` | int | `500` | Checkpoint every N optimizer steps |
| `logging_steps` | int | `1` | Log metrics every N optimizer steps |
| `bf16` | bool | `true` | Enable bf16 mixed precision |
| `gradient_checkpointing` | str | `selective` | AC mode: `selective`, `full`, `none` |

**Effective batch size** = `per_device_batch_size * gradient_accumulation_steps * world_size`

**LR recommendations for SFT**:
- Fine-tuning instruct models: `1e-5` to `5e-5`
- Training from base models: `2e-5` to `1e-4`
- Long-context adaptation: `5e-6` to `2e-5` (lower to avoid forgetting)

## Section: `parallel`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fsdp` | bool | `true` | Enable FSDP2 parameter sharding |
| `context_parallel` | bool | `false` | Enable Context Parallel (sequence sharding) |
| `cp_rotate_method` | str | `allgather` | CP rotation: `allgather` or `alltoall` |
| `cpu_offload` | bool | `false` | Offload params/optim to CPU (extreme memory saving) |
| `reshard_after_forward` | bool | `true` | Free params after forward (memory vs speed tradeoff) |

**Notes**:
- FSDP is automatically disabled for single-GPU runs
- Context Parallel requires `max_seq_length % world_size == 0`
- `allgather` rotation is better for most topologies (used by Llama 3)
- `alltoall` can be better when inter-node bandwidth is limited
- CPU offload reduces GPU memory by ~60% but slows training by 3-5x
- `reshard_after_forward=false` keeps params in memory (faster but uses more memory)

## Section: `memory`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chunked_loss` | bool | `true` | Chunk CE loss along sequence dim |
| `loss_num_chunks` | int | `8` | Number of chunks (higher = less memory, tiny overhead) |
| `float32_matmul_precision` | str | `high` | Matmul precision: `highest`, `high`, `medium` |

**Notes**:
- Chunked loss avoids materializing `[B, S, V]` logits (saves 0.5-4 GB)
- More chunks = less peak memory, marginal compute overhead (~1% per chunk)
- `high` matmul precision allows TF32 on Ampere/Hopper (1.5x faster matmul)
- `medium` allows even more approximation (slightly faster, slightly less precise)

## Section: `preprocess`

Offline data preparation driven by the same config as training (`pgs prepare --config <file>`). Reuses `model.name_or_path`, `data.dataset`, `data.dataset_split`, `data.messages_field`, and `data.max_seq_length` — this section only controls selection and output.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Training uses the prepared output in `output_dir` instead of the raw `data.dataset`. Fails loudly if nothing was prepared |
| `output_dir` | str | `./prepared` | Where `scored_data.parquet` + `prepared_meta.json` are written and looked up |
| `format` | str | `parquet` | `parquet` (order-preserving, fast, auto-fallback to jsonl on schema issues) or `jsonl` |
| `max_samples` | int | `0` | Cap on raw samples read before scoring (0 = all) |
| `budget` | int | `0` | Samples to keep after scoring/filtering (0 = all) |
| `strategy` | str | `optimal` | `optimal`, `curriculum` (order preserved at train time), `balanced`, `medium_focus`, `hard_focus`, `flow`, `random` |
| `batch_size` | int | `4` | Scoring batch size (inference only) |
| `hes` | bool | `false` | Also compute HES reasoning-quality scores (slower) |
| `hes_top_k_pct` | float | `0.5` | Top-k% highest-entropy tokens summed for HES |

**Workflow**:
```bash
pgs prepare --config my_config.yaml          # score + filter + dump parquet
pgs train   --config my_config.yaml \
    --preprocess.enabled true                # trains on the prepared parquet
```

## Section: `logging`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | str | `palingenesis` | Project name for wandb/trackio |
| `run_name` | str | `null` | Run name (auto-generated if null) |
| `use_wandb` | bool | `true` | Log to Weights & Biases |
| `use_trackio` | bool | `true` | Log to HuggingFace Trackio |
| `log_grad_norm` | bool | `true` | Track gradient norms |
| `health_tier2_every` | int | `10` | Tier-2 health cadence (grad cosine sim, GNS, CUDA memory). Multiple of `logging_steps` |
| `health_tier3_every` | int | `100` | Tier-3 health cadence (weight norms, stable rank, drift). Lower for short test runs |
| `rl_readiness` | bool | `false` | Log output entropy every logging step (SFT→RL readiness) |
| `rl_entropy_floor` | float | `1.0` | Warn when mean output entropy drops below this |

**Tracker behavior (automatic)**:
- The wandb run id is persisted to `{output_dir}/tracker_run_id.json` — resuming from a checkpoint (`train.resume_from`) continues the SAME wandb run, while a fresh start in the same `output_dir` mints a new run id (so wandb never drops metrics for being below the old run's history step). trackio resumes by project + run name.
- Tracker init/log failures never kill training: they degrade to warnings.
- All `train/*`, `eval/*`, `health/*` metrics share `train/global_step` as x-axis, aligned across restarts.

## Recommended Configs by Scenario

### Single A100 80GB, 8B model SFT

```yaml
train:
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: selective
parallel:
  fsdp: false
memory:
  chunked_loss: true
  loss_num_chunks: 8
data:
  max_seq_length: 8192
```

### 8x A100, 8B model, standard SFT

```yaml
train:
  per_device_batch_size: 2
  gradient_accumulation_steps: 4
  gradient_checkpointing: selective
parallel:
  fsdp: true
  context_parallel: false
memory:
  chunked_loss: true
data:
  max_seq_length: 8192
```

### 8x A100, 8B model, 64k long-context

```yaml
train:
  per_device_batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: selective
parallel:
  fsdp: true
  context_parallel: true
memory:
  chunked_loss: true
  loss_num_chunks: 16
data:
  max_seq_length: 65536
```

### 32x A100 (4 nodes), 8B model, 128k context

```yaml
train:
  per_device_batch_size: 1
  gradient_accumulation_steps: 2
  gradient_checkpointing: selective
parallel:
  fsdp: true
  context_parallel: true
memory:
  chunked_loss: true
  loss_num_chunks: 32
data:
  max_seq_length: 131072
```
