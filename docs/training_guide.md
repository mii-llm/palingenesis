# Training Guide

A practical guide for training agentic language models with `palingenesis`. From raw data to deployed model.

## Quick Start (5 minutes)

```bash
# Install
pip install -e ".[all]"

# Single GPU, default config
pgs train --config configs/qwen3_4b.yaml

# Multi-GPU
torchrun --nproc_per_node=4 -m palingenesis.train --config configs/qwen3_4b.yaml
```

## The Full Pipeline

```
┌──────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  CURATE  │───▶│ PREPARE │───▶│  TRAIN   │───▶│ EVALUATE │───▶│  DEPLOY  │
│  data    │    │ score   │    │  SFT     │    │  bench   │    │  merge   │
└──────────┘    └─────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## Step 1: Data Curation

Your training data should be multi-turn conversations in the standard chat format:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant with access to tools."},
  {"role": "user", "content": "Find files larger than 100MB in /home"},
  {"role": "assistant", "content": "I'll search for large files.\n```bash\nfind /home -size +100M -type f\n```"},
  {"role": "tool", "content": "/home/user/data/model.bin (2.1GB)\n/home/user/videos/clip.mp4 (450MB)"},
  {"role": "assistant", "content": "Found 2 files larger than 100MB:\n- `/home/user/data/model.bin` (2.1 GB)\n- `/home/user/videos/clip.mp4` (450 MB)"}
]}
```

### Data Quality Rules

| Rule | Why |
|------|-----|
| Verified correct outputs | Incorrect data teaches wrong behaviors permanently |
| Multi-turn > single-turn | Models learn error recovery, iteration, tool chaining |
| Include failed attempts | ECHO: training on tool outputs teaches world models |
| Diverse tools and tasks | Prevents overfitting to one tool/pattern |
| 500-5000 high-quality samples | Repetition on fewer good samples > one pass on many bad ones |

### Data Sources for Agentic Training

- **Self-generated**: Run your model, verify outputs, keep successes (best for distributional alignment)
- **Distilled from stronger model**: Generate with GPT-4/Claude, filter by execution verification
- **Human-written**: Highest quality but expensive. Best for hard edge cases.
- **Execution-verified**: Any source where outputs are verified by running the code/tool

---

## Step 2: Data Preparation

The `prepare` pipeline scores your data by model-relative difficulty and selects an optimal subset.

### Basic preparation

```bash
pgs prepare \
    --model Qwen/Qwen3.5-4B \
    --data my_traces.jsonl \
    --output prepared/ \
    --budget 5000 \
    --strategy optimal
```

This runs the target model over all samples and computes perplexity. The `optimal` strategy then selects a J-shaped difficulty distribution:

- 20% easy (maintains capabilities, stable gradients)
- 50% medium (maximum information content)
- 25% hard (pushes frontiers)
- 5% very hard (exposure without overwhelming)

### With HES reasoning quality scoring

For reasoning/code data, add `--hes` to compute the High-Entropy Sum metric. This identifies samples with genuine decision points (high-quality reasoning) vs template-following (low quality):

```bash
pgs prepare \
    --model Qwen/Qwen3.5-4B \
    --data reasoning_traces.jsonl \
    --output prepared/ \
    --budget 3000 \
    --strategy optimal \
    --hes
```

### Multi-source preparation

When training on multiple datasets (agentic + general + code), prepare each independently:

```yaml
# sources.yaml
- name: agentic_traces
  dataset: ./data/agentic.jsonl
  weight: 0.70
  split: train
- name: general_instruct
  dataset: ./data/general.jsonl
  weight: 0.20
  split: train
- name: code_solutions
  dataset: ./data/code.jsonl
  weight: 0.10
  split: train
```

```bash
pgs prepare-multi \
    --model Qwen/Qwen3.5-4B \
    --sources sources.yaml \
    --output prepared/ \
    --budget 3000 \
    --hes
```

This creates per-source scored data + a `manifest.json` with recommended compute allocation.

---

## Step 3: Training Configuration

### Choosing Your Model

| Model | Type | Best For | Key Config |
|-------|------|----------|------------|
| Qwen3.5-4B | Hybrid (DeltaNet+Attention) | Agentic, long context | `freeze_non_attention: true`, `attn: sdpa` |
| Qwen3-4B | Dense Transformer | General SFT | Standard config |
| Gemma 4 E4B | Dense (PLE arch) | Multilingual, reasoning | `llrd_decay: 0.9`, chunked loss essential |
| Gemma 4 E2B | Dense (PLE, small) | Edge deployment | Conservative LR, heavy regularization |
| LLaMA 3.1 8B | Dense Transformer | Baseline, well-studied | Standard recipe |

### Choosing Your Loss Function

```
                        ┌─────────────────────────────┐
                        │     What's your task?        │
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
            ┌──────────┐      ┌──────────────┐    ┌──────────────┐
            │ Reasoning│      │  Knowledge   │    │  Pre-GRPO    │
            │ Code/Math│      │  Medical/QA  │    │  Warm-start  │
            └─────┬────┘      └──────┬───────┘    └──────┬───────┘
                  │                  │                   │
                  ▼                  ▼                   ▼
            ┌──────────┐     ┌──────────────┐    ┌──────────────┐
            │  DEFT    │     │ DEFT + KL    │    │   Pre-RL     │
            │          │     │  anchor      │    │   mode       │
            └──────────┘     └──────────────┘    └──────────────┘
```

**DEFT** (default recommendation): Parameter-free adaptive loss. Automatically NLL on uncertain tokens, sharpening on confident ones. Best overall.

**DEFT + KL anchor**: For knowledge tasks where drift from base model hurts factual accuracy.

**Pre-RL mode**: When your pipeline is SFT → GRPO/DPO. Preserves exploration diversity.

### Choosing Your Optimizer

| Optimizer | Memory/Param | Best For | Config |
|-----------|-------------|----------|--------|
| AdamW | 16 bytes | Default, well-understood | `optimizer: adamw` |
| Muon | 8 bytes | Speed (1.5× convergence), memory | `optimizer: muon` |
| Lion 8-bit | 4 bytes | Absolute minimum memory | `optimizer: lion8bit` |
| AdamW 8-bit | 6 bytes | Memory savings, AdamW behavior | `optimizer: adamw8bit` |

**Decision tree:**
- Fits in memory with AdamW? → Use AdamW (safest)
- Tight on memory? → Muon (same quality, 50% less optimizer memory)
- Still tight? → Lion 8-bit (75% less memory than AdamW)

### The Optimal Config (Copy This)

```yaml
model:
  name_or_path: Qwen/Qwen3.5-4B
  torch_dtype: bfloat16
  attn_implementation: sdpa        # MUST be sdpa for hybrid models
  use_liger_kernel: true
  compile: true

data:
  dataset: your-org/raw-traces     # raw data; preprocess below replaces it at train time
  max_seq_length: 4096
  packing: true                    # 30-60% throughput improvement
  include_observations: true       # ECHO: train on tool outputs
  turn_scaling: progressive        # Later turns get more weight
  pretrain_replay_dataset: ""      # Set to HF dataset for anti-forgetting
  # Validation
  eval_dataset: your-org/traces
  eval_split: test
  eval_samples: 200
  eval_every: 50

# Run `pgs prepare --config <this file>` once; training then uses the parquet
preprocess:
  enabled: true
  output_dir: ./prepared
  budget: 5000
  strategy: optimal

train:
  per_device_batch_size: 2
  gradient_accumulation_steps: 8   # effective batch = 16
  ga_ramp_start: 2                 # Start small, grow late
  learning_rate: 1.5e-5
  weight_decay: 0.1
  warmup_ratio: 0.05
  lr_scheduler: cosine
  optimizer: muon
  gradient_checkpointing: selective
  # Stability
  adagc: true                      # Per-tensor adaptive gradient clipping
  spike_detection: true
  adamc: true                      # Corrected weight decay
  # Anti-forgetting
  ema: true                        # EMA of weights (CPU, zero GPU cost)
  base_merge: true                 # Periodic SLERP toward pretrained
  base_merge_method: slerp
  base_merge_every: 200
  # Hybrid-specific
  freeze_non_attention: true       # Only for Qwen3.5 / hybrid models

memory:
  chunked_loss: true
  loss_num_chunks: 4

plugins:
  deft: true                       # THE best loss function
  sym_noise: true                  # Embedding regularization
  sym_noise_alpha: 5.0
```

---

## Step 4: Running Training

### Single GPU (A100 80GB)

```bash
pgs train --config projects/qwen35_4b_agentic/config.yaml
```

### Multi-GPU (8× A100)

```bash
torchrun --standalone --nproc_per_node=8 \
    -m palingenesis.train --config configs/llama3_8b.yaml
```

### With overrides

```bash
pgs train --config configs/qwen3_4b.yaml \
    --train.learning_rate 2e-5 \
    --train.max_steps 5000 \
    --data.max_seq_length 8192
```

### Monitoring

Watch training in real-time:
```bash
pgs monitor --log_file checkpoints/train.log --brief
```

Check loss curve after training:
```bash
pgs loss --log_file checkpoints/train.log
```

---

## Step 5: What to Watch During Training

### Healthy training looks like:

```
step=100  loss=2.3400  lr=1.5e-05  tok/s=12000  grad_norm=0.45
step=200  loss=1.8200  lr=1.5e-05  tok/s=11800  grad_norm=0.52
step=500  loss=1.2100  lr=1.4e-05  tok/s=12200  grad_norm=0.41
step=1000 loss=0.9800  lr=1.2e-05  tok/s=12100  grad_norm=0.38
```

- **Loss**: Decreasing, not spiking
- **Grad norm**: Stable (0.1-2.0 range), no spikes
- **tok/s**: Consistent (not degrading)
- **LR**: Following cosine schedule

### Red flags:

| Signal | Problem | Fix |
|--------|---------|-----|
| Loss spike (10× jump) | Bad batch or LR too high | AdaGC handles this; if persistent, reduce LR |
| Loss NaN | Numerical overflow | Enable `adagc: true`, reduce LR |
| Loss plateaus early | LR too low or data too easy | Increase LR, check prepare pipeline |
| Grad norm exploding late | AdamC needed | `adamc: true` |
| tok/s dropping | Data loading bottleneck | Increase `num_workers` |
| eval_loss increasing | Overfitting | Enable EMA, reduce epochs, more data |

---

## Step 6: After Training

### Apply EMA weights

If EMA was enabled, the final checkpoint already uses EMA weights (applied automatically before save).

### S0 Tuning (Hybrid Models Only)

For Qwen3.5 and similar hybrids, you can apply S0 Tuning as a final specialization step:

```bash
pgs s0-tune \
    --model ./checkpoints/final \
    --data verified_solutions.jsonl \
    --output s0_states.pt \
    --alpha 0.07 \
    --epochs 50
```

This adds zero inference overhead and can give +10-23pp improvement with just ~50 verified samples.

### Merge with base model

For anti-forgetting, the `base_merge` SLERP during training already handles this. No post-hoc merge needed.

---

## Recipes

### Recipe 1: Agentic Tool-Use (Recommended)

Best for: training models to call tools, execute code, use APIs.

```yaml
# Key choices:
data:
  include_observations: true     # ECHO: learn from tool outputs
  turn_scaling: progressive      # Weight later turns (error recovery)
  packing: true
train:
  optimizer: muon
  freeze_non_attention: true     # If hybrid model
  adagc: true
  ema: true
  base_merge: true
plugins:
  deft: true
  sym_noise: true
```

### Recipe 2: Math/Code Reasoning

Best for: distilling reasoning traces from stronger models.

```yaml
data:
  include_observations: false
  turn_scaling: uniform
train:
  epochs: 20                     # Repetition helps (arxiv:2602.11149)
  weight_decay: 0.3              # Higher WD for multi-epoch (arxiv:2509.14786)
  optimizer: adamw
  adagc: true
  ema: true
plugins:
  deft: true
  sym_noise: true
  sym_noise_alpha: 5.0
```

### Recipe 3: Knowledge Fine-Tuning (Medical, Legal, Domain)

Best for: adapting to a domain without forgetting general capabilities.

```yaml
data:
  pretrain_replay_dataset: HuggingFaceH4/ultrachat_200k
  pretrain_replay_weight: 0.15   # 15% generic data for anti-forgetting
  turn_scaling: uniform
train:
  learning_rate: 1e-5            # Conservative
  optimizer: adamw
  base_merge: true
  base_merge_ratio: 0.15         # Stronger pull-back
  base_merge_every: 100
plugins:
  deft: true
  sym_noise: true
  sym_noise_alpha: 7.0           # Stronger regularization
```

### Recipe 4: Pre-GRPO Warm Start

Best for: SFT stage before reinforcement learning (GRPO/DPO/PPO).

```yaml
data:
  turn_scaling: uniform          # Don't bias RL exploration
train:
  epochs: 1                      # Light SFT (don't over-sharpen)
  learning_rate: 5e-6            # Very conservative
plugins:
  pre_rl: true
  pre_rl_entropy_coeff: 0.1      # Keep diversity
  pre_rl_kl_coeff: 0.5           # Don't drift from base
  sym_noise: true
```

### Recipe 5: Multi-Source with Adaptive Weighting

Best for: mixing multiple data types (agentic + general + code).

```yaml
data:
  sources:
    - dataset: ./prepared/agentic/scored_data.jsonl
      name: agentic
      weight: 0.70
      mode: sft
    - dataset: ./prepared/general/scored_data.jsonl
      name: general
      weight: 0.20
      mode: sft
    - dataset: ./prepared/code/scored_data.jsonl
      name: code
      weight: 0.10
      mode: sft
  msft_tracking: true            # Adaptive per-source weighting
  msft_eval_every: 50
  msft_decay_factor: 0.7
train:
  optimizer: muon
  adagc: true
plugins:
  deft: true
  sym_noise: true
```

---

## Memory Budget Reference

Approximate GPU memory for single-GPU training (bf16, selective AC, batch=1):

| Model | Weights | Optimizer (AdamW) | Optimizer (Muon) | Optimizer (Lion8bit) | Activations | Total |
|-------|---------|-------------------|------------------|---------------------|-------------|-------|
| Qwen3.5-4B | 8 GB | 16 GB | 8 GB | 4 GB | 8-15 GB | 32-39 GB |
| Gemma 4 E4B | 16 GB | 32 GB | 16 GB | 8 GB | 12-20 GB | 60-68 GB |
| LLaMA 3.1 8B | 16 GB | 32 GB | 16 GB | 8 GB | 15-25 GB | 63-73 GB |

With `freeze_non_attention` (Qwen3.5): trainable params drop to ~25%, reducing optimizer memory by 75%.

---

## Troubleshooting

### OOM

1. Reduce `per_device_batch_size` to 1
2. Enable `chunked_loss: true` with `loss_num_chunks: 8`
3. Switch to `optimizer: lion8bit`
4. Enable `gradient_checkpointing: selective` (or `full` for maximum savings)
5. Reduce `max_seq_length`

### Training too slow

1. Enable `compile: true`
2. Enable `packing: true`
3. Increase `num_workers` (4-8)
4. Check `pgs monitor` for data loading bottleneck

### Loss not decreasing

1. Check masking: `pgs inspect --config ...`
2. Verify data quality: `pgs validate --config ...`
3. Increase learning rate (try 2× current)
4. Check if model is too small for your data complexity

### Catastrophic forgetting

1. Enable `base_merge: true` with `base_merge_method: slerp`
2. Add `pretrain_replay_dataset` (10-15% generic data)
3. Enable EMA: `ema: true`
4. Consider LLRD: `llrd_decay: 0.9` (lower LR on early layers)


---

## SFT → RL Transition: Preserving Plasticity

If you plan to run GRPO, DPO, or any RL after SFT, these findings are critical:

### The Problem: Entropy Collapse Kills RL

Research shows (arxiv:2606.18487, 2606.09932) that **excessive SFT destroys the model's
capacity to benefit from subsequent RL**:

- SFT overtraining compresses the output distribution (entropy collapse)
- When output entropy drops too low, GRPO's group-relative advantage variance collapses
- Result: RL gradient signal vanishes, training stalls regardless of RL hyperparameters
- KL penalties and label smoothing do NOT rescue already-collapsed checkpoints

### Rules for SFT → RL Pipelines

1. **Monitor output entropy** during SFT:
   ```yaml
   logging:
     rl_readiness: true
     rl_entropy_floor: 1.0  # warn if entropy drops below this
   ```

2. **Stop SFT at 2-3 epochs**, not until loss plateaus. The "best" SFT checkpoint
   (highest pass@1) is NOT the best RL starting point.

3. **Keep SFT and RL data DISJOINT** (arxiv:2604.13515):
   - Overlapping data causes interference patterns
   - Use separate data splits for SFT and RL

4. **Use EMA/base-merge to preserve diversity**:
   ```yaml
   train:
     ema: true
     ema_decay: 0.999
     base_merge: true
     base_merge_ratio: 0.1
     base_merge_every: 500
   ```

5. **Enable the `pre_rl` plugin** if entropy starts declining:
   ```yaml
   plugins:
     pre_rl: true
     pre_rl_entropy_coeff: 0.1
     pre_rl_kl_coeff: 0.5
   ```

### Entropy Thresholds (Qwen 3B-scale models)

| Output Entropy | Status | Recommendation |
|----------------|--------|----------------|
| > 3.0 | Healthy | Continue SFT normally |
| 2.0 - 3.0 | Caution | Monitor closely, consider stopping soon |
| 1.0 - 2.0 | Warning | Stop SFT. Enable pre_rl or use EMA checkpoint |
| < 1.0 | Collapsed | Do NOT start RL. Use Rejuvenation (merge with base model) |

### Quick Diagnostic

```bash
# Check RL-readiness of a checkpoint
python -m agent_tooling.diagnose --config your_config.yaml --mode post --log_file outputs/train.log
```

The health metrics will show `health/output_entropy` and `health/rl_readiness_warning`
in your training logs when `logging.rl_readiness: true` is set.
