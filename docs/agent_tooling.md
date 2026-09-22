# Agent Tooling

## Purpose

The `agent_tooling/` directory contains standalone diagnostic scripts that an AI agent (or human developer) can run to assess training health without modifying the training loop. They're designed for:

1. **Pre-flight validation** — confirm everything will work before burning GPU hours
2. **Live monitoring** — check if a running training is healthy
3. **Post-mortem analysis** — diagnose what went wrong after a failed run
4. **Iterative improvement** — provide data for hyperparameter tuning decisions

## Tool Reference

### `diagnose` — All-in-One Health Check

The master diagnostic tool. Runs appropriate checks based on mode.

```bash
# Pre-training (no GPU needed): memory + masking + config sanity
python -m agent_tooling diagnose --config configs/llama3_8b.yaml --mode pre

# Post-training (analyzes log): loss curve pathologies
python -m agent_tooling diagnose --config configs/llama3_8b.yaml --mode post --log_file train.log

# Full (needs GPU): pre + gradient check
python -m agent_tooling diagnose --config configs/llama3_8b.yaml --mode full

# JSON output for programmatic parsing
python -m agent_tooling diagnose --config configs/llama3_8b.yaml --mode pre --json
```

**Exit codes**: 0 = healthy, 1 = issues found

**JSON output structure**:
```json
{
  "timestamp": "2025-06-23 14:30:00",
  "mode": "pre",
  "checks": {
    "memory": {"status": "pass", "estimated_gb": 32.5, "headroom_gb": 47.5},
    "masking": {"status": "pass", "train_ratio": 0.234, "issues": ["OK"]},
    "config": {"status": "pass", "issues": []}
  },
  "overall": "HEALTHY"
}
```

### `inspect_batch` — Visual Token Inspection

Shows actual tokenization with colored output — green tokens have loss, gray tokens are masked.

```bash
python -m agent_tooling inspect_batch --config configs/llama3_8b.yaml --num_samples 3
```

**Output**:
```
Sample 1 — 2847 tokens, 891 trained (31.3%)
────────────────────────────────────────────────
  [   0] <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
  [  50] You are a helpful assistant...<|eot_id|><|start_header_id|>user...
  [ 150] Hello! Can you help me...<|eot_id|><|start_header_id|>assistant...
  [ 200] Hi there! How can I help you today?<|eot_id|>  ← GREEN (trained)
```

**Use case**: Verify the chat template is applying correctly for your model.

### `validate_masking` — Statistical Masking Validation

Processes many samples and checks for masking anomalies.

```bash
python -m agent_tooling validate_masking --config configs/llama3_8b.yaml --num_samples 200
```

**Checks**:
- Train ratio is in expected range (5-80%)
- No samples have 0% trained tokens (broken masking)
- No samples have 100% trained tokens (masking disabled)
- Padding tokens never have loss
- Consistent ratios across samples

### `check_loss` — Loss Curve Analysis

Detects pathologies in training loss without needing the model.

```bash
# From a log file
python -m agent_tooling check_loss --log_file outputs/train.log

# From stdin (pipe from training)
grep "loss=" train.log | python -m agent_tooling check_loss --stdin

# From wandb
python -m agent_tooling check_loss --wandb_run user/project/run_id
```

**Detects**:
- NaN/Inf losses (catastrophic failure)
- Spikes (>3x rolling average)
- Plateaus (no improvement for extended periods)
- Divergence (loss increasing in tail)
- Abnormal initial loss (wrong model-data pairing)

### `check_gradients` — Per-Layer Gradient Analysis

Runs one forward-backward pass and reports gradient statistics.

```bash
python -m agent_tooling check_gradients --config configs/single_gpu.yaml
```

**Requires**: 1 GPU with enough memory for the model + 1 batch.

**Reports**:
- Total gradient L2 norm
- Per-layer norms (top 10 by magnitude)
- Dead layers (zero gradient)
- Vanishing layers (norm < 1e-7)
- Exploding layers (norm > 100)
- NaN/Inf detection
- Layer norm ratio (max/min across depth)

### `profile_memory` — Memory: exact where possible, measured on demand

```bash
pgs profile --config configs/qwen35_4b/a100_80gb.yaml --gpu 80      # static estimate, no GPU
pgs profile --config configs/qwen35_4b/a100_80gb.yaml --measure     # + two real optimizer steps
```

The static estimate has two parts:

- **Exact:** parameters (the architecture built on the meta device, so GQA, linear attention, MoE, tying and freezing all count), optimizer states per the optimizer's real state layout, gradients, and the DPO reference model.
- **Estimated:** logits (the trainer's per-batch chunk), the SeCO K/V cache, and activations. Activations are a rough approximation, because they depend on the architecture and its kernels.

`--measure` runs the **real trainer** (`palingenesis.train.train`) for two optimizer steps on worst-case rows (every row rendered to `max_seq_length`). It uses the configured batch size, gradient accumulation and evaluation (run once), and reports the true peak. A standalone trainer run with the same settings gave the same number (43.32 GiB), so the measurement is exact by construction. Single process only; FSDP sharding is not modelled.

If the model config cannot be loaded, the tool fails rather than guessing.

Example (Qwen3.5-0.8B, 2 × 8 × 4096 tokens, bf16, selective checkpointing, three eval sets, one A100):

```
  Model: Qwen/Qwen3.5-0.8B (0.75B params, bfloat16 weights, 100% trainable)
  Exact (single GPU, before any FSDP sharding):
    Model parameters:        1.5 GB
    Optimizer states:        3.0 GB  [adamw]
    Gradients:               1.5 GB
  Estimated:
    Logits (one chunk):      3.8 GB
    Activations (rough):     6.4 GB  [checkpointing: selective]
    Total (+10% overhead):  17.9 GB of 80 GB (+62.1 GB)
  Measured peak: 18.0 GiB of 79 GiB (real trainer: 2 optimizer steps x 2 micro-batches of 8 x 4096 tokens,
  plus one evaluation)
```

### `monitor_run` — Live Training Monitor

Parses training log output and reports status.

```bash
# Quick status (for agent polling)
python -m agent_tooling monitor_run --log_file outputs/train.log --brief
# Output: [HEALTHY] step=1234 loss=1.2345 trend=-0.5% tok/s=5432 eta=2.3h

# Full report
python -m agent_tooling monitor_run --log_file outputs/train.log --max_steps 5000

# Last N steps only
python -m agent_tooling monitor_run --log_file outputs/train.log --last 50
```

## Agent Decision Framework

```
START
  │
  ├── Before training
  │   └── python -m agent_tooling diagnose --mode pre --json
  │       ├── memory.status == "fail" → adjust config (reduce seq, add FSDP)
  │       ├── masking.status == "fail" → check chat template, inspect_batch
  │       └── all pass → proceed to training
  │
  ├── During training (poll every 5 min)
  │   └── python -m agent_tooling monitor_run --brief
  │       ├── status == "CRASHED" → check_loss for details, fix and restart
  │       ├── status == "WARNING" → read detailed issues, may need intervention
  │       └── status == "HEALTHY" → continue
  │
  ├── Training not converging
  │   ├── python -m agent_tooling check_loss --log_file ...
  │   │   ├── Plateaued → increase LR or check data
  │   │   ├── Diverging → decrease LR
  │   │   └── Spikes → check data quality, reduce LR
  │   ├── python -m agent_tooling validate_masking --num_samples 200
  │   │   └── Low train_ratio → data issue, consider packing
  │   └── python -m agent_tooling check_gradients
  │       ├── Dead layers → LR too low or frozen params
  │       └── Exploding → LR too high or grad clip too loose
  │
  └── After training
      └── python -m agent_tooling diagnose --mode post --log_file ...
          └── Report on convergence quality
```

## Implementation Details

### Path Resolution

All tools use `agent_tooling/_path_setup.py` to ensure `palingenesis` is importable regardless of how the tool is invoked:

```python
import agent_tooling._path_setup  # Adds src/ to sys.path
from palingenesis.config import Config
```

This works whether the package is installed (`pip install -e .`) or run from the repo root.

### Standalone Tools

`check_loss` and `monitor_run` have zero dependencies on `palingenesis` — they only parse text. They can be used with any training framework that outputs `step=N loss=X` format.

### Exit Codes

All tools follow Unix convention:
- `0`: healthy / pass / no issues
- `1`: issues found / fail / needs attention

This enables scripting:
```bash
python -m agent_tooling diagnose --mode pre --config ... || echo "Pre-flight failed"
```
