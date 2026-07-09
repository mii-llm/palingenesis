# Agent Instructions — palingenesis

This document instructs AI agents on how to debug and diagnose training runs.

## Available Diagnostic Tools

All tools are in `agent_tooling/` and can be run independently.

### 1. Pre-Training Validation

Before starting training, run the all-in-one diagnostic:

```bash
python -m agent_tooling.diagnose --config configs/llama3_8b.yaml --mode pre --json
```

This checks:
- Memory estimate (will it fit on the GPU?)
- Masking validation (are labels correct?)
- Config sanity (any obvious misconfigurations?)

### 2. Monitor Active Training

While training is running, check on it:

```bash
# Quick one-line status
python -m agent_tooling.monitor_run --log_file outputs/train.log --brief

# Detailed report
python -m agent_tooling.monitor_run --log_file outputs/train.log --max_steps 5000
```

### 3. Post-Training Analysis

After training completes or if something went wrong:

```bash
# Analyze loss curve
python -m agent_tooling.check_loss --log_file outputs/train.log

# Full post-mortem
python -m agent_tooling.diagnose --config configs/llama3_8b.yaml --mode post --log_file outputs/train.log
```

### 4. Deep Debugging (requires GPU)

```bash
# Check gradient flow (1 forward-backward pass)
python -m agent_tooling.check_gradients --config configs/llama3_8b.yaml

# Inspect actual tokenization and masking
python -m agent_tooling.inspect_batch --config configs/llama3_8b.yaml --num_samples 5

# Validate masking across many samples
python -m agent_tooling.validate_masking --config configs/llama3_8b.yaml --num_samples 200

# Memory profiling
python -m agent_tooling.profile_memory --config configs/long_context.yaml --gpu_memory_gb 80
```

## Decision Tree for Agents

```
Training issue?
├── Loss is NaN/Inf
│   ├── Check: python -m agent_tooling.check_gradients --config ...
│   ├── Likely causes: LR too high, data corruption, numerical overflow
│   └── Fix: Reduce LR, check data, enable gradient clipping
├── Loss not decreasing
│   ├── Check: python -m agent_tooling.check_loss --log_file ...
│   ├── Likely causes: LR too low, masking wrong, data issue
│   └── Debug: python -m agent_tooling.validate_masking --config ...
├── OOM error
│   ├── Check: python -m agent_tooling.profile_memory --config ... --gpu_memory_gb 80
│   ├── Likely causes: Seq too long, batch too big, no checkpointing
│   └── Fix: Enable chunked_loss, selective AC, reduce seq/batch, add FSDP/CP
├── Training too slow
│   ├── Check: python -m agent_tooling.monitor_run --log_file ... --brief
│   ├── Likely causes: Data loading bottleneck, no compile, small batch
│   └── Fix: Increase num_workers, enable compile, use packing
└── Want to verify everything is OK before starting
    └── Run: python -m agent_tooling.diagnose --config ... --mode pre --json
```

## Exit Codes

All tools return:
- `0` = healthy / pass
- `1` = issues found / fail

Use `--json` flag (where available) for structured output suitable for programmatic parsing.

## Common Fixes

| Problem | Config Change |
|---------|--------------|
| OOM | `memory.chunked_loss: true`, `train.gradient_checkpointing: selective` |
| Loss NaN | `train.learning_rate: 1e-5`, `train.max_grad_norm: 1.0` |
| No improvement | Increase LR, check masking with `validate_masking` |
| Slow throughput | `model.compile: true`, `data.packing: true`, increase `data.num_workers` |
| Long seq OOM | `parallel.context_parallel: true`, `memory.loss_num_chunks: 16` |
