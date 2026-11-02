# Agent Instructions — palingenesis

This document instructs AI agents on how to debug and diagnose training runs.

## Available Diagnostic Tools

All tools are in `src/palingenesis/agent_tooling/` and can be run independently (`python -m palingenesis.agent_tooling.<tool>`, or the `pgs` subcommands: `pgs diagnose`, `pgs monitor`, `pgs loss`, ...). From a checkout, `python -m palingenesis.agent_tooling.<tool>` still works through a thin shim.

### 1. Pre-Training Validation

Before starting training, run the all-in-one diagnostic:

```bash
python -m palingenesis.agent_tooling.diagnose --config configs/llama3_8b.yaml --mode pre --json
```

This checks:
- Memory estimate (will it fit on the GPU?)
- Masking validation (are labels correct?)
- Config sanity (any obvious misconfigurations?)

### 2. Monitor Active Training

While training is running, check on it:

```bash
# Quick one-line status
python -m palingenesis.agent_tooling.monitor_run --log_file outputs/train.log --brief

# Detailed report
python -m palingenesis.agent_tooling.monitor_run --log_file outputs/train.log --max_steps 5000
```

### 3. Post-Training Analysis

After training completes or if something went wrong:

```bash
# Analyze loss curve
python -m palingenesis.agent_tooling.check_loss --log_file outputs/train.log

# Full post-mortem
python -m palingenesis.agent_tooling.diagnose --config configs/llama3_8b.yaml --mode post --log_file outputs/train.log
```

### 4. Deep Debugging (requires GPU)

```bash
# Check gradient flow (1 forward-backward pass)
python -m palingenesis.agent_tooling.check_gradients --config configs/llama3_8b.yaml

# Inspect actual tokenization and masking
python -m palingenesis.agent_tooling.inspect_batch --config configs/llama3_8b.yaml --num_samples 5

# Validate masking across many samples
python -m palingenesis.agent_tooling.validate_masking --config configs/llama3_8b.yaml --num_samples 200

# Memory profiling
python -m palingenesis.agent_tooling.profile_memory --config configs/long_context.yaml --gpu_memory_gb 80
```

## Decision Tree for Agents

```
Training issue?
├── Loss is NaN/Inf
│   ├── Check: python -m palingenesis.agent_tooling.check_gradients --config ...
│   ├── Likely causes: LR too high, data corruption, numerical overflow
│   └── Fix: Reduce LR, check data, enable gradient clipping
├── Loss not decreasing
│   ├── Check: python -m palingenesis.agent_tooling.check_loss --log_file ...
│   ├── Likely causes: LR too low, masking wrong, data issue
│   └── Debug: python -m palingenesis.agent_tooling.validate_masking --config ...
├── OOM error
│   ├── Check: python -m palingenesis.agent_tooling.profile_memory --config ... --gpu_memory_gb 80
│   ├── Likely causes: Seq too long, batch too big, no checkpointing
│   └── Fix: Enable chunked_loss, selective AC, reduce seq/batch, add FSDP/CP
├── Trained model loops, rambles to the token limit, or never stops calling tools
│   ├── Measure: the loop share and how episodes end (stop / length limit / turn limit) at deployment sampling,
│   │   in both thinking modes; during SFT watch train/repeat_p and train/stop_p
│   ├── Likely cause: exposure bias: teacher-forced data never shows the student its own failure states
│   ├── SFT fix (removes enumerated loop forms): data.loop_recovery, data.reasoning_loop_recovery,
│   │   data.repeat_unlikelihood, data.stop_unlikelihood (need memory.seco; no packing / pretokenize)
│   ├── Root-cause fix: OPD on the student's own episodes (agent_traces source) with onset_branches,
│   │   error_patterns / error_weight / first_error_weight, cutoff_logp (see the distillation guide)
│   └── Stopgap at inference: presence_penalty (sampling)
├── Training too slow
│   ├── Check: python -m palingenesis.agent_tooling.monitor_run --log_file ... --brief
│   ├── Likely causes: Data loading bottleneck, no compile, small batch
│   └── Fix: Increase num_workers, enable compile, use packing
└── Want to verify everything is OK before starting
    └── Run: python -m palingenesis.agent_tooling.diagnose --config ... --mode pre --json
```

## In-process use (scripts, notebooks)

`palingenesis.train.train(Config)` runs SFT / DPO in-process on one GPU and returns a `TrainResult` (output_dir,
global_step, metrics, best_eval_loss); `RLTrainer(...).train()` and `OPDTrainer(...).train()` also run in-process.
Multi-GPU needs `torchrun` (`pgs train`). `pgs init <name>` writes an example config. See the "Training from a
notebook" docs page.

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
| Repetition loops after SFT | `data.loop_recovery: 0.2`, `data.reasoning_loop_recovery: 0.2`, `data.repeat_unlikelihood: 20`, `data.stop_unlikelihood: 5` (with `memory.seco: true`) |
| Agent drifts on its own errors | OPD `sources.<name>.format: agent_traces` on the student's episodes + `onset_branches`, `error_patterns`, `cutoff_logp` |
