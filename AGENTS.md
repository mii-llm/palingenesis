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

# RL / on-policy distillation: what the rollouts do (needs logging.dump_trajectories: N in the config)
python -m palingenesis.agent_tooling.rollouts --dir outputs/rl --json
```

`monitor_run` and `check_loss` read the logs of `pgs train`, `pgs rl` and `pgs distill` alike.

### 3. Post-Training Analysis

After training completes or if something went wrong:

```bash
# Analyze loss curve
python -m palingenesis.agent_tooling.check_loss --log_file outputs/train.log

# Full post-mortem
python -m palingenesis.agent_tooling.diagnose --config configs/llama3_8b.yaml --mode post --log_file outputs/train.log

# Evaluate a checkpoint on tasks (YAML; built-in: gsm8k, math500, mmlu_pro, ifeval, bfcl_simple) through the
# training stack: k samples, each thinking mode a separate result, files in the pgs compare format
pgs eval --model runs/sft/final --tasks gsm8k,ifeval --k 4 --modes plain,think --out eval/sft

# Did the candidate beat the baseline? Per-item results (JSONL: id + scores), paired tests, pass@k, exit 1 if not met
pgs compare eval/base/gsm8k.think.jsonl eval/sft/gsm8k.think.jsonl --require not_worse --margin 0.02
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
│   ├── Fix: Enable chunked_loss, selective AC, activation offload (memory.activation_offload), reduce
│   │   seq/batch, add FSDP/CP
│   ├── Fastest settings that fit: pgs fit --config ... --gpu 80 (measures checkpointing modes and SeCO, real steps)
│   └── Layers too large for FSDP alone (32B+, long sequences on many GPUs): parallel.tp_degree (inside a node),
│       parallel.pp_degree (across nodes); per-rank memory: pgs profile --config ... --world_size N
├── Trained model loops, rambles to the token limit, or never stops calling tools
│   ├── Measure: the loop share and how episodes end (stop / length limit / turn limit) at deployment sampling,
│   │   in both thinking modes; during SFT watch train/repeat_p and train/stop_p
│   ├── Likely cause: exposure bias: teacher-forced data never shows the student its own failure states
│   ├── SFT fix (removes enumerated loop forms): data.loop_recovery, data.reasoning_loop_recovery,
│   │   data.repeat_unlikelihood, data.stop_unlikelihood (need memory.seco; no pretokenize)
│   ├── Root-cause fix: OPD on the student's own episodes (agent_traces source) with onset_branches,
│   │   error_patterns / error_weight / first_error_weight, cutoff_logp (see the distillation guide)
│   └── Stopgap at inference: presence_penalty (sampling)
├── RL / on-policy distillation: reward flat, steps slow, or the policy misbehaves (truncation, tool errors, loops)
│   ├── Dump: logging.dump_trajectories: 10 (+ logging.dump_groups: 8 to sample groups; off the training thread)
│   ├── Check: pgs rollouts --dir <output_dir> --json (finish shares stop / length / turns, reward components,
│   │   zero-variance share, tool error rate and latency per tool, loop share, slowest turns, worst trajectories)
│   ├── Alarms (exit 1): length_share > 0.2 (budget or rambling), turns_share > 0.2 (never finishes),
│   │   tool_error_rate > 0.3 (schemas / arguments / timeouts), loop_share > 0.1, zero_variance_share > 0.5
│   ├── zero_variance_share high before the run: pgs screen --config <rl.yaml> --k 8 --output screened.jsonl
│   │   (keeps 0 < pass rate < 1 rows in screened.kept.jsonl)
│   ├── pass@1 up, pass@G down (sharpening): loss.pass_k (2 <= k < group_size), then a pass_k 1 stage
│   └── Rollout engine latency: vLLM OTLP traces via rollout.vllm_args (see the Observability guide)
├── Multi-role RL (pgs rl with a scenario: section; CPU-tested only, GPU validation pending)
│   ├── Data: an ordinary RL dataset plus a pattern (scenario.type solve_check | handoff | lead_workers, scored by
│   │   the rewards: section; plan_execute on Harbor rows); pgs scenario patterns lists row fields and options
│   ├── Before training: pgs scenario check --config <rl.yaml> (a pattern: fields, rewards, prompts per role, the
│   │   audit on your rows) or pgs scenario audit <name> (a generated scenario); oracle partners must earn,
│   │   degenerate ones must not; exit 1
│   ├── check fails on "oracle solver": the gold column does not pass the reward as a final answer (a worked
│   │   solution in the column, another answer format): scenario.args.gold_field / answer_tag, rewards.<name>.args
│   ├── Delegation adds nothing: role/<r>/rewards/metric/control close to the role's reward (lift near 0)
│   ├── Watch per role: role/<r>/reward, role/<r>/advantage_std (0 = that role's groups carry no signal),
│   │   role/<r>/truncated, episode/failed; pgs rollouts reports finish reasons, loops and rewards per role
│   ├── Sessions end after one turn: rollout.max_completion_tokens is a session's whole budget (default: one turn)
│   ├── One role dominates: loss.advantage_std: role + advantage_std_floor, roles.<name>.share
│   ├── Engines on other GPUs (rollout.backend remote): pool/lag growing or stale_groups_dropped rising = updates
│   │   travel longer than a step: push adapters (rollout.push fold / native), raise max_staleness, or add link speed
│   └── Always run a single-agent control at equal sampled tokens (see the multi-agent guide)
├── Distilling a larger teacher on fixed data (off-policy KD in pgs train), or QAT while distilling
│   ├── Setup: distill.enabled + distill.teachers.<name>: {model: ...} (online) or {cache: ...} (pgs distill-cache)
│   ├── Check before: pgs profile --config ... (counts teacher weights and targets); the teacher must share
│   │   the student's vocabulary (startup error otherwise: use OPD xtok across tokenizers)
│   ├── Watch: train/kd_loss falling, train/kd_coverage == 1.0, train/topk_mass/<teacher> (cached: > ~0.9)
│   ├── "no teacher target" error: the cache was built from other data / masking -> rebuild it from the
│   │   training config, or distill.missing: skip (uncovered tokens get only ce_weight * CE)
│   ├── OOM from the teacher: teachers.<name>.placement: shard (multi-GPU), dtype: bfloat16, or cache it
│   ├── QAT: plugins.custom: palingenesis.qat:TernaryQAT (watch plugins/ternary_qat/codes_changed, strength)
│   └── FP4 / FP8 QAT: palingenesis.qat:FloatQAT (format nvfp4|mxfp4|fp8|fp8_block), teacher = the bf16 model;
│       PTQ gap first: pgs quantize; watch plugins/float_qat/qerr_mean, saturated_blocks; vLLM: final-<format>/
├── MoE model (Qwen3-MoE, Qwen3.5-MoE, Mixtral, GPT-OSS, DeepSeek-V3 family): routers collapse, RL unstable
│   ├── Watch: moe/load_max (1 = balanced), moe/dead_experts, moe/router_entropy in the log line (moe[...])
│   ├── Balancing is on by default (moe.balance_loss_coef: the model config's router_aux_loss_coef, load measured
│   │   over the global batch; DeepSeek-V3 family: selection-bias update). Collapse anyway: moe.balance_loss_coef
│   │   0.01, moe.z_loss_coef 0.001, lower LR
│   ├── RL: rollout.routing_replay: auto (opt-in; vLLM needs rollout.prefix_caching) trains each token
│   │   through the experts the engine used; watch moe/route_mismatch and policy/mismatch_k3
│   └── Speed: python scripts/bench_moe_experts.py picks moe.experts_implementation; several GPUs:
│       parallel.expert_parallel (experimental)
├── Evaluate a checkpoint broadly / fast (pgs eval)
│   ├── One command: pgs eval --model <ckpt> --suite core --modes plain,think (table, summary.json, report.md)
│   ├── lm-eval task files run as they are: --tasks path/to/task.yaml (completion mode; --template chat);
│   │   agreement with lm-eval: python -m palingenesis.eval.conformance --pgs ... --lmeval ... --task ... --metric ...
│   ├── Slow: check gpu_util_mean in the summary (well under 90% on a small model: --replicas 2..4), the
│   │   engine's memory share (--engine.gpu_memory_utilization), time_to_first_result, cache_hits
│   ├── Multiple choice by log-likelihood (output: choices; mmlu, mmlu_pro_mc): one prefill per question; needs
│   │   the vllm or hf backend
│   ├── Leakage: --contamination <training data paths> (13-gram overlap; clean_score vs flagged_score)
│   └── Against lm-eval and lighteval on one GPU: scripts/gpu_checks/eval_bench.sh
├── Did my model improve? (a checkpoint against its base, a step against step 0)
│   ├── Run: pgs eval --model <base> --tasks <tasks> --k 4 --modes plain,think --out eval/base, the same command
│   │   for the candidate (same tasks, k, modes, sampling, environment), then per task and mode:
│   │   pgs compare eval/base/<task>.<mode>.jsonl eval/cand/<task>.<mode>.jsonl --require not_worse (exit 1 if not)
│   ├── Decide on the paired difference's interval (diff [lo, hi], p_perm), not on two means; an interval that
│   │   contains 0 is not an improvement. Use k >= 4 at temperature > 0, and --limit for large sets
│   ├── Before reading the score: truncated_share (max_tokens too small for the thinking mode?), loop_share,
│   │   unscored_samples (sandbox / judge outage, a missing optional checker) in <task>.<mode>.summary.json
│   ├── Sampling: unset = the model's generation_config.json (deployment sampling); greedy hides loops
│   ├── Your own task: a YAML file (dataset, fields, prompt, scorer; env: for agentic tasks, the RL config's
│   │   section); scorers: exact_match, numeric, mcq, regex_match, json_schema, function_call, ifeval, any RL
│   │   reward (math, code, judge with judge_validation), module:function (see the Evaluation guide)
│   ├── During training: eval_tasks: {tasks: [...], every: N, limit: 200} in the pgs train / rl / distill config;
│   │   watch eval/tasks/<task>/<mode>/score, compare <output_dir>/eval_tasks/step_<n>/ files with pgs compare
│   └── A gain on a public benchmark only: check the training data for its test items (contamination)
├── Training too slow
│   ├── Check: python -m palingenesis.agent_tooling.monitor_run --log_file ... --brief
│   ├── Likely causes: Data loading bottleneck, no compile, small batch
│   └── Fix: Increase num_workers, enable compile, use packing
└── Want to verify everything is OK before starting
    └── Run: python -m palingenesis.agent_tooling.diagnose --config ... --mode pre --json
```

## In-process use (scripts, notebooks)

`palingenesis.train.train(Config, plugins=[...])` runs SFT / DPO / off-policy distillation in-process on one GPU and
returns a `TrainResult` (output_dir, global_step, metrics, best_eval_loss); `plugins` are `palingenesis.hooks.TrainerPlugin`
objects (layer substitution, extra losses, optimizer groups, export folding). `RLTrainer(...).train()` and
`OPDTrainer(...).train()` also run in-process.
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
| OOM from activations, PCIe to spare | `memory.activation_offload: true` (saved activations in pinned host memory; with `gradient_checkpointing: full` only the layer inputs move) |
| Optimizer step slow (many parameter tensors) | `train.fused_optimizer: true` (adamw / adamw_bf16: one Triton launch per dtype, clipping folded in) |
| Loss NaN | `train.learning_rate: 1e-5`, `train.max_grad_norm: 1.0` |
| No improvement | Increase LR, check masking with `validate_masking` |
| Slow throughput | `model.compile: true`, `data.packing: true`, increase `data.num_workers` |
| Long seq OOM | `parallel.context_parallel: true` (multi-GPU, with `parallel.fsdp: true`; `parallel.cp_degree` GPUs per sequence), `memory.loss_num_chunks: 16`; one GPU: `memory.seco: true` |
| Context parallel refused (recurrent layers other than Gated DeltaNet, e.g. Mamba) | `memory.seco: true` for long sequences, or FSDP without context parallel (Qwen3.5 / Qwen3-Next are supported) |
| Images in the data are ignored (Qwen-VL family) | `multimodal.enabled: true` (+ `multimodal.max_pixels` to bound tokens per image; not with SeCO, context parallel, `data.pretokenize`, cached teachers); RL: `model.multimodal: true`, tools return `ToolOutput(text, images=[...])`, `env.image_results: tool`/`user`, `env.max_images` (see the multimodal guide) |
| Repetition loops after SFT | `data.loop_recovery: 0.2`, `data.reasoning_loop_recovery: 0.2`, `data.repeat_unlikelihood: 20`, `data.stop_unlikelihood: 5` (with `memory.seco: true`) |
| Agent drifts on its own errors | OPD `sources.<name>.format: agent_traces` on the student's episodes + `onset_branches`, `error_patterns`, `cutoff_logp` |
| Off-policy KD: teacher too large | `distill.teachers.<name>.placement: shard` (FSDP), or `pgs distill-cache` once and `cache: <dir>` |
| Off-policy KD: "no teacher target" | rebuild the cache with the training config, or `distill.missing: skip` + `distill.ce_weight: 0.1` |
| LoRA (many adapters, model too big for full-weight optimizer state) | `plugins.custom: [{class: palingenesis.lora:LoRA, args: {rank: 16, alpha: 32}}]`, `train.learning_rate` ~10x the full fine-tune's (see the LoRA guide) |
| LoRA in RL (`pgs rl`: policy too big for full-weight optimizer state next to the rollout engine) | `lora.rank: 16`, `lora.alpha: 32` in the RL config; watch `lora/push_survival` and `policy/mismatch_k3`; checkpoints are plain models plus `adapter/` (tested on CPU only; GPU validation pending, see the LoRA guide) |
| Large model, FSDP all-gathers dominate or activations OOM on many GPUs | `parallel.tp_degree: 2..8` (one node's GPUs; Llama / Mistral / Qwen2 / Qwen3 / Qwen3-MoE / Qwen3.5 / Gemma), `parallel.pp_degree` + `pp_schedule: 1f1b` / `zbv` across nodes (untied embeddings); see the multi-GPU guide's composition table |
| MoE routers collapse (`moe/load_max` rising, `moe/router_entropy` falling) | `moe.balance_loss_coef: 0.01`, `moe.z_loss_coef: 0.001` (see the MoE guide) |
| Rollouts slow with vLLM, or an engine on another node | `rollout.backend: sglang` (in-process) or `sglang_server` (+ `rollout.weight_sync`); check `policy/mismatch_k3_step0` and `rollout_tok_s` against the vLLM run (RL guide, Rollout engines) |
| MoE policy in RL: noisy ratios | `rollout.routing_replay: auto` with `rollout.prefix_caching: true` (vLLM returns the routed experts) |
| Quality lost after FP4 / FP8 quantisation (PTQ) | QAT with KD to the bf16 model, `train.learning_rate: 3.0e-6`, thousands of distinct self-generated rows each seen once (measured on a 0.8B model: over half of the PTQ held-out KL recovered; a learning rate near SFT's on a narrow set made it worse than PTQ, so verify on held-out KL and a task metric against `pgs quantize`): `plugins.custom: [{class: palingenesis.qat:FloatQAT, args: {format: nvfp4, scales: mse, freeze_scales_at: 0.9}}]` + `distill.teachers.<name>.model: <the bf16 model>`, `model.torch_dtype: float32`; serve `final-nvfp4/` (see the QAT guide) |
| Evaluation score dropped but samples look right | read `<task>.<mode>.samples.jsonl`: `finish: length` (raise `sampling.max_tokens`, thinking needs more), an answer format the scorer does not extract, or `unscored_samples` (see the Evaluation guide's pitfalls) |
| Off-policy KD: student copies the teacher's tail / rambles | `distill.divergence: reverse_kl` or `jsd`, add `distill.ce_weight` |
