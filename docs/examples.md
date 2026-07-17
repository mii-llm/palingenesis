# Practical Examples

Copy-paste-ready examples for common training scenarios.

---

## Example 1: Train Qwen3.5-4B on Agentic Tool-Use Data

**Goal**: Fine-tune Qwen3.5-4B to reliably call tools (bash, python, APIs) in multi-turn conversations.

**Hardware**: Single A100-80GB

### Step 1: Prepare data

```bash
# Score data with target model
pgs prepare \
    --model Qwen/Qwen3.5-4B \
    --data ./data/agentic_traces.jsonl \
    --output ./projects/qwen35_agentic/prepared/ \
    --budget 5000 \
    --strategy optimal \
    --hes
```

### Step 2: Create config

```yaml
# projects/qwen35_agentic/config.yaml
model:
  name_or_path: Qwen/Qwen3.5-4B
  torch_dtype: bfloat16
  attn_implementation: sdpa
  use_liger_kernel: true
  compile: true

data:
  dataset: ./projects/qwen35_agentic/prepared/scored_data.jsonl
  streaming: false
  max_seq_length: 4096
  packing: true
  include_observations: true
  turn_scaling: progressive
  eval_dataset: ./data/agentic_eval.jsonl
  eval_split: train
  eval_samples: 100
  eval_every: 50

train:
  output_dir: ./projects/qwen35_agentic/checkpoints
  epochs: 3
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
  ga_ramp_start: 2
  learning_rate: 1.5e-5
  weight_decay: 0.1
  lr_scheduler: cosine
  optimizer: muon
  gradient_checkpointing: selective
  adagc: true
  adamc: true
  ema: true
  ema_every: 10
  base_merge: true
  base_merge_method: slerp
  base_merge_every: 200
  freeze_non_attention: true
  save_steps: 200

memory:
  chunked_loss: true
  loss_num_chunks: 4

plugins:
  deft: true
  sym_noise: true
  sym_noise_alpha: 5.0

logging:
  project: palingenesis
  run_name: qwen35-4b-agentic-v1
```

### Step 3: Verify before training

```bash
# Check memory fits
pgs profile --config projects/qwen35_agentic/config.yaml --gpu_memory_gb 80

# Check masking is correct
pgs inspect --config projects/qwen35_agentic/config.yaml --num_samples 3
```

### Step 4: Train

```bash
pgs train --config projects/qwen35_agentic/config.yaml
```

### Step 5: S0 specialization (optional, +10-23pp on code tasks)

```bash
# Generate and verify solutions from the trained model
# (assumes you have a verification script)
python scripts/generate_verified.py \
    --model ./projects/qwen35_agentic/checkpoints/final \
    --problems coding_problems.jsonl \
    --output verified_solutions.jsonl

# Apply S0 tuning
pgs s0-tune \
    --model ./projects/qwen35_agentic/checkpoints/final \
    --data verified_solutions.jsonl \
    --output ./projects/qwen35_agentic/s0_code.pt \
    --alpha 0.07 \
    --epochs 50
```

---

## Example 2: Train Gemma 4 E4B with Memory Constraints

**Goal**: Fine-tune an 8B parameter model on a single A100-80GB (very tight fit).

**Key constraints**: 262K vocabulary = massive logit tensor. Lion 8-bit optimizer essential.

### Config

```yaml
model:
  name_or_path: google/gemma-4-E4B-it
  torch_dtype: bfloat16
  attn_implementation: flex_attention
  use_liger_kernel: true
  compile: true

data:
  dataset: ./prepared/scored_data.jsonl
  streaming: false
  max_seq_length: 4096    # Don't go higher — 262K vocab × seq = OOM
  packing: true
  include_observations: true
  turn_scaling: progressive

train:
  output_dir: ./checkpoints/gemma4_e4b
  epochs: 3
  per_device_batch_size: 1     # MUST be 1 for 8B + 262K vocab
  gradient_accumulation_steps: 16
  ga_ramp_start: 4
  learning_rate: 2e-5
  weight_decay: 0.1
  lr_scheduler: cosine
  optimizer: lion8bit           # 4 bytes/param (essential for fitting)
  gradient_checkpointing: selective
  adagc: true
  adamc: true
  ema: true
  base_merge: true
  base_merge_method: slerp
  llrd_decay: 0.9              # Protect PLE layers

memory:
  chunked_loss: true
  loss_num_chunks: 8           # 262K vocab needs more chunks
  float32_matmul_precision: high

plugins:
  deft: true
  sym_noise: true
```

### Memory budget breakdown

| Component | Size |
|-----------|------|
| Model weights (bf16) | 16 GB |
| Lion 8-bit optimizer | 4 GB |
| Gradients (bf16) | 16 GB |
| Activations (selective AC, batch=1, seq=4096) | 12-15 GB |
| Chunked loss peak (262K/8 chunks) | 0.5 GB |
| EMA shadow (CPU) | 0 GPU |
| Base merge (CPU) | 0 GPU |
| **Total** | **~50-52 GB** ✓ |

Without Lion 8-bit (using AdamW): 16+32+16+15 = 79 GB → barely fits, no headroom.

---

## Example 3: Multi-Source Training with Adaptive Weighting

**Goal**: Train on agentic + general + code data with automatic compute allocation.

### Step 1: Prepare each source

```yaml
# sources.yaml
- name: agentic
  dataset: ./data/agentic_traces.jsonl
  weight: 0.65
- name: general
  dataset: HuggingFaceH4/ultrachat_200k
  split: train_sft
  weight: 0.25
- name: code
  dataset: ./data/code_verified.jsonl
  weight: 0.10
```

```bash
pgs prepare-multi \
    --model Qwen/Qwen3-4B \
    --sources sources.yaml \
    --output prepared/ \
    --budget 3000 \
    --hes
```

### Step 2: Train with adaptive weighting

```yaml
data:
  sources:
    - dataset: ./prepared/agentic/scored_data.jsonl
      name: agentic
      weight: 0.65
      mode: sft
    - dataset: ./prepared/general/scored_data.jsonl
      name: general
      weight: 0.25
      mode: sft
    - dataset: ./prepared/code/scored_data.jsonl
      name: code
      weight: 0.10
      mode: sft
  max_seq_length: 4096
  packing: true
  include_observations: true
  msft_tracking: true
  msft_eval_every: 50
  msft_decay_factor: 0.7
  msft_floor_ratio: 0.1

train:
  optimizer: muon
  learning_rate: 2e-5
  adagc: true
  ema: true
```

During training, you'll see logs like:
```
step=50   msft: agentic=100%↑ | general=100%↑ | code=100%↑
step=100  msft: agentic=100%↑ | general=70%↓1 | code=100%↑
step=150  msft: agentic=100%↑ | general=49%↓2 | code=85%↓1
```

The general source (easier to learn) gets its weight reduced as it starts overfitting, while agentic (harder) maintains full weight.

---

## Example 4: Reasoning Distillation (Multi-Epoch)

**Goal**: Distill math reasoning from a stronger model into Qwen3-4B.

### Data: 500 verified math solutions × 50 epochs

```bash
# Only need ~500 high-quality solutions
pgs prepare \
    --model Qwen/Qwen3-4B \
    --data math_solutions_500.jsonl \
    --output prepared/ \
    --strategy optimal \
    --hes
```

### Config (key: high epochs, high weight decay)

```yaml
model:
  name_or_path: Qwen/Qwen3-4B
  compile: true

data:
  dataset: ./prepared/scored_data.jsonl
  streaming: false
  max_seq_length: 8192
  packing: true

train:
  epochs: 50                       # Key: repetition helps for reasoning
  per_device_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 3e-5
  weight_decay: 0.3                # Higher WD for multi-epoch (arxiv:2509.14786)
  lr_scheduler: cosine
  optimizer: adamw
  adagc: true
  ema: true
  ema_decay: 0.9999                # Slower EMA for many epochs

plugins:
  deft: true
  sym_noise: true
  sym_noise_alpha: 7.0             # Stronger noise for more epochs
```

### Why this works

From arxiv:2602.11149: "Data Repetition Beats Data Scaling in Long-CoT SFT"
- 400 samples × 128 epochs outperforms 51,200 × 1 epoch by 12-26pp
- No additional catastrophic forgetting from multi-epoch
- The model memorizes the data (train acc → 100%) — this is FINE for reasoning
- Stop when train accuracy saturates

---

## Example 5: Self-Improvement Loop

**Goal**: Iteratively improve a model using its own verified generations.

### Round 1: Generate

```bash
# Start vLLM server with current model
python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/base \
    --max-model-len 8192 &

# Generate 8 completions per prompt
python scripts/generate.py \
    --prompts training_prompts.jsonl \
    --endpoint http://localhost:8000/v1 \
    --n_completions 8 \
    --temperature 0.7 \
    --output generations/round_1.jsonl

# Kill server
kill %1
```

### Round 2: Score and verify

```bash
# For code: execute and check
python scripts/verify_code.py \
    --generations generations/round_1.jsonl \
    --output scored/round_1.jsonl

# For math: check answers
python scripts/verify_math.py \
    --generations generations/round_1.jsonl \
    --answers ground_truth.jsonl \
    --output scored/round_1.jsonl
```

### Round 3: Prepare

```bash
pgs prepare \
    --model ./checkpoints/base \
    --data scored/round_1.jsonl \
    --output prepared/round_1/ \
    --budget 2000 \
    --strategy optimal \
    --hes
```

### Round 4: Train

```bash
pgs train --config configs/self_improve.yaml \
    --data.dataset prepared/round_1/scored_data.jsonl
```

### Repeat

Each round: generate → verify → prepare → train. Typically 2-3 rounds give compound gains.

---

## Example 6: Knowledge Domain Adaptation (No Forgetting)

**Goal**: Adapt to medical/legal domain while preserving general capabilities.

### Key techniques:
- Pretraining replay (15% generic data)
- Base merge SLERP every 100 steps
- Conservative LR
- DEFT + KL anchor

```yaml
data:
  dataset: ./prepared/medical_qa.jsonl
  max_seq_length: 4096
  pretrain_replay_dataset: HuggingFaceH4/ultrachat_200k
  pretrain_replay_weight: 0.15

train:
  learning_rate: 1e-5              # Conservative
  weight_decay: 0.1
  optimizer: adamw
  base_merge: true
  base_merge_ratio: 0.15           # Stronger anti-forgetting
  base_merge_every: 100            # More frequent
  base_merge_method: slerp
  ema: true
  llrd_decay: 0.85                 # Aggressive LLRD to protect early layers

plugins:
  deft: true
  sym_noise: true
  sym_noise_alpha: 7.0             # Stronger regularization
```

### Validation

Always track both domain performance AND general benchmarks:

```yaml
data:
  eval_dataset: ./data/medical_eval.jsonl
  eval_every: 50
```

And separately evaluate on general benchmarks after training to verify no regression.

---

## Example 7: Quick Iteration (Development Mode)

**Goal**: Fast iteration cycle during data/config development.

```yaml
# configs/dev.yaml — fast feedback, not for production
model:
  name_or_path: Qwen/Qwen3-0.6B   # Smallest model for fast iteration
  compile: false                    # Skip compilation during dev
  use_liger_kernel: false

data:
  dataset: ./data/small_sample.jsonl
  streaming: false
  max_seq_length: 2048
  packing: false                   # Easier to debug without packing

train:
  max_steps: 100                   # Just enough to see if loss decreases
  per_device_batch_size: 4
  gradient_accumulation_steps: 1
  logging_steps: 1                 # Log every step
  save_steps: 0                    # Don't save during dev

plugins:
  deft: true
```

```bash
# Quick test
pgs train --config configs/dev.yaml

# Check masking
pgs inspect --config configs/dev.yaml --num_samples 5
```

---

## Config Reference Cheatsheet

### Must-set for every training:

```yaml
model.name_or_path        # Which model
data.dataset              # Which data
data.max_seq_length       # Sequence length budget
train.learning_rate       # Start with 1.5-2e-5
train.optimizer           # adamw, muon, or lion8bit
```

### Recommended defaults (always good):

```yaml
plugins.deft: true               # Best loss function
plugins.sym_noise: true          # Regularization
train.adagc: true                # Zero loss spikes
train.gradient_checkpointing: selective
memory.chunked_loss: true
```

### Model-specific:

```yaml
# Qwen3.5 (hybrid):
model.attn_implementation: sdpa
train.freeze_non_attention: true

# Gemma 4 (PLE):
model.attn_implementation: flex_attention
train.llrd_decay: 0.9
memory.loss_num_chunks: 8        # 262K vocab

# LLaMA 3.1:
model.attn_implementation: sdpa  # or flash_attention_2
```

### Scale-specific:

```yaml
# ≤ 3B params: conservative
train.learning_rate: 2e-5
data.max_seq_length: 4096       # No long CoT (degrades small models)

# 4-8B params: standard
train.learning_rate: 1.5e-5
data.max_seq_length: 8192

# > 8B params: aggressive
train.learning_rate: 1e-5
train.optimizer: lion8bit       # Memory constrained
```

## Example 6: On-Policy Distillation into a Small Student

**Goal**: Lift a 0.4B student toward a 3B teacher on an MCQA benchmark (real case: nesso-0.4B-agentic on ITALIC, official score 33.1% → 37.2%).

### Step 1: Annotate the pool with the teacher's answers

Pure reverse KL distills the teacher's *errors* too — the teacher's accuracy becomes a hard ceiling. Score first, filter before training:

```bash
pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl
# -> each row gains teacher_answer / teacher_correct (one forward per row, ~2h for 110k on an A100)
```

Then apply your policy (drop wrong rows, reweight weak domains) with a small script — the annotation is mechanism, the filtering is your call.

### Step 2: Train

```yaml
# configs/distill_opd.yaml (the winning knobs)
model:
  student: mii-llm/nesso-0.4B-agentic
  teacher: Coloss/nesso-3B
  gradient_checkpointing: true     # 0.4B student + 3B teacher on one 80GB GPU

bridge:
  eos_map: {"<|im_end|>": "<|eot_id|>"}   # ChatML student <- Llama-3 teacher
  extra_stop_tokens: ["<|end_of_text|>"]

data:
  prompts_path: data/prompts_filtered.jsonl
  p_reference_shots: 0.8           # train mostly on the benchmark's exact shot prefix
  shots_path: data/5_shots.jsonl
  # plus the benchmark's VERBATIM templates + system message via
  # data.fast_template / data.cot_template — exact prompt bytes are policy,
  # so they live in the config (see configs/distill_opd.yaml for ITALIC's)

sampling:
  max_new_tokens: 8                # terse supervision: no verbosity drift, no misparse tax

train:
  steps: 600
  score_micro_seqs: 8              # memory knob; grad accumulation keeps the math identical
  eval_every: 50
  save_steps: 50
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pgs distill --config configs/distill_opd.yaml
```

### Step 3: Pick the checkpoint by dev accuracy, not KL

KL keeps falling long after accuracy stops improving. With a teacher-correct-filtered pool the dev accuracy climbs monotonically instead of plateauing early — checkpoint densely and evaluate the top candidates on the real benchmark.

### Results (ITALIC, official harness, full 10k)

| Configuration | Score |
|---|---:|
| Baseline student | 33.1% |
| + on-policy KL, unfiltered pool | ~34.4% |
| + fast-only supervision | ~35.4%* |
| + teacher-correct filter, shot matching, terse budget | **37.2%** |
| Teacher (ceiling for pure KL) | 50.7% |

*informal harness. The same pipeline runs on generic chat data with `data.format: messages` (`configs/distill_chat.yaml`) — dev metric becomes held-out reverse KL.
