# Data Preparation Guide

Everything you need to know about preparing data for agentic SFT. This is the most impactful step — data quality determines 80% of final model performance.

## Core Principles (from 348 papers)

1. **Fewer good samples > many bad samples.** 400 verified samples × 128 epochs beats 51,200 samples × 1 epoch (arxiv:2602.11149).

2. **Difficulty matters, and the optimal is model-relative.** A sample that's "medium" for GPT-4 might be "impossible" for a 4B model. Always score with YOUR target model.

3. **The J-shaped distribution is optimal.** Not all easy, not all hard. The sweet spot: 20% easy (stability) + 50% medium (learning) + 25% hard (growth) + 5% very hard (exposure).

4. **Diversity > quantity.** 5K diverse samples across 50 tools outperforms 50K samples with 5 tools.

5. **Pretraining replay helps the TARGET task.** Mixing 10% generic data during SFT doesn't just prevent forgetting — it improves the target task too (arxiv:2603.04964).

6. **On-policy data generalizes better than off-policy.** Self-generated data (from your model) is distributionally aligned. Distilled data (from GPT-4) can be mismatched.

---

## Data Format

### Chat format (standard)

```json
{"messages": [
  {"role": "system", "content": "You are a coding assistant."},
  {"role": "user", "content": "Write a function to find primes."},
  {"role": "assistant", "content": "```python\ndef is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\n```"}
]}
```

### Agentic format (with tool calls)

```json
{"messages": [
  {"role": "system", "content": "You have access to: bash, python, web_search"},
  {"role": "user", "content": "What's the current disk usage on /data?"},
  {"role": "assistant", "content": "Let me check the disk usage.\n```bash\ndf -h /data\n```"},
  {"role": "tool", "content": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       500G  312G  188G  63% /data"},
  {"role": "assistant", "content": "The `/data` partition is 63% full (312 GB used out of 500 GB, with 188 GB available)."}
]}
```

### Key roles

| Role | Purpose | Gets loss? |
|------|---------|-----------|
| `system` | Context, tool descriptions | No |
| `user` | Input queries | No |
| `assistant` | Model responses (what we train) | **Yes** |
| `tool` / `observation` / `ipython` | Tool/environment outputs | Only with `include_observations: true` (ECHO) |

---

## The Prepare Pipeline

### Config-driven preparation (recommended)

The cleanest workflow: one YAML drives both preparation and training. `pgs prepare --config` reads `model.name_or_path` (scoring model), `data.dataset` / `data.dataset_split` / `data.messages_field` / `data.max_seq_length` (what to score, and how), and the `preprocess:` section (how to select and where to write):

```yaml
# In your training config
preprocess:
  enabled: true                    # training auto-uses the prepared output
  output_dir: ./prepared/qwen35_4b
  format: parquet                  # parquet (default) or jsonl
  budget: 5000                     # samples to keep (0 = all)
  strategy: optimal                # optimal | curriculum | balanced | flow | ...
  hes: true                        # optional reasoning-quality scoring
```

```bash
pgs prepare --config configs/qwen35_4b/a100_80gb.yaml   # score + filter + dump parquet
pgs train   --config configs/qwen35_4b/a100_80gb.yaml   # trains on the prepared parquet
```

Because scoring model and training model are the same config field, they can never drift apart. If `preprocess.enabled: true` but nothing was prepared yet, training fails immediately with the exact command to run — never a silent fallback to raw data. With `strategy: curriculum`, training skips shuffling so the easy→hard ordering is preserved.

The usual CLI overrides work: `pgs prepare --config cfg.yaml --preprocess.budget 2000`.

### Single-source preparation (standalone flags)

```bash
pgs prepare \
    --model Qwen/Qwen3.5-4B \
    --data raw_traces.jsonl \
    --output prepared/ \
    --budget 5000 \
    --strategy optimal \
    --format parquet \
    --hes
```

What happens:
1. **Score**: Runs your target model over each sample, computes response perplexity
2. **HES**: Computes High-Entropy Sum (identifies genuine reasoning vs template-following)
3. **Classify**: Buckets into easy/medium/hard based on perplexity percentiles
4. **Filter**: Removes outliers (ppl < 1.5 = trivial, ppl > 500 = impossible/corrupt)
5. **Select**: Picks J-shaped subset within budget

Output: `prepared/scored_data.parquet` (or `.jsonl` with `--format jsonl`) with fields:
```json
{
  "messages": [...],
  "_score_ppl": 12.4,
  "_score_response_ppl": 8.7,
  "_score_length": 1024,
  "_score_difficulty_bucket": "medium",
  "_score_flow_weight": 0.82,
  "_score_hes": 45.2,
  "_score_hes_normalized": 0.67
}
```

Plus `prepared/prepared_meta.json` — a provenance manifest (scoring model, source dataset, strategy, sample count, perplexity stats, difficulty distribution) that training logs at startup. Parquet is the default because it preserves sample order (needed for curriculum), loads much faster than JSONL, and is directly consumable by the training loader — `data.dataset` accepts `.parquet` files and prepared directories too.

### Multi-source preparation

For training on mixed datasets (recommended for production):

```yaml
# sources.yaml
- name: agentic_tool_calls
  dataset: ./data/agentic_traces.jsonl
  weight: 0.65
  messages_field: messages

- name: general_instruction
  dataset: HuggingFaceH4/ultrachat_200k
  split: train_sft
  weight: 0.20
  messages_field: messages

- name: code_solutions
  dataset: ./data/code_verified.jsonl
  weight: 0.15
  messages_field: messages
```

```bash
pgs prepare-multi \
    --model Qwen/Qwen3.5-4B \
    --sources sources.yaml \
    --output prepared/ \
    --budget 3000 \
    --strategy optimal \
    --hes
```

Output structure:
```
prepared/
├── manifest.json                    # Source metadata + MSFT recommendations
├── agentic_tool_calls/
│   └── scored_data.jsonl
├── general_instruction/
│   └── scored_data.jsonl
└── code_solutions/
    └── scored_data.jsonl
```

The `manifest.json` includes per-source overfit risk estimation and recommended epoch multipliers.

---

## Selection Strategies

| Strategy | When to use | Distribution |
|----------|-------------|--------------|
| `optimal` | Default. Research-backed J-shape. | 20% easy + 50% medium + 25% hard + 5% very hard |
| `balanced` | Maximum diversity across difficulty. | Equal parts each bucket |
| `medium_focus` | Small budget, need maximum info. | Centered on median difficulty |
| `flow` | Anti-forgetting priority. | Weighted by exp(-ppl/median) |
| `curriculum` | Multi-epoch with progressive difficulty. | Ordered easy → hard |
| `hard_focus` | Large data + strong model. | Prioritize hardest samples |

### When to use which:

- **First time training on a task**: `optimal` (safest, works everywhere)
- **Small budget (< 1000 samples)**: `medium_focus` (every sample must count)
- **Multi-epoch (> 5 epochs)**: `curriculum` or `flow` (prevent over-memorization of hard samples)
- **Already-strong model (fine-tuning further)**: `hard_focus` (easy data is redundant)
- **Domain adaptation (medical/legal)**: `flow` (prioritize anti-forgetting)

---

## The HES Metric

High-Entropy Sum identifies genuine reasoning quality by looking at decision points.

### How it works

1. Run the model over each sample
2. Compute per-token entropy (uncertainty at each position)
3. Take only the top 0.5% highest-entropy tokens
4. Sum their entropies → HES score

### Why it works

Most tokens in any response are predictable (articles, punctuation, common patterns). The CRITICAL tokens — where the model makes real decisions (which function to call, what logic to apply, which direction to take) — are rare but have high entropy.

**Good reasoning sample**: Many high-entropy decision points → high HES
**Template/boilerplate**: Few decisions, mostly pattern-matching → low HES
**Garbage/noise**: Uniformly confused → moderate HES but NOT high top-k (differs from average entropy!)

### Validation

From the paper (arxiv:2605.22389):
- Training on top 20% HES-ranked data matches FULL dataset performance
- Training on bottom 20% HES-ranked data actively degrades the model
- HES works across SFT, rejection fine-tuning, AND RL settings

---

## ECHO: Training on Tool Outputs

When `data.include_observations: true`, the model also gets loss on tool/environment output tokens (roles: `tool`, `observation`, `ipython`, `function`).

### Why this matters (arxiv:2605.24517, ICML 2026)

Standard SFT only trains on `assistant` tokens. But tool outputs contain rich information:
- What happens when you run `ls -la`
- What errors look like
- How APIs respond
- What file contents look like

By training on these, the model builds an internal "world model" — it learns to PREDICT what tools will return. This means:
- Better tool selection (knows what each tool does)
- Better error handling (knows what errors look like)
- Fewer unnecessary tool calls (can predict outputs without calling)

### Configuration

```yaml
data:
  include_observations: true   # Enable ECHO
  turn_scaling: progressive    # Weight later turns more (combined effect is powerful)
```

The combination of ECHO + DEFT is especially powerful: DEFT's adaptive weighting naturally focuses more on surprising tool outputs (high information content) and less on predictable ones.

---

## Turn Scaling

Multi-turn conversations are not uniform in difficulty or importance.

### `uniform` (default)
All turns get equal loss weight. Standard SFT.

### `progressive` (recommended for agentic)
Later turns get more weight: `w = sqrt(turn_idx / total_turns)`

Why: In agentic traces, later turns contain:
- Error recovery ("that didn't work, let me try...")
- Iterative refinement
- Harder reasoning (building on context)
- The final answer

### `last_heavy`
Final turn gets 2× weight, others 1×. For tasks where only the final answer matters.

---

## Pretraining Replay

```yaml
data:
  pretrain_replay_dataset: HuggingFaceH4/ultrachat_200k
  pretrain_replay_weight: 0.10  # 10% of training tokens
```

### The surprising finding

Stanford (arxiv:2603.04964) showed that replaying generic pretraining data during fine-tuning doesn't just prevent forgetting — it IMPROVES performance on the target task.

Hypothesis: Generic data acts as implicit regularization, keeping the model in a good region of the loss landscape where gradients are clean.

### Recommendations

| Scenario | Replay weight |
|----------|--------------|
| Large SFT dataset (> 10K samples) | 0.05 (5%) |
| Small SFT dataset (< 2K samples) | 0.15 (15%) |
| Domain adaptation (high forgetting risk) | 0.20 (20%) |
| Pre-GRPO (preserving diversity) | 0.10 (10%) |

---

## Adaptive Source Weighting (MSFT-inspired)

When training on multiple sources, they overfit at different rates. Our adaptive tracker handles this automatically.

```yaml
data:
  sources: [...]
  msft_tracking: true
  msft_eval_every: 50          # Check every 50 steps
  msft_decay_factor: 0.7       # Decay rate when overfitting
  msft_recovery_factor: 1.15   # Recovery rate when improving
  msft_floor_ratio: 0.1        # Never go below 10% of original weight
```

### How it works

1. Every 50 steps, evaluate per-source validation loss
2. Sources whose val loss is increasing → weight decays (×0.7)
3. Sources whose val loss is decreasing → weight recovers (×1.15)
4. Weight never drops below 10% of original (floor for anti-forgetting)
5. Weight never exceeds original (ceiling)

### Why not hard exclusion (like the MSFT paper)?

Excluding a source entirely causes:
- Catastrophic forgetting of that source's capabilities
- Loss of gradient diversity
- Loss of the pretraining replay benefit
- Irreversible decision (can't recover if wrong)

Our continuous decay with floor provides the same compute allocation signal while maintaining all beneficial properties.

---

## S0 Tuning (Hybrid Models)

After standard SFT, you can apply S0 Tuning for an additional zero-cost specialization:

```bash
# Prepare ~50 execution-verified solutions
python generate_and_verify.py --output verified.jsonl

# S0 Tune (50 epochs, learns a 48MB state file)
pgs s0-tune \
    --model ./checkpoints/final \
    --data verified.jsonl \
    --output s0_code.pt \
    --alpha 0.07 \
    --epochs 50
```

### Key points

- Only works on hybrid models (Qwen3.5, FalconH1) with matrix-valued recurrent states
- Needs execution-VERIFIED data (not just any data — correctness matters here)
- ~50 samples is enough (it's optimizing a tiny surface)
- Zero inference overhead (state absorbed into recurrence at first token)
- Task switching = loading a different 48MB file (no model reload)

### Alpha values

| Architecture | Alpha | Notes |
|--------------|-------|-------|
| Qwen3.5 (GatedDeltaNet) | 0.07 | Paper default, validated on 0.8B-9B |
| FalconH1 (Mamba-2) | 0.65 | Different gating dynamics |
| Other hybrids | Sweep 0.01-1.0 | Architecture-specific |

---

## Validation and Quality Checks

### Before training

```bash
# Check masking (are the right tokens getting loss?)
pgs inspect --config your_config.yaml --num_samples 5

# Validate masking across many samples
pgs validate --config your_config.yaml --num_samples 200

# Estimate memory usage
pgs profile --config your_config.yaml --gpu_memory_gb 80
```

### During training

```bash
# Quick status
pgs monitor --log_file checkpoints/train.log --brief

# Watch for: eval_loss not increasing, grad_norm stable, loss decreasing
```

### After training

```bash
# Analyze loss curve
pgs loss --log_file checkpoints/train.log

# Look for:
# - Smooth decrease (good)
# - Late plateau (might need more data or different LR)
# - Oscillations (LR too high or batch too small)
```

---

## Common Mistakes

### 1. Not scoring with the TARGET model

❌ Scoring difficulty with GPT-4, training a 4B model
✓ Scoring with the 4B model you'll actually train

Why: difficulty is model-relative. What's "medium" for GPT-4 might be impossible for 4B.

### 2. Training on ALL data regardless of quality

❌ `pgs train --data raw_dump.jsonl`
✓ `pgs prepare --data raw_dump.jsonl --budget 5000 --strategy optimal`

Why: 5K curated samples consistently outperforms 50K unfiltered.

### 3. Single epoch on large dataset

❌ 50K samples × 1 epoch
✓ 2K samples × 20 epochs (for reasoning tasks)

Why: Repetition helps for hard tasks. But increase weight_decay to 0.3 for multi-epoch.

### 4. Same config for different model families

❌ Using flex_attention for Qwen3.5 (crashes DeltaNet layers)
✓ Using sdpa for hybrid models, flex_attention for pure transformers

### 5. Ignoring tool outputs

❌ Standard masking (only assistant tokens get loss)
✓ `include_observations: true` (ECHO: model learns world model for free)

### 6. Uniform turn weighting

❌ `turn_scaling: uniform` for agentic data
✓ `turn_scaling: progressive` (later turns = harder = more important)
