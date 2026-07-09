# Training Strategies & Best Practices

Research-backed guidance for configuring agentic SFT runs. Every recommendation links to a published paper with ablations.

## Key Findings (2024-2026)

### 1. Data Repetition > Data Scaling for CoT/Reasoning

**Paper**: "Data Repetition Beats Data Scaling in Long-CoT SFT" (arxiv:2602.11149, Feb 2025)

**Finding**: Under a fixed compute budget, training for MORE EPOCHS on a SMALLER curated dataset outperforms single-epoch training on a larger dataset.

**Implication for agentic SFT**:
- Curate 5-20k high-quality agentic traces instead of gathering 100k+ noisy ones
- Train for 3-5 epochs on the curated set
- Quality >>> Quantity for reasoning/tool-use behaviors

**Config recommendation**:
```yaml
data:
  dataset: your-org/curated-agentic-traces  # 5-20k samples
train:
  epochs: 3-5
  learning_rate: 1.0e-5  # Lower LR for multi-epoch
```

### 2. Lower Learning Rate Preserves General Capabilities

**Paper**: "Revisiting Domain-Specific Fine-Tuning" (arxiv:2509.20758, 2025)
**Paper**: "Full Finetuning with Same Optimizer as Pretraining Forgets Less" (arxiv:2605.06654, May 2025)

**Finding**: Using a smaller LR substantially mitigates general performance degradation while preserving target-domain performance. The learning-forgetting tradeoff is heavily controlled by LR.

**For agentic SFT** (from instruct models):
- Use 5e-6 to 2e-5 (NOT the 1e-4 to 5e-4 range some tutorials suggest)
- The model already has the target behavior approximately; you're refining, not teaching from scratch

**Config**:
```yaml
train:
  learning_rate: 1.0e-5   # Conservative for instruct models
  min_learning_rate: 1.0e-6
```

### 3. Warmup is Optional for Small-Scale SFT

**Paper**: "A Guide For Supervised Fine-Tuning Small LLMs" (arxiv:2412.13337, Dec 2024)

**Finding**: Simplifications like omitting warmup and using constant LR do not compromise performance for SFT.

**Practical implication**: For short runs (<5k steps), warmup adds complexity without benefit. For longer runs, warmup helps stability.

**Config**:
```yaml
train:
  warmup_ratio: 0.0    # Skip for short runs
  lr_scheduler: constant  # Or cosine if epochs > 1
```

### 4. 1% Pretraining Data Injection Prevents Forgetting

**Paper**: "Scaling Laws for Forgetting during Finetuning" (NeurIPS 2025)

**Finding**: Injecting as little as 1% of pretraining-style data into the SFT mixture prevents catastrophic forgetting of pretraining capabilities.

**For agentic SFT**: Mix 1-5% general instruction-following data (e.g., from the instruct model's training set) into your agentic traces.

**How to do it**: Create a dataset that's 95% your agentic data + 5% general instructions (HF's ultrachat, Alpaca, etc.).

### 5. InfoSFT is Complementary with Standard SFT

**Paper**: "InfoSFT" (arxiv:2605.14967, May 2025)

**Finding**: For behaviors very different from the base model (e.g., new `<think>` format), do SFT first for 1 epoch to boost low-probability tokens, then switch to InfoSFT to refine.

**Two-stage strategy**:
```yaml
# Stage 1: Standard SFT (learn the format)
plugins:
  info_sft: false
train:
  epochs: 1

# Stage 2: InfoSFT (refine on informative tokens)
plugins:
  info_sft: true
train:
  epochs: 2
  resume_from: auto
```

### 6. Small Models Need Different Strategies

**Paper**: "Path to Effective Long CoT Training for Small Language Models" (arxiv:2506.07712, 2025)

**Finding**: Small models (< 4B) can LOSE performance from SFT on long CoT data. They need:
- Shorter training (1-2 epochs max)
- Lower LR (5e-6)
- Less data (overfitting faster)
- More aggressive filtering of hard samples

**For 1-4B models**:
```yaml
train:
  epochs: 1
  learning_rate: 5.0e-6
  gradient_accumulation_steps: 32  # Larger effective batch
data:
  max_seq_length: 4096  # Shorter for small models
```

### 7. Cosine Schedule vs Constant LR

**Research consensus 2025**:
- Cosine: best for fixed-budget training where you know total steps
- Constant: fine for SFT, simpler, especially with Schedule-Free
- WSD (Warmup-Stable-Decay): best for flexible stopping

**Our recommendation**: Use Schedule-Free AdamW (plugin) when dataset size is unknown. Use cosine when you know exactly how long to train.

## Hyperparameter Cheat Sheet

### For 7-8B Models (Llama 3.1, Qwen 2.5, Mistral)

| Scenario | LR | Epochs | Batch | Seq Len | Key Insight |
|----------|-----|--------|-------|---------|-------------|
| Standard SFT from instruct | 1-2e-5 | 1-2 | 128-256 | 4096-8192 | Conservative LR |
| Agentic traces (tool use) | 1e-5 | 3-5 | 64-128 | 8192-16384 | Repetition helps |
| Long-context adaptation | 5e-6 | 1 | 16-32 | 32768-65536 | Very low LR |
| Reasoning/CoT distillation | 2e-5 | 2-3 | 128 | 8192 | SFT then InfoSFT |
| Quick domain adaptation | 5e-5 | 1 | 256 | 2048 | Higher LR, 1 epoch |

### For 1-4B Models (Qwen 2.5 1.5B, Phi, SmolLM)

| Scenario | LR | Epochs | Batch | Key Insight |
|----------|-----|--------|-------|-------------|
| Standard SFT | 5e-6 to 1e-5 | 1-2 | 64-128 | Overfits fast |
| Agentic traces | 5e-6 | 2-3 | 32-64 | Filter hard samples |
| CoT distillation | 1e-5 | 1 | 128 | Can hurt if too long |

### Universal Settings

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| Weight decay | 0.1 | Standard for transformers |
| Adam beta1 | 0.9 | Standard |
| Adam beta2 | 0.95 | Slightly lower than default 0.999 for stability |
| Grad clip | 1.0 | Prevents explosions |
| bf16 | true | No-brainer on modern GPUs |
| Selective AC | true | Best memory/compute tradeoff |
| Liger Kernel | true | Free throughput + memory |

## Data Quality Signals

From "A Guide For Supervised Fine-Tuning Small LLMs" and our health metrics:

**Good data characteristics** (monitor via `health/token_efficiency`):
- 20-50% token efficiency (not all padding, not all trained)
- Consistent loss decrease in first epoch
- grad_cosine_sim > 0.2 (updates are coherent)
- loss_cv < 0.5 (not too noisy)

**Bad data signals**:
- loss_cv > 1.0 → data is too diverse/noisy, filter it
- token_efficiency < 5% → mostly system prompts, consider packing
- grad_cosine_sim < 0 → conflicting samples, curate better
- Loss doesn't decrease after 100 steps → data may be too easy (model already knows it)

## Training Duration Guidelines

From "Data Repetition Beats Data Scaling" + practical experience:

| Dataset Size | Recommended Epochs | Total Steps (bs=128) |
|-------------|-------------------|---------------------|
| 1k samples | 5-10 | 40-80 |
| 5k samples | 3-5 | 120-200 |
| 20k samples | 2-3 | 300-470 |
| 100k samples | 1-2 | 780-1560 |
| 500k+ samples | 1 | 3900+ |

**Rule of thumb**: For reasoning/agentic data, 3 epochs on 10k curated samples often beats 1 epoch on 100k noisy samples.

## Anti-Patterns to Avoid

1. **High LR + many epochs** = catastrophic forgetting + mode collapse
2. **1 epoch on huge dataset** = underfitting the hard samples (need repetition)
3. **Training on all tokens equally** = wasting compute on tokens model already knows (use InfoSFT)
4. **No validation** = silent overfitting (use our health metrics)
5. **Fixed LR schedule when dataset size unknown** = wrong decay point (use Schedule-Free)
6. **LoRA for agentic SFT** = inferior to full fine-tuning for behavior change (paper: arxiv:2605.06654)
