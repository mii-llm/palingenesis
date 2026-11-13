---
template: home.html
description: Post-train LLMs and agents from one config. SFT, multi-turn RL in tool environments, on-policy distillation, DPO and merging, with state-of-the-art defaults.
hide:
  - navigation
  - toc
---

<section class="pg-section pg-source" markdown>

## What it does

<div class="pg-body pg-rows" markdown>

<div class="pg-recipe" data-group="Train" data-keys="sft finetune fine-tune fine-tuning supervised chat train" markdown>

### Supervised fine-tuning

Including agentic traces with tool calls and reasoning.

```bash
pgs train --config configs/quickstart.yaml
```

<p class="pg-more" markdown>[Data](guides/data.md) [Agentic](guides/agentic-training.md)</p>

</div>

<div class="pg-recipe" data-group="Train" data-keys="rl rlvr grpo rewards rollouts policy" markdown>

### Reinforcement learning

Verifiable rewards, multi-turn tool use, GRPO-style groups, vLLM rollouts.

```bash
pgs rl --config configs/rl_math.yaml
```

<p class="pg-more" markdown>[RL guide](guides/rl.md)</p>

</div>

<div class="pg-recipe" data-group="Train" data-keys="agents tools mcp harbor docker harness swe" markdown>

### Agent environments

Python classes, MCP servers, OpenEnv, NeMo Gym, Harbor tasks in Docker, with emulated harness dialects (opencode, pi, OpenHands, …).

```bash
pgs rl --config configs/rl_harbor.yaml
```

<p class="pg-more" markdown>[Harbor](guides/harbor.md) [Harness profiles](guides/harness-profiles.md)</p>

</div>

<div class="pg-recipe" data-group="Train" data-keys="opd distill distillation teacher student kd" markdown>

### On-policy distillation

The student samples, one or more teachers score those tokens, reverse KL.

```bash
pgs distill --config configs/distill_math.yaml
```

<p class="pg-more" markdown>[Distillation](guides/distillation.md)</p>

</div>

<div class="pg-recipe" data-group="Train" data-keys="kd distill distillation teacher cache qat" markdown>

### Off-policy distillation

Fixed data against one or more teachers' distributions (online, or a cached top-k), context distillation, hidden-state targets, multi-GPU, plugins for QAT.

```bash
pgs init kd && pgs train --config kd.yaml
```

<p class="pg-more" markdown>[Off-policy distillation](guides/offpolicy-distillation.md)</p>

</div>

<div class="pg-recipe" data-group="Train" data-keys="dpo ipo slic preference pairs alignment" markdown>

### Preference optimization

DPO, IPO, SLiC-HF, robust and length-normalised DPO, SFT anchor.

```bash
pgs train --config configs/dpo_qwen35_08b.yaml
```

<p class="pg-more" markdown>[DPO](guides/dpo.md)</p>

</div>

<div class="pg-recipe" data-group="Train" data-keys="merge soup interpolate task arithmetic" markdown>

### Model merging

Interpolation, soups, task arithmetic.

```bash
pgs merge interpolate START END --lam 0.3 -o OUT
```

<p class="pg-more" markdown>[Merging](guides/merging.md)</p>

</div>

<div class="pg-recipe" data-group="Train" data-keys="data prepare filter score select clean" markdown>

### Data preparation

Score, filter and select with the same config you train with.

```bash
pgs prepare --config configs/qwen35_4b/a100_80gb.yaml
```

<p class="pg-more" markdown>[Data](guides/data.md)</p>

</div>

<div class="pg-recipe" data-group="Train" data-keys="autopilot auto sweep learning rate lr profile" markdown>

### Autopilot

Profiles the GPU, sweeps the learning rate, trains to completion.

```bash
pgs autopilot --model Qwen/Qwen3.5-4B --dataset data.jsonl
```

<p class="pg-more" markdown>[Autopilot](guides/autopilot.md)</p>

</div>

</div>

</section>

<section class="pg-section pg-source" markdown>

## When something is off

<div class="pg-body pg-rows" markdown>

<div class="pg-recipe" data-group="Fix" data-keys="nan inf diverge diverges exploding explode spike" markdown>

### Loss is NaN or spikes

**LR too high.** Loss NaN/spikes, `grad_cosine_sim < 0`.

<p class="pg-fact" markdown><span>Diagnosis</span> `train/grad_norm` exploding</p>

<p class="pg-fact" markdown><span>Fix</span> `learning_rate` ÷ 3</p>

<p class="pg-more" markdown>[Training dynamics troubleshooting](reference/health.md#training-dynamics-troubleshooting) [When things go wrong](guides/single-gpu.md#when-things-go-wrong)</p>

</div>

<div class="pg-recipe" data-group="Fix" data-keys="stuck plateau not decreasing no improvement slow convergence" markdown>

### Loss is flat

**LR too low.** Loss flat, slow convergence.

<p class="pg-fact" markdown><span>Diagnosis</span> eval not improving for 500+ steps</p>

<p class="pg-fact" markdown><span>Fix</span> `learning_rate` × 2</p>

```bash
pgs inspect --config your_config.yaml --num_samples 3
```

This shows you exactly what the model sees: tokenized text, which tokens have loss, how masking works. Most "loss not decreasing" issues are data masking bugs.

<p class="pg-more" markdown>[Training dynamics troubleshooting](reference/health.md#training-dynamics-troubleshooting) [When things go wrong](guides/single-gpu.md#when-things-go-wrong)</p>

</div>

<div class="pg-recipe" data-group="Fix" data-keys="oom cuda malloc memory vram crash" markdown>

### Out of memory

**Memory pressure.** OOM or CUDA malloc fails.

<p class="pg-fact" markdown><span>Diagnosis</span> `health/cuda_utilization_pct > 95%`</p>

<p class="pg-fact" markdown><span>Fix</span> Reduce `per_device_batch_size`, increase `loss_num_chunks`</p>

<p class="pg-more" markdown>[Single GPU](guides/single-gpu.md) [Long sequences](guides/long-sequences.md) [Troubleshooting](reference/health.md#training-dynamics-troubleshooting)</p>

</div>

<div class="pg-recipe" data-group="Fix" data-keys="slow throughput speed tok/s tokens per second" markdown>

### Training is slow

**Slow throughput.** tok/s below expected.

<p class="pg-fact" markdown><span>Diagnosis</span> `health/cuda_utilization_pct` low</p>

<p class="pg-fact" markdown><span>Fix</span> `model.compile: true`, increase batch, `num_workers`</p>

<p class="pg-more" markdown>[Training dynamics troubleshooting](reference/health.md#training-dynamics-troubleshooting)</p>

</div>

<div class="pg-recipe" data-group="Fix" data-keys="overfit overfitting generalisation generalization gap" markdown>

### Eval loss rises while train loss falls

**Overfitting.** eval↑ while train↓.

<p class="pg-fact" markdown><span>Diagnosis</span> `eval/loss` diverging from `train/loss`</p>

<p class="pg-fact" markdown><span>Fix</span> `epochs: 2`, `base_merge: true`</p>

<p class="pg-more" markdown>[Training dynamics troubleshooting](reference/health.md#training-dynamics-troubleshooting)</p>

</div>

<div class="pg-recipe" data-group="Fix" data-keys="repetition repeat loops collapse rank" markdown>

### Outputs become repetitive

**Representation collapse.** Model outputs become repetitive.

<p class="pg-fact" markdown><span>Diagnosis</span> `health/stable_rank_min < 3`</p>

<p class="pg-fact" markdown><span>Fix</span> Reduce LR, enable `sym_noise`</p>

<p class="pg-more" markdown>[Training dynamics troubleshooting](reference/health.md#training-dynamics-troubleshooting)</p>

</div>

<div class="pg-recipe" data-group="Fix" data-keys="entropy diversity rl sft overtrain" markdown>

### Output diversity disappears (RL)

**Entropy collapse (RL).** Output diversity disappearing.

<p class="pg-fact" markdown><span>Diagnosis</span> `health/output_entropy < 1.5`</p>

<p class="pg-fact" markdown><span>Fix</span> `pre_rl: true`, stop early</p>

<p class="pg-more" markdown>[SFT → RL transition](guides/sft-to-rl.md) [Troubleshooting](reference/health.md#training-dynamics-troubleshooting)</p>

</div>

</div>

</section>

<section class="pg-section" markdown>

## Memory, quality, scale

<div class="pg-body pg-cols" markdown>

<div markdown>

### Memory

A Qwen3.5-4B fine-tune (its attention pathway, 36% of the weights) trains in about 16 GiB, without LoRA. Chunked losses, gradient release, Lion 8-bit and selective checkpointing each remove a different kind of memory.

[How it works](guides/single-gpu.md)

</div>

<div markdown>

### Quality

Correctness first: the loss path matches a plain reference loop step by step, assistant turns are masked from the chat template itself (tool calls included), packed conversations never see each other. On top, research options (DEFT, Hyperball, power-decay) implemented to their papers' definitions, with their papers' claims clearly labelled as such.

[The research](architecture/research.md)

</div>

<div markdown>

### Scale

Single RTX 4090 to 32-node SLURM clusters. Same config file, same code path. FSDP2, Ring Attention, sharded checkpoints.

[Multi-node guide](guides/multi-node.md)

</div>

</div>

</section>

<section class="pg-section" markdown>

## The thesis

<div class="pg-body pg-prose" markdown>

<p class="pg-pull">Fine-tuning that reads the research so you don't have to.</p>

Most fine-tuning tools give you knobs. Hundreds of options, each requiring expertise to set correctly. The implicit message: "you figure it out."

Palingenesis takes the opposite stance. We read the papers, ran the ablations, found what works. The defaults are the result. You override them when you have a specific reason, not because you have to.

This means:

- The masking is right without you writing a template-specific rule: turns are located from the chat template itself.
- Memory is handled: chunked losses, checkpointing and `pgs profile` (static estimate, or a measured real step).
- The learning rate can be found for you: **Autopilot** sweeps it.
- Research options (DEFT, Hyperball, power-decay) are one line each, documented with what their papers claim.

When you *do* want control, it's all there. Every parameter, every plugin, every optimization is configurable. But the base case is: it just works.

</div>

</section>

<section class="pg-section" markdown>

## Install

<div class="pg-body pg-prose" markdown>

=== "From source"

    ```bash
    git clone https://github.com/mii-llm/palingenesis.git && cd palingenesis
    uv sync --extra train --extra logging && source .venv/bin/activate
    ./run.sh configs/quickstart.yaml
    ```

=== "From PyPI"

    ```bash
    pip install "torch>=2.11" --index-url https://download.pytorch.org/whl/cu129   # FIRST: the CUDA 12.9 build (driver >= 575)
    pip install "palingenesis[train,logging]"
    pgs init sft                     # writes ./sft.yaml: set your model and dataset
    pgs train --config sft.yaml
    ```

Any HuggingFace causal LM. Optimized for: Qwen 2.5/3/3.5, Llama 3, Gemma 4, Mistral. Hybrid architectures (Qwen3.5 DeltaNet) supported via `freeze_non_attention`.

[Full installation guide](getting-started/install.md)

</div>

</section>

<section class="pg-section" markdown>

## Start here

<div class="pg-body pg-ways" markdown>

<div markdown>

### First time?

Install, run, see results in 5 minutes.

[Quickstart](getting-started/quickstart.md)

</div>

<div markdown>

### Have agentic data?

Reasoning traces, tool calls, multi-turn. Native support.

[Agentic guide](guides/agentic-training.md)

</div>

<div markdown>

### Planning RL after SFT?

Monitor entropy. Don't overtrain. We explain why.

[SFT → RL](guides/sft-to-rl.md)

</div>

</div>

</section>
