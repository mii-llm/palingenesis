---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# palingenesis

<p class="tagline">
Fine-tuning that reads the research so you don't have to.<br>
One command. State-of-the-art quality. RTX 4090 to multi-node clusters.
</p>

```bash
./run.sh configs/quickstart.yaml
```

</div>

<div class="numbers">
  <div class="num"><span class="value">352</span><span class="label">papers read</span></div>
  <div class="num"><span class="value">15 GB</span><span class="label">for 4B full ft</span></div>
  <div class="num"><span class="value">62</span><span class="label">tests passing</span></div>
</div>

---

<div class="pillars" markdown>

<div class="pillar" markdown>

### Memory

A 4-billion parameter model trains in 15 GB. Full fine-tune, not LoRA. Gradient release + Lion 8-bit + selective checkpointing eliminate every byte of waste.

[How it works →](guides/single-gpu.md)

</div>

<div class="pillar" markdown>

### Quality

DEFT loss is a parameter-free token weighting aimed at reasoning tasks (the original paper reports gains, not independently reproduced; results vary by model and data). Hyperball adds 20-30% convergence speed (single paper, experimental). Power-decay schedule is provably optimal for the easy-task regime. All on by default in flagship configs.

[The research →](architecture/research.md)

</div>

<div class="pillar" markdown>

### Scale

Single RTX 4090 to 32-node SLURM clusters. Same config file, same code path. FSDP2, Ring Attention, sharded checkpoints.

[Multi-node guide →](guides/multi-node.md)

</div>

</div>

---

## The thesis

Most fine-tuning tools give you knobs. Hundreds of options, each requiring expertise to set correctly. The implicit message: "you figure it out."

Palingenesis takes the opposite stance. We read the papers, ran the ablations, found what works. The defaults are the result. You override them when you have a specific reason — not because you have to.

This means:

- You don't pick a loss function. **DEFT** is parameter-free and subsumes everything else.
- You don't pick a scheduler. **Power-decay** is theoretically optimal for SFT.
- You don't worry about memory. **Gradient release** handles it.
- You don't tune the learning rate (unless you want to). **Autopilot** finds it.

When you *do* want control — it's all there. Every parameter, every plugin, every optimization is configurable. But the base case is: it just works.

---

## Start here

<div class="pillars" markdown>

<div class="pillar" markdown>

### First time?

Install, run, see results in 5 minutes.

[Quickstart →](getting-started/quickstart.md)

</div>

<div class="pillar" markdown>

### Have agentic data?

Reasoning traces, tool calls, multi-turn. Native support.

[Agentic guide →](guides/agentic-training.md)

</div>

<div class="pillar" markdown>

### Planning RL after SFT?

Monitor entropy. Don't overtrain. We explain why.

[SFT → RL →](guides/sft-to-rl.md)

</div>

</div>

---

## Supported models

Any HuggingFace causal LM. Optimized for: Qwen 2.5/3/3.5, Llama 3, Gemma 4, Mistral. Hybrid architectures (Qwen3.5 DeltaNet) supported via `freeze_non_attention`.

## Install

```bash
git clone https://github.com/your-org/palingenesis.git && cd palingenesis
uv pip install -e ".[train]"
```

[Full installation guide →](getting-started/install.md)
