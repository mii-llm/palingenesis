---
hide:
  - navigation
  - toc
---

<style>
.hero {
  text-align: center;
  padding: 4rem 1rem 3rem;
  max-width: 680px;
  margin: 0 auto;
}
.hero h1 {
  font-size: 3.2rem !important;
  font-weight: 200 !important;
  font-style: italic !important;
  letter-spacing: -0.03em;
  margin-bottom: 0.3em;
  line-height: 1.1;
}
.hero .subtitle {
  font-size: 1.1rem;
  opacity: 0.6;
  margin-bottom: 2.5rem;
  font-style: italic;
}
.hero .tagline {
  font-size: 1.05rem;
  line-height: 1.7;
  margin-bottom: 2.5rem;
}
.hero code {
  font-size: 0.85rem !important;
  padding: 0.8em 1.5em !important;
  display: inline-block;
  margin: 1rem 0;
}
.pillars {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
  max-width: 720px;
  margin: 3rem auto;
  text-align: left;
}
.pillars .pillar h3 {
  font-size: 0.8rem !important;
  text-transform: uppercase !important;
  font-style: normal !important;
  letter-spacing: 0.08em;
  opacity: 0.5;
  margin-bottom: 0.5em;
}
.pillars .pillar p {
  font-size: 0.88rem;
  line-height: 1.6;
}
.numbers {
  display: flex;
  justify-content: center;
  gap: 3rem;
  margin: 3rem 0;
  flex-wrap: wrap;
}
.numbers .num {
  text-align: center;
}
.numbers .num .value {
  font-size: 2rem;
  font-weight: 200;
  font-style: italic;
  display: block;
}
.numbers .num .label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  opacity: 0.5;
}
</style>

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
