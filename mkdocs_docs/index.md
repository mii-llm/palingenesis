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
  <div class="num"><span class="value">~16 GiB</span><span class="label">Qwen3.5-4B fine-tune</span></div>
  <div class="num"><span class="value">530+</span><span class="label">tests</span></div>
</div>

---

<div class="pillars" markdown>

<div class="pillar" markdown>

### Memory

A Qwen3.5-4B fine-tune (its attention pathway, 36% of the weights) trains in about 16 GiB, without LoRA. Chunked losses, gradient release, Lion 8-bit and selective checkpointing each remove a different kind of memory.

[How it works →](guides/single-gpu.md)

</div>

<div class="pillar" markdown>

### Quality

Correctness first: the loss path matches a plain reference loop step by step, assistant turns are masked from the chat template itself (tool calls included), packed conversations never see each other. On top, research options (DEFT, Hyperball, power-decay) implemented to their papers' definitions, with their papers' claims clearly labelled as such.

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

- The masking is right without you writing a template-specific rule: turns are located from the chat template itself.
- Memory is handled: chunked losses, checkpointing and `pgs profile` (static estimate, or a measured real step).
- The learning rate can be found for you: **Autopilot** sweeps it.
- Research options (DEFT, Hyperball, power-decay) are one line each, documented with what their papers claim.

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
git clone https://github.com/mii-llm/palingenesis.git && cd palingenesis
uv sync --extra train --extra logging && source .venv/bin/activate
```

[Full installation guide →](getting-started/install.md)
