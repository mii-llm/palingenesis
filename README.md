# palingenesis

<p align="center">
  <img src="./assets/pgs.png" alt="palingenesis" width="100%">
</p>

**Fire-and-forget LLM fine-tuning with state-of-the-art defaults.**

Papers distilled into one command. Every optimization applied automatically.

```bash
git clone https://github.com/mii-llm/palingenesis.git && cd palingenesis
uv sync --extra train --extra logging     # creates .venv with CUDA-enabled torch
                                          # (+ --extra hybrid for Qwen3.5: faster, and needed to pack)
source .venv/bin/activate
./run.sh configs/quickstart.yaml
```

See [Installation](mkdocs_docs/getting-started/install.md) for pip, other CUDA versions and troubleshooting.

Short CLI alias: `pgs`

```bash
pgs train --config configs/quickstart.yaml
pgs autopilot --model Qwen/Qwen3.5-4B --dataset your_data.jsonl
```

---

## What you get

- **Correct by construction**: the loss path is checked against a plain reference loop step by step; packed conversations never see each other; saved models load with plain `from_pretrained`; multi-GPU gradients equal single-GPU ones (tests included)
- **Agentic masking from the chat template itself**: whole assistant turns, tool calls and end-of-turn tokens included; `tools` rendered into the prompt; per-message `loss: false`
- **Memory**: chunked losses (never the full logits), selective activation checkpointing, gradient release, 8-bit optimizers; a Qwen3.5-4B fine-tune (attention pathway trained, 36% of the weights) in ~16 GiB
- **Research options, one line each**: DEFT/DFT/InfoSFT token weighting, Hyperball, power-decay; implemented to their papers' definitions, benefits as reported by the papers (not reproduced here)
- **Best-model tracking**: eval before training, every `eval_every` steps and at the end; saves the lowest-eval-loss checkpoint
- **Auto-resume**: crash and re-run, picks up exactly where the last complete checkpoint left off
- **Multi-node SLURM**: sharded DCP checkpoints

## Hardware

| GPU | Model | Config |
|-----|-------|--------|
| RTX 4090 (24 GB) | Qwen3.5-4B, attention pathway (36% of weights) | `configs/qwen35_4b/a100_40gb.yaml` |
| A100-80GB | Qwen3.5-4B, batch=4 | `configs/qwen35_4b/a100_80gb.yaml` |
| 8× A100 | Qwen3.5-35B MoE | `configs/qwen35_35b_moe/a100_80gb_multigpu.yaml` |
| H100 | Qwen3.5-4B, FP8 | `configs/qwen35_4b/h100_80gb.yaml` |

## Data preparation (one config, closed loop)

Score, filter, and select your data with the *same* config you train with — the scoring model is `model.name_or_path`, the raw data is `data.dataset`, and the `preprocess:` section controls selection. Output is parquet plus a provenance manifest:

```bash
pgs prepare --config configs/qwen35_4b/a100_80gb.yaml            # score → filter → parquet
pgs train   --config configs/qwen35_4b/a100_80gb.yaml \
            --preprocess.enabled true                            # trains on the prepared data
```

Strategies: `optimal` (research-backed J-shaped difficulty mix), `curriculum` (easy→hard, ordering preserved during training), `balanced`, `flow`, and more.

## Training dynamics you can actually see

wandb + trackio, wired for real investigation: loss/ppl, grad norm, spike/clip counters, gradient noise scale, output entropy, and the generalization gap (`eval/gap`) — all on a single `train/global_step` axis. Crash-resume continues the *same* wandb run (run id persisted next to your checkpoints), and a tracker outage can never kill training.

## Agentic data support

Native support for reasoning traces with `reasoning` (or legacy `reasoning_content`), `tool_calls`, and tool responses. Each assistant turn is trained exactly as the chat template renders it, tool calls and end-of-turn token included. Tool definitions in a `tools` field are rendered into the prompt; `"loss": false` on a message keeps it as context only. ShareGPT, Alpaca and OpenAI formats (JSON-string tool arguments included) are normalized. Tool-call validation against declared schemas.

```yaml
data:
  include_observations: true  # ECHO: train on tool outputs (world model)
  turn_scaling: progressive   # Later turns weighted more
```

## Hyper-long sequences on one GPU (SeCO)

Train on sequences far beyond what fits in memory. The model runs chunk by chunk with a cache, keeping one chunk's activations at a time, and gradients are relayed back through the cache. The result is the exact gradient, verified against full backpropagation on 27 architectures, including the Qwen3.5 and LFM2 hybrids.

```yaml
memory:
  seco: true
  seco_chunk_size: 4096
```

See the [long-sequences guide](mkdocs_docs/guides/long-sequences.md).

## Preference optimization (DPO)

DPO and its variants (IPO, SLiC-HF, robust DPO, length-normalised, plus LD-DPO and an SFT anchor) run on the same trainer, chat templates and masking as SFT, with thinking and non-thinking pairs mixed in one dataset. The full `[B, S, V]` logits are never materialised.

```yaml
dpo:
  enabled: true
  beta: 0.1
  sft_weight: 0.2
```

See [docs/dpo.md](docs/dpo.md).

## Autopilot

Zero-config mode. Profiles your GPU, sweeps LR, trains to completion:

```bash
pgs autopilot --model Qwen/Qwen3.5-4B --dataset your_data.jsonl
```

## On-policy distillation

Shrink a teacher into a student by scoring the student's **own samples**: the student samples with its current weights, the teacher scores those exact tokens, and the student is pulled toward the teacher's distribution (reverse KL). Rollouts from vLLM or the trainer's own `generate`; teachers in-process or on a vLLM server; teachers with another tokenizer (byte-aligned chunks); one teacher per prompt source.

```bash
pgs distill --config configs/distill_math.yaml     # Qwen3-1.7B -> Qwen3-0.6B, vLLM rollouts
pgs distill --config configs/distill_xtok.yaml     # teacher with another tokenizer
pgs distill --config configs/distill_multi.yaml    # one teacher per prompt source
pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl  # annotate an mcqa pool
```

`distill-score` marks every multiple-choice pool row with the teacher's own answer so you can filter before training — pure KL faithfully distills the teacher's *errors* too, making its accuracy a hard ceiling. See the [distillation guide](https://mii-llm.github.io/palingenesis/guides/distillation/).

## Multi-GPU / Multi-Node

```bash
./scripts/train_multi_gpu.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
sbatch scripts/train_slurm.sh configs/qwen35_35b_moe/a100_80gb_multigpu.yaml
```

## Documentation

```bash
pip install mkdocs-material
mkdocs serve  # → http://localhost:8000
```

## Tests

```bash
pytest tests/
```

---

*A new form emerging from what came before.*
