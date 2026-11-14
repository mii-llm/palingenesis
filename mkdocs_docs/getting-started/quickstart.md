# Quickstart

*From zero to a trained model in 5 minutes.*

---

## 1. Install

```bash
git clone https://github.com/mii-llm/palingenesis.git
cd palingenesis
uv sync --extra train --extra logging
source .venv/bin/activate
```

Other ways to install (pip, other CUDA versions): see [Installation](install.md).

Installed from PyPI instead? `pgs init sft` writes the same config as `sft.yaml` in the current directory: use it
wherever this page says `configs/quickstart.yaml`.

## 2. Run

```bash
./run.sh configs/quickstart.yaml
```

The first run downloads the model (Qwen3-0.6B, ~1.5 GB) and starts streaming UltraChat. After a minute of setup, you'll see:

```
step=1   loss=4.12 lr=0.00e+00 tok/s=0     grad_norm=1.23 dt=45.2s   ← first step: compile warmup
step=2   loss=3.89 lr=4.00e-06 tok/s=5842  grad_norm=0.89 dt=1.3s    ← normal speed
step=3   loss=3.74 lr=8.00e-06 tok/s=6011  grad_norm=0.76 dt=1.2s
...
```

!!! note "The first step is slow"
    `torch.compile` traces and compiles the model on the first forward pass. Steps 2+ are the real speed. This is normal and only happens once (cached for the session).

How to read the line:

| Field | Meaning |
|---|---|
| `loss` | the training objective, averaged over the accumulation window: it should fall |
| `lr` | the learning rate from the schedule (it starts at 0 and warms up) |
| `tok/s` | training throughput |
| `grad_norm` | gradient norm before clipping; judge it against the run's own recent values |
| `dt` | wall-clock time of the step |
| `eval` | with an eval set (`data.eval_dataset` or `data.eval_sources`): the validation loss, every `data.eval_every` steps, and once as `step=0 eval=... (before training)` |
| `eval_sources` | with `data.eval_sources`: each source's own loss, `eval_sources=<name>:<loss>,...`, after the composite `eval` |

Checkpoints go to `train.output_dir` (`./output` here): a resumable `step-N/` every `train.save_steps` steps, of which the newest `train.keep_checkpoints` (default 5) are kept, and the final model in `final/`. See [Your First Training](first-training.md) for the layout and `best/`.

## 3. Use your own data

Edit `configs/quickstart.yaml`:

```yaml
model:
  name_or_path: your-org/your-model

data:
  dataset: path/to/your_data.jsonl
  messages_field: messages
```

Your data should be JSONL with chat messages:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

!!! tip "Clean data in two commands"
    Before training on raw data, let palingenesis drop broken rows and duplicates and report what your data holds — same config, one extra command, no GPU:

    ```bash
    pgs prepare --config configs/quickstart.yaml
    pgs train --config configs/quickstart.yaml --preprocess.enabled true
    ```

    See the [Data Preparation guide](../guides/data.md).

## 4. What happened?

Behind the scenes, palingenesis applied:

- **Chat-template masking** — only assistant turns get loss (tool calls and end-of-turn tokens included)
- **Cross-entropy loss** on those tokens (token-weighting objectives such as DEFT are [opt-in](../reference/loss.md))
- **Power-decay LR** schedule
- **Chunked loss** — the full logits are never materialized, whatever the vocabulary
- **Length-grouped batches** — rows of similar length batched together, so little compute goes to padding
- **Best-model tracking** — with an `eval_dataset`, the lowest-eval-loss checkpoint is saved to `best/`

All without configuring anything.

## Next

- [Your first real training](first-training.md)
- [Single GPU optimization](../guides/single-gpu.md): what fits, and `pgs fit` for the fastest memory setting
- [LoRA](../guides/lora.md): adapters on a frozen base (`pgs init lora`)
- [Autopilot mode](../guides/autopilot.md)
- [CLI reference](../reference/cli.md) and [configuration reference](../reference/configuration.md)
