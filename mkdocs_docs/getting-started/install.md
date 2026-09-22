# Installation

---

## Prerequisites

1. **Linux with an NVIDIA GPU.** The default install targets CUDA 12.8, which needs an NVIDIA driver ≥ 570 (`nvidia-smi` shows the driver version).
2. **[uv](https://docs.astral.sh/uv/)** (recommended): `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. **Python 3.11 or 3.12.** uv downloads one if needed.

---

## Install

=== "uv sync (recommended)"

    ```bash
    git clone https://github.com/mii-llm/palingenesis.git
    cd palingenesis
    uv sync --extra train --extra logging   # creates .venv, installs the locked versions
    source .venv/bin/activate
    ```

    This installs exactly the versions in `uv.lock`, the tested set. On Linux, torch comes from PyTorch's CUDA 12.8 index automatically (configured in `pyproject.toml`).

=== "uv pip (into your own venv)"

    ```bash
    git clone https://github.com/mii-llm/palingenesis.git
    cd palingenesis
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -e ".[train,logging]"
    ```

    `uv pip` needs an active virtual environment. It also reads the CUDA 12.8 torch source from `pyproject.toml`.

=== "pip"

    ```bash
    git clone https://github.com/mii-llm/palingenesis.git
    cd palingenesis
    python -m venv .venv && source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu128   # FIRST: CUDA build of torch
    pip install -e ".[train,logging]"
    ```

    Install torch first. Plain pip ignores uv's index configuration, so it would otherwise install PyPI's default torch. The same applies to installing a built wheel: wheel metadata cannot name a package index.

!!! warning "Torch installs, but sees no GPU?"
    PyPI's default Linux torch wheels target the newest CUDA (torch 2.14 needs CUDA 13.0, i.e. driver ≥ 580). On an older driver, `torch.cuda.is_available()` is `False`, or you get "The NVIDIA driver on your system is too old". Install torch from a CUDA index your driver supports:

    ```bash
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # driver >= 570
    ```

!!! warning "`cannot import name 'ScalingType' from 'torch.nn.functional'`"
    torchao (from the `train` extra) is newer than your torch. transformers imports torchao whenever it's installed, so model creation fails. palingenesis requires torch ≥ 2.11 for this reason. Upgrade torch as above rather than pinning an old one.

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# e.g. 2.11.0+cu128 True NVIDIA A100-SXM4-80GB
```

---

## Optional extras

| Extra | What it adds | Install when... |
|-------|-------------|-----------------|
| `train` | Liger Kernel (Linux), bitsandbytes, torchao | Always (core training dependencies) |
| `logging` | wandb, trackio | You want experiment tracking dashboards |
| `loss` | Cut Cross-Entropy (Triton kernel) | Training models with 256K+ vocabulary (Gemma) |
| `optim` | ScheduleFree | Using schedule-free mode |
| `prepare` | sentence-transformers | Running TFP semantic packing during data prep |

---

## Verify

```bash
pgs version
```

Expected:

```
palingenesis 0.3.0
  PyTorch: 2.7.0+cu124
  Transformers: 4.52.0
  CUDA: NVIDIA A100-SXM4-80GB (80 GB)
  Liger Kernel: 0.5.2
```

If you see `CUDA: not available`, your PyTorch installation doesn't have CUDA support. Reinstall PyTorch from the [official instructions](https://pytorch.org/get-started/locally/).

---

## Supported models

Any HuggingFace causal language model works. Tested and optimized for:

| Family | Models | Notes |
|--------|--------|-------|
| Qwen 3.5 | 0.8B, 4B, 35B-MoE | Hybrid attention-recurrent. Use `freeze_non_attention: true`. |
| Qwen 3 | 0.8B, 4B | Standard Transformer. |
| Qwen 2.5 | 1.5B, 3B, 7B, 14B, 72B | Standard Transformer. |
| Llama 3 | 8B, 70B | Standard Transformer. |
| Gemma 4 | 2B, 4B, 12B | 262K vocabulary — use `cut-cross-entropy` or chunked loss. |
| Mistral | 7B, 8x7B MoE | Standard Transformer. |

Any model loadable with `AutoModelForCausalLM.from_pretrained()` and supporting a chat template works. Palingenesis detects the architecture and applies appropriate optimizations (Liger kernels, activation checkpointing patterns, compile compatibility).

---

## Supported data formats

Your training data must be in one of:

- **JSONL** with chat messages (recommended):
  ```json
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ```

- **HuggingFace Dataset** with a messages field:
  ```bash
  pgs train --config config.yaml  # config points to HF dataset name
  ```

Multi-turn conversations (any number of user/assistant/system turns) are fully supported. Tool-calling traces with `tool` and `observation` roles work too — enable `include_observations: true` to train on tool outputs.

---

## What if I don't have a GPU?

Data preparation (`pgs prepare`) works on CPU but is very slow (~10× slower than GPU). Training requires a GPU — there's no CPU training mode because it would take weeks and produce an inferior result.

For GPU access without buying hardware: [Lambda Cloud](https://lambdalabs.com/), [RunPod](https://www.runpod.io/), or [Vast.ai](https://vast.ai/) offer A100/H100 instances at $1-3/hour.
