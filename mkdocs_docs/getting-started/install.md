# Installation

---

## Prerequisites

1. **Linux with an NVIDIA GPU.** The default install targets CUDA 12.9, which needs an NVIDIA driver ≥ 575 (`nvidia-smi` shows the driver version).
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

    This installs exactly the versions in `uv.lock`, the tested set. On Linux, torch comes from PyTorch's CUDA 12.9 index automatically (configured in `pyproject.toml`).

    Training Qwen3.5 or another model with linear-attention layers? Add `--extra hybrid`: it compiles causal-conv1d against your torch (a few minutes, needs `nvcc`), which makes those layers faster and is required to pack their sequences.

=== "uv pip (into your own venv)"

    ```bash
    git clone https://github.com/mii-llm/palingenesis.git
    cd palingenesis
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -e ".[train,logging]"
    ```

    `uv pip` needs an active virtual environment. It also reads the CUDA 12.9 torch source from `pyproject.toml`.

=== "pip"

    ```bash
    git clone https://github.com/mii-llm/palingenesis.git
    cd palingenesis
    python -m venv .venv && source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu129   # FIRST: CUDA build of torch
    pip install -e ".[train,logging]"
    ```

    Install torch first. Plain pip ignores uv's index configuration, so it would otherwise install PyPI's default torch. The same applies to installing a built wheel: wheel metadata cannot name a package index.

!!! warning "Torch installs, but sees no GPU?"
    PyPI's default Linux torch wheels target the newest CUDA (torch 2.14 needs CUDA 13.0, i.e. driver ≥ 580). On an older driver, `torch.cuda.is_available()` is `False`, or you get "The NVIDIA driver on your system is too old". Install torch from a CUDA index your driver supports:

    ```bash
    uv pip install torch --index-url https://download.pytorch.org/whl/cu129   # driver >= 575
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # driver >= 570, without the vllm extra (vLLM 0.26 has no cu128 build)
    ```

!!! warning "`cannot import name 'ScalingType' from 'torch.nn.functional'`"
    torchao (from the `float8` extra) is newer than your torch. transformers imports torchao whenever it's installed, so model creation fails. Upgrade torch as above rather than pinning an old one, or uninstall torchao if you don't train in float8.

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# e.g. 2.11.0+cu129 True NVIDIA A100-SXM4-80GB
```

---

## Optional extras

| Extra | What it adds | Install when... |
|-------|-------------|-----------------|
| `train` | Liger Kernel, flash-linear-attention (Linux), bitsandbytes | Always (core training dependencies) |
| `hybrid` | causal-conv1d (compiled against your torch: a few minutes, needs `nvcc`) | Qwen3.5 / Qwen3-Next: ~25% faster, and required for `data.packing` on these models |
| `float8` | torchao | Float8 training (`memory.float8_training`) on H100/B200 |
| `vllm` | vLLM 0.26 (CUDA 12.9 build) with the matching torchvision, torchaudio and torchcodec (Linux) | Fast rollouts and vLLM teachers in [`pgs distill`](../guides/distillation.md). The server rollout backend (`rollout.backend: vllm_server`, experimental) also needs `uv pip install ray`: vLLM 0.26's CUDA IPC weight transfer imports it |
| `logging` | wandb, trackio | You want experiment tracking dashboards |
| `loss` | Cut Cross-Entropy (Triton kernel) | Training models with 256K+ vocabulary (Gemma) |
| `optim` | ScheduleFree | Using schedule-free mode |
| `prepare` | sentence-transformers | Running TFP semantic packing during data prep |

---

## Verify

```bash
pgs version
```

Expected (versions from `uv.lock`):

```
palingenesis 0.3.0
  PyTorch: 2.11.0+cu129
  Transformers: 5.12.1
  CUDA: NVIDIA A100-SXM4-80GB (80 GB)
  Liger Kernel: installed
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
