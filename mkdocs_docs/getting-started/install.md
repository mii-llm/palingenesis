# Installation

---

## Prerequisites

Before palingenesis:

1. **NVIDIA GPU** with 24+ GB VRAM (RTX 3090, RTX 4090, A100, H100, or newer)
2. **CUDA toolkit** and a CUDA-enabled PyTorch installation
3. **Python 3.11+**

If you don't have PyTorch with CUDA yet:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA works:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print your GPU name, e.g. "NVIDIA A100-SXM4-80GB"
```

---

## Get the code

```bash
git clone https://github.com/your-org/palingenesis.git
cd palingenesis
```

---

## Install

=== "uv (recommended — fast, reproducible)"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh  # skip if you have uv
    uv pip install -e ".[train]"
    ```

=== "pip"

    ```bash
    pip install -e ".[train,logging]"
    ```

=== "Full (everything)"

    ```bash
    uv pip install -e ".[all]"
    ```

---

## Optional extras

| Extra | What it adds | Install when... |
|-------|-------------|-----------------|
| `train` | Liger Kernel, bitsandbytes, torchao | Always (core training dependencies) |
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
