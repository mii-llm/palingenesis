# Training on AMD GPUs

palingenesis runs on AMD Instinct GPUs through PyTorch's ROCm build. That build exposes AMD GPUs through the CUDA API (`torch.cuda.*`, `device="cuda"`, the `nccl` backend, which is RCCL on AMD), so configs and commands are the same as on NVIDIA. The library itself only changes the few decisions that depend on the GPU: FP8, the Cut Cross-Entropy kernel, the quack Muon kernels and attention backends (see [What works](#what-works)).

---

## Install

Install PyTorch's ROCm build **first**, then palingenesis and its extras with plain `pip`, in the same environment. pip keeps the ROCm torch because it already satisfies `torch>=2.11`.

```bash
git clone https://github.com/mii-llm/palingenesis.git
cd palingenesis
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2   # FIRST: ROCm torch (2.11+)
pip install -e ".[logging,rl]"
pip install bitsandbytes "flash-linear-attention[rocm]" kernels                     # kernels with ROCm builds
pip install --no-deps liger-kernel cut-cross-entropy                               # keep torch's ROCm Triton
python -c "import torch; print(torch.__version__, torch.version.hip)"              # must still say +rocm
```

- **Not `uv sync` and not `pip install -e ".[train]"` / `".[all]"`.** `uv.lock` and `[tool.uv.sources]` resolve torch and vLLM from the CUDA 12.9 indexes, and uv has no marker that tells a ROCm machine from a CUDA one, so `uv sync` would replace the ROCm torch with the CUDA one. (uv's [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) shows how to select a ROCm index with conflicting extras; palingenesis does not do that yet, so the lock stays unchanged for NVIDIA users.)
- **Why `--no-deps` for Liger and Cut Cross-Entropy:** their wheels require PyPI's `triton`, while ROCm torch ships its own Triton as `pytorch-triton-rocm`. Without `--no-deps`, pip installs PyPI's `triton` over it. The `train` extra installs Liger with dependencies, which is why the command above lists the packages one by one.
- **ROCm index:** `rocm7.2` has torch 2.11 and 2.12 for Python 3.12 (check [download.pytorch.org/whl/rocm7.2/torch](https://download.pytorch.org/whl/rocm7.2/torch/) for newer ones). The older `rocm6.x` indexes stop before torch 2.11. Liger and flash-linear-attention document the same index.
- **bitsandbytes** ships ROCm builds in its PyPI wheel for gfx90a, gfx942 and gfx950 ([installation guide](https://huggingface.co/docs/bitsandbytes/main/en/installation#rocm)), so the 8-bit optimizers are available.
- **flash-linear-attention** (Qwen3.5 / Qwen3-Next linear-attention layers) has a `rocm` extra that does not pin a Triton build ([README](https://github.com/fla-org/flash-linear-attention#installation)).
- **causal-conv1d** (the `hybrid` extra) is compiled against your torch; it is untested here on ROCm. Without it Qwen3.5 trains with transformers' slower convolution and cannot use `data.packing`.

### Container

`Dockerfile.rocm` builds on vLLM's official ROCm image `vllm/vllm-openai-rocm:v0.26.0` (ROCm torch, its Triton and vLLM 0.26.0, the version the `vllm` extra pins; AMD's older `rocm/vllm` images are [deprecated](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)). It installs palingenesis as above and fails the build if the ROCm torch was replaced. It is not built in CI.

```bash
docker build -f Dockerfile.rocm -t palingenesis:rocm .
docker run --rm -it --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=16g \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  -v "$PWD/configs:/config:ro" -v "$PWD/outputs:/outputs" -v "$PWD/cache:/cache" \
  palingenesis:rocm train --config /config/train.yaml
```

The image runs as UID 10001 like the [NVIDIA image](https://github.com/mii-llm/palingenesis/blob/main/docs/containers.md): `--group-add` with the host's `video` and `render` group ids lets that user open `/dev/kfd` and `/dev/dri`.

### Verify

```bash
pgs version
#   GPU: 8x AMD Instinct MI300X (192 GB), ROCm 7.2.x, arch gfx942
```

---

## What works

| | MI250 / MI250X (gfx90a) | MI300 (gfx942) | MI350 (gfx950) |
|---|---|---|---|
| bf16 SFT, DPO, OPD, RL (`rollout.backend: hf`) | yes | yes | yes |
| FSDP2 / multi-GPU (RCCL via `nccl`) | yes | yes | yes |
| `attn_implementation: sdpa` | yes (torch's ROCm SDPA) | yes | yes |
| `attn_implementation: kernels-community/aiter-flash-attn@v2` | untested | untested | untested |
| `attn_implementation: flash_attention_2` | only with a ROCm flash-attn build installed | same | same |
| Cut Cross-Entropy | chunked CE instead (see below) | yes | chunked CE instead |
| FP8 (`memory.float8_training`) | **no**: no FP8 hardware, refused | opt-in (torchao uses e4m3fnuz) | opt-in |
| Muon (`train.optimizer: muon`) | torch backend | torch backend | torch backend |
| vLLM rollouts (`rollout.backend: vllm`) | needs vLLM's ROCm build, untested | same | same |
| `rollout.backend: vllm_server` | untested, warns | untested, warns | untested, warns |

**Attention.** `sdpa` works everywhere. For flash attention on AMD use AMD's aiter kernel from the Hugging Face Hub, as [TRL recommends](https://github.com/huggingface/trl/blob/main/docs/source/amd_gpus.md); transformers loads Hub kernels given as `org/repo[@revision]` through the [`kernels`](https://github.com/huggingface/kernels) package:

```yaml
model:
  attn_implementation: kernels-community/aiter-flash-attn@v2   # pip install kernels
```

transformers runs it through its flash-attention path, so `data.packing` flattens packed rows exactly as with `flash_attention_2`. The aiter kernel has no attention-logit soft-capping, which transformers skips silently: models that rely on it (Gemma 2) would train with uncapped attention. With `flash_attention_2` and no flash-attn installed, training stops before loading the model with a config error naming these alternatives; it does not switch backend on its own. (On ROCm the `kernels` fallback transformers uses on NVIDIA, `kernels-community/flash-attn2`, has no AMD build.)

**Cut Cross-Entropy.** CCE's Triton kernels request tf32 dot products when `memory.float32_matmul_precision` is `high` (the default), and Triton's AMD backend accepts tf32 only on gfx942 ([source](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py)). On other AMD GPUs the trainer logs why and uses chunked cross-entropy instead (same loss, more memory). `memory.float32_matmul_precision: highest` makes CCE use ieee dots, which do compile there.

**FP8.** Decided by GPU architecture, not by `torch.cuda.get_device_capability()`, which on ROCm returns gfx-derived numbers (gfx90a reports (9, 0) and would pass an "SM89+" check). gfx90a has no FP8 and the trainer continues in bf16 with a warning. On gfx942 and gfx950 torchao's float8 training picks the right formats (e4m3fnuz on MI300), so an explicit `memory.float8_training: true` is honoured with a warning; autopilot never turns it on there.

**Muon.** The quack kernels are NVIDIA-only (CuTe DSL) and stay off on AMD; Muon uses its torch backend.

**vLLM.** Install vLLM's ROCm build (Python 3.12 only), or use the container:

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

vLLM supports MI200 (gfx90a), MI300 (gfx942) and MI350 (gfx950, ROCm ≥ 7.0) ([vLLM GPU install guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)). `rollout.backend: vllm` (in-process, colocated) needs nothing special. `vllm_server` pushes weights to the server through torch's CUDA IPC handles, which are HIP IPC on AMD; it runs but logs that it is untested: check that rollouts follow the updated policy (`policy/abs_ratio_dev` near 0 in RL).

---

## Environment variables

| Variable | Use |
|---|---|
| `HIP_VISIBLE_DEVICES` | Which GPUs the process sees (AMD's `CUDA_VISIBLE_DEVICES`). |
| `ROCR_VISIBLE_DEVICES` | Same, one level lower (the ROCr runtime); applies before `HIP_VISIBLE_DEVICES`, which then indexes into what it leaves. SLURM sites often set it. |
| `CUDA_VISIBLE_DEVICES` | Also honoured on ROCm. If it and `HIP_VISIBLE_DEVICES` are both set and differ, palingenesis stops at start-up (vLLM's ROCm platform refuses the same). An empty `CUDA_VISIBLE_DEVICES` does not hide GPUs while `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` is set ([TRL #7685](https://github.com/huggingface/trl/pull/7685)). |
| `PYTORCH_CUDA_ALLOC_CONF` | Allocator options, read on ROCm too. palingenesis sets `expandable_segments:True` unless it or `PYTORCH_ALLOC_CONF` is set. |

---

## Acceptance script

`scripts/acceptance_smoke.py` runs short SFT, DPO, on-policy distillation and RL runs from the repository's own configs (with overrides it writes to `<out>/configs`) on synthetic arithmetic data generated with a fixed seed, and prints losses and throughput per stage. Run it on an NVIDIA GPU and on the AMD GPU and compare the two `summary.json` files: each stage must exit 0, `loss_first` / `loss_last` should be close to the NVIDIA run's, and `tok_s_median` / `s_per_step_median` is the speed.

```bash
python scripts/acceptance_smoke.py --out runs/acceptance                        # sdpa, hf rollouts
python scripts/acceptance_smoke.py --out runs/acceptance-aiter --stages sft,dpo \
    --attn kernels-community/aiter-flash-attn@v2
python scripts/acceptance_smoke.py --out runs/acceptance-vllm --stages opd,rl --rollout vllm
```

It uses Qwen/Qwen3-0.6B (student) and Qwen/Qwen3-1.7B (OPD teacher) by default, 30 steps per stage, one GPU. `--dry-run` writes the data and configs and validates them without a GPU.
