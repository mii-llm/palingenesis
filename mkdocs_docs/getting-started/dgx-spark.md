# Training on NVIDIA DGX Spark

!!! warning "Not tested on hardware"
    No palingenesis run has been made on a DGX Spark. This page is put together from NVIDIA's and PyTorch's documentation, the packages' published wheels and the library's own code; the unit tests cover the Spark-specific code paths with a simulated GB10. Sentences marked *expected* are inferences. If you run it, please [report](https://github.com/mii-llm/palingenesis/issues) what you see.

DGX Spark is a desktop machine built on the GB10 chip: a 20-core Arm CPU and a Blackwell GPU that share 128 GB of LPDDR5x memory (273 GB/s) ([hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)). Three things differ from a server with an A100 or H100:

| | DGX Spark | What it changes |
|---|---|---|
| CPU architecture | aarch64 (Arm) | every compiled dependency needs an aarch64 build |
| GPU compute capability | 12.1 (`sm_121`) ([CUDA GPU list](https://developer.nvidia.com/cuda/gpus)) | kernels built for `sm_120` run on it; `sm_121` is rarely a build target of its own |
| Memory | one pool for CPU and GPU ("unified") | "GPU memory" is the host's RAM: memory readings and budgets mean something else |

The system ships DGX OS 7 with NVIDIA driver 580 and the CUDA 13.0 toolkit ([release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)).

---

## Install

The standard install from source is *expected* to work unchanged:

```bash
git clone https://github.com/mii-llm/palingenesis.git
cd palingenesis
uv sync --extra train --extra logging
source .venv/bin/activate
pgs version
#   GPU: 1x NVIDIA GB10 (122 GB, shared with the host), CUDA 12.9, arch sm_121     (expected)
```

Why it should:

- **torch.** `uv.lock` pins `torch 2.11.0+cu129` and the lock holds its `manylinux_2_28_aarch64` wheels (PyTorch publishes aarch64 CUDA wheels on the [cu129](https://download.pytorch.org/whl/cu129/torch/) and [cu130](https://download.pytorch.org/whl/cu130/torch/) indexes). The wheel brings its own CUDA 12.9 libraries and needs driver ≥ 575; the Spark has 580.
- **`sm_121`.** `torch.cuda.get_arch_list()` ends at `sm_120` and has no `sm_121` entry. That is fine: PyTorch's NVIDIA maintainer states that `sm_121` is binary compatible with `sm_120` and that "all of our binaries for ARM built with CUDA >= 12.8 support DGXSpark already" ([PyTorch forum](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744)).
- **Triton kernels** (Liger Kernel, flash-linear-attention, Cut Cross-Entropy, palingenesis's own). All four are pure-Python packages that compile at first use through Triton, and Triton 3.6.0 has an aarch64 wheel in the lock.
- **bitsandbytes** (8-bit optimizers) publishes aarch64 wheels with `sm120` targets, and `sm121` in its CUDA 13 build ([installation guide](https://huggingface.co/docs/bitsandbytes/main/en/installation)).

With pip instead of uv, install torch first, as on any Linux machine ([Installation](install.md)). Either CUDA build is *expected* to work:

```bash
pip install "torch>=2.11" --index-url https://download.pytorch.org/whl/cu129   # the tested version set
pip install "torch>=2.11" --index-url https://download.pytorch.org/whl/cu130   # CUDA 13, matches the system toolkit
pip install "palingenesis[train,logging]"
```

### Known rough edges

- **Triton and `ptxas`.** Triton compiles with the `ptxas` it bundles, and older bundles reject the Spark's target with `Value 'sm_121a' is not defined for option 'gpu-name'` ([NVIDIA forum](https://forums.developer.nvidia.com/t/running-llama-3-1-8b-fp4-get-triton-error-value-sm-121a-is-not-defined-for-option-gpu-name/348808)). If a kernel fails that way, point Triton at the system's CUDA 13 assembler: `export TRITON_PTXAS_PATH="$(which ptxas)"`. Whether Triton 3.6.0 still needs this was not verified.
- **causal-conv1d** (the `hybrid` extra, for Qwen3.5 / Qwen3-Next) has no wheels: it is compiled on the machine with `nvcc` against the installed torch. The Spark's `nvcc` is CUDA 13, so build it against a `cu130` torch rather than the locked `cu129` one (*expected*; mixing the two is untested). People have built it on a Spark for torch 2.11 + cu130 ([example](https://huggingface.co/N8Programs/dgx-causal-conv1d-wheel)). Without it Qwen3.5 trains with transformers' slower convolution and cannot use `data.packing`.
- **vLLM** (the `vllm` extra: rollouts for RL and on-policy distillation). The locked `vllm 0.26.0+cu129` has an aarch64 wheel. vLLM documents aarch64 builds for Grace-Blackwell systems and needs compute capability ≥ 7.5 ([vLLM installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)); on the Spark specifically, earlier releases needed `enforce_eager` for correct output and some FP8/FP4 kernels had no `sm_12x` variant ([NVIDIA forum](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)). If rollouts look wrong, set `rollout.enforce_eager: true` first.
- **Float8 training** (`memory.float8_training`). GB10 passes PyTorch's capability test for FP8 matmuls, but no palingenesis run has validated it there. It is therefore never switched on automatically (autopilot), and when a config asks for it the run warns and goes on: compare the loss with a bf16 run.
- **`nvidia-smi`** prints `Memory-Usage: Not Supported` on the Spark; NVIDIA lists this as expected for an integrated GPU ([known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)). The helper scripts in `scripts/` print the GPU's memory from `nvidia-smi` and will show `[N/A]`; training does not parse `nvidia-smi`.

---

## Unified memory

On the Spark the GPU allocates from the host's RAM. PyTorch reports the whole pool as the device's memory (about 122 GiB after the firmware's display reservation, which can be lowered in the UEFI setup: [porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/optimization.html)), and everything else on the machine draws on the same pool: the operating system, the training process's own RAM (Python, the tokenized dataset, data-loader workers), a vLLM engine, a browser.

Two readings are off as a result, and palingenesis corrects for both on GPUs that CUDA reports as integrated:

- **Free memory.** CUDA's free figure counts the Linux page cache as used, so after a model or a dataset has been read from disk it can be tens of GB below what an allocation would actually get ([known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)). Where the library decides by free GPU memory (margin surgery's `gram: auto`), it uses the kernel's `MemAvailable` instead.
- **The budget.** `pgs fit` without `--gpu` takes as its budget the host memory available when it starts, not the device total, and says so. Pass `--gpu` to set it yourself.

What stays as it is:

- **`pgs profile` and `pgs diagnose` assume an 80 GiB GPU** unless told otherwise (they run without a GPU). On a Spark pass a budget that leaves room for the host, for example `pgs profile --config my.yaml --gpu 100`.
- **Peaks are CUDA allocations.** `pgs fit`, `pgs profile --measure` and `health/cuda_peak_gb` report what torch allocated on the GPU. The process's host memory comes on top, from the same pool. Leave more headroom than on a discrete GPU.
- **Running out of memory may not raise.** A discrete GPU fails an allocation with `torch.OutOfMemoryError`. On unified memory the kernel can swap or stall instead, and users report a machine that stops responding rather than an error ([NVIDIA forum](https://forums.developer.nvidia.com/t/dgx-spark-becomes-unresponsive-zombie-instead-of-throwing-cuda-oom/353752)). The out-of-memory candidates of `pgs fit` rely on that error: start it with a `--gpu` budget well under the pool the first time.
- **vLLM's `gpu_memory_utilization`** (`rollout.gpu_memory_utilization`, a teacher's) is a share of the whole pool. Colocated with training, the trainer, the engine and the host must fit together: 0.3 is about 36 GiB.
- **`train.cpu_offload`** (RL) moves optimizer state to host RAM, which is the same memory here. The trainer warns; turn it off.
- If a job that fitted yesterday does not fit today, check what else holds memory (`free -h`). NVIDIA's workaround for a full page cache is `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` ([known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)).

---

## What to expect

The Spark has more memory than an 80 GB A100 or H100 and much less speed: one GPU, LPDDR5x at 273 GB/s where datacenter GPUs use HBM. Plan it as a machine for jobs that need memory more than throughput. Throughput on palingenesis has not been measured; third-party numbers give the order of magnitude:

- NVIDIA reports, at 2048 tokens: full fine-tuning of Llama 3.2 3B, LoRA on Llama 3.1 8B and QLoRA on Llama 3.3 70B. The article's table gives 13.5k, 7.0k and 0.76k tokens/s; its text gives figures about six times higher for the same runs ([NVIDIA blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks)).
- PyTorch's blog reports a full fine-tune of Llama 3.1 8B in bf16 at 16k tokens and batch size 16 with torchtune: about 80% of memory at the peak and about 8 hours per epoch over 11k conversations ([PyTorch blog](https://pytorch.org/blog/unlock-reasoning-in-llama-3-1-8b-via-full-fine-tuning-on-nvidia-dgx-spark/)).

| Method | On a Spark (*expected*) |
|---|---|
| Full fine-tuning (SFT) | Small models (≤ 4B) with room to spare, long sequences included. 7–8B is at the edge: measure with `pgs fit`, and use an [8-bit optimizer](../reference/optimizers.md) or LoRA if it does not fit. |
| [LoRA](../guides/lora.md) | The practical route above 8B. The frozen base still has to fit in bf16 (2 bytes per parameter): a 30B-class model does, a 70B one (140 GB) does not. |
| [Long sequences](../guides/long-sequences.md) | `memory.seco: true`, as on any single GPU. Context parallel needs several GPUs. |
| [DPO](../guides/dpo.md) | Two sequences per row: half the batch of the same SFT config. |
| [Off-policy distillation](../guides/offpolicy-distillation.md) | An online teacher shares the pool with the student. Cache the teacher once (`pgs distill-cache`) when both do not fit. |
| [On-policy distillation](../guides/distillation.md), [RL](../guides/rl.md) with vLLM | Trainer and vLLM engine in one pool (see above); depends on vLLM working on the machine, the least certain part of the stack. |
| Float8 | Unvalidated, off by default. |
| Multi-GPU (FSDP, context parallel) | One GPU. Two Sparks linked over their network ports are not covered here. |
