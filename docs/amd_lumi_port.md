# AMD / ROCm support for palingenesis — Port Analysis & Plan

> **Branch:** `amd-support` (cut from `main` @ `0f763dc`, 2026-10-26)
> **Goal:** make palingenesis run on **any AMD / ROCm GPU machine** — a portable
> deliverable, not a LUMI fork.
> **LUMI is only the test bed** — the sole AMD machine we have to validate against.
> **Method:** port **feature by feature**, each a self-contained commit that either works
> on ROCm or degrades cleanly, and that **never regresses the NVIDIA path**.

---

## 0. The guiding principle — two tracks, kept separate

This effort is deliberately split in two, and the split is enforced on every change:

### Track A — AMD/ROCm support *(the product; lives in the codebase)*
Written against **generic ROCm**, detected at runtime, never against a specific site.
A user on an MI300 workstation with no Slurm must be able to install and train.

### Track B — LUMI validation harness *(the test rig; lives in scripts/ + docs only)*
All HPC/site specifics — Slurm/srun binding, the Apptainer `.sif` image, Slingshot +
`aws-ofi-rccl`, `/scratch/...` paths, module environments — are **allowed only here**,
clearly fenced off, and `core palingenesis never imports or assumes them`.

> **Litmus test for any change:** *would it still make sense on a random MI300 box with no
> Slurm?* If yes → Track A (core). If it names LUMI, Slingshot, Apptainer or a scratch
> path → Track B (test rig). Nothing from B may leak into A.

---

## 1. The good news: the device layer is already portable

PyTorch's ROCm build **masquerades as CUDA** — `torch.cuda.*`, `device="cuda"`,
`init_device_mesh("cuda", …)`, and `backend="nccl"` all work unchanged on AMD (they
dispatch to HIP / RCCL under the hood). So the ~76 `torch.cuda.*` call sites and the
hardcoded `"cuda"`/`"nccl"` strings in [`distributed.py`](../src/palingenesis/distributed.py)
are **not** blockers. Most of the framework imports and runs once the right wheels are present.

The real work is concentrated in **(1)** the NVIDIA-pinned **dependency stack** and **(2)** a
handful of **capability gates and compiled/Triton kernels** that assume CUDA semantics.

---

## 2. Track A — the AMD/ROCm deliverable

### 2.1 Runtime capability detection *(foundation — do first)*
Every ROCm-aware decision must key off **`torch.version.hip is not None`**, not a hostname
or `sys_platform` alone. Centralized in [`accelerator.py`](../src/palingenesis/accelerator.py):
`is_available() / is_rocm() / is_cuda() / backend() / supports_fp8() / describe()`.
This is the primitive the guards below consume.

### 2.2 Capability gates to fix
| Site | Today | ROCm problem | Fix |
|---|---|---|---|
| `float8.py` | `torch.cuda.get_device_capability() >= (8,9)` | On ROCm, `get_device_capability()` returns gfx-derived numbers (gfx90a → `(9,0)`), so the gate **wrongly passes** on a GPU with **no FP8**. | Route through `accelerator.supports_fp8()` → **False on ROCm**. |
| `autopilot/profile.py` | same `(8,9)` check for `supports_float8` | same false-positive | same |
| `muon.py` (quack) | `get_device_capability()[0] < 9` | quack is NVIDIA CuTeDSL; must **not** activate on ROCm | guard with `is_cuda()` before the sm check |
| `kernels.py` (selective AC) | matches `aten._flash_attention_forward` op names | ROCm SDPA may dispatch to different ops → save-list silently misses | verify op names resolve on ROCm; widen if needed |

### 2.3 Dependency stack (`pyproject.toml`) — the big one
Today Linux forces the **CUDA 12.9** torch/vLLM indexes. AMD-agnostic approach:
- Make the CUDA index **not unconditional** — add a ROCm install path (a `rocm` extra and/or
  a documented "install ROCm torch first" flow, mirroring the existing pip fallback).
- Gate CUDA-only extras so a ROCm install doesn't try to pull them: `torchao` (float8 — N/A),
  and treat `bitsandbytes`, `causal-conv1d`, Triton kernels as **opt-in / validate-per-kernel**.
- Keep macOS/Windows on PyPI as today.

### 2.4 Per-dependency verdict
| Dependency | Extra | AMD / ROCm status |
|---|---|---|
| torch | core | ✅ ROCm wheels / ROCm container torch. |
| transformers / datasets / accelerate | core | ✅ pure Python. |
| bitsandbytes | train | ⚠️ ROCm backend less mature → optional; default `adamw`/`muon`. |
| liger-kernel | train | ⚠️ Triton-on-ROCm; validate per kernel, pure-torch fallback exists. |
| flash-linear-attention | train | ⚠️ Triton; fallback is 6–20× slower but correct. |
| causal-conv1d | hybrid | ❌ CUDA build; needs HIP build/fork. Only gates Qwen3.5 *packing*. |
| torchao (float8) | float8 | ❌ N/A on ROCm (no FP8 path we target). Keep gated off. |
| vLLM | vllm | ✅ **ROCm vLLM works** (proven on LUMI). Target generic ROCm vLLM; see 2.5. |
| quack-kernels | — | ❌ NVIDIA-only; must stay inactive on ROCm (already falls back). |

### 2.5 Rollout backend for distill / RL
`opd/rollout.py`, `opd/teachers.py`, `rl/` talk to vLLM. ROCm vLLM is a solved problem
(validated on LUMI), so this is **integration, not a porting epic**: ensure palingenesis's
rollout backend speaks to a generic ROCm vLLM over its normal interface, with **no LUMI
assumptions** in the code. We *test* against LUMI's instance; we don't depend on it.

### 2.6 Hardware reality for the guards
- **No FP8** on CDNA2 (gfx90a / MI250X). bf16 is the ceiling there. `supports_fp8()` returns
  False on all ROCm for now (revisit only if we explicitly target MI300/gfx942 FP8).
- A core, generic **ROCm Dockerfile** is fair game for Track A (reproducible install);
  a *LUMI* `.sif` is Track B.

---

## 3. Track B — LUMI validation harness (kept out of core)

LUMI is where we run Track A to prove it. Facts, for the test scripts/docs only:
- **GPU:** MI250X / **gfx90a** / CDNA2; each card = **2 GCDs** → torch sees **8 × 64 GB** per node.
- **Container:** Apptainer image (e.g. `/appl/local/laifs/containers/lumi-multitorch-latest.sif`)
  carrying ROCm torch + vLLM — our test vehicle, **not** palingenesis's install story.
- **Interconnect:** Slingshot-11 → RCCL + `aws-ofi-rccl` (not InfiniBand; `NCCL_IB_*` irrelevant).
- **Scheduler:** Slurm on Cray EX; login nodes have internet + a RAM cap, compute nodes have
  GPU/RAM but no internet.
- **Validated vLLM launch:** `--tensor-parallel-size <TP> --dtype bfloat16 --trust-remote-code
  --gpu-memory-utilization …` (bf16, no FP8 — consistent with gfx90a).
- Lands as `scripts/lumi/*` + a LUMI section in the install docs, explicitly labeled
  "validation environment, not a requirement."

---

## 4. Feature-by-feature order

Each step is an independent commit; NVIDIA behaviour stays intact throughout.

1. **[Track A] Accelerator detection** (`accelerator.py` + tests) — the foundation. ← *starting here*
2. **[Track A] Fix capability gates** — float8 & profile refuse FP8 on ROCm; quack guarded by `is_cuda()`.
3. **[Track A] ROCm install path** in `pyproject.toml` (ROCm extra / documented flow); CUDA-only extras gated.
4. **[Track A] Single-device SFT MVP** — bf16, `sdpa`, `adamw`; no FP8/bnb/causal-conv1d. Core loop + loss + checkpoint.
5. **[Track B] LUMI bring-up** — prove step 4 inside the LUMI container on one GCD (`scripts/lumi/`).
6. **[Track A] Multi-GCD / multi-node** — FSDP over RCCL (generic), validated on LUMI Slingshot via Track B env.
7. **[Track A] Triton kernels** (liger, flash-linear-attention) — validate, keep fallbacks.
8. **[Track A] Rollout backend → generic ROCm vLLM** for distill/RL; validated against LUMI's vLLM.
9. *(optional)* bitsandbytes 8-bit optimizers; causal-conv1d HIP build for Qwen3.5 packing.

**Milestone 1:** steps 1–4 = "SFT trains correctly on a single ROCm device, NVIDIA path unchanged."

---

## 5. Open questions
- Which ROCm version do we standardize Track A testing on (matching LUMI's container)?
- Do the LUMI container's torch/transformers satisfy `pyproject.toml` bounds (`torch>=2.11`,
  `transformers>=5.12`)? If not, which gives?
- Milestone-1 success = "it runs" or "loss matches a CUDA reference within tolerance"?
- Do we ever target MI300/gfx942 FP8, or is `supports_fp8()==False on ROCm` permanent?

---

*Status: Track A step 1 (accelerator detection) in progress on this branch.*
