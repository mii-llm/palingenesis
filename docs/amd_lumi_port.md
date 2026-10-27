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

### 2.3 Dependency stack (`pyproject.toml`) — DONE (feature 3)
uv has no marker to tell ROCm from CUDA on Linux, so the ROCm path is **explicit** rather
than auto-detected — and the CUDA pins are left **exactly as they were** (zero NVIDIA
regression):
- Added a **`rocm` extra** = ROCm-safe, pure-Python deps only (`logging, optim, prepare,
  rl, mcp`). It deliberately excludes the CUDA-only extras (float8/torchao, hybrid/
  causal-conv1d, vllm) and the bitsandbytes/Triton pieces in `train` (validate later).
- ROCm install = **plain pip** after a ROCm torch is present: `pip install -e ".[rocm]"`.
  Plain pip ignores `[tool.uv.sources]`, so the cu129 pins never bite; the ROCm torch is kept.
- Documented in `mkdocs_docs/getting-started/install.md` (AMD / ROCm section). macOS/Windows
  unchanged.

**transformers 5.10 vs ≥5.12 (resolved):** floor **kept at `>=5.12`** (it is a real need:
cache classes for SeCO/DPO, and Qwen3.5). Verified on LUMI: palingenesis's core modules
(`train`, `data`, `seco`, `dpo`) **import fine under the container's 5.10.1**, so basic SFT
is not blocked. For the 5.12-only features, overlay a newer transformers via
`pip install --target=<scratch> "transformers>=5.12,<6"` + `PYTHONPATH` (its deps bump
numpy/hub, which can clash with vLLM — fine for SFT, revisit for rollouts), or rebuild the
container. `--user` does **not** work (the container python is a venv).

### 2.4 Per-dependency verdict
| Dependency | Extra | AMD / ROCm status |
|---|---|---|
| torch | core | ✅ ROCm wheels / ROCm container torch. |
| transformers / datasets / accelerate | core | ✅ pure Python. |
| bitsandbytes | train | ⚠️ ROCm backend less mature → optional; default `adamw`/`muon`. |
| liger-kernel | train | ✅ validated on gfx90a (qwen3): RMSNorm/SwiGLU/RoPE fused kernels compile on Triton 3.6.0 and train; its CuteDSL/Ascend RoPE backends fail to import but Liger falls back internally (benign). |
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
3. **[Track A] ROCm install path** in `pyproject.toml` (`rocm` extra, documented pip flow); CUDA-only extras gated. ✅ *done — CUDA pins untouched.*
4. **[Track A] Single-device SFT MVP** — bf16, `sdpa`, `adamw`; no FP8/bnb/causal-conv1d. Core loop + loss + checkpoint. ✅ *done — validated on a LUMI MI250X GCD (loss → ~0, HF checkpoint saved). `configs/rocm_smoke.yaml` + `scripts/lumi/sft_single_gcd.sh`.*
5. **[Track B] LUMI bring-up** — prove step 4 inside the LUMI container on one GCD (`scripts/lumi/`).
6. **[Track A] Multi-GCD / multi-node** — FSDP over RCCL (generic), validated on LUMI Slingshot via Track B env. ✅ *done — 2-GCD (`train_multi_gcd.sh`) and 2-node×2-GCD (`train_multi_node.sbatch`, with `singularity-AI-bindings` for Slingshot) both train & checkpoint. OFI/CXI fast-path vs socket fallback not yet confirmed (needs NCCL_DEBUG=INFO) — perf follow-up.*
7. **[Track A] Triton kernels** (liger, flash-linear-attention, CCE) — validate, keep fallbacks. ✅ *done for liger (works on gfx90a) and CCE (crashed on ROCm — tf32 Triton kernel; now gated on `is_cuda()` → falls back to chunked CE, validated). flash-linear-attention pending (needs a Qwen3.5 model + transformers≥5.12; transformers uses a pure-torch fallback when absent).*
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

*Status: **Milestone 1 reached** — Track A steps 1–4 done (accelerator detection, capability
gates, ROCm install path, single-device SFT) and validated on a LUMI MI250X GCD: SFT loop
runs in bf16/SDPA/AdamW, loss converges, HF checkpoint saved. Next: step 6 — multi-GCD /
multi-node FSDP over RCCL — DONE (2-GCD and 2-node validated on LUMI). Next: step 7 —
Triton kernels — liger validated on ROCm, CCE now gated to NVIDIA (falls back to chunked
CE on AMD), flash-linear-attention still pending. Next: step 8 — rollout backend → generic
ROCm vLLM for distill/RL (validate against LUMI's working vLLM).*
