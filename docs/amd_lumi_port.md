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
| `train.py` (attn load) | `attn_implementation=flash_attention_2` crashed at load without a flash-attn build | ✅ done: load now retries with `sdpa` on an FA2 failure (warns). FA2 itself **works on the LUMI container's ROCm flash-attn** (validated on gfx90a); the fallback covers ROCm boxes without it. `sdpa` stays the default. |

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
| bitsandbytes | train | ✅ validated on gfx90a (0.50.2): `adamw8bit`, `lion8bit`, `paged_adamw8bit` all step correctly (params update, finite loss). ROCm needs a ROCm-capable build; absent → clear error, use `adamw`. |
| liger-kernel | train | ✅ validated on gfx90a (qwen3): RMSNorm/SwiGLU/RoPE fused kernels compile on Triton 3.6.0 and train; its CuteDSL/Ascend RoPE backends fail to import but Liger falls back internally (benign). |
| flash-linear-attention | train | ⚠️ Triton; fallback is 6–20× slower but correct. |
| causal-conv1d | hybrid | ❌ CUDA build; needs HIP build/fork. Only gates Qwen3.5 *packing*. |
| torchao (float8) | float8 | ❌ N/A on ROCm (no FP8 path we target). Keep gated off. |
| vLLM | vllm | ✅ **ROCm vLLM works** (proven on LUMI). Target generic ROCm vLLM; see 2.5. |
| quack-kernels | — | ❌ NVIDIA-only; must stay inactive on ROCm (already falls back). |

### 2.5 Rollout backend for distill / RL — on-policy works on ROCm (colocate)
`opd/rollout.py`, `opd/teachers.py`, `rl/` talk to vLLM. Three backends: `hf` (generate with
the training model — ROCm-OK, no vLLM), `vllm` (in-process colocate), `vllm_server` (separate
server). The split that matters on ROCm:
- **Generation / frozen-teacher scoring over HTTP** (`VLLMServer.complete` → `/v1/completions`,
  `VLLMTeacher.score` via `prompt_logprobs`): backend-agnostic, **works on ROCm** —
  ✅ validated by driving palingenesis's own `VLLMServer` client against a ROCm vLLM serving
  Qwen3-0.6B on a LUMI GCD (generation + prompt_logprobs round-trip). This is the path that
  reuses LUMI's working vLLM for teachers / off-policy.
- **On-policy via the in-process `vllm` colocate backend** (`VLLMColocateRollout`): ✅
  **validated on gfx90a** — generate, vLLM sleep/wake (the CuMemAllocator freed ~25 GiB and
  woke in 0.08 s), and the on-policy weight update (`apply_model` → `model.load_weights`, 226
  tensors) all work **in-process, with no CUDA IPC**. So **on-policy OPD/RL runs on ROCm via
  the colocate backend** (`rollout.backend='vllm'`), no code change.
- **On-policy via a separate server** (`VLLMServerRollout.update_weights`): pushes weights
  over **CUDA IPC** (`reduce_tensor` handles) — NVIDIA-only. **Guarded on `is_cuda()`** (clear
  ROCm error). A non-IPC ROCm weight-sync for this *server* topology (RCCL broadcast /
  `collective_rpc`) is now **optional** — colocate already covers on-policy on AMD.

No LUMI assumptions live in this code — we test against LUMI's instance, we don't depend on it.

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
8. **[Track A] Rollout backend → generic ROCm vLLM** for distill/RL; validated against LUMI's vLLM. ◑ *off-policy / frozen-teacher HTTP path done + validated on gfx90a; on-policy CUDA-IPC weight sync guarded to NVIDIA (ROCm → use `hf`). Remaining: a non-IPC ROCm weight-sync for on-policy vLLM, and validating the in-process `vllm` colocate backend on ROCm.*
9. *(optional)* bitsandbytes 8-bit optimizers — ✅ *validated on gfx90a (all three step correctly)*; causal-conv1d HIP build for Qwen3.5 packing — pending.

**Milestone 1:** steps 1–4 = "SFT trains correctly on a single ROCm device, NVIDIA path unchanged."

---

## 5. Open questions
- Which ROCm version do we standardize Track A testing on (matching LUMI's container)?
- Do the LUMI container's torch/transformers satisfy `pyproject.toml` bounds (`torch>=2.11`,
  `transformers>=5.12`)? If not, which gives?
- Milestone-1 success = "it runs" or "loss matches a CUDA reference within tolerance"?
- Do we ever target MI300/gfx942 FP8, or is `supports_fp8()==False on ROCm` permanent?

---

*Status (all validated on a LUMI MI250X / gfx90a): steps 1–7 done — accelerator detection,
FP8/quack guards, ROCm install path, single-device SFT (bf16/SDPA/AdamW), multi-GCD &
multi-node FSDP over RCCL+Slingshot, liger kernels (work) and CCE (gated to NVIDIA),
bitsandbytes 8-bit (work), flash_attention_2 (works on the container's ROCm flash-attn;
falls back to sdpa otherwise). **Real-model SFT validated**: Qwen3-4B full fine-tune on one
64 GB GCD (adamw8bit fits 4B, ~1750 tok/s, stable grad norms, HF checkpoint) — `configs/
rocm_qwen3_4b.yaml` + `scripts/lumi/sft_qwen3_4b.sh`. **DPO validated** on gfx90a
(`configs/rocm_dpo_smoke.yaml`): frozen reference + chunked-logprob margins, all
device-agnostic torch — reward accuracy 0→100%, chosen−rejected margin 0→~1.8, loss
0.69→0.15; config uses `torch_dtype: float32` (fp32 master) per the framework's DPO
guidance (bf16 weights round DPO's ~1e-6 updates to zero). **SeCO validated** on gfx90a
(`configs/rocm_seco_smoke.yaml`): chunk-by-chunk forward matches the full forward to
relative diff 7.9e-6 (< 1e-4 fp32 tol), then trains and checkpoints — SeCO's varlen flash
op (`aten._flash_attention_forward`) already runs on ROCm (used by every SFT run). fp32
weights + single-process, as the framework requires. **vLLM rollouts on ROCm**: off-policy /
frozen-teacher over HTTP works; the in-process colocate backend (`rollout.backend='vllm'`)
works end-to-end on gfx90a — generate, sleep/wake (CuMemAllocator), and on-policy weight
update (`apply_model`/`load_weights`, no IPC) — so **on-policy OPD/RL runs on AMD** via
colocate. The separate-server CUDA-IPC weight sync stays guarded to NVIDIA. **End-to-end OPD
distill validated** on gfx90a (`configs/rocm_opd_smoke.yaml`, Qwen3-1.7B→0.6B): the full
on-policy loop runs — colocate vLLM sampling (~609 tok/s) + per-step sleep/wake (freed 19 GiB,
woke ~10 ms) + hf teacher scoring + exact full-vocab reverse KL (residual mass ~1e-9) +
weight sync, staleness 0, dev KL reported, checkpoint saved. **Qwen3.5 (hybrid
linear-attention) validated** on gfx90a (`configs/rocm_qwen35_08b.yaml`, Qwen3.5-0.8B): the
container's transformers 5.10.1 already registers `qwen3_5`, and it trains via the
**pure-torch linear-attention fallback** (loss 0.64→0.31, checkpoint; ~6–20× slower than fla,
as warned). **flash-linear-attention acceleration is deferred**: a simple overlay install of
`flash-linear-attention` 0.6.0 came up incomplete and a *broken* fla makes Qwen3.5 loading
hard-fail (fla *absent* falls back cleanly) — it needs a known-good ROCm fla build / matching
version, a follow-up. **RL (GRPO) validated** on gfx90a (`configs/rocm_rl_smoke.yaml`): the
full loop runs — colocate vLLM rollouts (~1000 tok/s) + pure-Python reward + GRPO advantages
+ clipped policy update (informative groups give grad_norm up to ~300) + per-step sleep/wake
+ checkpoint. **Margin surgery core validated**: `pgs surgery margins` computes decision-token
margins on gfx90a (the `margins.py` forward machinery that **CMR** also rests on; full
CMR-in-OPD and `surgery run` weight-edit reuse that validated core + device-agnostic torch —
a light follow-up, they need curated tool-calling twins). Remaining: CI wiring; a known-good
ROCm fla build. (Non-IPC server-topology weight sync is optional, not a blocker.)*
