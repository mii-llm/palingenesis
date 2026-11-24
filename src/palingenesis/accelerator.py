"""NVIDIA (CUDA) vs AMD (ROCm): the few decisions that differ between the two.

PyTorch's ROCm build exposes AMD GPUs through the CUDA API: ``torch.cuda.*``,
``device="cuda"`` and the ``"nccl"`` backend (RCCL) work unchanged, so almost all of
palingenesis runs as is. What does not carry over are checks written against CUDA
semantics: ``torch.cuda.get_device_capability()`` returns gfx-derived numbers on ROCm
(gfx90a -> (9, 0), gfx942 -> (9, 4)), which pass any ``>= (8, 9)`` "FP8 / Hopper" test.
The build is told apart with ``torch.version.hip`` (a version string on ROCm, None
otherwise), as PyTorch's HIP notes recommend; the GPU generation with the device's
``gcnArchName`` (e.g. "gfx942:sramecc+:xnack-").

Also here: integrated GPUs that allocate from the host's RAM (NVIDIA GB10 in DGX Spark,
Jetson): ``is_unified_memory``, ``memory_info`` and ``memory_budget_bytes``. There the
device's "total" is the host's RAM and CUDA's "free" figure leaves out page cache the
kernel would reclaim, so neither is what a job can plan with.

Also here: keeping the *current* device right (``device_scope``, ``on_current_device``).
Triton kernels (Liger's, flash-linear-attention's, vLLM's, palingenesis's own) launch on
the current device, not on their tensors' device; the current device is per thread and a
new thread starts on GPU 0. A kernel launched on GPU 0 with another GPU's tensors reads
memory that is not mapped there: on AMD GPUs a "Memory access fault by GPU node-N"
that kills the process (measured on MI250X: OPD's teacher on cuda:1, RL ranks > 0).

Only code that needs one of these decisions imports this module.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
from collections.abc import Callable
from typing import TypeVar

import torch

_T = TypeVar("_T")

# FP8 training (torchao float8). NVIDIA: SM89+ (Ada, Hopper, Blackwell), validated on
# discrete GPUs. GB10 (DGX Spark, sm_121) passes torch's capability test for scaled
# matmuls, but no palingenesis run has validated it: treated like the AMD GPUs below.
# ROCm: torchao picks e4m3fnuz on MI300 (gfx942) and OCP e4m3fn on MI350 (gfx950)
# (torchao.float8.config), but no palingenesis run has validated it: allowed when the
# config asks for it, with a warning, and never turned on automatically. Every other
# AMD GPU (gfx90a / MI250 included) has no FP8 matrix cores.
_FP8_MIN_CUDA_CAPABILITY = (8, 9)
_CUDA_FP8_UNVALIDATED = {(12, 1): "GB10, DGX Spark"}
_ROCM_FP8_ARCHS = {"gfx942": "MI300, float8 e4m3fnuz", "gfx950": "MI350, float8 e4m3fn"}

# Cut Cross-Entropy's Triton kernels call tl.dot(input_precision="tf32") when
# torch.get_float32_matmul_precision() == "high". Triton's AMD backend accepts "tf32"
# only on gfx942 (triton third_party/amd/backend/compiler.py: "Enable XF32 (TF32) for
# CDNA3 GPUs"); on other AMD GPUs the kernel fails to compile.
_ROCM_TF32_DOT_ARCHS = {"gfx942"}

# transformers' pattern for an attention kernel on the Hub: "org/repo[@revision][:function]".
_HUB_KERNEL = re.compile(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$")
ROCM_ATTENTION_KERNEL = "kernels-community/aiter-flash-attn@v2"
_AITER_REPO = "kernels-community/aiter-flash-attn"
# aiter's Triton attention reads build/torch-rocm/configs/<arch>-MHA-DEFAULT.json at its first
# call and ships one only for these targets (v1 and v2): on gfx90a (MI250) it fails with
# FileNotFoundError mid-forward, measured on an MI250X.
_AITER_ARCHS = {"gfx942", "gfx950", "gfx1150", "gfx1151", "gfx1250"}

AMD_DOCS = "https://mii-llm.github.io/palingenesis/getting-started/amd/"


def is_rocm() -> bool:
    """True on a PyTorch ROCm (HIP) build, GPU present or not."""
    return getattr(torch.version, "hip", None) is not None


def gpu_arch(device: int = 0) -> str | None:
    """The GPU's architecture: "gfx90a", "gfx942", ... on ROCm, "sm_90" style on CUDA;
    None without a GPU."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device)
    arch = getattr(props, "gcnArchName", "") if is_rocm() else ""
    if arch:
        return arch.split(":")[0]  # drop feature flags (":sramecc+:xnack-")
    return f"sm_{props.major}{props.minor}"


def is_unified_memory(device: int | torch.device = 0) -> bool:
    """True when the GPU is integrated and allocates from the host's RAM (NVIDIA GB10 in
    DGX Spark, Jetson): CUDA's ``integrated`` device property. False without a GPU."""
    if not torch.cuda.is_available():
        return False
    return bool(getattr(torch.cuda.get_device_properties(device), "is_integrated", False))


def host_available_bytes(meminfo: str = "/proc/meminfo") -> int | None:
    """The kernel's MemAvailable: RAM a new allocation can get without swapping, reclaimable
    page cache included. None where it cannot be read (not Linux)."""
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def memory_info(device: int | torch.device = 0) -> tuple[int, int]:
    """(free, total) bytes of GPU memory, like ``torch.cuda.mem_get_info``.

    On a unified-memory GPU the CUDA figure counts the host's page cache as used (after
    reading a model or a dataset it is tens of GB too low), while the kernel gives that
    memory back on demand: there, free is the host's MemAvailable when that is larger.
    """
    free, total = torch.cuda.mem_get_info(device)
    if is_unified_memory(device):
        available = host_available_bytes()
        if available is not None:
            free = min(max(free, available), total)
    return int(free), int(total)


def memory_budget_bytes(device: int | torch.device = 0) -> int:
    """What one job can plan to allocate on `device` when no budget is given: the whole
    device on a discrete GPU, and on a unified-memory GPU only what the host has available
    now, since the operating system, this process's own RAM and every other process draw on
    the same pool."""
    total = int(torch.cuda.get_device_properties(device).total_memory)
    if is_unified_memory(device):
        available = host_available_bytes()
        if available is not None:
            return min(total, available)
    return total


def device_scope(device: str | torch.device) -> contextlib.AbstractContextManager:
    """Make `device` the current GPU inside the block (a no-op for CPU or a bare "cuda"),
    for work on a model that lives on another GPU than the thread's current one."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None and torch.cuda.is_available():
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def on_current_device(fn: Callable[..., _T]) -> Callable[..., _T]:
    """`fn`, run on the GPU that is current where it was wrapped: for thread targets and
    executor initializers, since a new thread starts on GPU 0 (see the module docstring)."""
    if not torch.cuda.is_available():
        return fn
    index = torch.cuda.current_device()

    @functools.wraps(fn)
    def run(*args, **kwargs):
        torch.cuda.set_device(index)
        return fn(*args, **kwargs)

    return run


def platform_name() -> str:
    """The torch build's GPU platform: "ROCm 7.2.x", "CUDA 12.9" or "CPU"."""
    if is_rocm():
        return f"ROCm {torch.version.hip}"
    if torch.version.cuda:
        return f"CUDA {torch.version.cuda}"
    return "CPU"


def describe() -> str:
    """One line for `pgs version` and startup logs: device, memory, platform, arch."""
    try:
        if not torch.cuda.is_available():
            return f"not available (torch built for {platform_name()})"
        props = torch.cuda.get_device_properties(0)
        count = torch.cuda.device_count()
        memory = f"{props.total_memory / 2**30:.0f} GB"
        if is_unified_memory():
            memory += ", shared with the host"
        return f"{count}x {props.name} ({memory}), {platform_name()}, arch {gpu_arch()}"
    except Exception as e:  # noqa: BLE001 - diagnostics never break a command
        return f"unknown ({e})"


def no_gpu_message() -> str:
    """Why torch sees no GPU, worded for the torch build that is installed."""
    if is_rocm():
        env = ", ".join(
            f"{k}={os.environ[k]!r}"
            for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            if k in os.environ
        )
        return (
            f"No usable AMD GPU (torch {torch.__version__}, built for ROCm {torch.version.hip}). Check that "
            "`amd-smi` (or `rocm-smi`) lists the GPU, that this user can open /dev/kfd and /dev/dri (groups video "
            "and render; in a container pass --device=/dev/kfd --device=/dev/dri), and that the visible-device "
            f"variables do not hide it ({env or 'none set'}). See {AMD_DOCS}"
        )
    if torch.version.cuda is None:
        return (
            f"No GPU: torch {torch.__version__} is a CPU-only build. Install a CUDA build (or, on AMD GPUs, a ROCm "
            "build) of torch. See docs: Installation."
        )
    return (
        f"No usable CUDA GPU (torch {torch.__version__}, built for CUDA {torch.version.cuda}). If `nvidia-smi` "
        "works, this torch build needs a newer NVIDIA driver than the machine has: reinstall torch from a CUDA "
        "index your driver supports, e.g. `uv pip install torch --index-url "
        "https://download.pytorch.org/whl/cu128` (driver >= 570). See docs: Installation."
    )


def check_visible_devices(environ=None) -> None:
    """On ROCm, refuse HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES that select different GPUs.

    The HIP runtime reads both, and vLLM's ROCm platform raises on the same conflict at
    import; failing here says so before any model loads. An empty value counts as unset,
    as in vLLM. ROCR_VISIBLE_DEVICES filters first, at the driver level, and the other two
    index into what it leaves, so it never conflicts with them.
    """
    if not is_rocm():
        return
    environ = os.environ if environ is None else environ
    hip, cuda = environ.get("HIP_VISIBLE_DEVICES") or None, environ.get("CUDA_VISIBLE_DEVICES") or None
    if hip is not None and cuda is not None and hip != cuda:
        raise RuntimeError(
            f"HIP_VISIBLE_DEVICES={hip!r} and CUDA_VISIBLE_DEVICES={cuda!r} select different GPUs. On ROCm set only "
            "one of them (HIP_VISIBLE_DEVICES, or ROCR_VISIBLE_DEVICES for the whole process tree), or make them equal."
        )


def fp8_support() -> tuple[str, str]:
    """(status, reason) for FP8 training on device 0.

    status: "supported" (validated: NVIDIA SM89+), "unvalidated" (AMD MI300/MI350 and NVIDIA
    GB10: torchao has the path, palingenesis has not validated it) or "unsupported".
    """
    if not torch.cuda.is_available():
        return "unsupported", "no GPU"
    arch = gpu_arch()
    if is_rocm():
        if arch in _ROCM_FP8_ARCHS:
            return "unvalidated", f"{arch} ({_ROCM_FP8_ARCHS[arch]}) through torchao; not validated on ROCm yet"
        return "unsupported", f"AMD {arch} has no FP8 matrix cores (FP8 needs MI300 gfx942 or MI350 gfx950)"
    capability = tuple(torch.cuda.get_device_capability())
    if capability in _CUDA_FP8_UNVALIDATED:
        return "unvalidated", f"NVIDIA {arch} ({_CUDA_FP8_UNVALIDATED[capability]}) through torchao; not validated yet"
    if capability >= _FP8_MIN_CUDA_CAPABILITY:
        return "supported", f"NVIDIA {arch}"
    return "unsupported", f"NVIDIA {arch}: FP8 needs SM89+ (L40S, H100, B200)"


def cce_support(matmul_precision: str | None = None) -> tuple[bool, str]:
    """Whether Cut Cross-Entropy's Triton kernels compile on device 0, and why not.

    Only AMD GPUs other than gfx942 are refused, and only at float32 matmul precision
    "high" (the one that makes CCE ask Triton for tf32 dots).
    """
    if not is_rocm() or not torch.cuda.is_available():
        return True, ""
    precision = matmul_precision or torch.get_float32_matmul_precision()
    arch = gpu_arch()
    if precision == "high" and arch not in _ROCM_TF32_DOT_ARCHS:
        return False, (
            f"Cut Cross-Entropy uses tf32 Triton dots at float32 matmul precision 'high', which Triton supports on "
            f"AMD only for gfx942; this GPU is {arch}. Using chunked CE instead "
            "(memory.float32_matmul_precision: highest lets CCE run with ieee dots)."
        )
    return True, ""


def is_hub_kernel(attn_implementation: str) -> bool:
    """True for a Hub attention kernel name, e.g. "kernels-community/aiter-flash-attn@v2"."""
    return _HUB_KERNEL.match(attn_implementation) is not None


FLASH_ATTENTION_NAMES = ("flash_attention_2", "flash_attention_3", "flash_attention_4")


def is_flash_varlen(attn_implementation: str) -> bool:
    """Attention that transformers runs through its flash (variable-length) path: packed
    rows are flattened and documents separated by cu_seq_lens."""
    return attn_implementation in FLASH_ATTENTION_NAMES or (
        is_hub_kernel(attn_implementation) and "flash" in attn_implementation
    )


# ─── The flash kernel behind SDPA (model.flash_kernel) ───────────────────────────────────────────────────────────
#
# torch's SDPA flash backend and aten._flash_attention_forward (the variable-length kernel the sdpa path runs on
# packed rows and SeCO chunks) are FlashAttention-2 by default. torch >= 2.11 can put FlashAttention-3 (Hopper) or
# FlashAttention-4 (Hopper and Blackwell, CuTe DSL) behind the same ops
# (torch.nn.attention.activate_flash_attention_impl), so every sdpa code path keeps working and runs the newer
# kernel. The kernels come from the flash-attn-3 (`flash_attn_interface`) and flash-attn-4 (`flash_attn.cute`)
# packages. torch marks the swap as a prototype, so it is checked at startup on the model's head shape against the
# built-in kernel before training uses it.

FLASH_KERNELS = ("builtin", "auto", "fa3", "fa4")
_FLASH_MODULES = {"fa3": "flash_attn_interface", "fa4": "flash_attn.cute.interface"}
_FLASH_ARCHS = {"fa3": (9,), "fa4": (9, 10)}  # compute-capability majors torch's registration accepts
_FLASH_INSTALL = {
    "fa3": "the flash-attn-3 package (FlashAttention's hopper/ build)",
    "fa4": "the flash-attn-4 package (pip install flash-attn-4)",
}


def flash_kernel_candidates(choice: str, major: int | None) -> list[str]:
    """The kernels to try for model.flash_kernel=`choice` on a GPU of compute-capability `major` (None: no NVIDIA
    GPU), in order. "auto": FA3 then FA4 on Hopper (FA3 is the mature Hopper kernel), FA4 on datacenter Blackwell
    (sm100); none elsewhere (consumer Blackwell, sm120, is not a target of either)."""
    if choice not in FLASH_KERNELS:
        raise ValueError(f"model.flash_kernel must be one of {', '.join(FLASH_KERNELS)}, got {choice!r}")
    if choice == "builtin" or major is None:
        return []
    if choice == "auto":
        return {9: ["fa3", "fa4"], 10: ["fa4"]}.get(major, [])
    return [choice]


def _flash_module_available(kernel: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(_FLASH_MODULES[kernel]) is not None
    except (ImportError, ValueError):  # a parent package is missing ("flash_attn" of "flash_attn.cute.interface")
        return False


def _probe_flash_kernel(impl: str, heads: int, kv_heads: int, head_dim: int, device: torch.device) -> None:
    """Forward and backward of the two ops training uses (dense causal SDPA and the variable-length kernel) on the
    model's head shape, under `impl` and under the built-in kernel on the same inputs. Raises on failure or when
    they disagree by more than bf16 rounding."""
    from torch.nn import attention
    from torch.nn.functional import scaled_dot_product_attention

    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, heads, 256, head_dim, generator=g).to(device, torch.bfloat16)
    k = torch.randn(1, kv_heads, 256, head_dim, generator=g).to(device, torch.bfloat16)
    v = torch.randn(1, kv_heads, 256, head_dim, generator=g).to(device, torch.bfloat16)
    cu = torch.tensor([0, 100, 256], dtype=torch.int32, device=device)

    def run() -> list[torch.Tensor]:
        qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        with attention.sdpa_kernel(attention.SDPBackend.FLASH_ATTENTION):
            dense = scaled_dot_product_attention(qq, kk, vv, is_causal=True, enable_gqa=heads != kv_heads)
        flat = [t[0].transpose(0, 1).contiguous() for t in (qq, kk, vv)]  # [L, heads, D], K/V heads grouped
        varlen = torch.ops.aten._flash_attention_forward(*flat, cu, cu, 156, 156, 0.0, True, False)[0]
        (dense.float().square().sum() + varlen.float().square().sum()).backward()
        return [dense.float(), varlen.float(), qq.grad.float(), kk.grad.float(), vv.grad.float()]

    attention.restore_flash_attention_impl(_raise_warn=False)
    reference = run()
    attention.activate_flash_attention_impl(impl)
    for name, a, b in zip(("output", "varlen output", "dq", "dk", "dv"), run(), reference):
        err = float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))
        if not err < 2e-2:
            raise RuntimeError(f"{name} differs from the built-in kernel's by {err:.3g} of its largest value")


def attention_heads(model_config) -> tuple[int, int, int]:
    """(query heads, K/V heads, head dim) of a transformers model config (its text config for multimodal models)."""
    text = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    heads = int(getattr(text, "num_attention_heads", 1) or 1)
    kv_heads = int(getattr(text, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(text, "head_dim", None) or text.hidden_size // heads)
    return heads, kv_heads, head_dim


def activate_flash_kernel(
    choice: str,
    attn_implementation: str,
    heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device | str = "cuda",
) -> str:
    """Put FlashAttention-3 / -4 behind SDPA's flash kernels for model.flash_kernel=`choice` (FLASH_KERNELS) and
    return the kernel now in use: "builtin", "fa3" or "fa4". Only the sdpa path has a kernel to swap. "auto" falls
    back to the built-in kernel quietly (and does nothing without an NVIDIA GPU); an explicit "fa3" / "fa4" that
    cannot run raises ConfigError. Each candidate is first probed on the model's head shape (heads, kv_heads,
    head_dim) against the built-in kernel."""
    from palingenesis.config import ConfigError

    if choice == "builtin" or attn_implementation != "sdpa":
        return "builtin"
    cuda = torch.cuda.is_available() and not is_rocm()
    major = torch.cuda.get_device_capability(torch.device(device))[0] if cuda else None
    explicit = choice != "auto"
    if explicit and major is None:
        raise ConfigError(f"model.flash_kernel={choice} needs an NVIDIA GPU; use builtin or auto.")
    if explicit and major not in _FLASH_ARCHS[choice]:
        archs = " or ".join(f"sm{a}0" for a in _FLASH_ARCHS[choice])
        raise ConfigError(f"model.flash_kernel={choice} runs on {archs}; this GPU is sm{major}x. Use builtin or auto.")
    from torch.nn import attention

    if not hasattr(attention, "activate_flash_attention_impl"):
        if explicit:
            raise ConfigError(f"model.flash_kernel={choice} needs torch >= 2.11 (this is {torch.__version__}).")
        return "builtin"
    for kernel in flash_kernel_candidates(choice, major):
        if not _flash_module_available(kernel):
            if explicit:
                raise ConfigError(f"model.flash_kernel={kernel} needs {_FLASH_INSTALL[kernel]}.")
            continue
        try:
            _probe_flash_kernel(kernel.upper(), heads, kv_heads, head_dim, torch.device(device))
            return kernel
        except Exception as e:  # registration or probe failure: keep the built-in kernel
            attention.restore_flash_attention_impl(_raise_warn=False)
            if explicit:
                raise ConfigError(f"model.flash_kernel={kernel} failed its startup check: {e}") from e
    return "builtin"


def aiter_supported(arch: str | None = None) -> bool:
    """Whether aiter's Hub flash attention has Triton configs for this AMD GPU (device 0 by default)."""
    return (arch or gpu_arch()) in _AITER_ARCHS


def _rocm_flash_alternatives(arch: str | None) -> str:
    """The flash attention options to suggest on this AMD GPU."""
    if aiter_supported(arch):
        return (
            f"Use model.attn_implementation: {ROCM_ATTENTION_KERNEL} (AMD's aiter flash attention from the Hub; "
            "`pip install kernels`) or sdpa, or install ROCm flash-attn."
        )
    return (
        f"Use model.attn_implementation: sdpa, or install a ROCm flash-attn build for {arch} "
        "(aiter's Hub kernel has no configs for this GPU)."
    )


def check_attn_implementation(attn_implementation: str) -> None:
    """Raise ConfigError when model.attn_implementation cannot load or run in this environment.

    flash_attention_2 needs the flash-attn package; on NVIDIA transformers also accepts the
    `kernels` package instead (it then loads kernels-community/flash-attn2, which has no
    ROCm build). A Hub kernel needs the `kernels` package, and aiter's only runs on the AMD
    GPUs it ships configs for. Other names are left to transformers.
    """
    from transformers.utils import is_flash_attn_2_available, is_kernels_available

    from palingenesis.config import ConfigError

    if attn_implementation == "flash_attention_2":
        if is_flash_attn_2_available() or (is_kernels_available() and not is_rocm()):
            return
        if is_rocm():
            raise ConfigError(
                "model.attn_implementation=flash_attention_2, but flash-attn is not installed for this ROCm torch "
                f"({platform_name()}). {_rocm_flash_alternatives(gpu_arch())} See {AMD_DOCS}"
            )
        raise ConfigError(
            "model.attn_implementation=flash_attention_2, but neither flash-attn nor the `kernels` package is "
            f"installed ({platform_name()}). `pip install flash-attn` (or `pip install kernels` for the Hub build), "
            "or set model.attn_implementation: sdpa."
        )
    if attn_implementation in ("flash_attention_3", "flash_attention_4"):
        from transformers.utils import import_utils

        version = attn_implementation[-1]
        if not getattr(import_utils, f"is_flash_attn_{version}_available", lambda: False)():
            package = "flash-attn-3 (FlashAttention's hopper/ build)" if version == "3" else "flash-attn-4"
            raise ConfigError(
                f"model.attn_implementation={attn_implementation}, but {package} is not installed "
                f"({platform_name()}). Install it, or use sdpa with model.flash_kernel: auto (sdpa then runs the "
                "same kernel through torch)."
            )
        return
    if is_hub_kernel(attn_implementation) and not is_kernels_available():
        raise ConfigError(
            f"model.attn_implementation={attn_implementation!r} is a Hub kernel and needs the `kernels` package: "
            "`pip install kernels`."
        )
    repo = attn_implementation.split("@")[0].split(":")[0]
    if repo == _AITER_REPO and is_rocm() and torch.cuda.is_available() and not aiter_supported():
        arch = gpu_arch()
        raise ConfigError(
            f"model.attn_implementation={attn_implementation!r}: aiter's flash attention ships Triton configs only "
            f"for {', '.join(sorted(_AITER_ARCHS))}; this GPU is {arch}. {_rocm_flash_alternatives(arch)} "
            f"See {AMD_DOCS}"
        )
