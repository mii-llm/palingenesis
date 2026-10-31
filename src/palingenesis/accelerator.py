"""NVIDIA (CUDA) vs AMD (ROCm): the few decisions that differ between the two.

PyTorch's ROCm build exposes AMD GPUs through the CUDA API: ``torch.cuda.*``,
``device="cuda"`` and the ``"nccl"`` backend (RCCL) work unchanged, so almost all of
palingenesis runs as is. What does not carry over are checks written against CUDA
semantics: ``torch.cuda.get_device_capability()`` returns gfx-derived numbers on ROCm
(gfx90a -> (9, 0), gfx942 -> (9, 4)), which pass any ``>= (8, 9)`` "FP8 / Hopper" test.
The build is told apart with ``torch.version.hip`` (a version string on ROCm, None
otherwise), as PyTorch's HIP notes recommend; the GPU generation with the device's
``gcnArchName`` (e.g. "gfx942:sramecc+:xnack-").

Only code that needs one of these decisions imports this module.
"""

from __future__ import annotations

import os
import re

import torch

# FP8 training (torchao float8). NVIDIA: SM89+ (Ada, Hopper, Blackwell), validated.
# ROCm: torchao picks e4m3fnuz on MI300 (gfx942) and OCP e4m3fn on MI350 (gfx950)
# (torchao.float8.config), but no palingenesis run has validated it: allowed when the
# config asks for it, with a warning, and never turned on automatically. Every other
# AMD GPU (gfx90a / MI250 included) has no FP8 matrix cores.
_FP8_MIN_CUDA_CAPABILITY = (8, 9)
_ROCM_FP8_ARCHS = {"gfx942": "MI300, float8 e4m3fnuz", "gfx950": "MI350, float8 e4m3fn"}

# Cut Cross-Entropy's Triton kernels call tl.dot(input_precision="tf32") when
# torch.get_float32_matmul_precision() == "high". Triton's AMD backend accepts "tf32"
# only on gfx942 (triton third_party/amd/backend/compiler.py: "Enable XF32 (TF32) for
# CDNA3 GPUs"); on other AMD GPUs the kernel fails to compile.
_ROCM_TF32_DOT_ARCHS = {"gfx942"}

# transformers' pattern for an attention kernel on the Hub: "org/repo[@revision][:function]".
_HUB_KERNEL = re.compile(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$")
ROCM_ATTENTION_KERNEL = "kernels-community/aiter-flash-attn@v2"

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
        return f"{count}x {props.name} ({props.total_memory / 2**30:.0f} GB), {platform_name()}, arch {gpu_arch()}"
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

    status: "supported" (validated: NVIDIA SM89+), "unvalidated" (AMD MI300/MI350: torchao
    has the path, palingenesis has not validated it) or "unsupported".
    """
    if not torch.cuda.is_available():
        return "unsupported", "no GPU"
    arch = gpu_arch()
    if is_rocm():
        if arch in _ROCM_FP8_ARCHS:
            return "unvalidated", f"{arch} ({_ROCM_FP8_ARCHS[arch]}) through torchao; not validated on ROCm yet"
        return "unsupported", f"AMD {arch} has no FP8 matrix cores (FP8 needs MI300 gfx942 or MI350 gfx950)"
    capability = tuple(torch.cuda.get_device_capability())
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


def is_flash_varlen(attn_implementation: str) -> bool:
    """Attention that transformers runs through its flash (variable-length) path: packed
    rows are flattened and documents separated by cu_seq_lens."""
    return attn_implementation == "flash_attention_2" or (
        is_hub_kernel(attn_implementation) and "flash" in attn_implementation
    )


def check_attn_implementation(attn_implementation: str) -> None:
    """Raise ConfigError when model.attn_implementation cannot load in this environment.

    flash_attention_2 needs the flash-attn package; on NVIDIA transformers also accepts the
    `kernels` package instead (it then loads kernels-community/flash-attn2, which has no
    ROCm build). A Hub kernel needs the `kernels` package. Other names are left to
    transformers.
    """
    from transformers.utils import is_flash_attn_2_available, is_kernels_available

    from palingenesis.config import ConfigError

    if attn_implementation == "flash_attention_2":
        if is_flash_attn_2_available() or (is_kernels_available() and not is_rocm()):
            return
        if is_rocm():
            raise ConfigError(
                "model.attn_implementation=flash_attention_2, but flash-attn is not installed for this ROCm torch "
                f"({platform_name()}). Use model.attn_implementation: {ROCM_ATTENTION_KERNEL} (AMD's aiter flash "
                f"attention from the Hub; `pip install kernels`) or sdpa, or install ROCm flash-attn. See {AMD_DOCS}"
            )
        raise ConfigError(
            "model.attn_implementation=flash_attention_2, but neither flash-attn nor the `kernels` package is "
            f"installed ({platform_name()}). `pip install flash-attn` (or `pip install kernels` for the Hub build), "
            "or set model.attn_implementation: sdpa."
        )
    if is_hub_kernel(attn_implementation) and not is_kernels_available():
        raise ConfigError(
            f"model.attn_implementation={attn_implementation!r} is a Hub kernel and needs the `kernels` package: "
            "`pip install kernels`."
        )
