"""Accelerator detection: tell CUDA (NVIDIA) and ROCm (AMD) apart.

PyTorch's ROCm build masquerades as CUDA — ``torch.cuda.*``, ``device="cuda"`` and the
``"nccl"`` distributed backend all work on AMD by dispatching to HIP / RCCL. That makes
most of the framework portable for free, but it also means ``torch.cuda.is_available()``
is True on an AMD GPU and ``torch.cuda.get_device_capability()`` returns gfx-derived
numbers (gfx90a → ``(9, 0)``). Any CUDA-semantics check — above all the SM-version gate
guarding FP8 — therefore needs to know which backend it is really running on.

The one source of truth is ``torch.version.hip``: it is a version string on a ROCm build
and ``None`` on a CUDA build. Everything here keys off that, never off a hostname or
``sys_platform`` (a ROCm box need not be an HPC login node, and a CUDA box can be Linux
too). Helpers import torch lazily so this module stays importable on a torch-less box.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

# FP8 tensor-core support we target lives on NVIDIA SM89+ (Ada / Hopper / Blackwell).
# CDNA2 (gfx90a / MI250X) has no FP8 at all; we do not target the MI300/gfx942 FP8 path
# yet, so FP8 is reported unsupported on every ROCm device for now.
_FP8_MIN_CUDA_CAPABILITY = (8, 9)


def is_rocm() -> bool:
    """True when running on a PyTorch ROCm (AMD/HIP) build, GPU present or not."""
    import torch

    return torch.version.hip is not None


def is_available() -> bool:
    """True when an accelerator (NVIDIA or AMD) is visible to torch.

    ROCm masquerades as CUDA, so ``torch.cuda.is_available()`` covers both.
    """
    import torch

    return torch.cuda.is_available()


def is_cuda() -> bool:
    """True only on a real NVIDIA CUDA device (available and not a ROCm build)."""
    return is_available() and not is_rocm()


def backend() -> str:
    """Coarse backend label for logging / branching: ``"rocm"``, ``"cuda"`` or ``"cpu"``."""
    if not is_available():
        return "cpu"
    return "rocm" if is_rocm() else "cuda"


def device_capability() -> tuple[int, int] | None:
    """``torch.cuda.get_device_capability()`` of device 0, or None without an accelerator.

    On ROCm the numbers are gfx-derived, not CUDA SM versions — do not compare them
    against CUDA SM thresholds. Use :func:`supports_fp8` for FP8 decisions.
    """
    if not is_available():
        return None
    import torch

    return tuple(torch.cuda.get_device_capability())  # type: ignore[return-value]


def gpu_arch() -> str | None:
    """Architecture name of device 0 for diagnostics.

    ROCm reports the gfx target (e.g. ``"gfx90a"``) via ``gcnArchName``; on CUDA we fall
    back to ``"sm_<major><minor>"``. None when no accelerator is present.
    """
    if not is_available():
        return None
    import torch

    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", None)  # ROCm only
    if arch:
        # gcnArchName can carry feature flags, e.g. "gfx90a:sramecc+:xnack-"; keep the base.
        return arch.split(":")[0]
    cap = (props.major, props.minor)
    return f"sm_{cap[0]}{cap[1]}"


def supports_fp8() -> bool:
    """True only where the FP8 training path is actually available: NVIDIA SM89+.

    Always False on ROCm (CDNA2 has no FP8; the MI300 FP8 path is not targeted yet) and
    False on CPU. This is the check FP8 gates must use instead of a raw SM comparison,
    because on ROCm a raw ``get_device_capability() >= (8, 9)`` wrongly passes.
    """
    if not is_cuda():
        return False
    cap = device_capability()
    return cap is not None and cap >= _FP8_MIN_CUDA_CAPABILITY


@functools.lru_cache(maxsize=1)
def describe() -> str:
    """One-line, cached human summary for startup logs. Never raises."""
    try:
        if not is_available():
            return "accelerator: none (CPU)"
        import torch

        name = torch.cuda.get_device_properties(0).name
        count = torch.cuda.device_count()
        return (
            f"accelerator: {backend()} ({count}x {name}, arch={gpu_arch()}, "
            f"fp8={'yes' if supports_fp8() else 'no'})"
        )
    except Exception as e:  # noqa: BLE001 — diagnostics must never break a run
        return f"accelerator: detection failed ({e})"
