"""Float8 training via torchao — 1.2-1.5x speedup on H100/B200.

Converts nn.Linear layers to Float8Linear which computes forward/backward
in float8 (e4m3/e5m2) while maintaining bf16 master weights. The float8
GEMMs are 2x faster than bf16 on H100 tensor cores.

Requirements:
    - torchao >= 0.9.0
    - CUDA compute capability >= 8.9 (H100, L40S, B200)
    - torch.compile enabled (needed for optimal float8 kernel fusion)

Usage:
    Set `memory.float8_training: true` in your config.
    Applied AFTER model load, BEFORE FSDP (so FSDP sees Float8Linear params).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_float8_training(model: nn.Module) -> bool:
    """Convert eligible nn.Linear layers to Float8Linear for faster training.

    Only converts layers where both dimensions are divisible by 16
    (hardware alignment requirement for float8 tensor cores).

    Returns True if conversion was applied.
    """
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError:
        logger.warning(
            "torchao not installed or too old for float8 training. " "Install torchao >= 0.9.0: pip install torchao"
        )
        return False

    # Check hardware capability
    if not torch.cuda.is_available():
        logger.warning("Float8 training requires CUDA. Skipping.")
        return False

    capability = torch.cuda.get_device_capability()
    if capability < (8, 9):
        logger.warning(
            f"Float8 training requires SM89+ (H100/B200). "
            f"Current device is SM{capability[0]}{capability[1]}. Skipping."
        )
        return False

    # Use rowwise recipe (best for SFT workloads)
    try:
        config = Float8LinearConfig.from_recipe_name("rowwise")
    except AttributeError:
        # Older torchao version
        config = Float8LinearConfig()

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        """Only convert Linear layers with dimensions divisible by 16."""
        if not isinstance(module, nn.Linear):
            return False
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
        # Skip embedding-tied layers (lm_head often shares with embed)
        if "embed" in fqn.lower():
            return False
        return True

    try:
        convert_to_float8_training(model, config=config, module_filter_fn=filter_fn)
        # Enable inductor precision cast emulation for rowwise float8
        torch._inductor.config.emulate_precision_casts = True
        logger.info("Float8 training enabled (rowwise recipe)")
        return True
    except Exception as e:
        logger.warning(f"Float8 conversion failed: {e}. Continuing with bf16.")
        return False
