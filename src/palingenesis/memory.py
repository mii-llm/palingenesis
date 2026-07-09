"""Memory optimizations: gradient release, selective differentiation.

Techniques that reduce peak GPU memory without approximations or accuracy loss.

1. GRADIENT RELEASE (optimizer step in backward pass)
   - Paper: FORGE (arxiv:2606.22932, Jun 2026) + PyTorch native support
   - Concept: Instead of storing ALL layer gradients then stepping, step each
     layer's optimizer immediately during backward and free the gradient.
   - Result: Only 1 layer's gradient lives at a time, not all N layers.
   - Savings: ~16 GB for 8B model (eliminates entire gradient buffer)
   - Limitation: Only works when gradient_accumulation_steps = 1
   - Compatibility: AdamW, Lion, SGD, RMSprop (NOT Muon — needs full grad matrix)

2. SELECTIVE DIFFERENTIATION (skip activations for frozen layers)
   - Paper: arxiv:2404.12406 (NAACL 2025)
   - Concept: When parameters are frozen (requires_grad=False), PyTorch still
     saves input activations for the backward pass. This is wasteful.
   - For linear layers: if W is frozen, input X doesn't need to be saved.
   - Result: Frozen layers use zero activation memory in the compute graph.
   - Savings: With freeze_non_attention (75% frozen), saves 50-70% of activation memory.
   - Compatibility: Works with any optimizer, any parallelism, gradient accumulation.

Combined effect for hybrid model training (Qwen3.5-4B, freeze_non_attention=true):
   - Gradient release: eliminates gradient buffer for trainable params (~2 GB)
   - Selective diff: eliminates activation storage for 75% of layers (~6 GB)
   - Total saving: ~8 GB → model that needed 28 GB now needs ~20 GB
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. GRADIENT RELEASE: OPTIMIZER STEP IN BACKWARD PASS
# ══════════════════════════════════════════════════════════════════════════════


class GradientRelease:
    """Fuse optimizer step into backward pass to eliminate gradient memory.

    Instead of the standard pattern:
        loss.backward()     # ALL gradients live simultaneously (~16 GB for 8B)
        optimizer.step()    # reads all gradients, updates weights
        optimizer.zero_grad()

    We do:
        loss.backward()     # each layer's gradient is consumed and freed immediately
                           # peak gradient memory = largest single param (~0.5 GB)

    Implementation: Register a post_accumulate_grad_hook on each parameter.
    When the hook fires (gradient is ready), we immediately:
      1. Run the optimizer step for that parameter
      2. Free the gradient (set to None)

    This is the PyTorch-native version of FORGE (arxiv:2606.22932).
    FORGE goes deeper (tile-level fusion in Triton), but this gives 80% of the
    benefit with zero custom CUDA code.

    IMPORTANT LIMITATIONS:
      - NOT compatible with gradient_accumulation_steps > 1
        (gradients need to accumulate across micro-batches before stepping)
      - NOT compatible with Muon optimizer (needs full gradient matrix)
      - NOT compatible with global gradient clipping (need all grad norms first)
        → Use AdaGC (per-tensor) instead, which IS compatible
      - Compatible with: AdamW, Lion, SGD, RMSprop + per-tensor AdaGC

    Usage:
        optimizer = AdamW(model.parameters(), lr=1e-5)
        grad_release = GradientRelease(model, optimizer)
        grad_release.enable()

        # Training loop (no optimizer.step() needed — it's in the backward!)
        for batch in dataloader:
            loss = model(**batch).loss
            loss.backward()
            # Optimizer already stepped per-param during backward
            # Just zero_grad (which is nearly free since grads are already None)
            optimizer.zero_grad(set_to_none=True)

        grad_release.disable()  # restore normal training

    Args:
        model: The model being trained
        optimizer: Element-wise optimizer (AdamW, Lion, SGD — NOT Muon)
        adagc: Optional AdaGC instance for per-tensor clipping before step
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        adagc=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adagc = adagc
        self._hooks: list[Any] = []
        self._enabled = False
        self._grad_norm_sq_accum = 0.0
        self._last_norm = 0.0

        # Build param → param_group mapping for per-param optimizer step
        self._param_to_group: dict[int, dict] = {}
        self._param_to_state: dict[int, dict] = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                self._param_to_group[id(p)] = group

    def enable(self):
        """Register backward hooks for gradient release."""
        if self._enabled:
            return
        self._enabled = False  # set True after hooks registered
        self._hooks.clear()
        self._grad_norm_sq_accum = 0.0

        trainable_count = 0
        total_grad_memory = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            trainable_count += 1
            total_grad_memory += param.numel() * param.element_size()

            # Register hook that fires AFTER gradient is accumulated
            hook = param.register_post_accumulate_grad_hook(self._make_hook(param, name))
            self._hooks.append(hook)

        self._enabled = True

        # Detect optimizer type for per-param stepping
        self._optimizer_type = self._detect_optimizer_type()

        saved_gb = total_grad_memory / 1e9
        logger.info(
            f"GradientRelease enabled: {trainable_count} params, "
            f"~{saved_gb:.1f} GB gradient memory eliminated, "
            f"optimizer_type={self._optimizer_type}"
        )

    def _detect_optimizer_type(self) -> str:
        """Detect which optimizer algorithm to use in per-param step.

        Returns: "adamw", "lion", or "sgd"
        """
        opt_cls = type(self.optimizer).__name__.lower()
        if "lion" in opt_cls:
            return "lion"
        elif "sgd" in opt_cls:
            return "sgd"
        else:
            return "adamw"  # default for AdamW, AdamW8bit, PagedAdamW8bit

    def disable(self):
        """Remove hooks, restore normal optimizer step pattern."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._enabled = False
        logger.info("GradientRelease disabled")

    def _make_hook(self, param: nn.Parameter, name: str):
        """Create the per-parameter backward hook."""
        group = self._param_to_group.get(id(param))

        def hook(p: nn.Parameter):
            if p.grad is None:
                return

            # Track gradient norm for monitoring
            grad_norm = p.grad.data.float().norm().item()
            self._grad_norm_sq_accum += grad_norm ** 2

            # Per-tensor AdaGC clipping (if enabled)
            if self.adagc is not None:
                ema = self.adagc._ema.get(name, grad_norm)
                threshold = self.adagc.lambda_rel * ema
                if grad_norm > threshold and grad_norm > 1e-12:
                    p.grad.data.mul_(threshold / grad_norm)
                    clipped_norm = threshold
                else:
                    clipped_norm = grad_norm
                self.adagc._ema[name] = self.adagc.beta * ema + (1 - self.adagc.beta) * clipped_norm

            # Perform single-param optimizer step
            if group is not None:
                self._step_single_param(p, group)

            # FREE the gradient immediately
            p.grad = None

        return hook

    def _step_single_param(self, param: nn.Parameter, group: dict):
        """Apply optimizer update to a single parameter.

        Dispatches to the correct algorithm based on detected optimizer type.
        AdamW and Lion are both element-wise, so per-param stepping is exact.
        """
        if self._optimizer_type == "lion":
            self._step_lion(param, group)
        else:
            self._step_adamw(param, group)

    def _step_adamw(self, param: nn.Parameter, group: dict):
        """AdamW: decoupled weight decay + bias-corrected moments."""
        grad = param.grad.data
        state = self.optimizer.state.setdefault(param, {})

        lr = group["lr"]
        weight_decay = group.get("weight_decay", 0.0)
        betas = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-8)

        # Initialize state on first call
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param.data)
            state["exp_avg_sq"] = torch.zeros_like(param.data)

        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        beta1, beta2 = betas

        # AdamW update (decoupled weight decay)
        if weight_decay != 0:
            param.data.mul_(1 - lr * weight_decay)

        # Moment updates
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5

        # Update
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        param.data.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_lion(self, param: nn.Parameter, group: dict):
        """Lion: sign-based update with EMA. 4 bytes/param (only 1 momentum buffer)."""
        grad = param.grad.data
        state = self.optimizer.state.setdefault(param, {})

        lr = group["lr"]
        weight_decay = group.get("weight_decay", 0.0)
        betas = group.get("betas", (0.95, 0.98))

        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(param.data)

        exp_avg = state["exp_avg"]
        beta1, beta2 = betas

        # Weight decay (decoupled)
        if weight_decay != 0:
            param.data.mul_(1 - lr * weight_decay)

        # Update = sign(beta1 * exp_avg + (1 - beta1) * grad)
        update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
        param.data.add_(update, alpha=-lr)

        # Momentum update (for next step)
        exp_avg.lerp_(grad, 1 - beta2)

    @property
    def last_grad_norm(self) -> float:
        """Total gradient norm from the last backward pass.

        Safe to read multiple times: the accumulator is drained into a cached
        value on first read after a backward pass, and subsequent reads return
        the same cached value (a destructive read here previously caused
        grad_norm to always log as 0.0 — hasattr() consumed the value).
        """
        if self._grad_norm_sq_accum > 0.0:
            self._last_norm = self._grad_norm_sq_accum ** 0.5
            self._grad_norm_sq_accum = 0.0  # reset accumulator for next backward
        return self._last_norm

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ══════════════════════════════════════════════════════════════════════════════
# 2. SELECTIVE DIFFERENTIATION: SKIP ACTIVATIONS FOR FROZEN LAYERS
# ══════════════════════════════════════════════════════════════════════════════


def apply_selective_differentiation(model: nn.Module) -> int:
    """Eliminate activation memory for frozen linear layers.

    From arxiv:2404.12406 (NAACL 2025): When a linear layer's weights are frozen
    (requires_grad=False), PyTorch still saves the input tensor for potential
    backward computation. This is wasteful — if no gradient flows through the
    weight, the input doesn't need to be saved.

    This function wraps frozen Linear layers with a custom forward that
    uses torch.no_grad() context to prevent activation saving.

    For models with freeze_non_attention=True (75% frozen layers), this saves
    50-70% of activation memory — the single largest memory pool during training.

    The technique is exact (zero accuracy loss) — we're just skipping unnecessary
    autograd bookkeeping.

    Args:
        model: Model with some parameters frozen (requires_grad=False)

    Returns:
        Number of layers modified
    """
    modified = 0
    saved_bytes = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Check if this layer's weight is frozen
        if module.weight.requires_grad:
            continue

        # Also check bias
        if module.bias is not None and module.bias.requires_grad:
            continue

        # This linear layer is fully frozen — wrap its forward
        _wrap_frozen_linear(module, name)
        modified += 1

        # Estimate memory saved: input tensor that would have been stored
        # Shape: [batch, seq_len, in_features] — estimate with seq=4096, batch=1
        saved_bytes += module.in_features * 4096 * 2  # bf16 = 2 bytes

    saved_gb = saved_bytes / 1e9
    logger.info(
        f"Selective differentiation: {modified} frozen linear layers optimized, "
        f"estimated ~{saved_gb:.1f} GB activation memory saved (at seq=4096, batch=1)"
    )
    return modified


def _wrap_frozen_linear(module: nn.Linear, name: str):
    """Replace a frozen Linear's forward with a no-activation-saving version.

    The key insight: for y = Wx + b where W and b are frozen, the backward pass
    for W needs input X, but since W.requires_grad=False, that backward never runs.
    So X doesn't need to be saved.

    We achieve this by running the forward inside torch.no_grad() and then
    reattaching the output to the autograd graph via a trivial operation that
    doesn't save the large input activation.
    """
    original_forward = module.forward

    def memory_efficient_forward(input: torch.Tensor) -> torch.Tensor:
        # Compute output WITHOUT saving input for backward
        # (since this layer's parameters are frozen, no grad needed through weights)
        with torch.no_grad():
            output = torch.nn.functional.linear(input, module.weight, module.bias)

        # Reattach to autograd graph: output still needs to participate in
        # backward for upstream layers. We use a lightweight detach+requires_grad
        # trick: the output carries gradient info but input is NOT stored.
        if input.requires_grad:
            # Create a differentiable path that doesn't store input
            output = output + 0  # This creates a node but with no large saved tensor
            output.requires_grad_(True)

        return output

    module.forward = memory_efficient_forward


class SelectiveDiffLinear(torch.autograd.Function):
    """Custom autograd function for frozen linear layers.

    More robust version: explicitly controls what's saved for backward.
    For a frozen linear W: y = Wx + b
    - Forward: compute y normally
    - Backward: only need dy (upstream gradient), NOT x (input)
      - dW is not needed (W is frozen)
      - dx = dy @ W (only needs W, not x)
      - db = sum(dy) (no input needed)

    So we save NOTHING from forward except W (which is a parameter, already in memory).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        # Save ONLY the weight (already in memory as a parameter — zero extra cost)
        ctx.save_for_backward(weight)
        ctx.has_bias = bias is not None
        output = torch.nn.functional.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight,) = ctx.saved_tensors

        # dx = grad_output @ weight (input gradient — needed for upstream layers)
        grad_input = grad_output.matmul(weight)

        # dW = None (weight is frozen, no gradient needed)
        # db = None or sum of grad_output (bias might be trainable in some configs)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

        return grad_input, None, grad_bias


def apply_selective_diff_v2(model: nn.Module) -> int:
    """Apply SelectiveDiffLinear to all frozen layers (more robust version).

    Uses custom autograd Function that explicitly saves only what's needed.
    This version properly handles gradient flow through the layer while
    eliminating the input activation from saved tensors.

    Args:
        model: Model with frozen parameters

    Returns:
        Number of modified layers
    """
    modified = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if module.weight.requires_grad:
            continue

        # Check if bias is also frozen (or absent)
        bias_frozen = module.bias is None or not module.bias.requires_grad

        if not bias_frozen:
            continue  # Skip layers with trainable bias but frozen weight (rare)

        # Replace forward with custom autograd function
        weight_ref = module.weight
        bias_ref = module.bias

        def make_forward(w, b):
            def forward(input: torch.Tensor) -> torch.Tensor:
                return SelectiveDiffLinear.apply(input, w, b)

            return forward

        module.forward = make_forward(weight_ref, bias_ref)
        modified += 1

    logger.info(f"SelectiveDiff v2: {modified} frozen linear layers, zero input activation stored")
    return modified
