"""S0 Tuning: Zero-Overhead Adaptation for Hybrid Recurrent-Attention Models.

Paper: "S0 Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models"
       (arxiv:2604.01168, Apr 2026)

Key insight: Hybrid models (Qwen3.5/GatedDeltaNet, FalconH1/Mamba-2) carry a per-layer
recurrent state matrix initialized to zero. Replacing that zero with a LEARNED value
steers the model toward a target task with:
  - Zero inference overhead (injected at t=0, absorbed into running state at t=1)
  - +23.6pp on HumanEval for Qwen3.5-4B (beats LoRA by +10.8pp)
  - Only 12.6M params (0.3% of model) — one state matrix per recurrent layer
  - Works with ~48 verified training samples

The method:
  1. Freeze all model weights
  2. For each recurrent layer, create a learnable S0 tensor (same shape as state)
  3. Inject α * S0 as the initial hidden state before the first token
  4. Optimize S0 with CE loss + L2 regularization
  5. At inference: zero additional cost (state absorbed into recurrence at t≥1)

Architecture support:
  - Qwen3.5 (GatedDeltaNet): state St ∈ R^{H×K×V} per layer, α=0.07
  - FalconH1 (Mamba-2): state St ∈ R^{H×K×V} per layer, α=0.65
  - Any hybrid with matrix-valued recurrent states (NOT diagonal like Mamba-1)

Usage:
    from palingenesis.s0_tuning import S0Trainer

    trainer = S0Trainer(model, alpha=0.07, weight_decay=1e-4)
    trainer.train(train_dataloader, epochs=50, lr=1e-3)
    trainer.save("s0_states.pt")

    # At inference: load and inject
    s0_states = torch.load("s0_states.pt")
    inject_s0_states(model, s0_states, alpha=0.07)
    # model now runs with zero overhead
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from palingenesis.loss import shift_labels

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STATE DISCOVERY: find recurrent layers and their state shapes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class RecurrentLayerInfo:
    """Metadata about a discovered recurrent layer."""

    name: str  # Module path (e.g., "model.layers.0.delta_net")
    state_shape: tuple[int, ...]  # Shape of the state matrix
    module: nn.Module  # Reference to the actual module
    inject_attr: str  # Attribute name to inject initial state


def discover_recurrent_layers(model: nn.Module) -> list[RecurrentLayerInfo]:
    """Discover all recurrent layers with matrix-valued states in a hybrid model.

    Looks for GatedDeltaNet, DeltaNet, Mamba2, and similar layers by checking:
      1. Module class names containing 'delta', 'mamba', 'rwkv', 'recurrent'
      2. Presence of state-related attributes (initial_state, state_size, etc.)

    Returns list of RecurrentLayerInfo with state shapes inferred from model config.
    """
    recurrent_keywords = (
        "deltanet",
        "delta_net",
        "gated_delta",
        "gateddelta",
        "mamba",
        "mamba2",
        "ssm",
        "rwkv",
        "recurrent",
        "linear_attn",
    )
    layers: list[RecurrentLayerInfo] = []

    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()

        # Check if this is a recurrent layer
        is_recurrent = any(kw in module_type for kw in recurrent_keywords)
        if not is_recurrent:
            # Also check parent module name
            is_recurrent = any(kw in name.lower() for kw in recurrent_keywords)

        if not is_recurrent:
            continue

        # Try to determine state shape
        state_shape = _infer_state_shape(module, model)
        if state_shape is None:
            continue

        # Determine injection attribute
        inject_attr = _find_inject_attribute(module)

        layers.append(
            RecurrentLayerInfo(
                name=name,
                state_shape=state_shape,
                module=module,
                inject_attr=inject_attr,
            )
        )

    return layers


def _infer_state_shape(module: nn.Module, model: nn.Module) -> tuple[int, ...] | None:
    """Infer the recurrent state shape from module attributes or config."""
    # Method 1: Module has explicit state_size or hidden_size attributes
    if hasattr(module, "state_size"):
        ss = module.state_size
        if isinstance(ss, (tuple, list)):
            return tuple(ss)
        return (ss,)

    # Method 2: Look for key/value dimensions in the module
    # GatedDeltaNet: state is (num_heads, key_dim, value_dim)
    num_heads = getattr(module, "num_heads", None) or getattr(module, "n_heads", None)
    key_dim = getattr(module, "key_dim", None) or getattr(module, "head_k_dim", None)
    value_dim = getattr(module, "value_dim", None) or getattr(module, "head_v_dim", None)

    if num_heads and key_dim and value_dim:
        return (num_heads, key_dim, value_dim)

    # Method 3: Check model config for Qwen3.5-style dimensions
    config = getattr(model, "config", None)
    if config:
        # Qwen3.5: 32 heads, key_dim=128, value_dim=128
        nh = getattr(config, "num_attention_heads", None) or getattr(config, "num_heads", None)
        hd = getattr(config, "head_dim", None)
        if hd is None:
            hidden = getattr(config, "hidden_size", None)
            if hidden and nh:
                hd = hidden // nh

        if nh and hd:
            return (nh, hd, hd)

    # Method 4: Mamba-style — look at d_state and d_inner
    d_state = getattr(module, "d_state", None) or getattr(module, "ssm_state_size", None)
    d_inner = getattr(module, "d_inner", None) or getattr(module, "d_model", None)
    if d_state and d_inner:
        return (d_inner, d_state)

    return None


def _find_inject_attribute(module: nn.Module) -> str:
    """Find the best attribute name for injecting the initial state."""
    # Check known attribute names
    for attr in ("initial_state", "init_state", "state_init", "s0", "hidden_state"):
        if hasattr(module, attr):
            return attr
    # Default: we'll use a hook-based injection
    return "__s0_inject__"


# ══════════════════════════════════════════════════════════════════════════════
# S0 STATE CONTAINER
# ══════════════════════════════════════════════════════════════════════════════


class S0States(nn.Module):
    """Container for learnable initial state tensors.

    One state tensor per recurrent layer, all optimized jointly.
    Total parameter count = sum(state_shapes) ≈ 12.6M for Qwen3.5-4B.
    """

    def __init__(self, layer_infos: list[RecurrentLayerInfo]):
        super().__init__()
        self.layer_names: list[str] = []
        self.states = nn.ParameterList()

        for info in layer_infos:
            # Initialize at zero (paper: "Initialize S0 = 0 for each recurrent layer")
            state = nn.Parameter(torch.zeros(*info.state_shape))
            self.states.append(state)
            self.layer_names.append(info.name)

        total_params = sum(s.numel() for s in self.states)
        logger.info(f"S0States: {len(self.states)} recurrent layers, " f"{total_params/1e6:.1f}M params total")

    def get_scaled(self, alpha: float) -> list[torch.Tensor]:
        """Return alpha-scaled states for injection."""
        return [alpha * s for s in self.states]

    def l2_penalty(self) -> torch.Tensor:
        """L2 regularization over all state tensors."""
        return sum(s.pow(2).sum() for s in self.states)


# ══════════════════════════════════════════════════════════════════════════════
# INJECTION: hooks to inject S0 into the forward pass
# ══════════════════════════════════════════════════════════════════════════════


class S0InjectionHooks:
    """Manages forward hooks that inject S0 states into recurrent layers.

    The hook modifies the layer's initial_state (or equivalent) attribute
    before each forward pass. After the first token, the state evolves
    naturally with zero additional cost.
    """

    def __init__(
        self,
        layer_infos: list[RecurrentLayerInfo],
        s0_states: S0States,
        alpha: float = 0.07,
    ):
        self.layer_infos = layer_infos
        self.s0_states = s0_states
        self.alpha = alpha
        self._handles: list[Any] = []

    def register(self):
        """Register forward pre-hooks on all recurrent layers."""
        self.remove()  # clear any existing hooks

        for i, info in enumerate(self.layer_infos):
            state_param = self.s0_states.states[i]

            def make_hook(idx: int, layer_info: RecurrentLayerInfo):
                def hook(module, args, kwargs=None):
                    # Inject the scaled state
                    scaled = self.alpha * self.s0_states.states[idx]
                    # Try to set as kwarg 'initial_state' or module attribute
                    if kwargs is not None and isinstance(kwargs, dict):
                        kwargs["initial_state"] = scaled.unsqueeze(0)  # add batch dim
                        return args, kwargs
                    # Fallback: set as module attribute
                    setattr(module, layer_info.inject_attr, scaled.unsqueeze(0))
                    return None

                return hook

            # Use forward_pre_hook with kwargs support (PyTorch 2.0+)
            handle = info.module.register_forward_pre_hook(make_hook(i, info), with_kwargs=True)
            self._handles.append(handle)

        logger.info(f"Registered {len(self._handles)} S0 injection hooks (α={self.alpha})")

    def remove(self):
        """Remove all injection hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ══════════════════════════════════════════════════════════════════════════════
# S0 TRAINER
# ══════════════════════════════════════════════════════════════════════════════


class S0Trainer:
    """Complete S0 tuning trainer for hybrid recurrent-attention models.

    Freezes all model weights, discovers recurrent layers, creates learnable
    S0 states, and optimizes them on completion-only loss.

    Args:
        model: The hybrid model (Qwen3.5, FalconH1, etc.)
        alpha: State scaling factor (0.07 for Qwen3.5, 0.65 for FalconH1)
        weight_decay: L2 penalty coefficient (paper uses 1e-4)
        lr: Learning rate for S0 optimization (paper uses 1e-3)
        device: Device for training

    Example:
        trainer = S0Trainer(model, alpha=0.07)
        for epoch in range(50):
            for batch in dataloader:
                loss = trainer.step(batch)
        trainer.save("s0_qwen35.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.07,
        weight_decay: float = 1e-4,
        lr: float = 1e-3,
        device: torch.device | str = "cuda",
    ):
        self.model = model
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.device = torch.device(device) if isinstance(device, str) else device

        # 1. Freeze all model weights
        for p in model.parameters():
            p.requires_grad = False
        model.eval()  # BN/dropout off, though we use train mode for state gradients

        # 2. Discover recurrent layers
        self.layer_infos = discover_recurrent_layers(model)
        if not self.layer_infos:
            raise RuntimeError(
                "No recurrent layers found. S0 tuning requires hybrid models "
                "(GatedDeltaNet, Mamba-2, etc.) with matrix-valued states."
            )
        logger.info(f"Discovered {len(self.layer_infos)} recurrent layers:")
        for info in self.layer_infos:
            logger.info(f"  {info.name}: state_shape={info.state_shape}")

        # 3. Create learnable S0 states
        self.s0_states = S0States(self.layer_infos).to(self.device)

        # 4. Set up injection hooks
        self.hooks = S0InjectionHooks(self.layer_infos, self.s0_states, alpha)
        self.hooks.register()

        # 5. Optimizer (AdamW on S0 states only)
        self.optimizer = torch.optim.AdamW(
            self.s0_states.parameters(), lr=lr, weight_decay=0.0  # we handle WD manually
        )

    @torch.enable_grad()
    def step(self, batch: dict[str, torch.Tensor]) -> float:
        """Single optimization step on a batch.

        Args:
            batch: Dict with 'input_ids', 'attention_mask', 'labels'
                   (labels should have IGNORE_INDEX=-100 on non-completion tokens)

        Returns:
            Loss value (float)
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        IGNORE_INDEX = -100

        # Shift for next-token prediction: logits[t] predicts input_ids[t+1]
        labels = shift_labels(batch["labels"].to(self.device))

        # Forward pass (model frozen, only S0 states get gradients via hooks)
        # We need gradients to flow through the model to S0
        self.model.train()  # enable gradient flow through layers
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Completion-only CE loss
        valid_mask = labels != IGNORE_INDEX
        valid_count = valid_mask.sum().clamp(min=1)

        loss = (
            F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )
            / valid_count
        )

        # L2 regularization on S0 states
        l2 = self.s0_states.l2_penalty()
        total_loss = loss + self.weight_decay * l2

        # Backward + step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path: str | Path):
        """Save S0 states to a file (~48MB for Qwen3.5-4B)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            "s0_states": self.s0_states.state_dict(),
            "alpha": self.alpha,
            "layer_names": self.s0_states.layer_names,
        }
        torch.save(state_dict, path)
        size_mb = path.stat().st_size / 1e6
        logger.info(f"S0 states saved to {path} ({size_mb:.1f} MB)")

    def load(self, path: str | Path):
        """Load pre-trained S0 states."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.s0_states.load_state_dict(state_dict["s0_states"])
        logger.info(f"S0 states loaded from {path}")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE-TIME INJECTION (standalone, no trainer needed)
# ══════════════════════════════════════════════════════════════════════════════


def inject_s0_states(
    model: nn.Module,
    s0_path: str | Path,
    alpha: float | None = None,
) -> list:
    """Load and inject S0 states into a model for inference.

    Zero-overhead: states are injected at t=0 only, absorbed into recurrence at t≥1.
    Returns hook handles (keep reference to prevent garbage collection).

    Args:
        model: The hybrid model
        s0_path: Path to saved S0 states file
        alpha: Override alpha (if None, uses saved value)

    Returns:
        List of hook handles (keep alive for the lifetime of the model)
    """
    state_dict = torch.load(s0_path, map_location="cpu", weights_only=True)
    saved_alpha = state_dict.get("alpha", 0.07)
    alpha = alpha if alpha is not None else saved_alpha

    layer_infos = discover_recurrent_layers(model)
    s0_states = S0States(layer_infos)
    s0_states.load_state_dict(state_dict["s0_states"])

    # Move to model device
    device = next(model.parameters()).device
    s0_states = s0_states.to(device)

    hooks = S0InjectionHooks(layer_infos, s0_states, alpha)
    hooks.register()

    logger.info(f"S0 states injected (α={alpha}, {len(layer_infos)} layers)")
    return hooks  # caller must keep reference
