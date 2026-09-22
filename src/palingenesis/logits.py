"""The model's output head, including its logit post-processing.

Memory-efficient losses (chunked CE, DPO, SeCO) project hidden states to logits
themselves, chunk by chunk, instead of calling the model's forward. Several
architectures transform the `lm_head` output inside that forward, and a loss
that skips the transform trains on the wrong logits:

    Cohere / Cohere2          logits * logit_scale
    Granite family            logits / logits_scaling
    HyperCLOVA X              logits * logits_scaling
    Falcon-H1                 logits * lm_head_multiplier
    Gemma 2/3/3n/4, VaultGemma, NanoChat
                              cap * tanh(logits / cap)   (final_logit_softcapping)

`output_head(model)` returns the head with that transform applied, and
`verify_output_head` checks it against the model's own forward, so an unknown
transform is caught instead of silently mis-training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PostProcessedHead(nn.Module):
    """`lm_head` followed by the model's logit transform. `.weight` is the
    underlying projection's (for code that reads it)."""

    def __init__(self, head: nn.Module, multiplier: float = 1.0, softcap: float | None = None):
        super().__init__()
        self.head = head
        self.multiplier = multiplier
        self.softcap = softcap

    @property
    def weight(self) -> torch.Tensor:
        return self.head.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.head(hidden)
        if self.multiplier != 1.0:
            logits = logits * self.multiplier
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def extra_repr(self) -> str:
        return f"multiplier={self.multiplier}, softcap={self.softcap}"


def _transform(model: nn.Module) -> tuple[float, float | None]:
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    model_type = (getattr(model.config, "model_type", "") or "").lower()
    multiplier = 1.0
    if getattr(config, "logit_scale", None) is not None:                    # Cohere
        multiplier *= float(config.logit_scale)
    if getattr(config, "logits_scaling", None) is not None:
        if model_type.startswith("hyperclovax"):
            multiplier *= float(config.logits_scaling)
        else:                                                                # Granite family
            multiplier /= float(config.logits_scaling)
    if getattr(config, "lm_head_multiplier", None) is not None:             # Falcon-H1
        multiplier *= float(config.lm_head_multiplier)
    softcap = getattr(config, "final_logit_softcapping", None)
    return multiplier, (float(softcap) if softcap else None)


def output_head(model: nn.Module) -> nn.Module | None:
    """The module mapping final hidden states to the logits the model's forward
    returns: the raw `lm_head` when the model applies no transform."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else getattr(model, "lm_head", None)
    if head is None:
        return None
    multiplier, softcap = _transform(model)
    if multiplier == 1.0 and softcap is None:
        return head
    return PostProcessedHead(head, multiplier, softcap)


@torch.no_grad()
def verify_output_head(model: nn.Module, head: nn.Module, backbone: nn.Module, tokens: int = 32) -> float:
    """Raise if head(backbone(x)) differs from model(x).logits. Random tokens,
    eval mode (restored afterwards). Returns the relative max difference."""
    device = next(model.parameters()).device
    vocab = head.weight.shape[0]
    ids = torch.randint(0, vocab, (1, tokens), generator=torch.Generator().manual_seed(0)).to(device)
    was_training = model.training
    model.eval()
    matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")       # compare implementations, not TF32 noise
    try:
        reference = model(input_ids=ids).logits.float()
        out = backbone(input_ids=ids)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        ours = head(hidden).float()
    finally:
        model.train(was_training)
        torch.set_float32_matmul_precision(matmul_precision)
    diff = float((ours - reference).abs().max() / reference.abs().max().clamp(min=1e-30))
    dtype = next(model.parameters()).dtype
    tolerance = 1e-4 if dtype in (torch.float32, torch.float64) else 2e-2
    if diff > tolerance:
        raise NotImplementedError(
            f"The model's forward transforms the lm_head output in a way palingenesis does not replicate "
            f"(relative difference {diff:.1e}). Chunked losses would train on the wrong logits; disable "
            "memory.chunked_loss (and DPO/SeCO, which need it) for this model."
        )
    return diff
