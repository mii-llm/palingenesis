"""Chunked gated objectives (memory.chunked_loss with DEFT, DFT, InfoSFT, CADFT) equal
their full-logit definitions in plugins.py, in value and in gradient.

Before the chunked path existed for all four, `memory.chunked_loss: true` (the default)
silently replaced DFT, InfoSFT and CADFT with plain chunked cross-entropy.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis import plugins  # noqa: E402

REFERENCE = {
    "deft": lambda logits, labels: plugins._deft_loss_fused(logits, labels),
    "dft": lambda logits, labels: plugins._dft_loss_fused(logits, labels),
    "info_sft": lambda logits, labels: plugins._infosft_fused(logits, labels, torch.logit(torch.tensor(0.93)).item()),
    "cadft": lambda logits, labels: plugins.cadft_loss(logits, labels, beta=1.0),
}


@pytest.mark.parametrize("objective", plugins.GATED_OBJECTIVES)
@pytest.mark.parametrize("num_chunks", [1, 3])
def test_chunked_equals_full_logits(objective, num_chunks):
    torch.manual_seed(0)
    B, S, D, V = 3, 11, 16, 50
    head = torch.nn.Linear(D, V, bias=False).double()
    hidden = torch.randn(B, S, D, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, V, (B, S))
    labels[:, :3] = -100
    labels[1, :8] = -100
    denom = 7.0

    ref = REFERENCE[objective](head(hidden), labels) / denom
    ref.backward()
    ref_h, ref_w = hidden.grad.clone(), head.weight.grad.clone()
    hidden.grad, head.weight.grad = None, None

    got = plugins.chunked_gated_loss(hidden, labels, head, objective, num_chunks=num_chunks, global_valid_tokens=denom)
    got.backward()
    torch.testing.assert_close(
        got, ref.float() if got.dtype == torch.float32 else ref, rtol=1e-5, atol=1e-7, check_dtype=False
    )
    torch.testing.assert_close(hidden.grad, ref_h, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(head.weight.grad, ref_w, rtol=1e-5, atol=1e-8)
