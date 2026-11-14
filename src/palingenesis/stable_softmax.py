"""Cross-entropy and forward KL whose gradients survive the two softmax rounding failures.

Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and Cross-Entropy" (arXiv:2610.31642):

  collapse      the target's gradient p_t - 1 rounds to exactly 0 once the target logit leads the
                others by more than a precision cliff (about 16.6 in fp32), while the other logits
                still get gradients of ~1e-7: the update pushes every other logit down and leaves
                the target alone.
  cancellation  with soft targets the gradient q - p is a difference of two separately rounded
                softmaxes, so it is not zero when student and teacher logits are equal.

`cross_entropy` sets the target logit to exactly 0 and takes everything relative to it, so the
target's gradient is minus the sum of the others' (never rounded away). `forward_kl` builds the
gradient q - p from the logit difference with log1p / expm1 and attaches it as a custom gradient,
so it is exactly 0 when the logits are equal. Both compute in fp32 and cost a few extra
element-wise passes over the logits. Opt-in (train.stable_softmax); the default paths are PyTorch's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

IGNORE_INDEX = -100


def cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = IGNORE_INDEX) -> Tensor:
    """-log softmax(logits)[target] per row [N] (0 where target == ignore_index); logits [N, V]."""
    z = logits.float()
    ignored = targets == ignore_index
    t = targets.masked_fill(ignored, 0).long()
    is_t = F.one_hot(t, z.shape[-1]).bool()
    shifted = torch.where(is_t, torch.zeros_like(z), z - z.gather(-1, t[:, None]))
    b = shifted.max(-1, keepdim=True).values.detach()  # >= 0: the target's entry is 0
    s = torch.where(is_t, torch.zeros_like(z), torch.exp(shifted - b)).sum(-1)
    b = b.squeeze(-1)
    loss = b + torch.log1p(s + torch.expm1(-b))
    return loss.masked_fill(ignored, 0.0)


def log_softmax(logits: Tensor) -> Tensor:
    """log softmax in fp32 with the largest logit's entry exactly 0 before the normaliser."""
    z = logits.float()
    i_max = z.argmax(-1, keepdim=True)
    is_max = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, i_max, True)
    shifted = torch.where(is_max, torch.zeros_like(z), z - z.gather(-1, i_max))
    s = torch.where(is_max, torch.zeros_like(z), torch.exp(shifted)).sum(-1, keepdim=True)
    return shifted - torch.log1p(s)


def forward_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """KL(p || q) per row [N], p = softmax(teacher), q = softmax(student), both [N, V] over the same
    vocabulary. The value is the usual one; the gradient with respect to the student's logits is
    q - p computed from their difference, exactly 0 when the logits are equal. No gradient reaches
    the teacher."""
    z_s = student_logits.float()
    with torch.no_grad():
        lp = log_softmax(teacher_logits)
        lq = log_softmax(z_s)
        p, q = lp.exp(), lq.exp()
        delta = z_s - teacher_logits.float()
        d = delta - (p * delta).sum(-1, keepdim=True)  # centred under p
        s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))  # log(q / p)
        u = torch.where(s.abs() < 1, p * torch.expm1(s), q - p)  # q - p
        g = u - q * u.sum(-1, keepdim=True)  # sums to 0, as a softmax gradient does
        value = torch.where(p > 0, p * (lp - lq), torch.zeros_like(p)).sum(-1)
    return value + (g * (z_s - z_s.detach())).sum(-1)
