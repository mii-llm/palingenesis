"""Tests for shift_labels (next-token alignment) and the DFT loss family.

shift_labels is the single most safety-critical function in the loss path:
without it the model is trained to COPY its input instead of predicting the
next token. These tests pin down:
- the basic left-shift semantics
- exact equivalence with the HuggingFace-internal shift convention
- IGNORE_INDEX propagation
- packed-sequence document boundaries (no cross-document prediction)

The DFT family (dft, cadft, info_sft) previously only had config-level
mutual-exclusivity tests; here we verify their numerics.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.loss import IGNORE_INDEX, cross_entropy_loss, shift_labels  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# SHIFT_LABELS: NEXT-TOKEN ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════


def test_shift_labels_basic():
    """Labels move one position left; the final position is always masked."""
    labels = torch.tensor([[10, 11, 12, 13]])
    shifted = shift_labels(labels)

    assert shifted.tolist() == [[11, 12, 13, IGNORE_INDEX]]
    print("✓ test_shift_labels_basic PASSED")


def test_shift_labels_matches_hf_convention():
    """shift_labels + our CE must equal the HuggingFace-internal shift.

    HF models with labels= compute:
        CE(logits[:, :-1], labels[:, 1:])
    We compute:
        CE(logits, shift_labels(labels))
    These must be numerically identical.
    """
    torch.manual_seed(0)
    B, S, V = 2, 12, 50
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    labels[0, :3] = IGNORE_INDEX  # masked prompt region

    # HF-style shift
    hf_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, V).float(),
        labels[:, 1:].reshape(-1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )

    # Our shift
    shifted = shift_labels(labels)
    ours = cross_entropy_loss(logits, shifted, global_valid_tokens=1.0)

    assert torch.allclose(ours, hf_loss, atol=1e-5), f"{ours.item()} != {hf_loss.item()}"
    print(f"  our={ours.item():.4f} hf={hf_loss.item():.4f}")
    print("✓ test_shift_labels_matches_hf_convention PASSED")


def test_shift_labels_perfect_predictor_gets_zero_loss():
    """A model whose logits at position t point to token t+1 must get ~0 loss
    after the shift — and a LARGE loss without it (the copy-objective bug)."""
    V, S = 20, 8
    tokens = torch.arange(1, S + 1) % V  # arbitrary sequence
    labels = tokens.unsqueeze(0).clone()

    # Perfect next-token predictor: logits[t] peaked at tokens[t+1]
    logits = torch.full((1, S, V), -10.0)
    for t in range(S - 1):
        logits[0, t, tokens[t + 1]] = 10.0
    logits[0, S - 1, 0] = 10.0  # last position: irrelevant, gets masked

    valid = (S - 1)
    loss_shifted = cross_entropy_loss(logits, shift_labels(labels), global_valid_tokens=valid)
    loss_unshifted = cross_entropy_loss(logits, labels, global_valid_tokens=S)

    assert loss_shifted.item() < 0.01, f"Perfect predictor should have ~0 loss, got {loss_shifted.item()}"
    assert loss_unshifted.item() > 5.0, "Unshifted (copy objective) should be heavily penalized"
    print(f"  shifted={loss_shifted.item():.4f} unshifted={loss_unshifted.item():.2f}")
    print("✓ test_shift_labels_perfect_predictor_gets_zero_loss PASSED")


def test_shift_labels_preserves_ignore_index():
    """Masked positions travel with the shift: a masked label at position k
    means position k-1 has nothing to predict."""
    labels = torch.tensor([[5, IGNORE_INDEX, IGNORE_INDEX, 8, 9]])
    shifted = shift_labels(labels)

    assert shifted.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 8, 9, IGNORE_INDEX]]
    print("✓ test_shift_labels_preserves_ignore_index PASSED")


def test_shift_labels_packed_document_boundaries():
    """In packed sequences, the last token of a document must NOT be trained
    to predict the first token of the next document."""
    # Two packed documents of 3 tokens each; position_ids reset at doc 2
    labels = torch.tensor([[10, 11, 12, 20, 21, 22]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])

    shifted = shift_labels(labels, position_ids)

    # Index 2 is the end of doc 1 → must be masked (would otherwise predict 20)
    assert shifted[0, 2].item() == IGNORE_INDEX, "Cross-document prediction must be masked"
    # Within-document shifts are normal
    assert shifted[0, :2].tolist() == [11, 12]
    assert shifted[0, 3:5].tolist() == [21, 22]
    # Final position always masked
    assert shifted[0, 5].item() == IGNORE_INDEX
    print("✓ test_shift_labels_packed_document_boundaries PASSED")


def test_shift_labels_does_not_mutate_input():
    labels = torch.tensor([[1, 2, 3]])
    original = labels.clone()
    shift_labels(labels)
    assert torch.equal(labels, original), "shift_labels must not modify its input"
    print("✓ test_shift_labels_does_not_mutate_input PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# DFT: p_t-WEIGHTED CROSS-ENTROPY
# ══════════════════════════════════════════════════════════════════════════════


def test_dft_loss_equals_pt_weighted_ce():
    """DFT is exactly sum(p_t * CE_t) over valid tokens."""
    from palingenesis.plugins import dft_loss

    torch.manual_seed(1)
    B, S, V = 2, 10, 40
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    labels[1, 5:] = IGNORE_INDEX

    loss = dft_loss(logits, labels)

    # Manual reference
    probs = torch.softmax(logits.float(), dim=-1)
    p_t = probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="none", ignore_index=IGNORE_INDEX)
    valid = labels != IGNORE_INDEX
    expected = (ce.view(B, S) * p_t * valid).sum()

    assert torch.allclose(loss, expected, atol=1e-4), f"{loss.item()} != {expected.item()}"
    print(f"  dft={loss.item():.4f} expected={expected.item():.4f}")
    print("✓ test_dft_loss_equals_pt_weighted_ce PASSED")


def test_dft_loss_masked_and_finite():
    from palingenesis.plugins import dft_loss

    torch.manual_seed(2)
    logits = torch.randn(2, 8, 30, requires_grad=True)
    labels = torch.randint(0, 30, (2, 8))
    labels[0, :] = IGNORE_INDEX

    loss = dft_loss(logits, labels)
    loss.backward()
    assert logits.grad[0].abs().sum().item() < 1e-6, "Masked sample must have zero gradient"
    assert torch.isfinite(loss), "DFT loss must be finite"

    # Extreme logits stay finite
    extreme = torch.full((1, 4, 30), 100.0, requires_grad=True)
    lbl = torch.randint(0, 30, (1, 4))
    loss2 = dft_loss(extreme, lbl)
    assert torch.isfinite(loss2)
    print("✓ test_dft_loss_masked_and_finite PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# CADFT: SAMPLE-LEVEL COMPATIBILITY WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════


def test_cadft_downweights_incompatible_samples():
    """A sample with much higher NLL than the rest of the batch gets weight < 1;
    compatible samples keep weight 1."""
    from palingenesis.plugins import compute_sample_compatibility

    torch.manual_seed(3)
    B, S, V = 4, 10, 40
    labels = torch.randint(0, V, (B, S))

    # Samples 0-2: model is confident on the correct tokens. Sample 3: hopeless.
    logits = torch.randn(B, S, V)
    for b in range(3):
        for t in range(S):
            logits[b, t, labels[b, t]] = 8.0
    logits[3] = torch.randn(S, V)  # near-uniform → high NLL

    weights = compute_sample_compatibility(logits, labels, beta=1.0)

    assert weights.shape == (B,)
    assert weights[3] < 0.5, f"Incompatible sample should be heavily down-weighted, got {weights[3].item()}"
    for b in range(3):
        assert weights[b] > 0.9, f"Compatible sample {b} should keep ~full weight, got {weights[b].item()}"
    print(f"  weights={[round(w, 3) for w in weights.tolist()]}")
    print("✓ test_cadft_downweights_incompatible_samples PASSED")


def test_cadft_loss_masked_and_finite():
    from palingenesis.plugins import cadft_loss

    torch.manual_seed(4)
    logits = torch.randn(3, 8, 30, requires_grad=True)
    labels = torch.randint(0, 30, (3, 8))
    labels[1, :] = IGNORE_INDEX

    loss = cadft_loss(logits, labels, beta=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[1].abs().sum().item() < 1e-6, "Fully masked sample must have zero gradient"
    assert logits.grad[0].abs().sum().item() > 0, "Unmasked samples must receive gradient"
    print("✓ test_cadft_loss_masked_and_finite PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# INFOSFT: INFORMATION-AWARE TOKEN WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════


def test_infosft_zeroes_overconfident_tokens():
    """Tokens where q > p_bar carry zero InfoSFT weight (nothing left to learn).
    A batch of ONLY over-confident tokens therefore has ~zero loss."""
    from palingenesis.plugins import infosft_weighted_loss

    V, S = 30, 6
    labels = torch.randint(0, V, (1, S))
    logits = torch.full((1, S, V), -10.0)
    for t in range(S):
        logits[0, t, labels[0, t]] = 10.0  # q ≈ 1 > p_bar

    loss = infosft_weighted_loss(logits, labels, p_bar=0.93)
    assert loss.item() < 1e-3, f"Over-confident tokens should contribute ~0, got {loss.item()}"
    print(f"  loss={loss.item():.2e}")
    print("✓ test_infosft_zeroes_overconfident_tokens PASSED")


def test_infosft_focuses_medium_confidence():
    """Gradient should concentrate on medium-confidence tokens, not on
    over-confident ones."""
    from palingenesis.plugins import infosft_weighted_loss

    torch.manual_seed(5)
    V, S = 30, 8
    labels = torch.randint(0, V, (1, S))
    logits = torch.zeros(1, S, V)
    # First 4 tokens: over-confident (q ≈ 1). Last 4: medium confidence.
    for t in range(4):
        logits[0, t, labels[0, t]] = 12.0
    for t in range(4, S):
        logits[0, t, labels[0, t]] = 2.0  # q ≈ 0.2, well below p_bar

    logits = logits.requires_grad_(True)
    loss = infosft_weighted_loss(logits, labels, p_bar=0.93)
    loss.backward()

    overconf_grad = logits.grad[0, :4].abs().sum().item()
    medium_grad = logits.grad[0, 4:].abs().sum().item()
    assert medium_grad > 10 * max(overconf_grad, 1e-8), (
        f"Medium-confidence tokens should dominate the gradient "
        f"(medium={medium_grad:.4f}, overconfident={overconf_grad:.4f})"
    )
    print(f"  medium_grad={medium_grad:.4f} overconf_grad={overconf_grad:.2e}")
    print("✓ test_infosft_focuses_medium_confidence PASSED")


def test_infosft_masked_and_finite():
    from palingenesis.plugins import infosft_weighted_loss

    torch.manual_seed(6)
    logits = torch.randn(2, 8, 30, requires_grad=True)
    labels = torch.randint(0, 30, (2, 8))
    labels[0, :] = IGNORE_INDEX

    loss = infosft_weighted_loss(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0].abs().sum().item() < 1e-6, "Masked sample must have zero gradient"
    print("✓ test_infosft_masked_and_finite PASSED")


def test_chunked_deft_matches_unchunked_and_reports_ce():
    """Chunked DEFT must equal unchunked DEFT exactly, and its free side
    stats (unweighted CE, trust gate) must be exact.

    The stats justify why train/loss looks 'very low' under DEFT: the gated
    objective is bounded above by the true CE (gate = p^alpha <= 1)."""
    from palingenesis.plugins import _deft_loss_fused, chunked_deft_loss

    torch.manual_seed(7)
    B, S, D, V = 2, 12, 16, 40
    hidden = torch.randn(B, S, D, requires_grad=True)
    lm_head = torch.nn.Linear(D, V, bias=False)
    labels = torch.randint(0, V, (B, S))
    labels[0, :4] = IGNORE_INDEX
    valid = (labels != IGNORE_INDEX).sum().item()

    stats: dict = {}
    loss_chunked = chunked_deft_loss(
        hidden, labels, lm_head, num_chunks=3, global_valid_tokens=valid, stats=stats
    )

    logits = lm_head(hidden.detach())
    loss_ref = _deft_loss_fused(logits, labels) / valid
    torch.testing.assert_close(loss_chunked.detach(), loss_ref, atol=1e-5, rtol=1e-4)

    # Side stats: exact CE and gate over valid tokens
    assert stats["valid"] == valid
    ce_ref = F.cross_entropy(
        logits.reshape(-1, V), labels.reshape(-1), reduction="sum", ignore_index=IGNORE_INDEX
    ).item()
    assert abs(stats["ce_sum"] - ce_ref) / ce_ref < 1e-3, f"{stats['ce_sum']} vs {ce_ref}"
    mean_gate = stats["gate_sum"] / stats["valid"]
    assert 0.0 < mean_gate <= 1.0

    # DEFT <= CE always (gate <= 1): this is why train/loss << eval CE
    assert loss_chunked.item() <= ce_ref / valid + 1e-6

    # Gradients still flow through the chunked path
    loss_chunked.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    print("✓ test_chunked_deft_matches_unchunked_and_reports_ce PASSED")


if __name__ == "__main__":
    test_shift_labels_basic()
    test_shift_labels_matches_hf_convention()
    test_shift_labels_perfect_predictor_gets_zero_loss()
    test_shift_labels_preserves_ignore_index()
    test_shift_labels_packed_document_boundaries()
    test_shift_labels_does_not_mutate_input()
    test_dft_loss_equals_pt_weighted_ce()
    test_dft_loss_masked_and_finite()
    test_cadft_downweights_incompatible_samples()
    test_cadft_loss_masked_and_finite()
    test_infosft_zeroes_overconfident_tokens()
    test_infosft_focuses_medium_confidence()
    test_infosft_masked_and_finite()
    test_chunked_deft_matches_unchunked_and_reports_ce()
    print("\nALL LABEL-SHIFT + LOSS-FAMILY TESTS PASSED ✓")
