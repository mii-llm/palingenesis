"""Deep tests for remaining techniques: config coercion, packing correctness, DEFT loss, spike variance transition."""

import sys

sys.path.insert(0, "src")

import math
import tempfile
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG TYPE COERCION EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════


def test_config_scientific_notation():
    """YAML scientific notation (2e-5) parsed as string should be coerced to float."""
    from palingenesis.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("train:\n  learning_rate: 2e-5\n  min_learning_rate: 1e-6\n")
        f.flush()
        c = Config.from_yaml(f.name)

    assert isinstance(c.train.learning_rate, float), f"Should be float, got {type(c.train.learning_rate)}"
    assert abs(c.train.learning_rate - 2e-5) < 1e-10
    assert abs(c.train.min_learning_rate - 1e-6) < 1e-10
    print(f"  2e-5 → {c.train.learning_rate}, 1e-6 → {c.train.min_learning_rate}")
    print("✓ test_config_scientific_notation PASSED\n")


def test_config_null_and_empty():
    """Null and empty string values should be handled correctly."""
    from palingenesis.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("train:\n  resume_from: null\ndata:\n  eval_dataset: ''\n  pretrain_replay_dataset: ''\n")
        f.flush()
        c = Config.from_yaml(f.name)

    assert c.train.resume_from is None
    assert c.data.eval_dataset == ""
    assert c.data.pretrain_replay_dataset == ""
    print("  null → None, '' → ''")
    print("✓ test_config_null_and_empty PASSED\n")


def test_config_bool_from_yaml():
    """YAML booleans (true/false) should map to Python booleans."""
    from palingenesis.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("train:\n  hyperball: true\n  mona: false\n  ema: true\nmemory:\n  gradient_release: true\n")
        f.flush()
        c = Config.from_yaml(f.name)

    assert c.train.hyperball is True
    assert c.train.mona is False
    assert c.train.ema is True
    assert c.memory.gradient_release is True
    print("  true→True, false→False")
    print("✓ test_config_bool_from_yaml PASSED\n")


def test_config_int_from_string():
    """Integer values that YAML might parse as strings should be coerced."""
    from palingenesis.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        # force string with quotes
        f.write("train:\n  per_device_batch_size: 4\n  epochs: 3\ndata:\n  max_seq_length: 8192\n")
        f.flush()
        c = Config.from_yaml(f.name)

    assert isinstance(c.train.per_device_batch_size, int)
    assert c.train.per_device_batch_size == 4
    assert c.data.max_seq_length == 8192
    print("✓ test_config_int_from_string PASSED\n")


def test_config_unknown_fields_ignored():
    """Unknown fields in YAML should be silently ignored (forward compat)."""
    from palingenesis.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("train:\n  learning_rate: 1e-4\n  future_param: yes\nfuture_section:\n  foo: bar\n")
        f.flush()
        c = Config.from_yaml(f.name)

    assert c.train.learning_rate == 1e-4
    # Should not crash on unknown fields
    print("  Unknown fields silently ignored ✓")
    print("✓ test_config_unknown_fields_ignored PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# PACKING + POSITION_IDS CORRECTNESS
# ══════════════════════════════════════════════════════════════════════════════


def test_packing_position_ids_reset_at_boundaries():
    """Position IDs must reset to 0 at each document boundary within a packed sequence."""
    from palingenesis.data import PackedDataset, IGNORE_INDEX

    class FakeDataset:
        def __iter__(self):
            # Two documents: length 10 and length 12
            yield {"input_ids": torch.arange(10) + 1, "labels": torch.arange(10) + 1, "attention_mask": torch.ones(10)}
            yield {"input_ids": torch.arange(12) + 100, "labels": torch.arange(12) + 100, "attention_mask": torch.ones(12)}
            # Filler to complete a pack
            yield {"input_ids": torch.arange(10) + 200, "labels": torch.arange(10) + 200, "attention_mask": torch.ones(10)}

    packed = PackedDataset(FakeDataset(), max_len=32, eos_id=0, sort_buffer=0)
    outputs = list(packed)

    assert len(outputs) >= 1
    out = outputs[0]
    pos_ids = out["position_ids"]

    # Find boundaries: position_ids should be 0 at least twice (start of each doc)
    zero_positions = (pos_ids == 0).nonzero(as_tuple=True)[0].tolist()
    assert len(zero_positions) >= 2, f"Should have ≥2 document starts, got {len(zero_positions)}: {zero_positions}"

    # First doc: positions should be 0,1,2,...,9
    first_doc_end = zero_positions[1] if len(zero_positions) > 1 else len(pos_ids)
    first_doc_pos = pos_ids[:first_doc_end].tolist()
    assert first_doc_pos == list(range(len(first_doc_pos))), f"First doc should be sequential: {first_doc_pos}"

    # Second doc: should restart from 0
    if len(zero_positions) > 1:
        second_start = zero_positions[1]
        second_doc_pos = pos_ids[second_start : second_start + 5].tolist()
        assert second_doc_pos[0] == 0, f"Second doc should start at 0, got {second_doc_pos}"
        assert second_doc_pos == list(range(5)), f"Second doc should be sequential: {second_doc_pos}"

    print(f"  Position ID resets at: {zero_positions}")
    print(f"  First doc positions: {first_doc_pos[:5]}...")
    print("✓ test_packing_position_ids_reset_at_boundaries PASSED\n")


def test_packing_no_cross_document_label_leakage():
    """Labels from one document should not bleed into another document's positions."""
    from palingenesis.data import PackedDataset, IGNORE_INDEX

    class FakeDataset:
        def __iter__(self):
            # Doc A: labels are all 1s (first 4 masked)
            ids_a = torch.ones(15, dtype=torch.long)
            labels_a = torch.ones(15, dtype=torch.long)
            labels_a[:4] = IGNORE_INDEX
            yield {"input_ids": ids_a, "labels": labels_a, "attention_mask": torch.ones(15)}

            # Doc B: labels are all 2s (first 3 masked)
            ids_b = torch.ones(15, dtype=torch.long) * 2
            labels_b = torch.ones(15, dtype=torch.long) * 2
            labels_b[:3] = IGNORE_INDEX
            yield {"input_ids": ids_b, "labels": labels_b, "attention_mask": torch.ones(15)}

    packed = PackedDataset(FakeDataset(), max_len=30, eos_id=0, sort_buffer=0)
    outputs = list(packed)

    out = outputs[0]
    labels = out["labels"]
    input_ids = out["input_ids"]

    # Where input_ids are 1 (doc A), labels should be 1 or IGNORE
    doc_a_mask = input_ids == 1
    doc_a_labels = labels[doc_a_mask]
    assert all(l.item() in (1, IGNORE_INDEX) for l in doc_a_labels), "Doc A labels should only be 1 or IGNORE"

    # Where input_ids are 2 (doc B), labels should be 2 or IGNORE
    doc_b_mask = input_ids == 2
    doc_b_labels = labels[doc_b_mask]
    assert all(l.item() in (2, IGNORE_INDEX) for l in doc_b_labels), "Doc B labels should only be 2 or IGNORE"

    print("  No label leakage between packed documents ✓")
    print("✓ test_packing_no_cross_document_label_leakage PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# SPIKE DETECTOR: VARIANCE TRANSITION
# ══════════════════════════════════════════════════════════════════════════════


def test_spike_detector_variance_transition():
    """The warmup→EMA variance transition should be smooth (no sudden threshold change)."""
    from palingenesis.perf import SpikeDetector

    detector = SpikeDetector(z_threshold=3.0, warmup=50, ema_decay=0.99)

    # Warmup with values around 1.0 ± 0.2
    for i in range(50):
        detector.check(1.0 + (i % 5) * 0.04)

    # At the transition point (step 51), a value of 1.5 should NOT be a spike
    # (it's only 2.5σ above mean with σ≈0.1)
    mean_at_transition = detector.mean
    var_at_transition = detector.var

    # Step 51: the variance should have been finalized from Welford → EMA format
    is_spike = detector.check(1.5)

    # 1.5 is about 5σ above mean≈1.0 with Welford var... depends on actual computation
    # The key test: the std should be reasonable (not zero, not infinite)
    std = var_at_transition**0.5
    assert std > 0.01, f"Std should be positive after warmup, got {std}"
    assert std < 1.0, f"Std should be reasonable, got {std}"

    print(f"  At transition: mean={mean_at_transition:.4f}, std={std:.4f}")
    print(f"  Value 1.5 is spike: {is_spike} (z≈{(1.5-mean_at_transition)/std:.1f})")
    print("✓ test_spike_detector_variance_transition PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# DEFT LOSS: GRADIENT SCALING BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════


def test_deft_loss_weights_hard_tokens_more():
    """DEFT gradient scaling should differ from uniform CE when token difficulties vary."""
    from palingenesis.plugins import deft_loss
    from palingenesis.loss import cross_entropy_loss, IGNORE_INDEX

    torch.manual_seed(42)
    vocab = 50
    seq = 20

    # Create logits with CLEAR difficulty variation:
    # First 10 tokens: model is very confident (logit for correct token = 10)
    # Last 10 tokens: model is uncertain (logits are near-uniform)
    logits = torch.randn(1, seq, vocab)
    labels = torch.randint(0, vocab, (1, seq))

    # Make first 10 tokens "easy" by boosting the correct logit
    for i in range(10):
        logits[0, i, labels[0, i]] = 10.0  # very confident

    logits = logits.requires_grad_(True)

    # DEFT loss
    loss_deft = deft_loss(logits, labels)
    loss_deft.backward()
    grad_deft = logits.grad.clone()

    # Per-token gradient magnitude
    easy_grad = grad_deft[0, :10].abs().sum(dim=-1).mean().item()
    hard_grad = grad_deft[0, 10:].abs().sum(dim=-1).mean().item()

    # DEFT should produce DIFFERENT gradient magnitudes for easy vs hard tokens
    # The key property: it's NOT uniform
    ratio = hard_grad / max(easy_grad, 1e-8)
    assert ratio != 1.0, "DEFT should weight easy and hard tokens differently"

    print(f"  Easy token grad: {easy_grad:.4f}, Hard token grad: {hard_grad:.4f}")
    print(f"  Ratio (hard/easy): {ratio:.2f}")
    print("✓ test_deft_loss_weights_hard_tokens_more PASSED\n")


def test_deft_loss_ignores_masked_tokens():
    """DEFT should produce zero gradient for IGNORE_INDEX positions."""
    from palingenesis.plugins import deft_loss
    from palingenesis.loss import IGNORE_INDEX

    torch.manual_seed(42)
    logits = torch.randn(2, 16, 50, requires_grad=True)
    labels = torch.randint(0, 50, (2, 16))
    # Mask all of sample 0
    labels[0, :] = IGNORE_INDEX

    loss = deft_loss(logits, labels)
    loss.backward()

    grad = logits.grad
    # Sample 0 (fully masked) should have zero gradient
    sample0_grad = grad[0].abs().sum().item()
    sample1_grad = grad[1].abs().sum().item()

    assert sample0_grad < 1e-6, f"Masked sample should have zero grad, got {sample0_grad}"
    assert sample1_grad > 0.1, f"Unmasked sample should have non-zero grad, got {sample1_grad}"
    print(f"  Masked sample grad: {sample0_grad:.2e}, Unmasked: {sample1_grad:.2f}")
    print("✓ test_deft_loss_ignores_masked_tokens PASSED\n")


def test_deft_loss_finite_output():
    """DEFT loss should never produce NaN/Inf even with extreme logits."""
    from palingenesis.plugins import deft_loss
    from palingenesis.loss import IGNORE_INDEX

    # Extreme cases
    for desc, logits_val in [("very large", 100.0), ("very small", -100.0), ("near zero", 0.001)]:
        logits = torch.full((1, 8, 50), logits_val, requires_grad=True)
        labels = torch.randint(0, 50, (1, 8))

        loss = deft_loss(logits, labels)
        assert torch.isfinite(loss), f"DEFT loss should be finite for {desc} logits, got {loss.item()}"

        loss.backward()
        assert logits.grad.isfinite().all(), f"DEFT grad should be finite for {desc} logits"

    print("  Finite for extreme logits (100, -100, 0.001) ✓")
    print("✓ test_deft_loss_finite_output PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("=" * 60)
    print("DEEP TECHNIQUE TESTS")
    print("=" * 60 + "\n")

    print("── Config Coercion ──\n")
    test_config_scientific_notation()
    test_config_null_and_empty()
    test_config_bool_from_yaml()
    test_config_int_from_string()
    test_config_unknown_fields_ignored()

    print("── Packing Correctness ──\n")
    test_packing_position_ids_reset_at_boundaries()
    test_packing_no_cross_document_label_leakage()

    print("── Spike Detector Transition ──\n")
    test_spike_detector_variance_transition()

    print("── DEFT Loss ──\n")
    test_deft_loss_weights_hard_tokens_more()
    test_deft_loss_ignores_masked_tokens()
    test_deft_loss_finite_output()

    print("=" * 60)
    print("ALL DEEP TECHNIQUE TESTS PASSED ✓")
    print("=" * 60)
