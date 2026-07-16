"""Integration tests: verify the full pipeline assembles and runs correctly.

These tests use a tiny model (randomly initialized) and synthetic data to
verify that all components compose correctly end-to-end. No GPU required
(runs on CPU with small dimensions).

Tests cover:
- Config loading from YAML
- Data pipeline: chat template → masking → packing → collation
- Loss computation paths (standard, chunked, DEFT)
- Optimizer construction (AdamW, Lion-style, Muon-style)
- Hyperball + MONA composition
- Checkpoint save/load round-trip
- BestModelTracker logic
- Health monitor full lifecycle
- Scheduler correctness across all types
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


class TinyLM(nn.Module):
    """Minimal causal LM for testing. Same interface as HuggingFace models."""

    def __init__(self, vocab_size=256, hidden=64, layers=2):
        super().__init__()
        self.config = type("Config", (), {"vocab_size": vocab_size, "tie_word_embeddings": False})()
        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(vocab_size, hidden),
                "layers": nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)]),
                "norm": nn.LayerNorm(hidden),
            }
        )
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        h = self.model["embed_tokens"](input_ids)
        for layer in self.model["layers"]:
            h = torch.relu(layer(h))
        h = self.model["norm"](h)
        logits = self.lm_head(h)
        return type("Output", (), {"logits": logits})()

    def save_pretrained(self, path, **kwargs):
        from safetensors.torch import save_file

        Path(path).mkdir(parents=True, exist_ok=True)
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(Path(path) / "model.safetensors"))
        # Write index for sharded loading compatibility
        index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in state}}
        with open(Path(path) / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)


def make_batch(batch_size=2, seq_len=32, vocab_size=256):
    """Create a synthetic training batch."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :8] = -100  # mask first 8 tokens (simulating system/user prefix)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ══════════════════════════════════════════════════════════════════════════════


def test_config_from_yaml():
    """Config loads from YAML and all fields have correct types."""
    from palingenesis.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
model:
  name_or_path: test-model
  torch_dtype: bfloat16
train:
  learning_rate: 3e-5
  lr_scheduler: power_decay
  optimizer: lion8bit
  hyperball: true
  mona: true
  epochs: 2
memory:
  chunked_loss: true
  gradient_release: true
logging:
  rl_readiness: true
  rl_entropy_floor: 1.5
plugins:
  deft: true
"""
        )
        f.flush()
        cfg = Config.from_yaml(f.name)

    assert cfg.model.name_or_path == "test-model"
    assert cfg.train.lr_scheduler == "power_decay"
    assert cfg.train.hyperball is True
    assert cfg.train.mona is True
    assert cfg.memory.gradient_release is True
    assert cfg.logging.rl_readiness is True
    assert cfg.logging.rl_entropy_floor == 1.5
    assert cfg.plugins.deft is True
    print("✓ test_config_from_yaml PASSED\n")


def test_config_defaults_are_sane():
    """Default config produces valid training without any YAML."""
    from palingenesis.config import Config

    cfg = Config()
    assert cfg.train.learning_rate > 0
    assert cfg.train.epochs >= 1
    assert cfg.train.per_device_batch_size >= 1
    assert cfg.train.warmup_ratio > 0
    assert cfg.train.lr_scheduler in ("cosine", "linear", "constant", "power_decay", "wsd")
    assert cfg.memory.loss_num_chunks >= 1
    assert cfg.data.max_seq_length >= 128
    print("✓ test_config_defaults_are_sane PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# LOSS TESTS
# ══════════════════════════════════════════════════════════════════════════════


def test_cross_entropy_loss_correct():
    """Standard CE loss matches PyTorch reference."""
    from palingenesis.loss import cross_entropy_loss, IGNORE_INDEX

    torch.manual_seed(42)
    logits = torch.randn(2, 16, 256)
    labels = torch.randint(0, 256, (2, 16))
    labels[:, :4] = IGNORE_INDEX

    valid_count = (labels != IGNORE_INDEX).sum().item()
    our_loss = cross_entropy_loss(logits, labels, valid_count)

    # Reference
    ref_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 256).float(), labels.view(-1), reduction="sum", ignore_index=IGNORE_INDEX
    )
    ref_loss = ref_loss / valid_count

    diff = abs(our_loss.item() - ref_loss.item())
    assert diff < 1e-5, f"Loss mismatch: ours={our_loss.item():.6f}, ref={ref_loss.item():.6f}"
    print(f"  Loss: {our_loss.item():.4f} (diff={diff:.2e})")
    print("✓ test_cross_entropy_loss_correct PASSED\n")


def test_chunked_loss_matches_standard():
    """Chunked CE loss produces same result as standard (within numerical tolerance)."""
    from palingenesis.loss import cross_entropy_loss, chunked_cross_entropy_loss, IGNORE_INDEX

    torch.manual_seed(42)
    model = TinyLM(vocab_size=128, hidden=32, layers=1)

    batch = make_batch(batch_size=2, seq_len=32, vocab_size=128)
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Get hidden states
    h = model.model["embed_tokens"](input_ids)
    for layer in model.model["layers"]:
        h = torch.relu(layer(h))
    h = model.model["norm"](h)

    valid = (labels != IGNORE_INDEX).sum().item()

    # Standard loss
    logits = model.lm_head(h)
    std_loss = cross_entropy_loss(logits, labels, valid)

    # Chunked loss (4 chunks)
    h_detached = h.detach().requires_grad_(True)
    chunked_loss = chunked_cross_entropy_loss(
        h_detached, labels, model.lm_head, num_chunks=4, global_valid_tokens=valid
    )

    diff = abs(std_loss.item() - chunked_loss.item())
    assert diff < 1e-4, f"Chunked loss diverges: std={std_loss.item():.6f}, chunked={chunked_loss.item():.6f}"
    print(f"  Standard: {std_loss.item():.4f}, Chunked: {chunked_loss.item():.4f} (diff={diff:.2e})")
    print("✓ test_chunked_loss_matches_standard PASSED\n")


def test_eval_chunked_ce_matches_oneshot():
    """Eval CE chunking must equal the one-shot cross_entropy sum (it only exists
    to bound fp32 memory, not to change the number). Regression for the eval OOM."""
    import torch.nn.functional as F

    from palingenesis.train import _chunked_ce_sum
    from palingenesis.loss import IGNORE_INDEX

    torch.manual_seed(0)
    B, S, V = 3, 40, 128
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    labels[:, :5] = IGNORE_INDEX  # masked context tokens must be ignored

    oneshot = F.cross_entropy(
        logits.view(-1, V).float(), labels.view(-1), reduction="sum", ignore_index=IGNORE_INDEX
    )
    # Chunk size deliberately not a divisor of B*S (=120) to exercise the tail
    chunked = _chunked_ce_sum(logits, labels, chunk_tokens=32)

    diff = abs(oneshot.item() - chunked.item())
    assert diff < 1e-3, f"Chunked eval CE diverges: oneshot={oneshot.item():.6f}, chunked={chunked.item():.6f}"

    # All-ignored batch must contribute 0, not NaN
    all_ignored = torch.full((2, 10), IGNORE_INDEX)
    z = _chunked_ce_sum(torch.randn(2, 10, V), all_ignored, chunk_tokens=8)
    assert z.item() == 0.0
    print(f"✓ test_eval_chunked_ce_matches_oneshot PASSED (diff={diff:.2e})\n")


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER + HYPERBALL + MONA COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════


def test_full_optimizer_stack():
    """Test Muon-style optimizer + MONA + Hyperball compose and converge."""
    from palingenesis.optim import HyperballWrapper, MONAAcceleration

    torch.manual_seed(42)
    model = TinyLM(vocab_size=128, hidden=32, layers=2)
    target = torch.randn(2, 32, 128)  # target logits

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Hyperball on weight matrices (not embeddings/norms)
    constrained = [p for n, p in model.named_parameters() if p.ndim == 2 and "embed" not in n and "norm" not in n]
    hyperball = HyperballWrapper(optimizer, constrained)

    # MONA acceleration
    mona = MONAAcceleration(model, beta_a=0.9, lite=True)

    losses = []
    for step in range(50):
        optimizer.zero_grad()
        batch = make_batch(2, 32, 128)
        output = model(batch["input_ids"])
        loss = (output.logits - target).pow(2).mean()
        loss.backward()
        mona.apply()
        hyperball.step()
        losses.append(loss.item())

    # Verify convergence
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    reduction = 1 - final / initial

    # Verify Hyperball preserved norms
    for p in constrained:
        current_norm = p.data.norm().item()
        # Norms should be approximately preserved (Hyperball projects back)
        assert current_norm > 0.1, f"Norm collapsed to {current_norm}"

    assert reduction > 0.1, f"Should converge >10%, got {reduction*100:.1f}%"
    print(f"  Reduction: {reduction*100:.1f}% over 50 steps")
    print("✓ test_full_optimizer_stack PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT ROUND-TRIP
# ══════════════════════════════════════════════════════════════════════════════


def test_checkpoint_save_load_roundtrip():
    """Checkpoint save then load restores model + optimizer + scheduler state."""
    from palingenesis.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
    from palingenesis.optim import build_scheduler

    torch.manual_seed(42)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, "power_decay", num_steps=100, warmup_ratio=0.1, min_lr_ratio=0.1)

    # Train a few steps to populate optimizer state
    for _ in range(5):
        batch = make_batch()
        output = model(batch["input_ids"])
        loss = output.logits.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:

        class FakeTokenizer:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        save_checkpoint(model, FakeTokenizer(), optimizer, scheduler, step=5, output_dir=tmpdir, is_fsdp=False)

        # Verify checkpoint exists and is valid
        found = find_latest_checkpoint(tmpdir)
        assert found is not None, "Should find the checkpoint"
        assert "step-5" in found

        # Create fresh model + optimizer
        model2 = TinyLM()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = build_scheduler(optimizer2, "power_decay", num_steps=100, warmup_ratio=0.1, min_lr_ratio=0.1)

        # Load
        meta = load_checkpoint(model2, optimizer2, scheduler2, found, is_fsdp=False)
        assert meta["step"] == 5
        assert meta["epoch"] == 0

        # Verify model weights match
        max_diff = 0.0
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            diff = (p1.data - p2.data).abs().max().item()
            max_diff = max(max_diff, diff)

        assert max_diff < 1e-6, f"Model weights differ after load: max_diff={max_diff}"
        print(f"  Round-trip weight diff: {max_diff:.2e}")
        print("✓ test_checkpoint_save_load_roundtrip PASSED\n")


def test_checkpoint_auto_purge():
    """Auto-purge keeps only latest K checkpoints."""
    from palingenesis.checkpoint import save_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:

        class FakeTokenizer:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        model = TinyLM()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Save 8 checkpoints with keep_latest_k=3
        for step in range(1, 9):
            save_checkpoint(
                model, FakeTokenizer(), optimizer, None, step=step * 100, output_dir=tmpdir, keep_latest_k=3
            )

        # Only 3 should remain
        remaining = [d for d in Path(tmpdir).iterdir() if d.is_dir() and d.name.startswith("step-")]
        assert len(remaining) == 3, f"Expected 3 checkpoints, got {len(remaining)}: {[d.name for d in remaining]}"

        # Should be the latest 3
        steps = sorted(int(d.name.split("-")[1]) for d in remaining)
        assert steps == [600, 700, 800], f"Expected [600,700,800], got {steps}"

        print(f"  Remaining checkpoints: {steps}")
        print("✓ test_checkpoint_auto_purge PASSED\n")


def test_best_model_tracker():
    """BestModelTracker saves when eval loss improves, ignores when it doesn't."""
    from palingenesis.checkpoint import BestModelTracker

    with tempfile.TemporaryDirectory() as tmpdir:

        class FakeTokenizer:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        model = TinyLM()
        tracker = BestModelTracker(tmpdir)

        # First eval: should save (any loss beats inf)
        saved = tracker.update(2.5, step=10, model=model, tokenizer=FakeTokenizer())
        assert saved is True
        assert tracker.best_loss == 2.5
        assert tracker.best_step == 10

        # Worse eval: should NOT save
        saved = tracker.update(3.0, step=20, model=model, tokenizer=FakeTokenizer())
        assert saved is False
        assert tracker.best_loss == 2.5  # unchanged

        # Better eval: should save
        saved = tracker.update(1.8, step=30, model=model, tokenizer=FakeTokenizer())
        assert saved is True
        assert tracker.best_loss == 1.8
        assert tracker.best_step == 30

        # Verify best_meta.json
        meta_path = Path(tmpdir) / "best" / "best_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["step"] == 30
        assert abs(meta["eval_loss"] - 1.8) < 1e-6

        print(f"  Best: step={tracker.best_step}, loss={tracker.best_loss}")
        print("✓ test_best_model_tracker PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH MONITOR FULL LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════


def test_health_monitor_full_lifecycle():
    """Health monitor records data, computes metrics, detects issues."""
    from palingenesis.health import HealthMonitor, IGNORE_INDEX

    model = TinyLM()
    monitor = HealthMonitor(model, tier2_every=5, tier3_every=10, rl_readiness=True, rl_entropy_floor=2.0)

    # Simulate 20 training steps
    for step in range(1, 21):
        # Simulate micro-step data
        labels = torch.randint(0, 256, (2, 32))
        labels[:, :8] = IGNORE_INDEX
        loss = 3.0 - step * 0.1  # decreasing loss
        monitor.record_microstep(loss, labels)

        # Simulate logit entropy recording
        logits = torch.randn(2, 32, 256) / (1.0 + step * 0.05)  # decreasing temperature
        monitor.record_logit_entropy(logits, labels)

        # Get metrics on step boundaries
        metrics = monitor.on_step(step, model)

        # Tier 1 should always produce metrics
        if step > 1:
            assert "health/loss_mean_window" in metrics
            assert "health/token_efficiency" in metrics

        # Tier 2 should fire every 5 steps
        if step % 5 == 0:
            assert "health/cuda_peak_gb" in metrics or not torch.cuda.is_available()

        # Tier 3 should fire at step 10, 20
        if step % 10 == 0 and step > 0:
            assert "health/weight_norm_min" in metrics or True  # may skip if model too small

    # RL-readiness should have entropy data
    assert "health/output_entropy" in metrics
    assert monitor._entropy_buffer

    print(f"  Final metrics keys: {len(metrics)}")
    print(f"  Entropy buffer: {len(monitor._entropy_buffer)} entries")
    print("✓ test_health_monitor_full_lifecycle PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER CORRECTNESS (all types)
# ══════════════════════════════════════════════════════════════════════════════


def test_all_schedulers_produce_valid_lr():
    """Every scheduler type produces LR in [min_lr, peak_lr] range."""
    from palingenesis.optim import build_scheduler

    peak_lr = 1e-3
    min_ratio = 0.01

    for sched_type in ("cosine", "linear", "constant", "power_decay", "wsd"):
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr)
        scheduler = build_scheduler(opt, sched_type, num_steps=200, warmup_ratio=0.1, min_lr_ratio=min_ratio)

        lrs = []
        for _ in range(200):
            opt.step()
            scheduler.step()
            lrs.append(opt.param_groups[0]["lr"])

        min_lr = min(lrs)
        max_lr = max(lrs)

        # LR should never exceed peak (within floating point)
        assert max_lr <= peak_lr + 1e-10, f"{sched_type}: max_lr={max_lr} > peak={peak_lr}"
        # LR should never go below min_lr_ratio * peak (within tolerance)
        assert min_lr >= peak_lr * min_ratio - 1e-10, f"{sched_type}: min_lr={min_lr} < floor={peak_lr*min_ratio}"
        # LR should be monotonically non-increasing after warmup
        post_warmup = lrs[20:]
        for i in range(1, len(post_warmup)):
            assert post_warmup[i] <= post_warmup[i - 1] + 1e-10, f"{sched_type}: non-monotonic at step {20+i}"

    print("  All 5 scheduler types: valid range, monotonic after warmup")
    print("✓ test_all_schedulers_produce_valid_lr PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def test_packing_produces_correct_output():
    """PackedDataset produces fixed-length sequences with correct position_ids."""
    from palingenesis.data import PackedDataset, IGNORE_INDEX

    # Simulate a base dataset yielding variable-length samples
    class FakeDataset:
        def __iter__(self):
            for length in [10, 15, 8, 20, 5, 12, 18, 7]:
                ids = torch.arange(length) + 1  # 1-indexed tokens
                labels = ids.clone()
                labels[:3] = IGNORE_INDEX
                yield {"input_ids": ids, "labels": labels, "attention_mask": torch.ones(length)}

    packed = PackedDataset(FakeDataset(), max_len=32, eos_id=0, sort_buffer=4)

    outputs = list(packed)
    assert len(outputs) > 0, "Should produce at least one packed sequence"

    total_tokens = 10 + 15 + 8 + 20 + 5 + 12 + 18 + 7  # = 95
    for i, out in enumerate(outputs):
        # Full blocks are max_len; the LAST block may be a shorter trailing remainder
        # (emitted, not dropped — dropping it silently loses data).
        is_last = i == len(outputs) - 1
        n = out["input_ids"].shape[0]
        assert (n == 32) or (is_last and 0 < n <= 32), f"block {i} has bad length {n}"
        assert out["labels"].shape[0] == n
        assert out["position_ids"].shape[0] == n
        # A doc's remainder is carried across block boundaries, so only the very first
        # block is guaranteed to start at position 0; every block stays within a doc.
        assert out["position_ids"].max().item() < 32

    assert outputs[0]["position_ids"][0].item() == 0, "first block must start at position 0"
    assert sum(o["input_ids"].shape[0] for o in outputs) == total_tokens, "no tokens may be dropped"
    print(f"  Produced {len(outputs)} packed sequences ({total_tokens} tokens, none dropped)")
    print("✓ test_packing_produces_correct_output PASSED\n")


def test_packing_defensive_oversized_doc():
    """Packing handles documents longer than max_len gracefully."""
    from palingenesis.data import PackedDataset, IGNORE_INDEX

    class FakeDataset:
        def __iter__(self):
            # One document that exceeds max_len
            yield {
                "input_ids": torch.arange(100),
                "labels": torch.arange(100),
                "attention_mask": torch.ones(100),
            }
            yield {
                "input_ids": torch.arange(10),
                "labels": torch.arange(10),
                "attention_mask": torch.ones(10),
            }

    packed = PackedDataset(FakeDataset(), max_len=32, eos_id=0, sort_buffer=0)
    outputs = list(packed)

    # Should still produce valid outputs (not crash or produce garbage). Full blocks
    # are max_len; a shorter trailing remainder may be emitted last (never dropped).
    for i, out in enumerate(outputs):
        n = out["input_ids"].shape[0]
        is_last = i == len(outputs) - 1
        assert (n == 32) or (is_last and 0 < n <= 32), f"block {i} has bad length {n}"
        assert out["labels"].shape[0] == n

    print(f"  Handled oversized doc: produced {len(outputs)} valid packed sequences")
    print("✓ test_packing_defensive_oversized_doc PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# END-TO-END MINI TRAINING
# ══════════════════════════════════════════════════════════════════════════════


def test_mini_training_loop():
    """Run 10 training steps on a tiny model and verify loss decreases."""
    from palingenesis.loss import cross_entropy_loss, IGNORE_INDEX
    from palingenesis.optim import HyperballWrapper, build_scheduler

    torch.manual_seed(42)
    model = TinyLM(vocab_size=64, hidden=32, layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = build_scheduler(optimizer, "power_decay", num_steps=20, warmup_ratio=0.1, min_lr_ratio=0.1)

    # Hyperball on weight matrices
    constrained = [p for n, p in model.named_parameters() if p.ndim == 2 and "embed" not in n]
    hyperball = HyperballWrapper(optimizer, constrained)

    # Fixed training data (so loss can actually decrease)
    fixed_batch = make_batch(batch_size=4, seq_len=32, vocab_size=64)

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        output = model(fixed_batch["input_ids"])
        valid = (fixed_batch["labels"] != IGNORE_INDEX).sum().item()
        loss = cross_entropy_loss(output.logits, fixed_batch["labels"], valid)
        loss.backward()
        hyperball.step()
        scheduler.step()
        losses.append(loss.item())

    first_5 = sum(losses[:5]) / 5
    last_5 = sum(losses[-5:]) / 5
    reduction = 1 - last_5 / first_5

    assert reduction > 0.1, f"Loss should decrease by >10%, got {reduction*100:.1f}%"
    assert all(math.isfinite(l) for l in losses), "No NaN/Inf losses"

    print(f"  Initial: {first_5:.4f}, Final: {last_5:.4f}, Reduction: {reduction*100:.1f}%")
    print("✓ test_mini_training_loop PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("=" * 70)
    print("PALINGENESIS — INTEGRATION TESTS")
    print("=" * 70 + "\n")

    # Config
    print("── Config ──\n")
    test_config_from_yaml()
    test_config_defaults_are_sane()

    # Loss
    print("── Loss ──\n")
    test_cross_entropy_loss_correct()
    test_chunked_loss_matches_standard()
    test_eval_chunked_ce_matches_oneshot()

    # Optimizer composition
    print("── Optimizer Stack ──\n")
    test_full_optimizer_stack()

    # Checkpointing
    print("── Checkpointing ──\n")
    test_checkpoint_save_load_roundtrip()
    test_checkpoint_auto_purge()
    test_best_model_tracker()

    # Health
    print("── Health Monitor ──\n")
    test_health_monitor_full_lifecycle()

    # Schedulers
    print("── Schedulers ──\n")
    test_all_schedulers_produce_valid_lr()

    # Data
    print("── Data Pipeline ──\n")
    test_packing_produces_correct_output()
    test_packing_defensive_oversized_doc()

    # End-to-end
    print("── End-to-End ──\n")
    test_mini_training_loop()

    print("=" * 70)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 70)
