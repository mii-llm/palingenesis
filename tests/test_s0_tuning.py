"""Test S0 Tuning with a minimal synthetic model mimicking hybrid architecture."""

import sys

sys.path.insert(0, "src")

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Minimal GatedDeltaNet-style recurrent layer for testing
# ══════════════════════════════════════════════════════════════════════════════


class FakeGatedDeltaNet(nn.Module):
    """Minimal recurrent layer that mimics GatedDeltaNet state mechanics.

    State: S_t = alpha_t * S_{t-1} + beta_t * v_t @ k_t^T
    The key property: starts from S_0 (initial state) which can be tuned.
    """

    def __init__(self, hidden_size: int = 128, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        # Learnable projection (frozen during S0 tuning)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # Initial state (default zero, this is what S0 tuning replaces)
        self.initial_state = None  # Will be set by S0 hook

    @property
    def state_size(self):
        return (self.num_heads, self.head_dim, self.head_dim)

    def forward(self, x, initial_state=None, **kwargs):
        B, S, D = x.shape
        h = self.in_proj(x)

        # Simulate recurrence: just a simple state evolution for testing
        if initial_state is not None:
            # S0 was injected!
            state = initial_state.to(h.dtype).to(h.device)
            if state.dim() == 3:
                state = state.unsqueeze(0).expand(B, -1, -1, -1)
        elif self.initial_state is not None:
            state = self.initial_state.to(h.dtype).to(h.device)
            if state.dim() == 3:
                state = state.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=h.device, dtype=h.dtype)

        # Simple evolution: state contributes to output through trace
        state_contribution = state.sum(dim=(-1, -2)).unsqueeze(1)  # [B, 1, num_heads]
        # Pad/project state contribution to hidden size
        state_signal = torch.zeros_like(h)
        state_signal[:, :, : self.num_heads] = state_contribution.expand(B, S, -1)

        out = self.out_proj(h + state_signal * 0.1)
        return out


class FakeHybridModel(nn.Module):
    """Minimal hybrid model with recurrent + attention layers for testing S0."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, num_layers: int = 4):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_attention_heads": 4,
                "head_dim": 32,
            },
        )()

        self.embed = nn.Embedding(vocab_size, hidden_size)
        # 3 recurrent layers + 1 attention (mimics Qwen3.5 3:1 ratio)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 4 != 3:  # 3 GDN per 1 attention
                self.layers.append(FakeGatedDeltaNet(hidden_size, num_heads=4, head_dim=32))
            else:
                self.layers.append(
                    nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=256, batch_first=True)
                )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            if isinstance(layer, FakeGatedDeltaNet):
                x = x + layer(x)
            else:
                x = layer(x)
        logits = self.lm_head(x)
        return type("Output", (), {"logits": logits})()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════


def test_discover_layers():
    """Test that we can discover recurrent layers in a hybrid model."""
    from palingenesis.s0_tuning import discover_recurrent_layers

    model = FakeHybridModel()
    layers = discover_recurrent_layers(model)

    print(f"✓ Discovered {len(layers)} recurrent layers")
    for info in layers:
        print(f"  {info.name}: shape={info.state_shape}")

    assert len(layers) == 3, f"Expected 3 GatedDeltaNet layers, got {len(layers)}"
    for info in layers:
        assert info.state_shape == (4, 32, 32), f"Expected (4, 32, 32), got {info.state_shape}"

    print("✓ test_discover_layers PASSED\n")


def test_s0_states_container():
    """Test S0States container creation and L2 penalty."""
    from palingenesis.s0_tuning import discover_recurrent_layers, S0States

    model = FakeHybridModel()
    layers = discover_recurrent_layers(model)
    s0 = S0States(layers)

    # Check initialization
    total_params = sum(p.numel() for p in s0.parameters())
    expected = 3 * 4 * 32 * 32  # 3 layers × (4, 32, 32)
    assert total_params == expected, f"Expected {expected} params, got {total_params}"

    # Check all initialized to zero
    for state in s0.states:
        assert state.abs().sum() == 0, "States should be initialized to zero"

    # Check L2 penalty at zero = 0
    assert s0.l2_penalty() == 0, "L2 at zero should be 0"

    # Check scaled output
    scaled = s0.get_scaled(0.07)
    assert len(scaled) == 3
    assert all(s.abs().sum() == 0 for s in scaled)

    print("✓ test_s0_states_container PASSED\n")


def test_s0_training_step():
    """Test that S0 optimization actually changes the states."""
    from palingenesis.s0_tuning import S0Trainer

    model = FakeHybridModel()
    device = "cpu"

    # Create trainer
    trainer = S0Trainer(model, alpha=0.07, weight_decay=1e-4, lr=1e-2, device=device)

    # Check model is frozen
    for p in model.parameters():
        assert not p.requires_grad, "Model params should be frozen"

    # Check S0 states are learnable
    for p in trainer.s0_states.parameters():
        assert p.requires_grad, "S0 states should require grad"

    # Create a fake batch
    batch_size = 2
    seq_len = 32
    IGNORE_INDEX = -100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, : seq_len // 2] = IGNORE_INDEX  # mask first half (system/user)

    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Get initial state values
    initial_states = [s.clone() for s in trainer.s0_states.states]

    # Run a few steps
    losses = []
    for _ in range(5):
        loss = trainer.step(batch)
        losses.append(loss)

    # States should have changed
    for i, (initial, current) in enumerate(zip(initial_states, trainer.s0_states.states)):
        diff = (current - initial).abs().sum().item()
        assert diff > 0, f"State {i} didn't change after optimization"
        print(f"  State {i} moved by {diff:.6f}")

    # Loss should generally decrease (not guaranteed with random data, but should not explode)
    print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
    assert all(l < 100 for l in losses), "Losses exploded"

    print("✓ test_s0_training_step PASSED\n")


def test_s0_save_load():
    """Test saving and loading S0 states."""
    import tempfile
    from palingenesis.s0_tuning import S0Trainer

    model = FakeHybridModel()
    trainer = S0Trainer(model, alpha=0.07, lr=1e-2, device="cpu")

    # Modify states
    for s in trainer.s0_states.states:
        s.data.normal_(0, 0.1)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    trainer.save(path)

    # Load into fresh trainer
    model2 = FakeHybridModel()
    trainer2 = S0Trainer(model2, alpha=0.07, lr=1e-2, device="cpu")
    trainer2.load(path)

    # Check states match
    for s1, s2 in zip(trainer.s0_states.states, trainer2.s0_states.states):
        assert torch.allclose(s1, s2), "Loaded states don't match saved states"

    import os

    os.unlink(path)
    print("✓ test_s0_save_load PASSED\n")


def test_hes_metric():
    """Test HES metric computation logic (unit test without model)."""
    # Simulate token entropies for a sample
    torch.manual_seed(42)

    # Simulate: 100 tokens, most low entropy, a few high (decision points)
    entropies = torch.rand(100) * 0.5  # mostly low (0-0.5)
    # Add 5 high-entropy decision points
    entropies[10] = 3.5
    entropies[25] = 4.2
    entropies[50] = 3.8
    entropies[75] = 4.0
    entropies[90] = 3.3

    # HES with top 0.5% = top 1 token (floor to 1)
    k = max(1, int(len(entropies) * 0.5 / 100))
    assert k == 1
    top_k = entropies.topk(k).values
    hes_05 = top_k.sum().item()
    assert abs(hes_05 - 4.2) < 0.01, f"Expected ~4.2, got {hes_05}"

    # HES with top 5% = top 5 tokens
    k5 = max(1, int(len(entropies) * 5.0 / 100))
    assert k5 == 5
    top_k5 = entropies.topk(k5).values
    hes_5 = top_k5.sum().item()
    expected_5 = 3.5 + 4.2 + 3.8 + 4.0 + 3.3  # = 18.8
    assert abs(hes_5 - expected_5) < 0.01, f"Expected ~{expected_5}, got {hes_5}"

    # Bad sample: all uniform high entropy (garbage)
    garbage_entropies = torch.ones(100) * 2.0  # uniformly medium
    k5_garbage = max(1, int(100 * 5.0 / 100))
    hes_garbage = garbage_entropies.topk(k5_garbage).values.sum().item()
    # 5 tokens × 2.0 = 10.0 (LOWER than good reasoning sample's 18.8)
    assert hes_garbage < hes_5, "Good reasoning should have higher HES than garbage"

    print(f"  Good sample HES@5%: {hes_5:.2f}")
    print(f"  Garbage sample HES@5%: {hes_garbage:.2f}")
    print("✓ test_hes_metric PASSED\n")


def test_pretraining_replay_config():
    """Test that pretraining replay config fields exist."""
    sys.path.insert(0, "src")
    from palingenesis.config import DataConfig

    config = DataConfig()
    assert hasattr(config, "pretrain_replay_dataset")
    assert hasattr(config, "pretrain_replay_weight")
    assert config.pretrain_replay_dataset == ""
    assert config.pretrain_replay_weight == 0.1

    # MSFT fields
    assert hasattr(config, "msft_tracking")
    assert hasattr(config, "msft_decay_factor")
    assert hasattr(config, "msft_eval_every")
    assert config.msft_tracking is False
    assert config.msft_decay_factor == 0.7

    print("✓ test_pretraining_replay_config PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("S0 TUNING & DATA PIPELINE TESTS")
    print("=" * 60 + "\n")

    test_discover_layers()
    test_s0_states_container()
    test_s0_training_step()
    test_s0_save_load()
    test_hes_metric()
    test_pretraining_replay_config()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
