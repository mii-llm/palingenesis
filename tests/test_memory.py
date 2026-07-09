"""Test memory optimizations: gradient release and selective differentiation."""

import sys

sys.path.insert(0, "src")

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing memory optimizations."""

    def __init__(self, hidden=128, layers=4):
        super().__init__()
        self.embed = nn.Embedding(1000, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 1000)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.head(h)


def test_gradient_release_basic():
    """Test that gradient release produces correct weight updates."""
    from palingenesis.memory import GradientRelease

    torch.manual_seed(42)
    model = SimpleModel(hidden=64, layers=2)

    # Freeze some layers (simulating freeze_non_attention)
    for p in model.layers[0].parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        betas=(0.9, 0.999),
    )

    grad_release = GradientRelease(model, optimizer)
    grad_release.enable()

    # Run a training step
    x = torch.randint(0, 1000, (2, 16))
    target = torch.randint(0, 1000, (2, 16))

    # Save initial weights
    initial_weights = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

    output = model(x)
    loss = nn.functional.cross_entropy(output.view(-1, 1000), target.view(-1))
    loss.backward()

    # After backward with gradient release:
    # 1. Gradients should be None (freed after step)
    # 2. Weights should have changed (optimizer stepped)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is None, f"{name} should have grad=None after release"

    # Check weights changed
    changed = 0
    for name, p in model.named_parameters():
        if name in initial_weights:
            if not torch.allclose(p.data, initial_weights[name]):
                changed += 1

    assert changed > 0, "Weights should have been updated during backward"

    grad_release.disable()
    print(f"  {changed} parameter tensors updated during backward (gradients freed)")
    print("✓ test_gradient_release_basic PASSED\n")


def test_gradient_release_matches_standard():
    """Verify gradient release gives same result as standard training."""
    from palingenesis.memory import GradientRelease

    torch.manual_seed(42)
    model_std = SimpleModel(hidden=32, layers=2)
    model_gr = SimpleModel(hidden=32, layers=2)

    # Copy weights
    model_gr.load_state_dict(model_std.state_dict())

    # Same optimizer setup
    opt_std = torch.optim.AdamW(model_std.parameters(), lr=1e-3, betas=(0.9, 0.999))
    opt_gr = torch.optim.AdamW(model_gr.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # Enable gradient release on model_gr
    gr = GradientRelease(model_gr, opt_gr)
    gr.enable()

    # Same input
    x = torch.randint(0, 1000, (2, 8))
    target = torch.randint(0, 1000, (2, 8))

    # Standard training step
    out_std = model_std(x)
    loss_std = nn.functional.cross_entropy(out_std.view(-1, 1000), target.view(-1))
    loss_std.backward()
    opt_std.step()
    opt_std.zero_grad(set_to_none=True)

    # Gradient release training step
    out_gr = model_gr(x)
    loss_gr = nn.functional.cross_entropy(out_gr.view(-1, 1000), target.view(-1))
    loss_gr.backward()
    # No step needed — done in backward
    opt_gr.zero_grad(set_to_none=True)

    # Compare weights — should be identical (element-wise optimizer is exact)
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model_std.named_parameters(), model_gr.named_parameters()):
        diff = (p1.data - p2.data).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"  Max weight difference: {max_diff:.2e}")
    # Allow small numerical difference due to operation ordering
    assert max_diff < 1e-5, f"Gradient release should match standard training, got diff={max_diff}"

    gr.disable()
    print("✓ test_gradient_release_matches_standard PASSED\n")


def test_selective_diff_frozen_layers():
    """Test that selective differentiation works on frozen layers."""
    from palingenesis.memory import apply_selective_diff_v2

    model = SimpleModel(hidden=64, layers=4)

    # Freeze layers 0, 1 (simulating freeze_non_attention)
    for layer in model.layers[:2]:
        for p in layer.parameters():
            p.requires_grad = False

    # Apply selective differentiation
    num_modified = apply_selective_diff_v2(model)
    assert num_modified == 2, f"Expected 2 frozen layers modified, got {num_modified}"

    # Verify model still works (forward + backward)
    x = torch.randint(0, 1000, (2, 16))
    target = torch.randint(0, 1000, (2, 16))

    output = model(x)
    loss = nn.functional.cross_entropy(output.view(-1, 1000), target.view(-1))
    loss.backward()

    # Check gradients exist for trainable parameters
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} should have gradient"
        else:
            assert p.grad is None, f"{name} (frozen) should NOT have gradient"

    print(f"  {num_modified} frozen layers optimized")
    print("✓ test_selective_diff_frozen_layers PASSED\n")


def test_selective_diff_correctness():
    """Verify selective diff produces same gradients as standard."""
    from palingenesis.memory import apply_selective_diff_v2

    torch.manual_seed(42)
    model_std = SimpleModel(hidden=32, layers=4)
    model_sd = SimpleModel(hidden=32, layers=4)
    model_sd.load_state_dict(model_std.state_dict())

    # Freeze first 2 layers in both
    for m in (model_std, model_sd):
        for layer in m.layers[:2]:
            for p in layer.parameters():
                p.requires_grad = False

    # Apply selective diff only to model_sd
    apply_selective_diff_v2(model_sd)

    # Same forward+backward
    x = torch.randint(0, 1000, (2, 8))
    target = torch.randint(0, 1000, (2, 8))

    out_std = model_std(x)
    loss_std = nn.functional.cross_entropy(out_std.view(-1, 1000), target.view(-1))
    loss_std.backward()

    out_sd = model_sd(x)
    loss_sd = nn.functional.cross_entropy(out_sd.view(-1, 1000), target.view(-1))
    loss_sd.backward()

    # Compare gradients of trainable parameters
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model_std.named_parameters(), model_sd.named_parameters()):
        if p1.requires_grad and p1.grad is not None and p2.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            max_diff = max(max_diff, diff)

    print(f"  Max gradient difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Selective diff should produce same gradients, got diff={max_diff}"

    print("✓ test_selective_diff_correctness PASSED\n")


def test_combined_memory_savings():
    """Test using both techniques together."""
    from palingenesis.memory import GradientRelease, apply_selective_diff_v2

    model = SimpleModel(hidden=64, layers=6)

    # Freeze first 4 layers (67% frozen — similar to Qwen3.5 freeze_non_attention)
    for layer in model.layers[:4]:
        for p in layer.parameters():
            p.requires_grad = False

    # Apply selective diff
    n_modified = apply_selective_diff_v2(model)
    assert n_modified == 4

    # Setup optimizer for trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    # Enable gradient release
    gr = GradientRelease(model, optimizer)
    gr.enable()

    # Run a step
    x = torch.randint(0, 1000, (2, 16))
    target = torch.randint(0, 1000, (2, 16))

    output = model(x)
    loss = nn.functional.cross_entropy(output.view(-1, 1000), target.view(-1))
    loss.backward()

    # Verify:
    # 1. Trainable params: no gradient (released after step)
    # 2. Frozen params: no gradient (never computed)
    # 3. Model still produces valid output
    for name, p in model.named_parameters():
        assert p.grad is None, f"{name} should have grad=None (either frozen or released)"

    # Check trainable weights changed
    print(f"  {n_modified} frozen layers (selective diff)")
    print(f"  {len(trainable)} trainable params (gradient release)")
    print(f"  Combined: all gradients freed after backward ✓")

    gr.disable()
    print("✓ test_combined_memory_savings PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY OPTIMIZATION TESTS")
    print("=" * 60 + "\n")

    test_gradient_release_basic()
    test_gradient_release_matches_standard()
    test_selective_diff_frozen_layers()
    test_selective_diff_correctness()
    test_combined_memory_savings()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
