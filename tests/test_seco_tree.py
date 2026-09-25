"""The tree forward/backward (palingenesis.seco_tree) equals running every branch as
its own sequence: the loss, every parameter gradient, and the no-grad hidden states,
for a pure attention decoder (GPT-2) and a Qwen3.5 hybrid (Gated DeltaNet +
attention), with branches at chunk boundaries, inside chunks, at 0, at the trunk's
end, and several at one point.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from test_seco import MODELS, TOL, VOCAB, _max_rel_err  # noqa: E402

from palingenesis.logits import output_head  # noqa: E402
from palingenesis.seco_tree import Branch, tree_forward_backward, tree_hidden_states  # noqa: E402

TRUNK = 61
# (start, length): mid-chunk, a shared start, the trunk's very start and end, a long one, a
# one-token branch and adjacent starts (a one-token trunk chunk: the layers' decode path)
BRANCHES = [(17, 9), (17, 4), (32, 12), (0, 5), (TRUNK, 7), (45, 20), (60, 3), (33, 1)]


def _tree(seed=2):
    g = torch.Generator().manual_seed(seed)
    trunk = torch.randint(1, VOCAB, (1, TRUNK), generator=g)
    branches = [Branch(s, torch.randint(1, VOCAB, (1, n), generator=g)) for s, n in BRANCHES]
    targets = [torch.randint(0, VOCAB, (n,), generator=g) for _, n in BRANCHES]
    weights = torch.rand(len(BRANCHES), generator=g, dtype=torch.float64)
    return trunk, branches, targets, weights


def _branch_loss(head, hidden, target, weight):
    logits = head(hidden[0]).double()
    return weight * torch.nn.functional.cross_entropy(logits, target, reduction="sum")


def _naive(model, trunk, branches, targets, weights):
    """Every branch as its own sequence (its context + its tokens), summed."""
    model.zero_grad()
    head = output_head(model)
    total = 0.0
    for b, t, w in zip(branches, targets, weights):
        ids = torch.cat([trunk[:, :b.start], b.input_ids], 1)
        hidden = model.base_model(input_ids=ids).last_hidden_state[:, b.start:]
        loss = _branch_loss(head, hidden, t, w)
        loss.backward()
        total += loss.item()
    return total, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _treed(model, trunk, branches, targets, weights, chunk_size, branch_tokens=64, min_gap=1):
    model.zero_grad()
    head = output_head(model)
    result = tree_forward_backward(model, trunk, branches,
                                   lambda i, h: _branch_loss(head, h, targets[i], weights[i]), chunk_size=chunk_size,
                                   branch_tokens=branch_tokens, min_gap=min_gap)
    return result, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("arch", MODELS)
@pytest.mark.parametrize("chunk_size", [8, 16, 64])
@pytest.mark.parametrize("branch_tokens", [0, 64])      # one branch at a time; batched (several groups)
@pytest.mark.parametrize("min_gap", [1, 16])            # a cut at every start; some branches re-read gaps
def test_tree_equals_every_branch_as_its_own_sequence(arch, chunk_size, branch_tokens, min_gap):
    model = MODELS[arch]()
    trunk, branches, targets, weights = _tree()
    want_loss, want = _naive(model, trunk, branches, targets, weights)
    result, got = _treed(model, trunk, branches, targets, weights, chunk_size, branch_tokens, min_gap)
    assert result.loss == pytest.approx(want_loss, rel=1e-6)
    assert result.branches == len(branches) and result.trunk_chunks >= 1
    assert _max_rel_err(got, want) < TOL[arch]


@pytest.mark.parametrize("arch", MODELS)
def test_gradient_reaches_the_trunk(arch):
    """Dropping the trunk's relay (branches trained on a frozen context) is measurably different:
    the exactness above is not an accident of tiny contributions."""
    model = MODELS[arch]()
    trunk, branches, targets, weights = _tree()
    _, want = _naive(model, trunk, branches, targets, weights)
    model.zero_grad()
    head = output_head(model)
    with torch.no_grad():
        ctx = [model.base_model(input_ids=trunk[:, :b.start]) if b.start else None for b in branches]
    for b, c, t, w in zip(branches, ctx, targets, weights):     # context detached: truncated backprop
        past = c.past_key_values if c is not None else None
        hidden = model.base_model(input_ids=b.input_ids, past_key_values=past).last_hidden_state
        _branch_loss(head, hidden, t, w).backward()
    detached = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert _max_rel_err(detached, want) > 100 * TOL[arch]


@pytest.mark.parametrize("arch", MODELS)
@pytest.mark.parametrize("branch_tokens", [0, 64])
@pytest.mark.parametrize("min_gap", [1, 16])
def test_hidden_states_equal_full_forwards(arch, branch_tokens, min_gap):
    model = MODELS[arch]().eval()
    trunk, branches, _, _ = _tree()
    positions = [0, 5, 16, 17, 40, TRUNK - 1]
    got, trunk_hidden = tree_hidden_states(model, trunk, branches, trunk_positions=positions, chunk_size=16,
                                           branch_tokens=branch_tokens, min_gap=min_gap)
    with torch.no_grad():
        for b, h in zip(branches, got):
            ids = torch.cat([trunk[:, :b.start], b.input_ids], 1)
            want = model.base_model(input_ids=ids).last_hidden_state[:, b.start:]
            torch.testing.assert_close(h, want, rtol=1e-5, atol=1e-6)
        full = model.base_model(input_ids=trunk).last_hidden_state[0, positions]
        torch.testing.assert_close(trunk_hidden, full, rtol=1e-5, atol=1e-6)


def test_one_branch_at_the_end_is_plain_backprop_of_the_whole_sequence():
    model = MODELS["gpt2"]()
    trunk, _, _, _ = _tree()
    ids = torch.randint(1, VOCAB, (1, 6), generator=torch.Generator().manual_seed(5))
    target = torch.randint(0, VOCAB, (6,), generator=torch.Generator().manual_seed(6))
    one = [Branch(TRUNK, ids)]
    want_loss, want = _naive(model, trunk, one, [target], torch.ones(1, dtype=torch.float64))
    result, got = _treed(model, trunk, one, [target], torch.ones(1, dtype=torch.float64), 16)
    assert result.loss == pytest.approx(want_loss, rel=1e-6)
    assert _max_rel_err(got, want) < TOL["gpt2"]


def test_rejects_malformed_trees():
    model = MODELS["gpt2"]()
    trunk = torch.randint(1, VOCAB, (1, 10))
    with pytest.raises(ValueError, match="outside the trunk"):
        tree_hidden_states(model, trunk, [Branch(11, torch.ones(1, 2, dtype=torch.long))])
    with pytest.raises(ValueError, match="one trunk at a time"):
        tree_hidden_states(model, trunk.repeat(2, 1), [])
    with pytest.raises(ValueError, match="L >= 1"):
        tree_hidden_states(model, trunk, [Branch(3, torch.ones(1, 0, dtype=torch.long))])


@pytest.mark.parametrize("arch", MODELS)
def test_trunk_loss_is_the_full_sequence_loss_of_the_trunk(arch):
    """A loss on trunk positions, computed in the reverse sweep, equals the same loss on a full
    forward of the trunk (added to the branches' naive losses)."""
    model = MODELS[arch]()
    trunk, branches, targets, weights = _tree()
    head = output_head(model)
    g = torch.Generator().manual_seed(9)
    trunk_targets = torch.randint(0, VOCAB, (TRUNK,), generator=g)
    scored = torch.rand(TRUNK, generator=g) < 0.4              # positions that carry a trunk loss

    def trunk_loss(lo, hi, hidden):
        keep = scored[lo:hi]
        if not keep.any():
            return None
        return 0.7 * torch.nn.functional.cross_entropy(head(hidden[0][keep]).double(), trunk_targets[lo:hi][keep],
                                                       reduction="sum")

    want_loss, _ = _naive(model, trunk, branches, targets, weights)     # leaves the branch gradients in .grad
    hidden = model.base_model(input_ids=trunk).last_hidden_state
    full = trunk_loss(0, TRUNK, hidden)
    full.backward()                                                     # adds the trunk loss's
    want_loss += full.item()
    want = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    result = tree_forward_backward(model, trunk, branches, lambda i, h: _branch_loss(head, h, targets[i], weights[i]),
                                   trunk_loss_fn=trunk_loss, chunk_size=16, min_gap=8)
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert result.loss == pytest.approx(want_loss, rel=1e-6)
    assert _max_rel_err(got, want) < TOL[arch]


def test_differentiable_decode_patches_and_restores():
    from palingenesis.seco import differentiable_decode

    model = MODELS["qwen35_hybrid"]()
    layers = [m for m in model.modules() if "recurrent_gated_delta_rule" in m.__dict__]
    assert layers
    before = [(m.recurrent_gated_delta_rule, m.causal_conv1d_update) for m in layers]
    with differentiable_decode(model):
        assert all(m.recurrent_gated_delta_rule is m.chunk_gated_delta_rule for m in layers)
        assert all(m.causal_conv1d_update.__name__ == "torch_causal_conv1d_update" for m in layers)
    assert [(m.recurrent_gated_delta_rule, m.causal_conv1d_update) for m in layers] == before


def test_bounds_merge_close_cuts():
    from palingenesis.seco_tree import _bounds

    assert _bounds(100, [10, 15, 40, 45, 99], chunk_size=64, min_gap=1) == [
        (0, 10), (10, 15), (15, 40), (40, 45), (45, 64), (64, 99), (99, 100)]
    # cut at a start only 20+ past the previous cut; the grid (every 64) always stays
    assert _bounds(100, [10, 15, 40, 45, 99], chunk_size=64, min_gap=20) == [(0, 40), (40, 64), (64, 99), (99, 100)]
