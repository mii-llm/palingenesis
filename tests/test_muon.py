"""Muon (palingenesis.muon): Newton-Schulz orthogonalization (standard and Gram), Moonlight update scaling, parameter
routing, the AdamW groups, checkpoints. CPU, float32."""

import copy

import pytest
import torch
import torch.nn as nn

from palingenesis.muon import MuonAdamW, build_muon, gram_newton_schulz, muon_param_groups, newton_schulz, orthogonalize


def _polar(G):
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    return (U @ Vh).float()


@pytest.mark.parametrize("shape", [(64, 256), (256, 64), (96, 96), (16, 1024)])
def test_orthogonalize_approximates_the_polar_factor(shape):
    torch.manual_seed(0)
    G = torch.randn(*shape)
    orth = orthogonalize(G, "gram")
    s = torch.linalg.svdvals(orth.double())
    # Polar Express (with its 1.05 safety factor) lands every singular value near 1
    assert s.min() > 0.7 and s.max() < 1.15, (s.min(), s.max())
    cos = torch.nn.functional.cosine_similarity(orth.flatten(), _polar(G).flatten(), dim=0)
    assert cos > 0.98


def test_gram_matches_standard_newton_schulz():
    torch.manual_seed(1)
    for shape in [(3, 64, 512), (2, 512, 64), (4, 128, 384)]:
        X = torch.randn(*shape)
        a, b = gram_newton_schulz(X), newton_schulz(X)
        assert torch.allclose(a, b, atol=2e-3, rtol=2e-3), (shape, (a - b).abs().max())


def test_batched_equals_one_by_one():
    torch.manual_seed(2)
    X = torch.randn(5, 48, 160)
    batched = orthogonalize(X, "gram")
    single = torch.stack([orthogonalize(x, "gram") for x in X])
    assert torch.allclose(batched, single, atol=1e-5)


class _Hybrid(nn.Module):
    """Parameter names / shapes like Qwen3.5's (embedding, Gated DeltaNet, attention, MLP, norms, tied head)."""

    def __init__(self, d=64, vocab=500):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        layer = nn.Module()
        layer.linear_attn = nn.Module()
        layer.linear_attn.in_proj_qkv = nn.Linear(d, 6 * d, bias=False)
        layer.linear_attn.in_proj_a = nn.Linear(d, 16, bias=False)  # rank-16 gate: thin
        layer.linear_attn.conv1d = nn.Conv1d(6 * d, 6 * d, 4, groups=6 * d, bias=False)  # 3-D kernel
        layer.linear_attn.A_log = nn.Parameter(torch.zeros(16))
        layer.linear_attn.out_proj = nn.Linear(2 * d, d, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.up_proj = nn.Linear(d, 3 * d, bias=False)
        layer.input_layernorm = nn.LayerNorm(d)
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight  # tied


def test_routing_sends_only_hidden_matrices_to_muon():
    m = _Hybrid()
    groups, counts = muon_param_groups(m, weight_decay=0.1, min_dim=32)
    names = {id(p): n for n, p in m.named_parameters()}
    muon = sorted(names[id(p)] for g in groups if g["use_muon"] for p in g["params"])
    assert muon == ["model.layers.0.linear_attn.in_proj_qkv.weight", "model.layers.0.linear_attn.out_proj.weight",
                    "model.layers.0.mlp.up_proj.weight"]
    no_decay = sorted(names[id(p)] for g in groups if not g["use_muon"] and g["weight_decay"] == 0 for p in g["params"])
    assert "model.layers.0.linear_attn.A_log" in no_decay and "model.layers.0.input_layernorm.weight" in no_decay
    assert counts["muon"] + counts["adamw"] == sum(p.numel() for p in m.parameters())


def test_moonlight_scale_matches_adamw_update_rms():
    torch.manual_seed(3)
    lin = nn.Linear(256, 1024, bias=False)
    w0 = lin.weight.detach().clone()
    opt = MuonAdamW([{"params": [lin.weight], "use_muon": True}], lr=1e-3, weight_decay=0.0, momentum=0.0,
                    nesterov=False)
    lin.weight.grad = torch.randn_like(lin.weight)
    opt.step()
    rms = ((lin.weight - w0) / 1e-3).pow(2).mean().sqrt()
    assert 0.17 < rms < 0.23, rms  # AdamW-like update RMS ~0.2, so Muon shares AdamW's learning rate


def test_adamw_groups_match_torch_adamw():
    torch.manual_seed(4)
    a, b = nn.Linear(8, 8), nn.Linear(8, 8)
    b.load_state_dict(a.state_dict())
    ours = MuonAdamW([{"params": list(a.parameters()), "use_muon": False}], lr=1e-2, weight_decay=0.1)
    ref = torch.optim.AdamW(b.parameters(), lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    for _ in range(5):
        x = torch.randn(4, 8)
        for m, o in ((a, ours), (b, ref)):
            o.zero_grad()
            m(x).pow(2).sum().backward()
            o.step()
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_training_reduces_loss_and_checkpoint_round_trips():
    torch.manual_seed(5)
    net = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 32))
    opt = build_muon(net, lr=2e-3, weight_decay=0.01, min_dim=16)
    x, y = torch.randn(64, 32), torch.randn(64, 32)
    losses = []
    for i in range(60):
        opt.zero_grad()
        loss = (net(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i == 29:
            snap_net, snap_opt = copy.deepcopy(net), copy.deepcopy(opt.state_dict())
    assert losses[-1] < 0.7 * losses[0]
    # resume from step 30 and replay: identical parameters
    net2 = snap_net
    opt2 = build_muon(net2, lr=2e-3, weight_decay=0.01, min_dim=16)
    opt2.load_state_dict(snap_opt)
    for _ in range(30):
        opt2.zero_grad()
        (net2(x) - y).pow(2).mean().backward()
        opt2.step()
    for p, q in zip(net.parameters(), net2.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_muon_weight_decay_is_decoupled():
    lin = nn.Linear(64, 64, bias=False)
    w0 = lin.weight.detach().clone()
    opt = MuonAdamW([{"params": [lin.weight], "use_muon": True}], lr=0.1, weight_decay=0.5)
    lin.weight.grad = torch.zeros_like(lin.weight)  # zero gradient: only the decay acts
    opt.step()
    assert torch.allclose(lin.weight, w0 * (1 - 0.1 * 0.5), atol=1e-6)
