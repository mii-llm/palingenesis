"""palingenesis.bf16_adamw: stochastic rounding with counter-based noise, AdamW on bf16 weights and moments against
fp32-master AdamW, determinism and exact resume, per-parameter stepping (gradient release), wiring into
build_optimizer and config validation. CPU."""

import copy

import pytest
import torch
import torch.nn as nn

from palingenesis import bf16_adamw as B
from palingenesis.bf16_adamw import BF16AdamW, hash32, round_bf16
from palingenesis.config import Config, ConfigError
from palingenesis.optim import build_optimizer


def _neighbours(x: torch.Tensor):
    """The two bf16 values around each fp32 x (toward zero, away from zero), as float64."""
    r = x.to(torch.bfloat16)
    below = r.double().abs() <= x.double().abs()
    bits = r.view(torch.int16)
    other = torch.where(below, bits + 1, bits - 1).view(torch.bfloat16).double()
    return torch.where(below, r.double(), other), torch.where(below, other, r.double())


# ── rounding and noise ──────────────────────────────────────────────────────────────────────────────────────────


def test_hash_is_deterministic_and_uniform():
    idx = torch.arange(1 << 18)
    a, b = hash32(idx, 7), hash32(idx, 7)
    assert torch.equal(a, b)
    assert int(a.min()) >= 0 and int(a.max()) < 1 << 32
    for draw in (a & 0xFFFF, a >> 16):
        u = draw.double() / 65536
        assert abs(u.mean().item() - 0.5) < 0.005
        hist = torch.histc(u.float(), bins=16, min=0, max=1)
        chi2 = (((hist - hist.mean()) ** 2) / hist.mean()).sum().item()
        assert chi2 < 40  # 15 degrees of freedom: p ~ 5e-4
    c = hash32(idx, 8)
    corr = torch.corrcoef(torch.stack([(a >> 16).double(), (c >> 16).double()]))[0, 1]
    assert abs(corr.item()) < 0.01


def test_round_bf16_is_unbiased_and_within_one_ulp():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, generator=g) * torch.logspace(-6, 3, 4096)
    x = torch.cat([x, -x])
    lo, hi = _neighbours(x)
    acc = torch.zeros_like(x, dtype=torch.float64)
    steps = 400
    for s in range(steps):
        r = round_bf16(x, hash32(torch.arange(x.numel()), s * 1_000_003) & 0xFFFF).double()
        assert bool(((r == lo) | (r == hi)).all())
        acc += r
    ulp = (hi - lo).abs()
    err = ((acc / steps - x.double()).abs() / ulp).mean().item()
    assert err < 0.03, err  # nearest rounding would be ~0.25 ulp off on average
    on_grid = torch.tensor([1.0, -2.5, 0.0, 3.0e38])
    assert torch.equal(round_bf16(on_grid, torch.full((4,), 65535)).float(), on_grid.to(torch.bfloat16).float())


def test_round_bf16_keeps_inf_nan_and_saturates():
    big = torch.finfo(torch.bfloat16).max
    x = torch.tensor([float("inf"), float("-inf"), float("nan"), big, -big, float(big) * 1.001])
    out = round_bf16(x, torch.full((6,), 65535, dtype=torch.int64))
    assert out[0] == float("inf") and out[1] == float("-inf") and torch.isnan(out[2])
    assert out[3] == big and out[4] == -big and torch.isfinite(out[5])


# ── the optimizer ───────────────────────────────────────────────────────────────────────────────────────────────


def _pair(dtype=torch.bfloat16, n=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, generator=g) * 0.02
    return nn.Parameter(w.to(dtype)), g


def test_fp32_parameters_match_torch_adamw():
    torch.manual_seed(0)
    a = nn.Parameter(torch.randn(64, 32))
    b = nn.Parameter(a.detach().clone())
    opt_a = BF16AdamW([a], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    opt_b = torch.optim.AdamW([b], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
    for _ in range(20):
        grad = torch.randn(64, 32)
        a.grad, b.grad = grad.clone(), grad.clone()
        opt_a.step()
        opt_b.step()
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    assert opt_a.state[a]["exp_avg"].dtype == torch.float32


def test_one_step_lands_on_a_neighbour_of_the_fp32_update():
    p, g = _pair()
    grad = (torch.randn(p.numel(), generator=g) * 1e-3).to(torch.bfloat16)
    w32 = p.detach().float().clone()
    ref = torch.optim.AdamW([nn.Parameter(w32)], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
    ref.param_groups[0]["params"][0].grad = grad.float()
    ref.step()
    expected = ref.param_groups[0]["params"][0].detach()
    opt = BF16AdamW([p], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    p.grad = grad
    opt.step()
    lo, hi = _neighbours(expected)
    got = p.detach().double()
    assert bool(((got == lo) | (got == hi)).all())
    assert opt.state[p]["exp_avg"].dtype == torch.bfloat16 and opt.state[p]["exp_avg_sq"].dtype == torch.bfloat16


def _drift(opt_factory, dtype, steps=200, lr=1e-5):
    p, _ = _pair(dtype)
    start = p.detach().double().clone()
    opt = opt_factory([p], lr)
    for _ in range(steps):
        p.grad = torch.ones_like(p)
        opt.step()
    return (p.detach().double() - start).mean().item()


def test_small_updates_accumulate_like_fp32_masters():
    """lr 1e-5 on weights of RMS 0.02: each step is ~1/16 of a bf16 ulp. Nearest rounding loses most of it."""

    def adamw(params, lr):
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)

    def bf16(params, lr):
        return BF16AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    fp32 = _drift(adamw, torch.float32)
    nearest = _drift(adamw, torch.bfloat16)
    sr = _drift(bf16, torch.bfloat16)
    assert fp32 == pytest.approx(-200 * 1e-5, rel=1e-3)
    assert abs(nearest / fp32) < 0.5  # most of the update is lost
    assert sr / fp32 == pytest.approx(1.0, abs=0.02)


def test_second_moment_is_unbiased():
    """v tracks the fp32 EMA of g^2 when each step's increment is far below v's ulp; nearest rounding on store
    stalls (here at 0.25, 4% low after 300 steps)."""
    p, g = _pair(n=8192)
    opt = BF16AdamW([p], lr=0.0, betas=(0.9, 0.999))
    v32 = torch.zeros(p.numel(), dtype=torch.float64)
    nearest = torch.zeros(p.numel(), dtype=torch.bfloat16)
    for _ in range(300):
        grad = (1.0 + 0.01 * torch.randn(p.numel(), generator=g)).to(torch.bfloat16)
        p.grad = grad
        opt.step()
        v32 = 0.999 * v32 + 0.001 * grad.double() ** 2
        nearest = (nearest.float() * 0.999 + 0.001 * grad.float() ** 2).to(torch.bfloat16)
    v = opt.state[p]["exp_avg_sq"].double()
    assert ((v - v32) / v32).mean().abs().item() < 0.003
    assert ((nearest.double() - v32) / v32).mean().abs().item() > 0.02


def _mlp(dtype, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 1)).to(dtype)


def _train(model, opt, steps=150, seed=0):
    g = torch.Generator().manual_seed(seed)
    teacher = torch.randn(32, 1, generator=g)
    dtype = next(model.parameters()).dtype
    losses = []
    for _ in range(steps):
        x = torch.randn(64, 32, generator=g)
        y = torch.tanh(x @ teacher)
        loss = ((model(x.to(dtype)).float() - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return sum(losses[-20:]) / 20


def test_training_matches_fp32_master_adamw():
    """A small regression at a fine-tuning-like LR: bf16 + SR ends where fp32 weights do; bf16 with nearest
    rounding does not."""
    lr = 3e-4
    m32 = _mlp(torch.float32)
    fp32 = _train(m32, torch.optim.AdamW(m32.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0))
    m_sr = _mlp(torch.bfloat16)
    sr = _train(m_sr, BF16AdamW(m_sr.parameters(), lr=lr, betas=(0.9, 0.95), named=m_sr.named_parameters()))
    m_rn = _mlp(torch.bfloat16)
    rn = _train(m_rn, torch.optim.AdamW(m_rn.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False))
    assert sr == pytest.approx(fp32, rel=0.05), (sr, fp32, rn)
    assert rn > fp32 * 1.05, (sr, fp32, rn)


def _run(model, opt, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(8, 32, generator=g).to(torch.bfloat16)
        opt.zero_grad()
        model(x).float().pow(2).mean().backward()
        opt.step()


def test_deterministic_and_resume_is_bit_exact(tmp_path):
    from palingenesis.checkpoint import _load_optimizer_sharded, _save_optimizer_sharded

    def build(model):
        return BF16AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, named=model.named_parameters(), seed=3)

    full = _mlp(torch.bfloat16)
    opt = build(full)
    _run(full, opt, 10)
    again = _mlp(torch.bfloat16)
    _run(again, build(again), 10)
    assert all(torch.equal(a, b) for a, b in zip(full.parameters(), again.parameters()))

    half = _mlp(torch.bfloat16)
    opt_half = build(half)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):  # the first 5 steps of _run, drawing the same inputs
        x = torch.randn(8, 32, generator=g).to(torch.bfloat16)
        opt_half.zero_grad()
        half(x).float().pow(2).mean().backward()
        opt_half.step()
    _save_optimizer_sharded(opt_half, tmp_path / "opt")
    resumed = copy.deepcopy(half)
    opt_resumed = build(resumed)
    _load_optimizer_sharded(opt_resumed, tmp_path / "opt", None)
    assert opt_resumed.state[next(resumed.parameters())]["step"] == 5
    for _ in range(5):
        x = torch.randn(8, 32, generator=g).to(torch.bfloat16)
        opt_resumed.zero_grad()
        resumed(x).float().pow(2).mean().backward()
        opt_resumed.step()
    assert all(torch.equal(a, b) for a, b in zip(full.parameters(), resumed.parameters()))


def test_other_seed_changes_rounding_only():
    a, b = _mlp(torch.bfloat16), _mlp(torch.bfloat16)
    _run(a, BF16AdamW(a.parameters(), lr=1e-3, seed=0), 5)
    _run(b, BF16AdamW(b.parameters(), lr=1e-3, seed=1), 5)
    wa, wb = a[0].weight.double(), b[0].weight.double()
    assert not torch.equal(wa, wb)
    assert ((wa - wb).norm() / wa.norm()).item() < 0.02


def test_names_key_the_noise_not_the_order():
    a, b = _mlp(torch.bfloat16), _mlp(torch.bfloat16)
    opt_a = BF16AdamW(list(a.parameters()), lr=1e-3, named=a.named_parameters())
    opt_b = BF16AdamW(list(b.parameters())[::-1], lr=1e-3, named=b.named_parameters())
    _run(a, opt_a, 3)
    _run(b, opt_b, 3)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_chunking_does_not_change_the_result(monkeypatch):
    a, b = _mlp(torch.bfloat16), _mlp(torch.bfloat16)
    _run(a, BF16AdamW(a.parameters(), lr=1e-3), 3)
    monkeypatch.setattr(B, "EAGER_CHUNK", 100)
    _run(b, BF16AdamW(b.parameters(), lr=1e-3), 3)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_skips_parameters_without_gradient_and_mixed_dtypes():
    model = _mlp(torch.bfloat16)
    model[2].float()  # an fp32 head next to bf16 layers
    for p in model[0].parameters():
        p.requires_grad_(False)
    opt = BF16AdamW(model.parameters(), lr=1e-3)
    before = model[0].weight.detach().clone()
    x = torch.randn(4, 32).to(torch.bfloat16)
    model[2](model[1](model[0](x)).float()).sum().backward()
    opt.step()
    assert torch.equal(model[0].weight, before) and model[0].weight not in opt.state
    assert opt.state[model[2].weight]["exp_avg"].dtype == torch.float32


def test_step_parameter_equals_step_and_gradient_release():
    from palingenesis.memory import GradientRelease

    a, b = _mlp(torch.bfloat16), _mlp(torch.bfloat16)
    opt_a = BF16AdamW(a.parameters(), lr=1e-3, weight_decay=0.1, named=a.named_parameters())
    opt_b = BF16AdamW(b.parameters(), lr=1e-3, weight_decay=0.1, named=b.named_parameters())
    release = GradientRelease(b, opt_b)
    release.enable()
    g = torch.Generator().manual_seed(0)
    for _ in range(3):
        x = torch.randn(8, 32, generator=g).to(torch.bfloat16)
        opt_a.zero_grad()
        a(x).float().pow(2).mean().backward()
        opt_a.step()
        b(x).float().pow(2).mean().backward()  # steps inside the backward
        opt_b.zero_grad()
    release.disable()
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_rejects_bad_arguments():
    p = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    for kwargs in ({"lr": -1}, {"betas": (1.0, 0.9)}, {"eps": 0.0}, {"weight_decay": -0.1}, {"compile": "yes"}):
        with pytest.raises(ValueError):
            BF16AdamW([p], **kwargs)
    q = nn.Parameter(torch.zeros(8, 8, dtype=torch.bfloat16).t())
    q.grad = torch.ones(8, 8, dtype=torch.bfloat16).t()
    with pytest.raises(RuntimeError, match="contiguous"):
        BF16AdamW([q]).step()


def test_compiled_update_matches_eager():
    """The fused (compiled) update and the eager one round the same elements the same way, up to FMA contraction
    in the fp32 math (an element may differ by one ulp)."""
    p, g = _pair(n=5000)
    q = nn.Parameter(p.detach().clone())
    grad = (torch.randn(p.numel(), generator=g) * 1e-2).to(torch.bfloat16)
    eager = BF16AdamW([p], lr=1e-3, compile=False)
    try:
        compiled = BF16AdamW([q], lr=1e-3, compile=True)
        for _ in range(3):
            p.grad, q.grad = grad.clone(), grad.clone()
            eager.step()
            compiled.step()
    except Exception as e:  # no C++ toolchain for inductor on this machine
        pytest.skip(f"torch.compile unavailable on CPU: {type(e).__name__}")
    differ = (p.detach() != q.detach()).float().mean().item()
    assert differ < 0.01, differ


# ── wiring ──────────────────────────────────────────────────────────────────────────────────────────────────────


def test_build_optimizer_adamw_bf16_groups_and_llrd():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 8)
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
            self.norm = nn.LayerNorm(8)

    model = Tiny().to(torch.bfloat16)
    opt = build_optimizer(model, 1e-3, 0.1, optimizer_name="adamw_bf16", seed=5)
    assert isinstance(opt, BF16AdamW) and opt.seed == 5
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
    llrd = build_optimizer(model, 1e-3, 0.1, llrd_decay=0.9, optimizer_name="adamw_bf16")
    assert isinstance(llrd, BF16AdamW) and len({g["lr"] for g in llrd.param_groups}) > 1


def test_config_validation():
    cfg = Config()
    cfg.train.optimizer = "adamw_bf16"
    assert cfg.validate() == []
    cfg.model.torch_dtype = "float32"
    with pytest.raises(ConfigError, match="adamw_bf16"):
        cfg.validate()
    cfg = Config()
    cfg.train.optimizer = "adamw_bf16"
    cfg.plugins.schedule_free = True
    with pytest.raises(ConfigError, match="Schedule-Free"):
        cfg.validate()


def test_state_bytes_per_param():
    assert B.state_bytes_per_param(torch.bfloat16) == 4
    assert B.state_bytes_per_param(torch.float32) == 8


# ── rounding="kahan" ────────────────────────────────────────────────────────────────────────────────────────────


def _rl_like(rounding, steps=300, lr=1e-6, n=4096):
    """RL-sized updates (~1/100 of a bf16 ulp, random signs). Returns (the stored bf16 weights, the value the optimizer
    holds: with kahan the weights plus their compensation, the fp32-master weights, the start)."""
    p, g = _pair(n=n)
    start = p.detach().double().clone()
    ref = nn.Parameter(p.detach().float().clone())
    opt = BF16AdamW([p], lr=lr, betas=(0.9, 0.95), rounding=rounding)
    ref_opt = torch.optim.AdamW([ref], lr=lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    for _ in range(steps):
        grad = torch.randn(n, generator=g).to(torch.bfloat16)
        p.grad, ref.grad = grad, grad.float()
        opt.step()
        ref_opt.step()
    held = p.detach().double()
    if rounding == "kahan":
        held = held + opt.state[p]["compensation"].double()
    return p.detach().double(), held, ref.detach().double(), start


def test_kahan_tracks_fp32_masters_without_rounding_noise():
    """At RL learning rates stochastic rounding's random walk is larger than the update itself (measured: 4x the
    distance the fp32 weights moved); Kahan holds the fp32 trajectory (2%), and its stored bf16 weights are the
    fp32 masters rounded to nearest, as a bf16 forward of fp32 masters would see them."""
    stored, held, ref, start = _rl_like("kahan")
    sr, _, ref_sr, _ = _rl_like("stochastic")
    assert torch.equal(ref, ref_sr)
    moved = (ref - start).norm()
    assert ((held - ref).norm() / moved).item() < 0.05
    assert ((sr - ref).norm() / moved).item() > 1.0
    assert (stored == ref.float().to(torch.bfloat16).double()).float().mean().item() > 0.95


def test_kahan_small_updates_and_state():
    def bf16(params, lr):
        return BF16AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, rounding="kahan")

    drift = _drift(bf16, torch.bfloat16)
    assert drift / (-200 * 1e-5) == pytest.approx(1.0, abs=0.03)  # the stored weights: within half an ulp
    p = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
    opt = BF16AdamW([p], lr=1e-3, rounding="kahan")
    p.grad = torch.ones(8, dtype=torch.bfloat16)
    opt.step()
    assert opt.state[p]["compensation"].dtype == torch.bfloat16
    assert B.state_bytes_per_param(torch.bfloat16, "kahan") == 6
    with pytest.raises(ValueError, match="rounding"):
        BF16AdamW([p], rounding="nearest")


def test_kahan_resume_is_bit_exact():
    def build(model):
        return BF16AdamW(model.parameters(), lr=1e-3, named=model.named_parameters(), rounding="kahan")

    full = _mlp(torch.bfloat16)
    _run(full, build(full), 6)
    half = _mlp(torch.bfloat16)
    opt = build(half)
    g = torch.Generator().manual_seed(0)

    def steps(model, optimizer, n):
        for _ in range(n):
            x = torch.randn(8, 32, generator=g).to(torch.bfloat16)
            optimizer.zero_grad()
            model(x).float().pow(2).mean().backward()
            optimizer.step()

    steps(half, opt, 3)
    resumed = copy.deepcopy(half)
    opt_resumed = build(resumed)
    opt_resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    steps(resumed, opt_resumed, 3)
    assert all(torch.equal(a, b) for a, b in zip(full.parameters(), resumed.parameters()))


def test_kahan_compiled_matches_eager():
    p, g = _pair(n=5000)
    q = nn.Parameter(p.detach().clone())
    grad = (torch.randn(p.numel(), generator=g) * 1e-2).to(torch.bfloat16)
    eager = BF16AdamW([p], lr=1e-3, rounding="kahan", compile=False)
    try:
        compiled = BF16AdamW([q], lr=1e-3, rounding="kahan", compile=True)
        for _ in range(3):
            p.grad, q.grad = grad.clone(), grad.clone()
            eager.step()
            compiled.step()
    except Exception as e:
        pytest.skip(f"torch.compile unavailable on CPU: {type(e).__name__}")
    assert (p.detach() != q.detach()).float().mean().item() < 0.01


def test_config_bf16_rounding():
    cfg = Config()
    cfg.train.optimizer, cfg.train.bf16_rounding = "adamw_bf16", "kahan"
    assert cfg.validate() == []
    cfg.train.bf16_rounding = "nearest"
    with pytest.raises(ConfigError, match="bf16_rounding"):
        cfg.validate()
    model = _mlp(torch.bfloat16)
    opt = build_optimizer(model, 1e-3, 0.0, optimizer_name="adamw_bf16", bf16_rounding="kahan")
    assert opt.rounding == "kahan"


# ── pgs rl ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _rl_config(**sets):
    from palingenesis.rl.config import RLConfig

    config = RLConfig()
    for key, value in {"model.policy": "x", "data.dataset": "d.jsonl", "rewards.m.fn": "math", **sets}.items():
        config.set(key, value)
    return config


def test_rl_validation():
    from palingenesis.rl.config import RLConfigError

    config = _rl_config(**{"train.optimizer": "adamw_bf16"})
    assert config.train.bf16_rounding == "kahan"
    config.validate()
    for key, value, match in [
        ("train.fsdp", True, "FSDP"),
        ("lora.rank", 16, "adapters"),
        ("train.bf16_rounding", "nearest", "bf16_rounding"),
    ]:
        with pytest.raises(RLConfigError, match=match):
            _rl_config(**{"train.optimizer": "adamw_bf16", key: value}).validate()


def test_rl_optimizer_is_bf16_adamw_with_rl_eps():
    from palingenesis.rl.trainer import _optimizer

    model = _mlp(torch.bfloat16)
    config = _rl_config(**{"train.optimizer": "adamw_bf16"})
    names = [n for n, _ in model.named_parameters()]
    opt = _optimizer(list(model.parameters()), config.train, "cpu", names)
    assert isinstance(opt, BF16AdamW) and opt.rounding == "kahan"
    assert opt.param_groups[0]["eps"] == 1e-15
    _run(model, opt, 2)
