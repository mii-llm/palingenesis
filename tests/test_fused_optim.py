"""palingenesis.fused_optim (train.fused_optimizer): the multi-tensor block map, FusedAdamW against torch AdamW
(update, parameter groups, state_dict interchange), BF16AdamW's multi-tensor mode and folded gradient clipping, the
fallback without CUDA / Triton, config validation and the trainer path. CPU.

The Triton kernels themselves need a GPU (scripts/gpu_checks/offload_fused_optim.sh checks them bit for bit against
the eager reference). What runs here instead is an emulation of the kernels' integer formulation (uint32 hash,
rounding on the top 16 bits), which the tests compare bit for bit with palingenesis.bf16_adamw's reference functions
over the block map, so the reformulation and the bookkeeping (offsets, per-tensor seeds and scalars) are covered."""

import copy

import pytest
import torch
import torch.nn as nn

from palingenesis import bf16_adamw as B
from palingenesis import fused_optim as F
from palingenesis.bf16_adamw import BF16AdamW
from palingenesis.config import Config, ConfigError
from palingenesis.fused_optim import FusedAdamW, block_map
from palingenesis.optim import build_optimizer

M32 = 0xFFFFFFFF

# ── block map ───────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("numels", [[1], [8192], [8193, 0, 5, 16384], [0, 0], [3, 70000, 1]])
def test_block_map_covers_every_element_once(numels):
    chunk = 8192
    tensor, start = block_map(numels, chunk)
    assert tensor.dtype == torch.int32 and start.dtype == torch.int64
    covered = [torch.zeros(n, dtype=torch.int64) for n in numels]
    for t, s in zip(tensor.tolist(), start.tolist()):
        assert s % chunk == 0 and s < numels[t]
        covered[t][s : s + chunk] += 1
    assert all(bool((c == 1).all()) for c in covered)
    assert tensor.tolist() == sorted(tensor.tolist())  # tensor order: one tensor's programs are contiguous


def test_block_map_empty():
    tensor, start = block_map([])
    assert tensor.numel() == 0 and start.numel() == 0


# ── the kernels' integer formulation, emulated (uint32 values held in int64) ────────────────────────────────────


def _hash32_u32(x):
    """The kernel's _hash32 on uint32 (products wrap mod 2^32, logical shifts)."""
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x5BD1E995) & M32
    return x ^ (x >> 16)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int32).to(torch.int64) & M32


def _from_hi(hi: torch.Tensor) -> torch.Tensor:
    """uint16 pattern (in int64) -> bf16."""
    return (hi & 0xFFFF).to(torch.int32).to(torch.int16).view(torch.bfloat16)


def _round_stochastic_u32(x, noise):
    """The kernel's _round_stochastic, line by line."""
    bits = _bits(x)
    finite = (bits & 0x7F800000) != 0x7F800000
    hi = ((bits + noise) & M32) >> 16
    hi = torch.where((hi & 0x7FFF) == 0x7F80, bits >> 16, hi)
    return torch.where(finite, _from_hi(hi), x.to(torch.bfloat16))


def _round_nearest_u32(x):
    bits = _bits(x)
    finite = (bits & 0x7F800000) != 0x7F800000
    hi = ((bits + 0x7FFF + ((bits >> 16) & 1)) & M32) >> 16
    out = torch.where(finite, _from_hi(hi), x.to(torch.bfloat16))
    as_f32 = torch.where(finite, ((hi << 16) & M32).to(torch.int64).to(torch.int32).view(torch.float32), x)
    return out, as_f32


def _special_values() -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    big = torch.finfo(torch.bfloat16).max
    tiny = torch.finfo(torch.float32).tiny
    x = torch.cat(
        [
            torch.randn(20000, generator=g) * torch.logspace(-40, 38, 20000).clamp(max=3e38),
            torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), big, -big, float(big) * 1.003]),
            torch.tensor([tiny, -tiny, tiny / 4, -tiny / 1024, 3.4028235e38, -3.4028235e38, 1.0, -1.0]),
        ]
    )
    return torch.cat([x, -x])


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    nan = torch.isnan(a.float())
    return bool((nan == torch.isnan(b.float())).all()) and torch.equal(
        a[~nan].view(torch.int16), b[~nan].view(torch.int16)
    )


def test_kernel_hash_matches_reference():
    idx = torch.arange(0, 1 << 16, dtype=torch.int64) * 977 + (1 << 31)
    for seed in (0, 1, 0xDEADBEEF, M32):
        ref = B.hash32(idx, seed)
        assert torch.equal(_hash32_u32((idx + seed) & M32), ref)


def test_kernel_stochastic_rounding_matches_reference_bit_for_bit():
    x = _special_values()
    for s in range(4):
        noise = B.hash32(torch.arange(x.numel()), s * 7919) & 0xFFFF
        assert _same_bits(_round_stochastic_u32(x, noise), B.round_bf16(x, noise))
    for noise in (0, 0x7FFF, 0x8000, 0xFFFF):
        n = torch.full_like(x, noise, dtype=torch.int64)
        assert _same_bits(_round_stochastic_u32(x, n), B.round_bf16(x, n))


def test_kernel_nearest_rounding_matches_reference_bit_for_bit():
    x = _special_values()
    out, as_f32 = _round_nearest_u32(x)
    ref_out, ref_f32 = B.round_bf16_nearest(x)
    assert _same_bits(out, ref_out)
    nan = torch.isnan(x)
    assert torch.equal(as_f32[~nan].view(torch.int32), ref_f32[~nan].view(torch.int32))


def _emulate_bf16_kernel(rows, kahan: bool, scale=None, chunk: int = 4096, block: int = 1024):
    """The bf16 kernel's program loop over the block map, with its integer formulation, on CPU tensors."""
    tensor, start = block_map([r.p.numel() for r in rows], chunk)
    for t, s0 in zip(tensor.tolist(), start.tolist()):
        r = rows[t]
        decay, beta1, beta2, step_size, inv_bc2_sqrt, eps = (torch.tensor(v, dtype=torch.float32) for v in r.hp)
        n = r.p.numel()
        for i in range(chunk // block):
            lo, hi = s0 + i * block, min(s0 + (i + 1) * block, n)
            if lo >= n:
                break
            sl = slice(lo, hi)
            gf = r.g[sl].float()
            if scale is not None:
                gf = gf * scale
            mf = r.m[sl].float() * beta1 + gf * (1.0 - beta1)
            vf = r.v[sl].float() * beta2 + gf * gf * (1.0 - beta2)
            pf = r.p[sl].float()
            if kahan:
                pf = pf + r.c[sl].float()
            pf = pf * decay - step_size * mf / (vf.sqrt() * inv_bc2_sqrt + eps)
            x = _hash32_u32((torch.arange(lo, hi, dtype=torch.int64) + r.seed) & M32)
            if kahan:
                stored, stored_f32 = _round_nearest_u32(pf)
                r.p[sl] = stored
                r.c[sl] = (pf - stored_f32).to(torch.bfloat16)
            else:
                r.p[sl] = _round_stochastic_u32(pf, x & 0xFFFF)
            r.v[sl] = _round_stochastic_u32(vf, x >> 16)
            r.m[sl] = mf.to(torch.bfloat16)


def _model(seed=0, sizes=((300, 41), (5000,), (7, 9), (13000,))):
    g = torch.Generator().manual_seed(seed)
    params = [nn.Parameter((torch.randn(*s, generator=g) * 0.02).to(torch.bfloat16)) for s in sizes]
    return params, g


@pytest.mark.parametrize("rounding", ["stochastic", "kahan"])
@pytest.mark.parametrize("scaled", [False, True])
def test_emulated_multi_tensor_step_equals_per_parameter_step(rounding, scaled):
    """Tables, block map, per-tensor scalars and seeds: the multi-tensor formulation over all parameters gives the
    per-parameter reference's bits (BF16AdamW with compile=False), parameter group scalars included (LLRD-like)."""
    params, g = _model()
    ref_params = [nn.Parameter(p.detach().clone()) for p in params]

    def groups(ps):
        return [
            {"params": ps[:2], "lr": 1e-3, "weight_decay": 0.1},
            {"params": ps[2:], "lr": 3e-4, "weight_decay": 0.0},
        ]

    names = [(f"w{i}", p) for i, p in enumerate(params)]
    ref_names = [(f"w{i}", p) for i, p in enumerate(ref_params)]
    ref = BF16AdamW(groups(ref_params), named=ref_names, seed=5, rounding=rounding, compile=False)
    emu = BF16AdamW(groups(params), named=names, seed=5, rounding=rounding, compile=False)
    scale = torch.tensor([0.37]) if scaled else None
    for step in range(3):
        grads = [(torch.randn(p.shape, generator=g) * 1e-2).to(torch.bfloat16) for p in params]
        for p, q, gr in zip(params, ref_params, grads):
            p.grad, q.grad = gr.clone(), gr.clone()
        if not scaled:
            ref.step()
        else:  # the folded factor multiplies the fp32 gradient: the reference update gets that fp32 gradient
            update = B._bf16_adamw_kahan_update if rounding == "kahan" else B._bf16_adamw_update
            for group in ref.param_groups:
                for q in group["params"]:
                    st = ref._init_state(q)
                    st["step"] += 1
                    hp = torch.tensor(ref._scalars(group, st["step"]), dtype=torch.float32)
                    seed = torch.tensor(ref._step_seed(q, st["step"]))
                    names_ = ("exp_avg", "exp_avg_sq") + (("compensation",) if rounding == "kahan" else ())
                    update(
                        q.data.view(-1), q.grad.float().view(-1) * scale, *(st[k].view(-1) for k in names_), hp, seed
                    )
        rows = []
        for group in emu.param_groups:
            for p in group["params"]:
                st = emu._init_state(p)
                st["step"] += 1
                rows.append(
                    F.Bf16Row(
                        p.data.view(-1),
                        p.grad.view(-1),
                        st["exp_avg"].view(-1),
                        st["exp_avg_sq"].view(-1),
                        st["compensation"].view(-1) if rounding == "kahan" else None,
                        emu._scalars(group, st["step"]),
                        emu._step_seed(p, st["step"]),
                    )
                )
        _emulate_bf16_kernel(rows, rounding == "kahan", scale)
        for p, q in zip(params, ref_params):
            assert torch.equal(p.detach().view(torch.int16), q.detach().view(torch.int16)), step
            for key in ("exp_avg", "exp_avg_sq") + (("compensation",) if rounding == "kahan" else ()):
                assert torch.equal(emu.state[p][key].view(torch.int16), ref.state[q][key].view(torch.int16)), key


def test_step_seed_equals_seed_chain():
    p = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    opt = BF16AdamW([p], named=[("layer.w", p)], seed=1234)
    for step in (1, 2, 1000):
        assert opt._step_seed(p, step) == B._seed(opt._key(p), step, 1234)
    opt.load_state_dict({**opt.state_dict(), "bf16_adamw_seed": 99})
    assert opt._step_seed(p, 3) == B._seed(opt._key(p), 3, 99)  # the cached chain follows a loaded seed


# ── FusedAdamW (fp32) ───────────────────────────────────────────────────────────────────────────────────────────


def _fp32_params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.randn(*s, generator=g)) for s in ((64, 32), (32,), (1000,))], g


def test_fused_adamw_matches_torch_adamw_with_groups():
    ps, g = _fp32_params()
    qs = [nn.Parameter(p.detach().clone()) for p in ps]

    def groups(x):
        return [{"params": x[:1], "lr": 1e-2, "weight_decay": 0.1}, {"params": x[1:], "lr": 3e-3, "weight_decay": 0.0}]

    fused = FusedAdamW(groups(ps), lr=1e-3, betas=(0.9, 0.95))
    torch_opt = torch.optim.AdamW(groups(qs), lr=1e-3, betas=(0.9, 0.95), foreach=False)
    for _ in range(20):
        for p, q in zip(ps, qs):
            grad = torch.randn(p.shape, generator=g)
            p.grad, q.grad = grad.clone(), grad.clone()
        fused.step()
        torch_opt.step()
    for p, q in zip(ps, qs):
        torch.testing.assert_close(p.detach(), q.detach(), rtol=1e-6, atol=1e-7)


def test_fused_adamw_state_dict_interchanges_with_torch_adamw():
    ps, g = _fp32_params(1)
    fused = FusedAdamW(ps, lr=1e-2)
    for _ in range(3):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g)
        fused.step()
    sd = fused.state_dict()
    assert all(torch.is_tensor(s["step"]) and s["step"].item() == 3 for s in sd["state"].values())
    assert all(isinstance(fused.state[p]["step"], int) for p in ps)  # the live state keeps host ints
    qs = [nn.Parameter(p.detach().clone()) for p in ps]
    torch_opt = torch.optim.AdamW(qs, lr=1e-2, foreach=False)
    torch_opt.load_state_dict(copy.deepcopy(sd))
    back = FusedAdamW([nn.Parameter(p.detach().clone()) for p in ps], lr=1e-2)
    back.load_state_dict(torch_opt.state_dict())
    assert all(isinstance(s["step"], int) and s["step"] == 3 for s in back.state.values())
    for _ in range(2):
        grads = [torch.randn(p.shape, generator=g) for p in ps]
        for p, q, gr in zip(ps, qs, grads):
            p.grad, q.grad = gr.clone(), gr.clone()
        fused.step()
        torch_opt.step()
    for p, q in zip(ps, qs):
        torch.testing.assert_close(p.detach(), q.detach(), rtol=1e-6, atol=1e-7)


def test_fused_adamw_rejects_bad_arguments_and_sparse_gradients():
    p = nn.Parameter(torch.zeros(3))
    for kwargs in ({"lr": -1}, {"betas": (1.0, 0.9)}, {"eps": 0}, {"weight_decay": -0.1}):
        with pytest.raises(ValueError):
            FusedAdamW([p], **kwargs)
    p.grad = torch.zeros(3).to_sparse()
    with pytest.raises(RuntimeError, match="sparse"):
        FusedAdamW([p]).step()


# ── folded clipping ─────────────────────────────────────────────────────────────────────────────────────────────


def _grads_on(params, grads):
    for p, gr in zip(params, grads):
        p.grad = gr.clone()


@pytest.mark.parametrize("kind", ["fused_adamw", "bf16"])
def test_folded_clipping_matches_clip_then_step(kind):
    if kind == "bf16":
        ps, g = _model(2)
        qs = [nn.Parameter(p.detach().clone()) for p in ps]
        make = lambda x: BF16AdamW(x, lr=1e-3, compile=False, multi_tensor=True)  # noqa: E731
    else:
        ps, g = _fp32_params(2)
        qs = [nn.Parameter(p.detach().clone()) for p in ps]
        make = lambda x: FusedAdamW(x, lr=1e-3)  # noqa: E731
    folded, plain = make(ps), make(qs)
    for _ in range(4):
        grads = [(torch.randn(p.shape, generator=g) * 3).to(p.dtype) for p in ps]
        _grads_on(ps, grads)
        _grads_on(qs, grads)
        norm = folded.clip_grad_norm_(1.0)
        ref_norm = torch.nn.utils.clip_grad_norm_(qs, 1.0)
        assert norm.item() == pytest.approx(ref_norm.float().item(), rel=1e-2)  # torch rounds bf16 norms to bf16
        assert folded.grad_scale is not None and folded.grad_scale.item() < 1.0
        folded.step()
        plain.step()
        assert folded.grad_scale is None  # consumed by the step
    for p, q in zip(ps, qs):
        torch.testing.assert_close(p.detach().float(), q.detach().float(), rtol=2e-2, atol=2e-5)


def test_zero_grad_drops_a_pending_clip_factor():
    """A skipped step (spike detection zeroes the gradients) must not carry its clip factor to the next step."""
    ps, g = _fp32_params(3)
    opt = FusedAdamW(ps, lr=1e-3)
    _grads_on(ps, [torch.randn(p.shape, generator=g) * 100 for p in ps])
    opt.clip_grad_norm_(1.0)
    assert opt.grad_scale is not None
    opt.zero_grad(set_to_none=True)
    assert opt.grad_scale is None
    norm = opt.clip_grad_norm_(0.0)  # the norm only
    assert norm.item() == 0.0 and opt.grad_scale is None


def test_bf16_multi_tensor_without_triton_is_the_per_parameter_path():
    ps, g = _model(4)
    qs = [nn.Parameter(p.detach().clone()) for p in ps]
    a = BF16AdamW(ps, lr=1e-3, compile=False, multi_tensor=True, named=[(str(i), p) for i, p in enumerate(ps)])
    b = BF16AdamW(qs, lr=1e-3, compile=False, named=[(str(i), p) for i, p in enumerate(qs)])
    assert not a.triton  # CPU
    for _ in range(3):
        grads = [(torch.randn(p.shape, generator=g) * 1e-2).to(torch.bfloat16) for p in ps]
        _grads_on(ps, grads)
        _grads_on(qs, grads)
        a.step()
        b.step()
    for p, q in zip(ps, qs):
        assert torch.equal(p.detach().view(torch.int16), q.detach().view(torch.int16))
    assert a.last_launches == 0


# ── wiring ──────────────────────────────────────────────────────────────────────────────────────────────────────


class _Tiny(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8).to(dtype)
        self.layers = nn.ModuleList([nn.Linear(8, 8).to(dtype) for _ in range(2)])
        self.norm = nn.LayerNorm(8).to(dtype)


def test_build_optimizer_fused_falls_back_without_cuda(caplog):
    with caplog.at_level("WARNING"):
        opt = build_optimizer(_Tiny(), 1e-3, 0.1, fused=True)
    assert type(opt) is torch.optim.AdamW
    assert "train.fused_optimizer" in caplog.text
    opt = build_optimizer(_Tiny(torch.bfloat16), 1e-3, 0.1, optimizer_name="adamw_bf16", fused=True)
    assert isinstance(opt, BF16AdamW) and not opt.multi_tensor


def test_build_optimizer_fused_when_usable(monkeypatch):
    """With the kernels usable, adamw gives FusedAdamW (LLRD groups kept) and adamw_bf16 the multi-tensor mode."""
    import palingenesis.optim as O

    monkeypatch.setattr(O, "_fused_kernels_usable", lambda model, name: True)
    opt = build_optimizer(_Tiny(), 1e-3, 0.1, llrd_decay=0.9, fused=True)
    assert isinstance(opt, FusedAdamW)
    assert len({g["lr"] for g in opt.param_groups}) > 1
    assert {g["weight_decay"] for g in opt.param_groups} == {0.1, 0.0}
    opt = build_optimizer(_Tiny(torch.bfloat16), 1e-3, 0.1, optimizer_name="adamw_bf16", fused=True)
    assert isinstance(opt, BF16AdamW) and opt.multi_tensor


def test_config_validation():
    c = Config()
    c.train.fused_optimizer = True
    c.validate()  # adamw
    c.train.optimizer = "muon"
    with pytest.raises(ConfigError, match="fused_optimizer does not support train.optimizer=muon"):
        c.validate()
    c = Config()
    c.train.fused_optimizer = True
    c.memory.gradient_release = True
    c.train.gradient_accumulation_steps = 1
    with pytest.raises(ConfigError, match="fused_optimizer does not support memory.gradient_release"):
        c.validate()
    c = Config()
    c.train.fused_optimizer = True
    c.plugins.schedule_free = True
    with pytest.raises(ConfigError, match="schedule_free"):
        c.validate()
    assert Config().train.fused_optimizer is False  # default unchanged


def test_importing_does_not_import_triton():
    import subprocess
    import sys

    code = (
        "import sys, palingenesis.fused_optim, palingenesis.bf16_adamw, palingenesis.config, "
        "palingenesis.activation_offload; print('triton' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_trainer_with_fused_optimizer_on_cpu_trains_like_without(tmp_path):
    """On the CPU train.fused_optimizer falls back to the usual optimizer: the same weights as without it."""
    import distill_helpers as H

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok)
    data = H.write_rows(tmp_path / "rows.jsonl")
    losses = []
    for fused in (False, True):
        c = H.cpu_config(tmp_path / f"run_{fused}", tmp_path / "student", data, steps=3)
        c.train.fused_optimizer = fused
        c.train.save_final = False
        losses.append(H.run_cpu(c).metrics["train/loss"])
    assert losses[0] == losses[1]
