"""train.stable_softmax (palingenesis.stable_softmax): the same values and gradients as PyTorch on
ordinary logits, a target gradient that survives softmax collapse, a zero distillation gradient at
equal logits, and the wiring into cross-entropy, chunked cross-entropy, SeCO and distillation."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from palingenesis import stable_softmax as S
from palingenesis.config import Config
from palingenesis.loss import chunked_cross_entropy_loss, cross_entropy_loss

sys.path.insert(0, str(Path(__file__).parent))

IGNORE = -100


def _grad(fn, x):
    x = x.clone().requires_grad_(True)
    return torch.autograd.grad(fn(x), x)[0]


def test_cross_entropy_matches_pytorch_and_ignores_masked_rows():
    torch.manual_seed(0)
    z = torch.randn(32, 300, dtype=torch.float64) * 3
    t = torch.randint(0, 300, (32,))
    t[[2, 9]] = IGNORE
    ours = S.cross_entropy(z, t)
    ref = F.cross_entropy(z.float(), t, reduction="none", ignore_index=IGNORE)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)
    assert ours[2] == 0 and ours[9] == 0
    g = _grad(lambda x: S.cross_entropy(x, t).sum(), z.float())
    g_ref = _grad(lambda x: F.cross_entropy(x, t, reduction="sum", ignore_index=IGNORE), z.float())
    torch.testing.assert_close(g, g_ref, rtol=1e-4, atol=1e-6)
    assert g[2].abs().sum() == 0


def test_the_target_gradient_survives_softmax_collapse():
    # the target leads by 20 (past fp32's cliff of ~16.6): PyTorch's target gradient rounds to exactly 0
    z = torch.zeros(1, 50)
    z[0, 7] = 20.0
    t = torch.tensor([7])
    ref = _grad(lambda x: F.cross_entropy(x, t, reduction="sum"), z)
    ours = _grad(lambda x: S.cross_entropy(x, t).sum(), z)
    assert ref[0, 7] == 0 and ref[0].sum() > 0
    assert ours[0, 7] < 0
    assert abs(float(ours[0].sum())) < 1e-12  # the target's gradient balances the others'
    torch.testing.assert_close(ours[0, 7], -49 * torch.exp(torch.tensor(-20.0)), rtol=1e-3, atol=0)


def test_log_softmax_matches_pytorch():
    torch.manual_seed(1)
    z = torch.randn(4, 100) * 5
    torch.testing.assert_close(S.log_softmax(z), F.log_softmax(z, -1), rtol=1e-5, atol=1e-5)


def test_forward_kl_matches_pytorch_and_is_flat_at_equal_logits():
    torch.manual_seed(2)
    zs, zt = torch.randn(8, 400) * 4, torch.randn(8, 400) * 4

    def ref(x, teacher):
        return F.kl_div(F.log_softmax(x, -1), F.log_softmax(teacher, -1), log_target=True, reduction="none").sum(-1)

    torch.testing.assert_close(S.forward_kl(zs, zt), ref(zs, zt), rtol=1e-4, atol=1e-5)
    g = _grad(lambda x: S.forward_kl(x, zt).sum(), zs)
    torch.testing.assert_close(g, _grad(lambda x: ref(x, zt).sum(), zs), rtol=1e-4, atol=1e-6)
    # equal logits: PyTorch leaves a rounding residue, the stable gradient is exactly 0
    assert _grad(lambda x: ref(x, zt).sum(), zt).abs().max() > 0
    assert _grad(lambda x: S.forward_kl(x, zt).sum(), zt).abs().max() == 0
    # the teacher gets no gradient
    s, t = zs.clone().requires_grad_(True), zt.clone().requires_grad_(True)
    S.forward_kl(s, t).sum().backward()
    assert s.grad is not None and t.grad is None


@pytest.mark.parametrize("weighted", [False, True])
def test_cross_entropy_losses_take_the_stable_path(weighted):
    torch.manual_seed(3)
    head = torch.nn.Linear(16, 64, bias=False)
    hidden = torch.randn(2, 12, 16)
    labels = torch.randint(0, 64, (2, 12))
    labels[0, :4] = IGNORE
    weights = torch.rand(2, 12) if weighted else None
    denom = float((labels != IGNORE).sum())

    def run(fn):
        h = hidden.clone().requires_grad_(True)
        head.zero_grad()
        loss = fn(h)
        loss.backward()
        return float(loss.detach()), h.grad.clone(), head.weight.grad.clone()

    for stable in (False, True):
        plain = run(lambda h: cross_entropy_loss(head(h), labels, denom, weights=weights, stable=stable))
        chunked = run(
            lambda h: chunked_cross_entropy_loss(
                h, labels, head, num_chunks=3, global_valid_tokens=denom, weights=weights, stable=stable
            )
        )
        if not stable:
            reference = plain
        for got in (plain, chunked):
            assert got[0] == pytest.approx(reference[0], rel=1e-5)
            torch.testing.assert_close(got[1], reference[1], rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(got[2], reference[2], rtol=1e-4, atol=1e-6)


def test_seco_passes_the_option_to_its_chunk_loss(monkeypatch):
    import distill_helpers as H

    from palingenesis import loss as loss_module
    from palingenesis.seco import chunkwise_forward_backward, use_chunk_attention

    tok = H.tokenizer()
    calls = []
    real = loss_module.stable_softmax.cross_entropy
    monkeypatch.setattr(loss_module.stable_softmax, "cross_entropy", lambda *a, **k: calls.append(1) or real(*a, **k))
    torch.manual_seed(4)
    ids = torch.randint(0, len(tok), (1, 40))
    grads = []
    for stable in (False, True):
        model = H.tiny_llama(len(tok), seed=0)
        use_chunk_attention(model)
        chunkwise_forward_backward(model, ids, ids.clone(), chunk_size=16, stable_softmax=stable)
        grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
    assert calls  # the stable run went through stable_softmax
    for n in grads[0]:
        torch.testing.assert_close(grads[1][n], grads[0][n], rtol=1e-3, atol=1e-6, msg=n)


def test_distillation_uses_the_stable_kl_and_cross_entropy(tmp_path, monkeypatch):
    import distill_helpers as H

    from palingenesis.distill import distiller as distiller_module
    from palingenesis.distill.distiller import build_distiller
    from palingenesis.loss import shift_labels
    from palingenesis.train import _get_hidden_states, _get_lm_head

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    H.save_model(tmp_path / "teacher", tok, seed=1, hidden=48)
    data = H.write_rows(tmp_path / "rows.jsonl")
    calls = {"kl": 0, "ce": 0}
    kl, ce = distiller_module.S.forward_kl, distiller_module.S.cross_entropy

    def count(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)

        return wrapped

    monkeypatch.setattr(distiller_module.S, "forward_kl", count("kl", kl))
    monkeypatch.setattr(distiller_module.S, "cross_entropy", count("ce", ce))
    torch.manual_seed(5)
    ids = torch.randint(0, len(tok), (2, 24))
    raw = ids.clone()
    raw[:, :6] = IGNORE
    labels = shift_labels(raw)
    denom = float((labels != IGNORE).sum())
    results = []
    for stable in (False, True):
        c = H.cpu_config(tmp_path / f"run{stable}", tmp_path / "student", data)
        c.distill.enabled, c.distill.ce_weight = True, 0.5
        c.distill.teachers = {"t": {"model": str(tmp_path / "teacher"), "dtype": "float32"}}
        c.train.stable_softmax = stable
        student = H.tiny_llama(len(tok), seed=0)
        distiller = build_distiller(c, tok, torch.device("cpu"))
        targets = distiller.targets({}, ids, labels, {"attention_mask": None}, compute_dtype=torch.float32)
        loss, _ = distiller.loss(_get_hidden_states(student, ids), targets, _get_lm_head(student), denom)
        loss.backward()
        results.append((float(loss.detach()), {n: p.grad.clone() for n, p in student.named_parameters()}))
    assert calls["kl"] > 0 and calls["ce"] > 0
    assert results[1][0] == pytest.approx(results[0][0], rel=1e-4)
    for n, g in results[0][1].items():
        torch.testing.assert_close(results[1][1][n], g, rtol=1e-3, atol=1e-6, msg=n)


def test_config_refuses_objectives_it_does_not_cover():
    from palingenesis.config import ConfigError

    c = Config()
    c.train.stable_softmax = True
    c.validate()
    c.dpo.enabled = True
    with pytest.raises(ConfigError, match=r"stable_softmax .* dpo\.enabled"):
        c.validate()
