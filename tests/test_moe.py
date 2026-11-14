"""Mixture-of-Experts training (palingenesis.moe): router balancing.

The balancer injects the gradient of the load-balancing loss and of the router z-loss at the
router logits. These tests compare it with the same losses written out by hand and added to
the task loss (one scalar, plain autograd), for every tiny MoE family, and through the
trainers' memory paths: activation checkpointing, the chunked loss, SeCO, gradient
accumulation, padding, several ranks.
"""

import contextlib
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402
from seco_archs import torch_kernels  # noqa: E402
from test_multi_rank_agreement import _run  # noqa: E402

from palingenesis.moe import (  # noqa: E402
    TINY_MOE,
    RouterBalancer,
    RoutingRecorder,
    balance_loss,
    find_moe_layers,
    router_probabilities,
    tiny_moe,
    z_loss,
)

IGNORE_INDEX = -100
COEF, Z_COEF = 0.05, 0.002
FAMILIES = list(TINY_MOE)


def _model(family: str = "qwen3_moe", **kw):
    with torch_kernels(TINY_MOE[family][0]):  # fp64 references: no Triton / CUDA linear-attention kernels
        model = tiny_moe(family, **kw).double().train()
    model.config.use_cache = False
    model.set_experts_implementation("eager")  # torch's grouped GEMM has no fp64
    return model


def _batch(rows: int = 3, length: int = 20, seed: int = 1):
    """Token ids, the real-token mask (right padding of different lengths) and labels."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 256, (rows, length), generator=g)
    mask = torch.ones(rows, length, dtype=torch.long)
    for row in range(1, rows):
        mask[row, length - 3 * row :] = 0
    labels = ids.clone()
    labels[mask == 0] = IGNORE_INDEX
    labels[:, :2] = IGNORE_INDEX
    return ids, mask, labels


def _ce(model, ids, labels, denom):
    logits = model(input_ids=ids).logits
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="sum") / (
        denom
    )


def _denom(labels) -> int:
    return int((labels[:, 1:] != IGNORE_INDEX).sum())


@contextlib.contextmanager
def _captured(model):
    """Router logits (with their graph) and chosen experts of every MoE layer, by plain hooks."""
    layers = find_moe_layers(model)
    logits, chosen, handles = {}, {}, []
    for layer in layers:
        handles.append(
            layer.router.register_forward_hook(
                lambda m, a, out, i=layer.index: logits.__setitem__(i, out[0] if isinstance(out, tuple) else out)
            )
        )
        handles.append(layer.experts.register_forward_pre_hook(lambda m, a, i=layer.index: chosen.__setitem__(i, a[1])))
    try:
        yield layers, logits, chosen
    finally:
        for h in handles:
            h.remove()


def _hand_losses(layers, logits, chosen, mask, loads=None):
    """(balance loss, z-loss, counts [layers, E]) written out: per layer E * sum_i f_i P_i with f the
    share of real tokens routed to expert i (or `loads`) and P the mean router probability; the mean
    over real tokens of logsumexp^2; both averaged over the layers."""
    real = mask.reshape(-1).bool()
    balance, z, counts = [], [], []
    for layer in layers:
        x = logits[layer.index].reshape(-1, layer.num_experts)[real]
        p = torch.softmax(x, -1) if layer.scoring == "softmax" else torch.sigmoid(x) / torch.sigmoid(x).sum(-1, True)
        c = torch.bincount(chosen[layer.index].reshape(real.numel(), -1)[real].reshape(-1), minlength=layer.num_experts)
        c = c.double()
        f = (c / real.sum()) if loads is None else loads[layer.index]
        balance.append(layer.num_experts * (f.detach() * p.mean(0)).sum())
        z.append((torch.logsumexp(x, -1) ** 2).mean())
        counts.append(c)
    return torch.stack(balance).mean(), torch.stack(z).mean(), torch.stack(counts)


def _reference(model, ids, mask, labels, coef=COEF, z_coef=Z_COEF):
    """Gradients of CE + coef * balance + z_coef * z-loss as one scalar; also the two loss values."""
    model.zero_grad()
    with _captured(model) as (layers, logits, chosen):
        loss = _ce(model, ids, labels, _denom(labels))
        balance, z, _ = _hand_losses(layers, logits, chosen, mask)
        (loss + coef * balance + z_coef * z).backward()
    return _grads(model), balance.item(), z.item()


def _grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _assert_same(got: dict, ref: dict, rtol=1e-7, atol=1e-10):
    assert got.keys() == ref.keys()
    for name in ref:
        torch.testing.assert_close(got[name], ref[name], rtol=rtol, atol=atol, msg=name)


def _balancer(model, coef=COEF, z_coef=Z_COEF, **kw):
    return RouterBalancer(find_moe_layers(model), balance_coef=coef, z_coef=z_coef, **kw)


# ── detection, the tiny families ────────────────────────────────────────────────


@pytest.mark.parametrize("family", FAMILIES)
def test_tiny_families_have_routed_layers(family):
    model = _model(family)
    layers = find_moe_layers(model)
    assert len(layers) >= 3 and all(layer.num_experts == 8 for layer in layers)
    assert layers[0].scoring == ("sigmoid" if family == "deepseek_v3" else "softmax")
    assert (layers[0].selection_bias is not None) == (family == "deepseek_v3")
    assert find_moe_layers(H.tiny_llama(64)) == []


def test_losses_at_uniform_routing():
    """Uniform router: the balancing loss is top_k (its minimum), the z-loss is log(E)^2."""
    logits = torch.zeros(10, 8)
    load = torch.full((8,), 2 / 8)  # top-2 routing, balanced
    assert balance_loss(logits, load).mean().item() == pytest.approx(2.0)
    assert z_loss(logits).mean().item() == pytest.approx(math.log(8) ** 2)
    # all the load on the experts the router prefers: larger
    skewed = torch.zeros(10, 8)
    skewed[:, :2] = 3.0
    load = torch.tensor([1.0, 1.0, 0, 0, 0, 0, 0, 0])
    assert balance_loss(skewed, load).mean().item() > 6.0


# ── the injected gradient is the gradient of the written-out losses ─────────────


@pytest.mark.parametrize("family", FAMILIES)
def test_gradients_and_values_match_the_written_out_losses(family):
    model = _model(family)
    ids, mask, labels = _batch()
    ref, ref_balance, ref_z = _reference(model, ids, mask, labels)

    model.zero_grad()
    balancer = _balancer(model)
    with balancer.batch(mask):
        loss = _ce(model, ids, labels, _denom(labels))
        balancer.sync()
        loss.backward()
    balancer.step()
    # DeepSeek-V3's router computes its logits in fp32 whatever the model's dtype
    tol = dict(rtol=1e-3, atol=1e-7) if family == "deepseek_v3" else {}
    _assert_same(_grads(model), ref, **tol)
    metrics = balancer.metrics()
    assert metrics["moe/balance_loss"] == pytest.approx(ref_balance, rel=1e-6)
    assert metrics["moe/z_loss"] == pytest.approx(ref_z, rel=1e-6)
    # the routers do get a gradient from the two losses
    balancer.detach()
    model.zero_grad()
    _ce(model, ids, labels, _denom(labels)).backward()
    router = next(n for n, _ in model.named_parameters() if n.endswith(("gate.weight", "router.weight")))
    assert not torch.allclose(_grads(model)[router], ref[router], rtol=1e-3)


def test_padding_is_left_out():
    """Other padding tokens, same real tokens: same loads, losses and gradients."""
    model = _model()
    ids, mask, labels = _batch()
    results = []
    for fill in (0, 77):
        padded = ids.masked_fill(mask == 0, fill)
        model.zero_grad()
        balancer = _balancer(model)
        with balancer.batch(mask):
            loss = _ce(model, padded, labels, _denom(labels))
            balancer.sync()
            loss.backward()
        balancer.step()
        results.append((_grads(model), balancer.metrics(), balancer.expert_loads()))
        balancer.detach()
    _assert_same(results[0][0], results[1][0])
    assert results[0][1] == pytest.approx(results[1][1])
    torch.testing.assert_close(results[0][2], results[1][2])
    # without the mask the padding is routed and counted: other loads
    balancer = _balancer(model)
    with balancer.batch(None), torch.no_grad():
        model(input_ids=ids)
    balancer.step()
    assert not torch.allclose(balancer.expert_loads(), results[0][2])


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_activation_checkpointing(mode):
    """The layers run twice (no grad, then recomputed in the backward): counted once, same gradients."""
    from palingenesis.kernels import apply_activation_checkpointing

    model = _model()
    ids, mask, labels = _batch()
    ref, ref_balance, _ = _reference(model, ids, mask, labels)
    balancer = _balancer(model)
    apply_activation_checkpointing(model, mode)
    model.zero_grad()
    with balancer.batch(mask):
        loss = _ce(model, ids, labels, _denom(labels))
        balancer.sync()
        loss.backward()
    balancer.step()
    got = {n.replace("._checkpoint_wrapped_module", ""): g for n, g in _grads(model).items()}
    _assert_same(got, ref)
    assert balancer.metrics()["moe/balance_loss"] == pytest.approx(ref_balance, rel=1e-9)


def test_chunked_loss_from_hidden_states():
    """The trainer's default path: hidden states, the head applied chunk by chunk outside the model."""
    from palingenesis.logits import final_hidden_states, output_head
    from palingenesis.loss import chunked_cross_entropy_loss, shift_labels

    model = _model()
    ids, mask, labels = _batch()
    ref, _, _ = _reference(model, ids, mask, labels)
    model.zero_grad()
    balancer = _balancer(model)
    with balancer.batch(mask):
        hidden = final_hidden_states(model, ids, None)
        loss = chunked_cross_entropy_loss(
            hidden, shift_labels(labels), output_head(model), num_chunks=3, global_valid_tokens=_denom(labels)
        )
        balancer.sync()
        loss.backward()
    _assert_same(_grads(model), ref, rtol=1e-5, atol=1e-8)  # the chunked loss computes its logits in fp32


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_5_moe", "mixtral"])
@pytest.mark.parametrize("chunk", [5, 8])
def test_seco_chunks(family, chunk):
    """SeCO runs its own backward chunk by chunk: the loads are those of the whole micro-batch
    (measured in its no-grad pass), and every chunk's routers get their gradient once."""
    from palingenesis.seco import chunkwise_forward_backward

    model = _model(family)
    ids, mask, labels = _batch()
    ref, ref_balance, ref_z = _reference(model, ids, mask, labels)
    model.zero_grad()
    balancer = _balancer(model)
    with balancer.batch(mask):
        chunkwise_forward_backward(model, ids, labels, chunk_size=chunk, loss_denom=_denom(labels))
    balancer.step()
    _assert_same(_grads(model), ref, rtol=1e-5, atol=1e-8)  # fp32 logits in the chunk losses
    metrics = balancer.metrics()
    assert metrics["moe/balance_loss"] == pytest.approx(ref_balance, rel=1e-9)
    assert metrics["moe/z_loss"] == pytest.approx(ref_z, rel=1e-9)


def test_gradient_accumulation_measures_the_load_over_the_step():
    """Two micro-batches of one optimizer step: the second one's load covers both (the step so far),
    each loss is averaged over the micro-batches, and the buffer restarts with the next step."""
    model = _model()
    a, b = _batch(seed=1), _batch(seed=2)
    # by hand: micro-batch 1 against its own counts, micro-batch 2 against the counts of both
    model.zero_grad()
    counts = None
    for ids, mask, labels in (a, b):
        with _captured(model) as (layers, logits, chosen):
            loss = _ce(model, ids, labels, _denom(labels) * 2)
            _, _, c = _hand_losses(layers, logits, chosen, mask)
            counts = c if counts is None else counts + c
            real = mask.sum() if ids is a[0] else a[1].sum() + b[1].sum()
            balance, z, _ = _hand_losses(layers, logits, chosen, mask, loads=counts / real)
            (loss + (COEF * balance + Z_COEF * z) / 2).backward()
    ref = _grads(model)

    model.zero_grad()
    balancer = _balancer(model)
    for ids, mask, labels in (a, b):
        with balancer.batch(mask, grad_accum=2):
            loss = _ce(model, ids, labels, _denom(labels) * 2)
            balancer.sync()
            loss.backward()
    _assert_same(_grads(model), ref)
    loads_step = balancer.expert_loads()
    balancer.step()
    with balancer.batch(a[1]), torch.no_grad():
        model(input_ids=a[0])
    assert not torch.allclose(balancer.expert_loads(), loads_step)  # the new step: micro-batch 1 alone


def test_frozen_routers_are_measured_not_trained():
    model = _model()
    for layer in find_moe_layers(model):
        layer.router.weight.requires_grad_(False)
    ids, mask, labels = _batch()
    balancer = _balancer(model)
    with balancer.batch(mask), torch.no_grad():
        model(input_ids=ids)
    balancer.step()
    metrics = balancer.metrics()
    assert 1.0 <= metrics["moe/load_max"] <= 4.0 and 0.0 < metrics["moe/router_entropy"] <= 1.0


# ── defaults from the model config, opting out ──────────────────────────────────


def test_defaults_follow_the_model_config():
    from palingenesis.config import ConfigError, MoEConfig

    model = _model()
    balancer = RouterBalancer.attach(model)
    assert balancer.balance_coef == pytest.approx(model.config.router_aux_loss_coef) and balancer.z_coef == 0.0
    assert balancer.bias_update_rate == 0.0  # no selection bias in this family's checkpoints
    balancer.detach()

    model.config.router_aux_loss_coef = 0.9  # a placeholder some checkpoints ship: capped
    assert RouterBalancer.attach(model).balance_coef == pytest.approx(0.01)
    assert RouterBalancer.attach(model, MoEConfig(balance_loss_coef=0.2, z_loss_coef=1e-3)).balance_coef == 0.2
    assert RouterBalancer.attach(model, MoEConfig(balance=False)) is None
    assert RouterBalancer.attach(H.tiny_llama(64)) is None
    with pytest.raises(ConfigError, match="selection bias"):
        RouterBalancer.attach(model, MoEConfig(bias_update_rate=1e-3))

    deepseek = RouterBalancer.attach(_model("deepseek_v3"))
    assert deepseek.bias_update_rate == pytest.approx(1e-3) and deepseek.balance_coef == 0.0


def test_selection_bias_moves_against_the_load():
    """Auxiliary-loss-free balancing: after a step, overloaded experts' bias goes down by the rate,
    underloaded experts' up; applied again and again it evens out the loads."""
    model = _model("deepseek_v3", router_std=0.3)
    ids, mask, _ = _batch(rows=4, length=48)
    balancer = RouterBalancer(find_moe_layers(model), bias_update_rate=0.01)
    layer = balancer.layers[0]
    with balancer.batch(mask), torch.no_grad():
        model(input_ids=ids)
    counts = balancer._buffer[0, :8].clone()
    first = balancer.metrics()["moe/load_cv"]
    balancer.step()
    torch.testing.assert_close(layer.selection_bias, 0.01 * torch.sign(counts.mean() - counts).to(layer.selection_bias))
    assert "e_score_correction_bias" in dict(layer.router.named_buffers())  # it is part of the checkpoint
    for _ in range(60):
        with balancer.batch(mask), torch.no_grad():
            model(input_ids=ids)
        last = balancer.metrics()["moe/load_cv"]
        balancer.step()
    assert last < 0.5 * first


# ── the global batch: several ranks ─────────────────────────────────────────────


def _rank_gradients(rank, world, fsdp):
    """This rank's rows of the batch through the balancer; the gradients summed over the ranks."""
    import torch.distributed as dist

    torch.set_num_threads(1)
    model = _model()
    ids, mask, labels = _batch(rows=6, length=16)
    denom = _denom(labels)
    balancer = _balancer(model)
    if fsdp:
        from torch.distributed.device_mesh import init_device_mesh

        from palingenesis.config import ParallelConfig
        from palingenesis.distributed import apply_fsdp

        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        model = apply_fsdp(model, mesh, ParallelConfig(), bf16=False)
    rows = slice(rank * 6 // world, (rank + 1) * 6 // world)
    with balancer.batch(mask[rows]):
        loss = _ce(model, ids[rows], labels[rows], denom)
        balancer.sync()
        loss.backward()
    balancer.step()
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad.clone()
        if not fsdp:
            dist.all_reduce(g)
        grads[name] = g.flatten().tolist()
    return {"grads": grads, "metrics": balancer.metrics(), "loads": balancer.expert_loads().tolist()}


@pytest.mark.parametrize("world,fsdp", [(2, False), (3, False), (3, True)])
def test_global_batch_over_ranks(tmp_path, world, fsdp):
    """Each rank holds some rows: the loads are those of the whole batch (one all-reduce of the
    counts), so the summed gradients and the logged values are a single process's on all the rows."""
    model = _model()
    ids, mask, labels = _batch(rows=6, length=16)
    ref, ref_balance, ref_z = _reference(model, ids, mask, labels)
    got = _run(_rank_gradients, tmp_path, fsdp, world=world, timeout=300.0)
    assert all("ok" in g for g in got), got
    for result in got:
        result = result["ok"]
        assert result["metrics"]["moe/balance_loss"] == pytest.approx(ref_balance, rel=1e-9)
        assert result["metrics"]["moe/z_loss"] == pytest.approx(ref_z, rel=1e-9)
        assert result["loads"] == got[0]["ok"]["loads"]
    grads = got[0]["ok"]["grads"]
    assert grads.keys() == ref.keys()
    for name in ref:
        torch.testing.assert_close(
            torch.tensor(grads[name], dtype=torch.float64),
            ref[name].flatten(),
            rtol=1e-4 if fsdp else 1e-6,  # FSDP reduces the gradients in fp32
            atol=1e-7 if fsdp else 1e-9,
            msg=name,
        )

    # a rank balancing against its own rows only would differ: the per-rank loss is another loss
    local = _model()
    local.zero_grad()
    balancer = _balancer(local)
    for rows in (slice(0, 3), slice(3, 6)):
        with balancer.batch(mask[rows]):
            loss = _ce(local, ids[rows], labels[rows], _denom(labels))
            balancer.sync()
            loss.backward()
        balancer.step()
    router = next(n for n in ref if n.endswith("gate.weight"))
    assert not torch.allclose(_grads(local)[router], ref[router], rtol=1e-4)
