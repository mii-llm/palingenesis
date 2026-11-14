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


# ── routing replay ──────────────────────────────────────────────────────────────


def _routing(model, ids):
    """[B, S, layers, top_k]: the experts `model` routes each token of `ids` to."""
    with RoutingRecorder(model) as recorder, torch.no_grad():
        model(input_ids=ids)
    return recorder.indices(rows=ids.shape[0]).long()


@pytest.mark.parametrize("family", FAMILIES)
def test_replaying_the_models_own_routing_changes_nothing(family):
    """The gate values recomputed from the router logits for the recorded experts are the model's
    own when the experts are its own: same loss, same gradients, no mismatch."""
    model = _model(family)
    ids, mask, labels = _batch()
    model.zero_grad()
    ref_loss = _ce(model, ids, labels, _denom(labels))
    ref_loss.backward()
    ref = _grads(model)

    recorded = _routing(model, ids)
    model.zero_grad()
    balancer = RouterBalancer(find_moe_layers(model))
    with balancer.batch(mask, replay=recorded):
        loss = _ce(model, ids, labels, _denom(labels))
        balancer.sync()
        loss.backward()
    balancer.step()
    assert all(layer.gate_normalised is not None for layer in balancer.layers)
    assert loss.item() == pytest.approx(ref_loss.item(), rel=1e-6)
    _assert_same(_grads(model), ref, rtol=1e-4, atol=1e-7)  # gate values pass through fp32 in the models
    assert balancer.metrics()["moe/route_mismatch"] == 0.0


@pytest.mark.parametrize("family", ["qwen3_moe", "gpt_oss", "deepseek_v3"])
def test_replay_routes_through_the_recorded_experts(family):
    """Routing recorded with other weights (a stale rollout): every token runs through the recorded
    experts, with gate values from the trained model's own router logits (so the router trains),
    and the share of choices the trained model would not have made is reported."""
    from palingenesis.moe import gate_values, route_mismatch

    model, sampler = _model(family), _model(family, seed=3)
    ids, mask, labels = _batch()
    recorded = _routing(sampler, ids)
    real = mask.reshape(-1).bool()

    seen = {}
    # hooks run in registration order: `chosen` holds the model's own choices, `seen` what the experts got
    with _captured(model) as (_, logits, chosen):
        balancer = RouterBalancer(find_moe_layers(model))
        handles = [
            layer.experts.register_forward_pre_hook(lambda m, a, i=layer.index: seen.__setitem__(i, (a[1], a[2])))
            for layer in balancer.layers
        ]
        with balancer.batch(mask, replay=recorded):
            loss = _ce(model, ids, labels, _denom(labels))
            balancer.sync()
            loss.backward()
        for h in handles:
            h.remove()
    balancer.step()
    mismatches = []
    for layer in balancer.layers:  # they know how the family turns scores into gate values
        index, weights = seen[layer.index]
        want = recorded[:, :, layer.index].reshape(-1, recorded.shape[-1])
        own = chosen[layer.index].reshape(want.shape)
        assert torch.equal(index.reshape(want.shape)[real], want[real])
        assert torch.equal(index.reshape(want.shape)[~real], own[~real])  # padding keeps the model's own routing
        flat = logits[layer.index].reshape(-1, layer.num_experts)
        torch.testing.assert_close(
            weights.reshape(want.shape)[real].double(),
            gate_values(layer, flat, want)[real].double(),
            rtol=1e-5,
            atol=1e-8,
        )
        mismatches.append(route_mismatch(want[real], own[real]))
    expected = sum(mismatches) / len(mismatches)
    assert 0.2 < expected < 1.0
    if family != "deepseek_v3":  # softmax routers: R3's gate, the softmax restricted to the recorded experts
        layer = balancer.layers[0]
        flat = logits[0].reshape(-1, layer.num_experts)
        want = recorded[:, :, 0].reshape(-1, recorded.shape[-1])
        restricted = torch.softmax(flat.gather(1, want), -1)
        torch.testing.assert_close(
            seen[0][1].reshape(want.shape)[real].double(), restricted[real], rtol=1e-5, atol=1e-8
        )
    assert balancer.metrics()["moe/route_mismatch"] == pytest.approx(expected, rel=1e-9)
    router = next(p for n, p in model.named_parameters() if n.endswith(("gate.weight", "router.weight")))
    assert router.grad is not None and router.grad.abs().sum() > 0


def test_positions_without_a_record_keep_the_models_routing():
    model = _model()
    ids, mask, labels = _batch()
    own = _routing(model, ids)
    recorded = _routing(_model(seed=3), ids)
    recorded[:, 10:] = -1  # the engine never forwarded these positions
    seen = {}
    balancer = RouterBalancer(find_moe_layers(model))
    handle = balancer.layers[0].experts.register_forward_pre_hook(lambda m, a: seen.__setitem__(0, a[1]))
    with balancer.batch(mask, replay=recorded), torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    index = seen[0].reshape(ids.shape[0], ids.shape[1], -1)
    assert torch.equal(index[:, 10:], own[:, 10:, 0])
    assert torch.equal(index[0, :10], recorded[0, :10, 0])


def test_hf_rollouts_record_the_routing_of_every_forwarded_position():
    """The hf backend's record lines up with the trainer's rows (prompt + completion but its last
    token), left padding removed: a teacher-forced forward routes the same way."""
    from palingenesis.opd.rollout import HFRollout

    model = tiny_moe("qwen3_moe").eval()
    engine = HFRollout(model, stop_ids=(2,), pad_id=0, micro_seqs=4)
    engine.record_routing = True
    torch.manual_seed(0)
    prompts = [[5, 6, 7, 8, 9, 10], [11, 12, 13], [14, 15, 16, 17]]
    rollouts = engine.generate(prompts, [6, 6, 6], 1.0)
    total = differ = 0
    for prompt, rollout in zip(prompts, rollouts):
        row = prompt + rollout.completion_ids[:-1]
        assert rollout.routed_experts.shape == (len(row), 4, 2)
        own = _routing(model, torch.tensor([row]))[0]
        differ += int((torch.from_numpy(rollout.routed_experts).long().sort(-1).values != own.sort(-1).values).sum())
        total += own.numel()
    assert differ / total < 0.02  # cached decoding against one full forward: numerics on near-ties only
    engine.record_routing = False
    assert engine.generate(prompts[:1], [3], 1.0)[0].routed_experts is None


def test_trajectories_keep_the_routing_their_tokens_were_sampled_under():
    import numpy as np

    from palingenesis.opd.rollout import decode_routed_experts
    from palingenesis.rl.trainer import _routing_batch
    from palingenesis.rl.trajectory import Trajectory

    def routed(n, value):
        return np.full((n, 3, 2), value, dtype=np.uint8)

    t = Trajectory(row={}, group=0, prompt_ids=[1, 2, 3], messages=[])
    t.append_generated([4, 5], [-0.1, -0.2], version=0)
    t.extend_routing(routed(4, 1))  # prompt + the turn's tokens but the last
    assert t.routing.shape == (4, 3, 2)
    t.append_context([6, 7])
    t.append_generated([8, 9, 10], [-0.1] * 3, version=1)
    t.extend_routing(routed(9, 2))  # the whole context again (re-prefilled) + the new turn
    assert t.routing.shape == (9, 3, 2)
    assert (t.routing[:4] == 1).all() and (t.routing[4:] == 2).all()  # the first turn keeps its record
    t.extend_routing(None)
    assert t.routing.shape == (9, 3, 2)

    other = Trajectory(row={}, group=0, prompt_ids=[1, 2], messages=[])
    other.append_generated([4, 5], [-0.1, -0.2], version=0)  # an engine that returned no routing
    batch = _routing_batch([t, other], width=10)
    assert batch.shape == (2, 10, 3, 2) and batch.dtype == torch.long
    assert (batch[0, :9] >= 0).all() and (batch[0, 9:] == -1).all() and (batch[1] == -1).all()
    assert _routing_batch([other], width=4) is None

    import base64
    import io

    buffer = io.BytesIO()
    np.save(buffer, routed(5, 7))
    assert (decode_routed_experts(base64.b64encode(buffer.getvalue()).decode()) == 7).all()
    assert decode_routed_experts(None) is None


# ── expert compute: dispatch and grouped GEMM against transformers' eager experts ──


def _experts_case(family, dtype, tokens=37, seed=0):
    """An experts module of a tiny model, tokens, their routing (some experts get no token) and gates."""
    model = _model(family).to(dtype)
    layer = find_moe_layers(model)[0]
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, 64, generator=g, dtype=dtype, requires_grad=True)
    scores = torch.randn(tokens, 8, generator=g)
    scores[:, 5] = -1e9  # an expert without tokens
    scores[:, 2] += 2.0  # a busy one
    index = scores.topk(2, dim=-1).indices
    gates = torch.rand(tokens, 2, generator=g, dtype=dtype, requires_grad=True)
    return layer.experts, hidden, index, gates


def _experts_grads(module, forward, hidden, index, gates):
    for t in (hidden, gates, *module.parameters()):
        t.grad = None
    out = forward(module, hidden, index, gates)
    (out * torch.linspace(-1, 1, out.numel(), dtype=out.dtype).view_as(out)).sum().backward()
    return out.detach(), [hidden.grad.clone(), gates.grad.clone()] + [p.grad.clone() for p in module.parameters()]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("padded", [False, True], ids=["grouped", "padded"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
def test_experts_forward_matches_the_eager_experts(family, padded, dtype):
    """Same output and same gradients (tokens, gate values, every expert weight and bias) as
    transformers' per-expert loop, for the gated / transposed / biased layouts of the families;
    fp32 runs torch's grouped GEMM, fp64 the per-group fallback."""
    from palingenesis.moe_experts import experts_forward

    module, hidden, index, gates = _experts_case(family, dtype)
    eager = type(module).forward  # the model was set to "eager"
    assert module.config._experts_implementation == "eager"
    ref_out, ref_grads = _experts_grads(module, eager, hidden, index, gates)
    out, grads = _experts_grads(
        module, lambda m, h, i, w: experts_forward(m, h, i, w, padded=padded), hidden, index, gates
    )
    tol = dict(rtol=1e-9, atol=1e-12) if dtype == torch.float64 else dict(rtol=2e-4, atol=1e-6)
    torch.testing.assert_close(out, ref_out, **tol)
    for got, ref in zip(grads, ref_grads):
        torch.testing.assert_close(got, ref, **tol)


def test_dispatch_orders_pairs_by_expert_and_keeps_a_tokens_order():
    from palingenesis.moe_experts import dispatch

    index = torch.tensor([[3, 0], [0, 3], [1, 3], [3, 2]])
    order, experts, counts = dispatch(index, 5)
    assert experts.tolist() == [0, 0, 1, 2, 3, 3, 3, 3] and counts.tolist() == [2, 1, 1, 4, 0]
    assert order.tolist() == [1, 2, 4, 7, 0, 3, 5, 6]  # stable: pairs of an expert in token order
    assert dispatch(torch.randint(0, 300, (50, 4)), 300)[2].sum() == 200  # more than 256 experts


@pytest.mark.parametrize("name", ["grouped", "padded"])
def test_experts_implementation_is_set_on_the_whole_model(name):
    from palingenesis.config import ConfigError
    from palingenesis.moe_experts import set_experts_implementation

    model = _model().float()
    ids, mask, labels = _batch()
    ref_loss = _ce(model, ids, labels, _denom(labels))
    ref_loss.backward()
    ref = _grads(model)
    model.zero_grad()
    set_experts_implementation(model, name)
    assert model.config._experts_implementation.startswith("palingenesis")
    loss = _ce(model, ids, labels, _denom(labels))
    loss.backward()
    assert loss.item() == pytest.approx(ref_loss.item(), rel=1e-5)
    _assert_same(_grads(model), ref, rtol=1e-3, atol=1e-6)
    set_experts_implementation(model, "auto")  # keeps what is set
    assert model.config._experts_implementation.startswith("palingenesis")
    with pytest.raises(ConfigError, match="experts_implementation"):
        set_experts_implementation(model, "no_such_kernel")
    set_experts_implementation(H.tiny_llama(64), "grouped")  # a dense model: nothing to set


# ── expert parallelism ──────────────────────────────────────────────────────────


def _ep_gradients(rank, world, family, padded):
    """FSDP + expert parallel on gloo ranks: this rank's rows; the full gradients and the loss."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.config import ParallelConfig
    from palingenesis.distributed import apply_fsdp
    from palingenesis.moe_experts import ExpertParallel

    torch.set_num_threads(1)
    model = _model(family)
    ids, mask, labels = _batch(rows=4, length=16)
    denom = _denom(labels)
    balancer = _balancer(model)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
    ep = ExpertParallel.apply(model, mesh, padded=padded)
    experts = find_moe_layers(model)[0].experts
    local_shape = tuple(experts.down_proj.to_local().shape)
    model = apply_fsdp(model, mesh, ParallelConfig(), bf16=False, ignored_params=ep.parameters)
    rows = slice(rank * 4 // world, (rank + 1) * 4 // world)
    with balancer.batch(mask[rows]):
        loss = _ce(model, ids[rows], labels[rows], denom)
        balancer.sync()
        loss.backward()
    balancer.step()
    total = loss.detach().clone()
    dist.all_reduce(total)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)  # sharded and expert-parallel gradients together
    grads = {n: p.grad.full_tensor().flatten().tolist() for n, p in model.named_parameters() if p.grad is not None}
    return {
        "loss": total.item(),
        "grads": grads,
        "local_shape": local_shape,
        "norm": float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm),
    }


@pytest.mark.parametrize(
    "world,family,padded",
    [(2, "qwen3_moe", False), (4, "qwen3_moe", True), (2, "gpt_oss", False), (2, "deepseek_v3", False)],
)
def test_expert_parallel_gradients_match_one_process(tmp_path, world, family, padded):
    """Every rank keeps E / ranks experts and the tokens are exchanged: the loss and every gradient
    (experts, routers with the balancing losses, the rest under FSDP) are a single process's."""
    model = _model(family)
    ids, mask, labels = _batch(rows=4, length=16)
    ref, _, _ = _reference(model, ids, mask, labels)
    ref_loss = _ce(model, ids, labels, _denom(labels)).item()
    ref_norm = torch.sqrt(sum((g**2).sum() for g in ref.values())).item()

    got = _run(_ep_gradients, tmp_path, family, padded, world=world, timeout=300.0)
    assert all("ok" in g for g in got), got
    for rank, result in enumerate(got):
        result = result["ok"]
        assert result["local_shape"][0] == 8 // world  # this rank's experts only
        assert result["loss"] == pytest.approx(ref_loss, rel=1e-6)
        assert result["norm"] == pytest.approx(ref_norm, rel=1e-4)
        assert result["grads"].keys() == ref.keys()
        for name in ref:
            torch.testing.assert_close(
                torch.tensor(result["grads"][name], dtype=torch.float64),
                ref[name].flatten(),
                rtol=1e-3 if family == "deepseek_v3" else 1e-4,  # FSDP reduces in fp32
                atol=1e-7,
                msg=f"rank {rank}: {name}",
            )


def test_expert_parallel_is_validated():
    from palingenesis.config import Config, ConfigError

    config = Config()
    config.parallel.expert_parallel = True
    config.memory.seco = True
    with pytest.raises(ConfigError, match="expert_parallel does not support memory.seco"):
        config.validate()
    config.memory.seco = False
    config.parallel.fsdp = False
    with pytest.raises(ConfigError, match="expert_parallel does not support parallel.fsdp: false"):
        config.validate()


# ── the trainer: runs, exports, expert parallel ─────────────────────────────────

ROWS = [
    {
        "messages": [
            {"role": "user", "content": f"What is {i} plus {i + 1}?"},
            {"role": "assistant", "content": f"The answer is {2 * i + 1}, since {i} + {i + 1} = {2 * i + 1}."},
        ]
    }
    for i in range(96)
]


def _saved_student(tmp_path, family="qwen3_moe", **kw):
    """A tiny MoE student on disk with the test tokenizer, and the training rows."""
    tok = H.tokenizer()
    model = tiny_moe(
        family,
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        **kw,
    )
    model.save_pretrained(tmp_path / "student")
    tok.save_pretrained(tmp_path / "student")
    return tmp_path / "student", H.write_rows(tmp_path / "rows.jsonl", ROWS)


def _file_keys(directory) -> dict[str, tuple]:
    from safetensors import safe_open

    keys = {}
    for file in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(file), "pt") as f:
            keys.update({k: tuple(f.get_slice(k).get_shape()) for k in f.keys()})
    return keys


@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3", "gpt_oss"])
def test_trained_model_exports_in_the_checkpoints_layout(tmp_path, family):
    """A run of the trainer on a MoE student (balancing on by default, activation checkpointing,
    gradient accumulation): the log carries the MoE metrics, and the exported model has the tensors
    of the checkpoint it started from, name by name (what vLLM and from_pretrained read)."""
    from transformers import AutoModelForCausalLM

    student, data = _saved_student(tmp_path, family)
    config = H.cpu_config(tmp_path, student, data, steps=3)
    if family == "gpt_oss":  # transformers has no sdpa attention for it
        config.model.attn_implementation = "eager"
    config.train.gradient_checkpointing = "full"
    config.moe.z_loss_coef = 1e-3
    config.moe.log_layers = True
    result = H.run_cpu(config)
    for key in ("moe/load_max", "moe/load_cv", "moe/dead_experts", "moe/router_entropy", "moe/z_loss"):
        assert key in result.metrics, key
    assert "moe/layer_0/load_max" in result.metrics and 1.0 <= result.metrics["moe/load_max"] <= 4.0

    final = Path(config.train.output_dir) / "final"
    assert _file_keys(final) == _file_keys(student)
    before = AutoModelForCausalLM.from_pretrained(student)
    after = AutoModelForCausalLM.from_pretrained(final)
    assert type(after) is type(before)
    router = next(n for n, _ in before.named_parameters() if n.endswith(("gate.weight", "router.weight")))
    assert not torch.equal(after.state_dict()[router], before.state_dict()[router])
    if family == "deepseek_v3":  # the selection bias moved against the loads, and is in the export
        bias = next(n for n in after.state_dict() if n.endswith("e_score_correction_bias"))
        assert before.state_dict()[bias].abs().sum() == 0 and after.state_dict()[bias].abs().sum() > 0
        assert after.state_dict()[bias].abs().max() <= 3 * 1e-3 + 1e-6  # three steps of the default rate
    ids = torch.randint(0, 100, (1, 8))
    assert torch.isfinite(after(input_ids=ids).logits).all()


def test_opting_out_leaves_the_routers_to_the_task_loss(tmp_path):
    student, data = _saved_student(tmp_path)
    config = H.cpu_config(tmp_path, student, data, steps=2)
    config.moe.balance = False
    result = H.run_cpu(config)
    assert not any(key.startswith("moe/") for key in result.metrics)


def _train_ranks(rank, world, config):
    from test_multi_rank_agreement import _train

    return _train(rank, world, config)


def test_expert_parallel_run_matches_fsdp(tmp_path):
    """The same run on 2 ranks with and without expert parallelism: same loss, same exported weights
    (experts in the checkpoint's stacked layout), up to the order of the floating-point sums."""
    from transformers import AutoModelForCausalLM

    student, data = _saved_student(tmp_path)
    runs = {}
    for name, expert_parallel in (("fsdp", False), ("ep", True)):
        config = H.cpu_config(tmp_path / name, student, data, steps=4)
        config.parallel.fsdp = True
        config.parallel.expert_parallel = expert_parallel
        got = _run(_train_ranks, tmp_path, config, world=2, timeout=600.0)
        assert all("ok" in g for g in got), got
        runs[name] = got[0]["ok"]
    assert runs["ep"]["steps"] == runs["fsdp"]["steps"] == 4
    assert runs["ep"]["loss"] == pytest.approx(runs["fsdp"]["loss"], rel=1e-3)
    start = AutoModelForCausalLM.from_pretrained(student).state_dict()
    assert runs["ep"]["weights"].keys() == start.keys()
    for name, initial in start.items():
        ep = torch.tensor(runs["ep"]["weights"][name])
        fsdp = torch.tensor(runs["fsdp"]["weights"][name])
        assert ep.numel() == initial.numel(), name  # all the experts, gathered
        moved = (fsdp - initial.flatten()).abs().mean()
        if moved > 0:
            assert (ep - fsdp).abs().mean() < 0.02 * moved, name


def test_balancing_keeps_the_expert_load_from_collapsing(tmp_path):
    """60 steps on a student whose routers start out peaked. On the task loss alone the routing
    collapses (nearly every token on the same experts, a quarter of the experts without a token);
    with the balancing loss every expert keeps receiving tokens and the load stays spread."""
    runs = {}
    for name, coef in (("task_loss_only", 0.0), ("balanced", 0.05)):
        student, data = _saved_student(tmp_path / name, router_std=0.5)
        config = H.cpu_config(tmp_path / name, student, data, steps=60)
        config.train.learning_rate = config.train.min_learning_rate = 3e-3
        config.train.per_device_batch_size = 8
        config.train.gradient_accumulation_steps = 1
        config.train.logging_steps = 10
        config.train.save_final = False
        config.moe.balance_loss_coef = coef
        runs[name] = H.run_cpu(config).metrics
    alone, balanced = runs["task_loss_only"], runs["balanced"]
    assert alone["moe/load_max"] > 3.5 and alone["moe/dead_experts"] > 0.1  # 4.0: all tokens on the same two experts
    assert balanced["moe/load_max"] < 2.5 and balanced["moe/dead_experts"] < 0.1
    assert balanced["moe/load_cv"] < 0.5 * alone["moe/load_cv"]
    assert balanced["moe/balance_loss"] < alone["moe/balance_loss"]
    assert balanced["train/loss"] == pytest.approx(alone["train/loss"], rel=0.05)  # at no cost to the task loss here


def test_compiled_layers_keep_the_router_gradient():
    """model.compile compiles every layer in place: the balancer's hooks and this library's experts
    forward stay Python (graph breaks), and the gradients are those of the uncompiled model."""
    from palingenesis.moe_experts import set_experts_implementation
    from palingenesis.train import _compile_layers

    model = _model().float()
    ids, mask, labels = _batch()
    ref, ref_balance, _ = _reference(model, ids, mask, labels)
    model.zero_grad()
    set_experts_implementation(model, "grouped")
    balancer = _balancer(model)
    _compile_layers(model)
    try:
        for _ in range(2):  # the second call runs the compiled code
            model.zero_grad()
            with balancer.batch(mask):
                loss = _ce(model, ids, labels, _denom(labels))
                balancer.sync()
                loss.backward()
            balancer.step()
    finally:
        torch._dynamo.reset()
    _assert_same(_grads(model), ref, rtol=1e-3, atol=1e-6)
    assert balancer.metrics()["moe/balance_loss"] == pytest.approx(ref_balance, rel=1e-5)


# ── the RL trainer ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "replay,fsdp", [("auto", False), ("off", False), ("auto", True)], ids=["replay", "off", "fsdp2"]
)
def test_rl_trainer_balances_and_replays_a_moe_policy(tmp_path, replay, fsdp):
    """An RL run on a MoE policy (hf rollouts): the trajectories carry the engine's routing, the
    trainer replays it and logs the mismatch and the expert loads; with replay off it only balances."""
    import json

    from test_rl import tiny_policy, tokenizer

    from palingenesis.rl.config import RLConfig, RLConfigError
    from palingenesis.rl.trainer import RLTrainer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    policy = tiny_moe(
        "qwen3_moe",
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    policy.save_pretrained(tmp_path / "policy")
    tok.save_pretrained(tmp_path / "policy")
    rows = [{"prompt": f"What is {i} plus {i + 1}?", "answer": str(2 * i + 1)} for i in range(30)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    config = RLConfig()
    for key, value in {
        "model.policy": str(tmp_path / "policy"),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "data.dataset": str(tmp_path / "data.jsonl"),
        "data.eval_size": 3,
        "rollout.backend": "hf",
        "rollout.batch_prompts": 3,
        "rollout.group_size": 3,
        "rollout.max_new_tokens": 8,
        "rollout.routing_replay": replay,
        "rewards.m.fn": "math",
        "rewards.m.weight": 0.0,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "train.save_steps": 0,
        "train.fsdp": fsdp,
        "model.gradient_checkpointing": "full" if fsdp else "none",
        "moe.z_loss_coef": 1e-3,
    }.items():
        config.set(key, value)
    trainer = RLTrainer(config, rewards=[lambda completion: len(completion) / 40])
    assert trainer.routing_replay == (replay == "auto") and trainer.engine.record_routing == (replay == "auto")
    assert trainer.router_balancer.balance_coef == pytest.approx(1e-3)  # the model config's router_aux_loss_coef
    logged, seen = [], []
    trainer._log = lambda kind, metrics, step: logged.append((kind, dict(metrics)))
    step = trainer._train_step
    trainer._train_step = lambda groups: (seen.extend(t for g in groups for t in g), step(groups))[1]

    def local(p):  # FSDP2 parameters are DTensors; on one rank the local shard is the whole tensor
        return p.detach().to_local() if hasattr(p, "to_local") else p.detach()

    router = find_moe_layers(trainer.model)[0].router.weight
    before = local(router).clone()
    trainer.train()

    steps = [m for kind, m in logged if kind == "step"]
    assert len(steps) == 2
    for metrics in steps:
        assert 1.0 <= metrics["moe/load_max"] <= 4.0 and 0.0 < metrics["moe/router_entropy"] <= 1.0
        assert metrics["moe/balance_loss"] >= 2.0 - 1e-6 and metrics["moe/z_loss"] > 0
        assert ("moe/route_mismatch" in metrics) == (replay == "auto")
    assert not torch.equal(local(router), before)
    if replay == "auto":
        for t in seen:  # every forwarded position has its record
            assert t.routing.shape == (len(t.prompt_ids) + len(t.tokens) - 1, 2, 2)
        # step 0 trains the weights that sampled: only the numerics of cached decoding differ
        assert steps[0]["moe/route_mismatch"] < 0.02
    else:
        assert all(t.routing is None for t in seen)

    config.set("model.policy", tiny_policy(tmp_path / "dense"))
    config.set("rollout.routing_replay", "on")
    with pytest.raises(RLConfigError, match="no MoE layers"):
        RLTrainer(config, rewards=[lambda completion: 0.0])


# ── on-policy distillation ──────────────────────────────────────────────────────


def test_opd_trainer_balances_a_moe_student(tmp_path):
    """On-policy distillation of a MoE student: the balancing losses are applied over the step's
    micro-batches and the MoE metrics are logged. No routing replay: the loss has no importance ratio."""
    from test_opd_trainer import tiny_model, write_prompts
    from test_rl import tokenizer

    from palingenesis.opd.config import OPDConfig
    from palingenesis.opd.trainer import OPDTrainer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    student = tiny_moe(
        "qwen3_moe",
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    student.save_pretrained(tmp_path / "student")
    tok.save_pretrained(tmp_path / "student")
    config = OPDConfig()
    prompts = write_prompts(tmp_path / "prompts.jsonl")
    for key, value in {
        "model.student": str(tmp_path / "student"),
        "model.chat_template_kwargs": {"enable_thinking": False},
        "model.use_liger_kernel": False,
        "teachers.same.model": tiny_model(tmp_path, "Qwen/Qwen3-0.6B", "teacher", 1),
        "sources.math.path": prompts,
        "sources.math.teacher": "same",
        "sources.math.max_new_tokens": 8,
        "sources.math.dev_size": 4,
        "rollout.batch_prompts": 6,
        "rollout.group_size": 2,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "train.score_micro_seqs": 4,
        "moe.balance_loss_coef": 0.01,
        "moe.z_loss_coef": 1e-3,
    }.items():
        config.set(key, value, "test")
    trainer = OPDTrainer(config)
    assert trainer.router_balancer.balance_coef == 0.01
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, metrics))
    scores, micro_batches = trainer._scores, []
    trainer._scores = lambda batch, train: (
        scores(batch, train),
        micro_batches.append(trainer._moe_micro_batches) if train else None,
    )[0]
    router = find_moe_layers(trainer.student)[0].router.weight
    before = router.detach().clone()
    trainer.train()
    steps = [m for kind, m in logged if kind == "step"]
    assert len(steps) == 2
    for metrics in steps:
        assert 1.0 <= metrics["moe/load_max"] <= 4.0 and metrics["moe/balance_loss"] >= 2.0 - 1e-6
        assert "moe/route_mismatch" not in metrics
    assert micro_batches == [3, 3]  # 12 samples in micro-batches of 4: the losses are averaged over them
    assert not torch.equal(router.detach(), before)
