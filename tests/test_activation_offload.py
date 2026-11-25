"""palingenesis.activation_offload (memory.activation_offload): gradients identical to a run without offload under
every checkpointing mode, packing, SeCO, gradient accumulation, retain_graph and compile; what is and is not moved
(last layer, parameters, small tensors, layer inputs only); the forward in-flight bound and the backward prefetch
order (recorded with a mock copier standing in for the CUDA streams); the pinned arena; config validation; the
trainer path. CPU: copies are synchronous, the scheduling is checked through the mock."""

import contextlib

import pytest
import torch
import torch.nn as nn
from seco_archs import VOCAB, build

from palingenesis.activation_offload import (
    ActivationOffloader,
    HostArena,
    OffloadOptions,
    SyncCopies,
    _Saved,
    describe,
)
from palingenesis.config import Config, ConfigError
from palingenesis.distributed import _find_transformer_layers
from palingenesis.kernels import apply_activation_checkpointing, sac_routes_saved_tensors_hooks


def _model(arch="qwen3", mode="none", respect=None):
    model = build(arch, dtype=torch.float32)
    model.config.use_cache = False
    apply_activation_checkpointing(model, mode=mode, respect_saved_tensors_hooks=respect)
    return model


def _grads(model, ids, offloader=None, **fwd):
    model.zero_grad(set_to_none=True)
    with offloader.saving() if offloader is not None else contextlib.nullcontext():
        loss = model(input_ids=ids, labels=ids, use_cache=False, **fwd).loss
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _offloader(model, copies=None, **opts):
    opts.setdefault("min_bytes", 0)
    return ActivationOffloader(
        _find_transformer_layers(model), "cpu", OffloadOptions(**opts), params=model.parameters(), copies=copies
    )


def _ids(rows=2, seq=40, seed=0):
    return torch.randint(3, VOCAB, (rows, seq), generator=torch.Generator().manual_seed(seed))


def _assert_same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


# ── exactness ───────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arch", ["qwen3", "qwen3_5"])
@pytest.mark.parametrize("mode", ["none", "full", "selective"])
def test_gradients_are_identical(arch, mode):
    model = _model(arch, mode, respect=True if mode == "selective" else None)
    ids = _ids()
    ref = _grads(model, ids)
    off = _offloader(model)
    got = _grads(model, ids, off)
    _assert_same(ref, got)
    assert off.stats()["tensors"] > 0
    off.close()


def test_what_moves_per_checkpointing_mode():
    layers = 4  # the tiny models' depth; the last one stays
    counts = {}
    for mode in ("none", "full", "selective"):
        model = _model("qwen3", mode, respect=True if mode == "selective" else None)
        off = _offloader(model)
        _grads(model, _ids(), off)
        counts[mode] = off.stats()["tensors"]
        off.close()
    assert counts["full"] == layers - 1  # one input per checkpointed layer, the last layer kept
    if sac_routes_saved_tensors_hooks():
        assert counts["full"] < counts["selective"] < counts["none"]
    else:
        assert counts["selective"] == counts["full"]  # this torch keeps selective checkpointing's tensors itself


def test_boundaries_only_moves_the_layer_inputs():
    model = _model("qwen3", "selective", respect=True)
    ids = _ids()
    ref = _grads(model, ids)
    off = _offloader(model, boundaries_only=True)
    _assert_same(ref, _grads(model, ids, off))
    assert off.stats()["tensors"] == 3
    off.close()


def test_parameters_small_tensors_and_the_last_layer_stay():
    model = _model("qwen3")
    seen = []
    off = _offloader(model, min_bytes=2048)
    pack = off._pack

    def spy(t):
        out = pack(t)
        seen.append((t, isinstance(out, _Saved), off._group))
        return out

    off._pack = spy
    _grads(model, _ids(), off)
    params = {p.data_ptr() for p in model.parameters()}
    moved = [(t, g) for t, s, g in seen if s]
    assert moved
    assert all(t.data_ptr() not in params for t, _ in moved)
    assert all(t.numel() * t.element_size() >= 2048 for t, _ in moved)
    assert all(g is not None and g < 3 for _, g in moved)  # layers 0-2; layer 3 (the last) and outside kept
    assert any(g == 3 for t, s, g in seen if not s)
    off.close()


class _Square(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        h = self.lin(x)
        return h * h  # mul saves h twice


def test_one_copy_of_a_tensor_saved_twice():
    torch.manual_seed(0)
    layers = nn.ModuleList([_Square(), nn.Identity()])
    off = ActivationOffloader(list(layers), "cpu", OffloadOptions(min_bytes=0), params=layers.parameters())
    x = torch.randn(4, 8)
    with off.saving():
        y = layers[1](layers[0](x)).sum()
    assert off.tensors == 2  # x (saved by the linear) and h, once
    handles = [r() for refs in off._groups.values() for r in refs]
    assert sorted(h.refs for h in handles) == [1, 2]
    y.backward()
    got = layers[0].lin.weight.grad.clone()
    layers.zero_grad(set_to_none=True)
    layers[1](layers[0](x)).sum().backward()
    assert torch.equal(layers[0].lin.weight.grad, got)


def test_restored_tensors_keep_dense_layouts():
    off = ActivationOffloader([nn.Identity(), nn.Identity()], "cpu", OffloadOptions(min_bytes=0))
    off._new_pass()
    off._group = 0
    base = torch.randn(8, 6, 4)
    cases = [base.permute(2, 0, 1), base.transpose(0, 1), base[:, :3], base[1:5]]
    handles = [off._pack(t) for t in cases]
    assert all(isinstance(h, _Saved) for h in handles)
    for t, h in zip(cases, handles):
        out = off._unpack(h)
        assert torch.equal(out, t)
        dense = torch.empty_like(t, device="meta").stride() == t.stride()
        assert out.stride() == (t.stride() if dense else out.contiguous().stride())
    off.close()


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a, self.b = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return x + self.b(torch.tanh(self.a(x)))


class _Stack(nn.Module):
    def __init__(self, n=5, d=16):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.square().mean()


def _stack_grads(model, x, off=None, passes=1, retain=False):
    model.zero_grad(set_to_none=True)
    for _ in range(passes):
        with off.saving() if off is not None else contextlib.nullcontext():
            loss = model(x)
        loss.backward(retain_graph=retain)
        if retain:
            loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def test_gradient_accumulation_and_retain_graph():
    torch.manual_seed(0)
    model, x = _Stack(), torch.randn(3, 16)
    for passes, retain in ((2, False), (1, True)):
        ref = _stack_grads(model, x, passes=passes, retain=retain)
        off = ActivationOffloader(list(model.layers), "cpu", OffloadOptions(min_bytes=0), params=model.parameters())
        _assert_same(ref, _stack_grads(model, x, off, passes=passes, retain=retain))
        off.close()


def test_a_forward_that_raised_leaves_nothing_behind():
    torch.manual_seed(0)
    model, x = _Stack(), torch.randn(3, 16)
    off = ActivationOffloader(list(model.layers), "cpu", OffloadOptions(min_bytes=0), params=model.parameters())

    def boom(module, args):
        raise RuntimeError("boom")

    hook = model.layers[3].register_forward_pre_hook(boom)
    with pytest.raises(RuntimeError, match="boom"), off.saving():
        model(x)
    hook.remove()
    ref = _stack_grads(model, x)
    _assert_same(ref, _stack_grads(model, x, off))


# ── scheduling (a mock copier in place of the CUDA streams) ─────────────────────────────────────────────────────


class Recorder(SyncCopies):
    """Synchronous copies that log what the offloader asks of the streams."""

    def __init__(self, off_ref):
        self.log, self.off_ref = [], off_ref
        self.events = 0

    def to_host(self, src, dst):
        super().to_host(src, dst)
        self.events += 1
        self.log.append(("out", self.off_ref()._group, self.events))
        return self.events

    def to_device(self, pairs):
        super().to_device(pairs)
        self.events += 1
        self.log.append(("in", len(pairs), self.events))
        return self.events

    def wait(self, event):
        if event is not None:
            self.log.append(("wait", event))


def _scheduled(prefetch, in_flight, n=6):
    torch.manual_seed(0)
    model, x = _Stack(n), torch.randn(3, 16)
    holder = {}
    rec = Recorder(lambda: holder["off"])
    off = ActivationOffloader(
        list(model.layers), "cpu", OffloadOptions(min_bytes=0, prefetch=prefetch, in_flight=in_flight), copies=rec
    )
    holder["off"] = off
    ref = _stack_grads(model, x)
    _assert_same(ref, _stack_grads(model, x, off))
    return rec.log, off


def test_forward_waits_for_copies_beyond_the_in_flight_layers():
    for in_flight in (1, 2):
        log, _ = _scheduled(prefetch=1, in_flight=in_flight)
        fwd = log[: next(i for i, e in enumerate(log) if e[0] == "in")]
        last_event = {}
        for e in fwd:
            if e[0] == "out":
                last_event[e[1]] = e[2]
        # entering layer g waits for the copies of layer g - in_flight - 1: those of layer 0 are waited for when
        # layer in_flight + 1 starts, i.e. after the copies of layer in_flight have been issued
        first_wait = next(i for i, e in enumerate(fwd) if e[0] == "wait")
        assert fwd[first_wait] == ("wait", last_event[0])
        assert max(e[1] for e in fwd[:first_wait] if e[0] == "out") == in_flight


@pytest.mark.parametrize("prefetch", [0, 1, 2])
def test_backward_fetches_whole_layers_ahead(prefetch):
    n = 6
    log, _ = _scheduled(prefetch=prefetch, in_flight=1, n=n)
    bwd = log[next(i for i, e in enumerate(log) if e[0] == "in") :]
    fetches = [e for e in bwd if e[0] == "in"]
    # one fetch per offloaded layer (0..4; layer 5 stays), whatever the prefetch depth
    assert len(fetches) == n - 1
    # the first offloaded layer the backward reaches fetches itself and the `prefetch` layers below it back to back
    # (no wait in between); after that, each layer finds itself fetched and fetches the one `prefetch` below
    first = bwd.index(fetches[0])
    assert all(e[0] == "in" for e in bwd[first : first + prefetch + 1])
    assert bwd[first + prefetch + 1][0] == "wait"
    # every wait in the backward is on a fetch issued earlier
    issued = set()
    for e in bwd:
        if e[0] == "in":
            issued.add(e[2])
        elif e[0] == "wait":
            assert e[1] in issued


# ── composition ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_packed_rows_are_exact():
    model = _model("qwen3", "full")
    ids = _ids(rows=1, seq=48)
    pos = torch.cat([torch.arange(20), torch.arange(28)]).unsqueeze(0)
    ref = _grads(model, ids, position_ids=pos)
    off = _offloader(model)
    _assert_same(ref, _grads(model, ids, off, position_ids=pos))
    off.close()


def test_seco_chunks_are_exact():
    from palingenesis.seco import chunkwise_forward_backward

    model = build("qwen3", dtype=torch.float64)
    model.config.use_cache = False
    apply_activation_checkpointing(model, mode="full")
    ids = _ids(rows=2, seq=48)

    def run(off):
        model.zero_grad(set_to_none=True)
        with off.saving() if off is not None else contextlib.nullcontext():
            chunkwise_forward_backward(model, ids, ids.clone(), chunk_size=16, budget=None, loss_denom=ids.numel())
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ref = run(None)
    off = _offloader(model)
    _assert_same(ref, run(off))
    assert off.stats()["tensors"] > 0  # the last chunk's pass, layers 0-2
    off.close()


def test_compiled_layers_are_exact():
    model = _model("qwen3", "full")
    ids = _ids()
    ref = _grads(model, ids)
    for layer in _find_transformer_layers(model):
        layer.compile(backend="aot_eager")
    off = _offloader(model)
    try:
        got = _grads(model, ids, off)
    except Exception as e:  # noqa: BLE001 - a torch without dynamo support here
        pytest.skip(f"torch.compile unavailable: {type(e).__name__}: {e}")
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
    assert off.stats()["tensors"] >= 3  # each compiled layer's saved inputs (hidden states, rotary tables)
    off.close()
    torch._dynamo.reset()


# ── host memory ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_arena_reuses_released_chunks_and_bounds_its_size():
    arena = HostArena(pin=False, limit_bytes=4 << 20, chunk_bytes=1 << 20)
    a, ca = arena.take(400_000)
    b, cb = arena.take(400_000)
    assert ca is cb and a.data_ptr() % 512 == 0 and b.data_ptr() - a.data_ptr() >= 400_000
    c, cc = arena.take(400_000)  # does not fit the first chunk
    assert cc is not ca and arena.reserved_bytes == 2 << 20
    for chunk in (ca, cb, cc):
        arena.release(chunk)
    d, cd = arena.take(400_000)
    assert cd is ca and d.data_ptr() == a.data_ptr()  # rewound
    big, cbig = arena.take(1_500_000)  # larger than a chunk: its own (a power of two)
    assert cbig.size == 2 << 20 and arena.reserved_bytes == 4 << 20
    with pytest.raises(RuntimeError, match="activation_offload_boundaries_only"):
        arena.take(1_500_000)  # a third chunk would pass the limit


def test_arena_leaves_host_memory_available():
    # 3 MiB available, 1 MiB must stay: a second 1 MiB chunk would leave less than the headroom
    left = [3 << 20]
    arena = HostArena(pin=False, chunk_bytes=1 << 20, headroom_bytes=1 << 20, available=lambda: left[0])
    _, first = arena.take(900_000)
    left[0] = (2 << 20) - 1
    with pytest.raises(RuntimeError, match="must stay free"):
        arena.take(900_000)
    assert arena.reserved_bytes == 1 << 20
    arena.release(first)
    arena.take(900_000)  # the released chunk is reused: no new memory, no check
    unknown = HostArena(pin=False, chunk_bytes=1 << 20, headroom_bytes=1 << 20, available=lambda: None)
    unknown.take(900_000)  # MemAvailable cannot be read (not Linux): only the share-of-RAM limit applies


def test_arena_release_during_take_is_safe():
    """The garbage collector may free a graph, and with it handles of the arena, while take() allocates on the
    thread that holds the arena's lock: no deadlock, and the range being handed out is not rewound."""
    arena = HostArena(pin=False, chunk_bytes=1 << 20, headroom_bytes=0)
    _, chunk = arena.take(600_000)
    arena._available = lambda: arena.release(chunk)  # runs inside take(), under the lock
    view, second = arena.take(600_000)  # does not fit: a new chunk is asked for, the old one is released meanwhile
    assert second is chunk and chunk.live == 1 and chunk.offset >= 600_000
    assert view.data_ptr() == chunk.buf.data_ptr()
    other, third = arena.take(300_000)
    assert third is chunk and other.data_ptr() >= view.data_ptr() + 600_000  # no overlap with the live view


def test_a_forward_that_raised_inside_a_layer_resets_the_layer_tag():
    torch.manual_seed(0)
    model, x = _Stack(), torch.randn(3, 16)
    off = ActivationOffloader(list(model.layers), "cpu", OffloadOptions(min_bytes=0), params=model.parameters())

    def boom(module, args, output):
        raise RuntimeError("boom")

    hook = model.layers[1].register_forward_hook(boom, prepend=True)  # before the offloader's exit hook
    with pytest.raises(RuntimeError, match="boom"), off.saving():
        model(x)
    hook.remove()
    assert off._group == 1
    with off.saving():
        assert off._group is None  # what is saved before the first layer is not charged to a stale layer
    off.close()


def test_handles_release_their_chunk():
    torch.manual_seed(0)
    model, x = _Stack(), torch.randn(3, 16)
    off = ActivationOffloader(list(model.layers), "cpu", OffloadOptions(min_bytes=0), params=model.parameters())
    for _ in range(3):
        _stack_grads(model, x, off)
    assert len(off.arena._chunks) == 1 and off.arena._chunks[0].live == 0  # one chunk, reused every step
    off.close()


def test_options_and_description():
    with pytest.raises(ValueError):
        OffloadOptions(in_flight=0)
    with pytest.raises(ValueError):
        OffloadOptions(prefetch=-1)
    with pytest.raises(ValueError):
        OffloadOptions(min_bytes=-1)
    assert "input" in describe(OffloadOptions(), "full", True)
    assert "every activation" in describe(OffloadOptions(), "none", True)
    assert "selective checkpointing keeps" in describe(OffloadOptions(), "selective", True)
    assert "keeps" not in describe(OffloadOptions(boundaries_only=True), "selective", True)


# ── config and trainer ──────────────────────────────────────────────────────────────────────────────────────────


def test_config_validation():
    c = Config()
    assert c.memory.activation_offload is False  # default unchanged
    c.memory.activation_offload = True
    c.validate()
    c.memory.activation_offload_boundaries_only = True
    c.train.gradient_checkpointing = "none"
    with pytest.raises(ConfigError, match="boundaries_only"):
        c.validate()
    for field, value in (("activation_offload_in_flight", 0), ("activation_offload_prefetch", -1)):
        c = Config()
        c.memory.activation_offload = True
        setattr(c.memory, field, value)
        with pytest.raises(ConfigError, match=field):
            c.validate()
    c = Config()
    c.memory.activation_offload_prefetch = 3
    assert any("activation_offload is false" in w for w in c.validate())


@pytest.mark.parametrize("mode", ["none", "full"])
def test_trainer_with_offload_trains_like_without(tmp_path, mode):
    import distill_helpers as H

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, layers=3)
    data = H.write_rows(tmp_path / "rows.jsonl")
    out = []
    for offload in (False, True):
        c = H.cpu_config(tmp_path / f"run_{offload}", tmp_path / "student", data, steps=3)
        c.train.gradient_checkpointing = mode
        c.train.save_final = False
        c.memory.activation_offload = offload
        c.memory.activation_offload_min_bytes = 0
        res = H.run_cpu(c)
        out.append(res)
    assert out[0].metrics["train/loss"] == out[1].metrics["train/loss"]
    assert out[1].metrics["memory/offload_gb"] > 0
