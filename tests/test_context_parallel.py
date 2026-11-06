"""Context parallelism (palingenesis.context_parallel) is exact: CPU processes (gloo) that each hold a
slice of the sequence give the same loss and gradients as one process on the whole sequence, for
full-causal and sliding-window attention, grouped K/V heads, padding to the layout's multiple and
uneven valid-token counts. Also: the layout's bookkeeping, and the refusal of layer types that
cannot be split (recurrent / linear attention, soft-capping)."""

import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis import context_parallel as C  # noqa: E402
from palingenesis.config import ConfigError  # noqa: E402

IGNORE = -100
VOCAB = 64


# ── layout bookkeeping (one process) ───────────────────────────────────────────


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_load_balanced_indices_partition_the_sequence(size):
    total = 2 * size * 5
    parts = [C.load_balanced_indices(total, size, r) for r in range(size)]
    assert all(p.numel() == total // size for p in parts)
    assert torch.equal(torch.cat(parts).sort().values, torch.arange(total))
    # head-tail: rank r holds chunk r and chunk 2N-1-r, so every rank has the same causal work
    work = [int((p + 1).sum()) for p in parts]
    assert len(set(work)) == 1


def test_indivisible_length_is_rejected():
    with pytest.raises(ValueError, match="multiple"):
        C.load_balanced_indices(10, 4, 0)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_global_and_rank_order_are_inverse(size):
    b, h, local, d = 2, 3, 6, 4
    parts = torch.randn(size, b, h, local, d)
    full = C.to_global_order(parts)
    assert full.shape == (b, h, size * local, d)
    for r in range(size):  # each rank's block lands at its global positions
        idx = C.load_balanced_indices(size * local, size, r)
        assert torch.equal(full[:, :, idx], parts[r])
    assert torch.equal(C.to_rank_order(full, size), parts)


def test_shard_pads_then_selects():
    cp = C.ContextParallel(group=None, size=2, rank=1)
    t = torch.arange(2 * 10).reshape(2, 10)
    local = cp.shard(t, pad_value=-1)
    assert cp.padded_length(10) == 12 and local.shape == (2, 6)
    idx = C.load_balanced_indices(12, 2, 1)  # chunks 1 and 2 of 4 (3 tokens each)
    padded = torch.cat([t, torch.full((2, 2), -1)], dim=1)
    assert torch.equal(local, padded[:, idx])
    assert torch.equal(cp.position_ids(10, 2, "cpu"), idx.expand(2, -1))


def test_labels_shifted_before_the_split_score_across_chunk_boundaries():
    """The label of the last token of a chunk is the first token of the next chunk, even when another
    rank holds that chunk: shifting happens on the whole sequence."""
    from palingenesis.loss import shift_labels

    ids = torch.arange(12)[None]
    labels = shift_labels(ids.clone())
    for r in range(2):
        cp = C.ContextParallel(group=None, size=2, rank=r)
        local_ids, local_labels = cp.shard(ids), cp.shard(labels, IGNORE)
        nxt = local_ids + 1
        nxt[local_ids == 11] = IGNORE
        assert torch.equal(local_labels, nxt)


def _qwen35_tiny():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    return Qwen3_5ForCausalLM(cfg)


def test_linear_attention_layers_are_refused():
    pytest.importorskip("transformers.models.qwen3_5")
    with pytest.raises(ConfigError, match="linear_attention"):
        C.check_model_support(_qwen35_tiny())


def test_softcapping_is_refused():
    from transformers import Gemma2Config, Gemma2ForCausalLM

    model = Gemma2ForCausalLM(
        Gemma2Config(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            attn_logit_softcapping=50.0,
        )
    )
    with pytest.raises(ConfigError, match="soft-capping"):
        C.check_model_support(model)


def test_layer_without_attention_interface_is_caught_at_runtime():
    """A decoder layer that never calls the attention interface (here: attention replaced by a
    module of its own) makes the forward raise instead of silently attending within the slice."""
    model = _model("llama")
    layers = list(model.model.layers)
    C.route_attention(model, layers)

    class Identity(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    model.model.layers[1].self_attn = Identity()
    cp = C.ContextParallel(group=None, size=1, rank=0)  # a group of one: no process group needed
    ids = torch.randint(0, VOCAB, (1, 8))
    cp.activate(8)
    try:
        with pytest.raises(C.ContextParallelError, match=r"layer\(s\) \[1\]"):
            model.model(input_ids=ids, position_ids=cp.position_ids(8, 1, "cpu"))
    finally:
        cp.deactivate()


def test_outside_context_parallel_the_original_attention_runs():
    model = _model("qwen3_sliding")
    ids = torch.randint(0, VOCAB, (2, 11))
    mask = torch.ones_like(ids)
    mask[1, :3] = 0  # left padding: a mask the original implementation must still apply
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask).logits
        C.route_attention(model, list(model.model.layers))
        got = model(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(got, ref)


@pytest.fixture
def single():
    """A context-parallel group of one rank (no process group needed), active for the test."""
    cp = C.ContextParallel(group=None, size=1, rank=0)
    yield cp
    cp.deactivate()


@pytest.mark.parametrize("kind", ["llama", "qwen3_sliding", "gemma3"])
@pytest.mark.parametrize("fused", [True, False])
def test_a_group_of_one_equals_the_plain_model(kind, fused, single, monkeypatch):
    """Both attention paths (gathered causal kernel, SDPA under autograd) reproduce the model's own
    attention, sliding windows included, values and gradients."""
    monkeypatch.setattr(C, "FUSED", fused)
    ids = torch.randint(0, VOCAB, (2, 14))
    ref_model, model = _model(kind), _model(kind)
    ref = ref_model(input_ids=ids).logits
    ref.square().mean().backward()
    C.route_attention(model, list(model.model.layers))
    single.activate(14)
    got = model(input_ids=ids, position_ids=single.position_ids(14, 2, "cpu")).logits
    got.square().mean().backward()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    for (name, p), q in zip(model.named_parameters(), ref_model.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_activate_and_deactivate_scope_the_routing(single):
    assert C._ACTIVE is None
    single.activate()
    assert C._ACTIVE is single
    single.deactivate()
    assert C._ACTIVE is None


def test_mask_function_returns_none_only_under_context_parallel(single):
    from transformers.masking_utils import causal_mask_function

    mask = C._mask_function("sdpa")
    kwargs = dict(
        batch_size=1,
        q_length=4,
        kv_length=4,
        q_offset=0,
        kv_offset=0,
        mask_function=causal_mask_function,
        attention_mask=torch.tensor([[0, 1, 1, 1]]).bool(),
    )
    assert mask(**kwargs) is not None  # the original (sdpa) mask: left padding must be masked
    single.activate()
    assert mask(**kwargs) is None


def _attention_inputs():
    q = torch.randn(1, 4, 6, 8)
    kv = torch.randn(1, 2, 6, 8)
    return q, kv, kv.clone()


class _Module(torch.nn.Module):
    is_causal = True


def test_attention_refuses_a_mask_bidirectional_attention_and_soft_capping(single):
    attention = C._make_attention("sdpa")
    q, k, v = _attention_inputs()
    single.activate()
    with pytest.raises(C.ContextParallelError, match="attention mask"):
        attention(_Module(), q, k, v, torch.ones(1, 1, 6, 6, dtype=torch.bool))
    with pytest.raises(C.ContextParallelError, match="bidirectional"):
        attention(_Module(), q, k, v, None, is_causal=False)
    with pytest.raises(C.ContextParallelError, match="softcap"):
        attention(_Module(), q, k, v, None, softcap=30.0)
    with pytest.raises(C.ContextParallelError, match="sinks"):
        attention(_Module(), q, k, v, None, sinks=torch.zeros(4))


def test_attention_refuses_an_odd_local_length(single):
    q, k, v = torch.randn(1, 4, 5, 8), torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    with pytest.raises(C.ContextParallelError, match="two equal chunks"):
        single.attention(_Module(), q, k, v, None, 0.0, None)


def test_window_mask_follows_the_sliding_window_convention():
    """transformers: key k is visible from query q iff 0 <= q - k < window."""
    cp = C.ContextParallel(group=None, size=2, rank=1)
    mask = cp._window_mask(chunk=3, lo=2, hi=9, window=4, device="cpu")[0, 0]
    for i, q in enumerate(range(6, 9)):
        for j, k in enumerate(range(2, 9)):
            assert bool(mask[i, j]) == (0 <= q - k < 4), (q, k)


@pytest.mark.parametrize(
    "rank,window,expected",
    [
        (0, None, [(0, 3), (0, 12)]),  # chunks 0 and 3 of a 2-rank, 12-token layout
        (1, None, [(0, 6), (0, 9)]),
        (1, 2, [(2, 6), (5, 9)]),  # only the keys within the window of the chunk's first query
        (0, 100, [(0, 3), (0, 12)]),
    ],
)
def test_key_ranges(rank, window, expected):
    assert C.ContextParallel(group=None, size=2, rank=rank)._key_ranges(3, window) == expected


def test_gather_and_scatter_gradient_with_one_rank_are_identities():
    cp = C.ContextParallel(group=None, size=1, rank=0)
    x = torch.randn(2, 3, 4, 5)
    assert torch.equal(cp.gather(x), x)
    assert torch.equal(cp.scatter_gradient(x), x)


def test_recurrent_module_names_are_refused():
    model = _model("llama")
    model.model.layers[0].add_module("mamba", torch.nn.Linear(2, 2))
    with pytest.raises(ConfigError, match="recurrent"):
        C.check_model_support(model)


def test_supported_models_pass_the_check():
    for kind in ("llama", "qwen3_sliding", "gemma3"):
        C.check_model_support(_model(kind))


def test_registered_names_avoid_substrings_transformers_branches_on():
    for original in ("sdpa", "flash_attention_2", "eager"):
        name = C._registered_name(original)
        assert "flash" not in name and "sdpa" not in name
        assert C._registered_name(original) == name  # registered once


# ── configuration and mesh ─────────────────────────────────────────────────────


def test_config_validation_of_context_parallel():
    from palingenesis.config import Config

    c = Config()
    c.parallel.context_parallel = True
    c.validate()
    c.parallel.fsdp = False
    with pytest.raises(ConfigError, match="needs parallel.fsdp"):
        c.validate()
    c.parallel.fsdp = True
    c.parallel.cp_rotate_method = "alltoall"
    with pytest.raises(ConfigError, match="only allgather"):
        c.validate()
    c.parallel.cp_rotate_method = "allgather"
    c.parallel.cp_degree = -1
    with pytest.raises(ConfigError, match="cp_degree"):
        c.validate()


def test_cp_degree_must_divide_the_world():
    from palingenesis.config import ParallelConfig
    from palingenesis.distributed import build_mesh

    with pytest.raises(ConfigError, match="must divide"):
        build_mesh(4, ParallelConfig(context_parallel=True, cp_degree=3))
    assert build_mesh(1, ParallelConfig(context_parallel=True)) is None


# ── multi-process equivalence ──────────────────────────────────────────────────


def _model(kind: str):
    torch.manual_seed(0)
    common = dict(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="sdpa",
    )
    if kind == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        model = LlamaForCausalLM(LlamaConfig(**common))
    elif kind == "qwen3_sliding":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        model = Qwen3ForCausalLM(
            Qwen3Config(
                **common,
                head_dim=8,
                use_sliding_window=True,
                sliding_window=5,
                layer_types=["sliding_attention", "full_attention"],
            )
        )
    elif kind == "gemma3":
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        model = Gemma3ForCausalLM(
            Gemma3TextConfig(
                **common, head_dim=8, sliding_window=4, layer_types=["sliding_attention", "full_attention"]
            )
        )
    else:
        raise ValueError(kind)
    model.config.use_cache = False
    return model


def _batch(seq_len: int):
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (2, seq_len))
    labels = ids.clone()
    labels[:, :3] = IGNORE  # a prompt
    labels[1, seq_len - 6 :] = IGNORE  # row 1 is right-padded
    return ids, labels


def _loss_sum(model, ids, labels, position_ids=None):
    from palingenesis.loss import chunked_cross_entropy_loss

    kwargs = {} if position_ids is None else {"position_ids": position_ids}
    hidden = model.model(input_ids=ids, **kwargs).last_hidden_state
    return chunked_cross_entropy_loss(hidden, labels, model.lm_head, num_chunks=2, global_valid_tokens=1.0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, kind, seq_len, fused, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from palingenesis.loss import shift_labels

        C.FUSED = fused
        model = _model(kind)
        C.route_attention(model, list(model.model.layers))
        cp = C.ContextParallel(dist.group.WORLD, world, rank)
        ids, labels = _batch(seq_len)
        labels = shift_labels(labels)
        local_ids, local_labels = cp.shard(ids), cp.shard(labels, IGNORE)
        cp.activate(seq_len)
        try:
            loss = _loss_sum(model, local_ids, local_labels, cp.position_ids(seq_len, ids.shape[0], "cpu"))
            loss.backward()
        finally:
            cp.deactivate()
        total = loss.detach().clone()
        dist.all_reduce(total)
        grads = {}
        for name, p in model.named_parameters():
            dist.all_reduce(p.grad)
            grads[name] = p.grad.flatten().tolist()
        if rank == 0:
            Path(out).write_text(json.dumps({"loss": total.item(), "grads": grads}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "kind,world,seq_len,fused",
    [
        ("llama", 2, 24, True),
        ("llama", 2, 24, False),  # every layer through SDPA under autograd
        ("llama", 4, 21, True),  # padded to 24 (a multiple of 2 x 4)
        ("llama", 4, 21, False),
        ("qwen3_sliding", 2, 22, True),
        ("qwen3_sliding", 3, 30, True),
        ("gemma3", 2, 20, True),
    ],
)
def test_context_parallel_matches_one_process(kind, world, seq_len, fused):
    from palingenesis.loss import shift_labels

    model = _model(kind)
    ids, labels = _batch(seq_len)
    ref = _loss_sum(model, ids, shift_labels(labels))
    ref.backward()
    ref_grads = {n: p.grad.flatten() for n, p in model.named_parameters()}

    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/result.json"
        mp.spawn(_worker, args=(world, kind, seq_len, fused, _free_port(), out), nprocs=world, join=True)
        got = json.loads(Path(out).read_text())

    assert got["loss"] == pytest.approx(ref.item(), rel=1e-5)
    assert got["grads"].keys() == ref_grads.keys()
    for name, g in ref_grads.items():
        torch.testing.assert_close(torch.tensor(got["grads"][name]), g, rtol=1e-4, atol=1e-6, msg=name)
