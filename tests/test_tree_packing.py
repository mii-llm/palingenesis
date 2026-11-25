"""Packed micro-batches (palingenesis.tree_packing, train.packing varlen / tree): the trie and its layout, the
micro-batch planner, and exactness against unpacked forwards and the padded RL trainer step (log-probs, loss
statistics and gradients) on tiny GPT-2, Qwen3 and Qwen3.5 (Gated DeltaNet) models."""

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from palingenesis.logits import final_hidden_states, output_head
from palingenesis.rl.config import RLConfig, RLConfigError
from palingenesis.rl.losses import LowPrecisionWeight
from palingenesis.rl.parallel import Parallel
from palingenesis.rl.trajectory import Trajectory
from palingenesis.seco import PACKED_VARLEN
from palingenesis.tree_packing import PackedRow, PackingError, _Trie, plan, prepare_model

VOCAB = 97


@pytest.fixture(autouse=True)
def small_trees(request, monkeypatch):
    """The trainer shares only runs that save a tree level's cost (TREE_MIN_SAVED forwarded tokens), far more than
    these tests' sequences hold: lowered, so their trees have shared segments and copied runs both."""
    if "worth_a_tree_level" not in request.node.name:
        monkeypatch.setattr("palingenesis.tree_packing.TREE_MIN_SAVED", 8)


# ── the trie and its layout ────────────────────────────────────────────────────


def _check_layout(row: PackedRow, seqs: list[list[int]]) -> None:
    """Invariants every packed row must satisfy."""
    n = row.num_tokens
    assert row.seg_starts[0] == 0 and row.seg_starts[-1] == n
    for i, s in enumerate(seqs):
        pos = row.positions(i)
        assert row.tokens[pos].tolist() == list(s)  # every sequence is recovered from its packed positions
        assert row.position_ids[pos].tolist() == list(range(len(s)))  # positions as in the unpacked sequence
        # a token's predecessors in its sequence are on its segment's path: what it attends to
        segment = np.searchsorted(row.seg_starts, pos, side="right") - 1
        for j in range(len(s)):
            path = row.paths[segment[j]]
            assert set(pos[: j + 1].tolist()) <= set(path.tolist())
            # and nothing else before it: the path up to the token is exactly its sequence's prefix
            assert path[: np.flatnonzero(path == pos[j])[0] + 1].tolist() == pos[: j + 1].tolist()
    # levels are contiguous, depth-major; a segment's parent is on the previous level
    for d, level in enumerate(row.levels):
        assert level.lo == (row.levels[d - 1].hi if d else 0)
        assert sum(level.lengths) == level.hi - level.lo
    assert row.levels[-1].hi == n
    for s, parent in enumerate(row.seg_parents):
        assert row.paths[s][-1] == row.seg_starts[s + 1] - 1
        if parent >= 0:
            assert row.paths[s][: len(row.paths[parent])].tolist() == row.paths[parent].tolist()
    mask = row.attention_mask("cpu").numpy()
    for i in range(n):  # the reference mask is the path relation
        segment = int(np.searchsorted(row.seg_starts, i, side="right") - 1)
        path = row.paths[segment]
        allowed = set(path[: np.flatnonzero(path == i)[0] + 1].tolist())
        assert set(np.flatnonzero(mask[i]).tolist()) == allowed


CASES = {
    "one": [[5, 6, 7]],
    "one_token": [[5]],
    "identical": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    "prefixes": [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3]],  # truncated sequences end inside a segment
    "prefix_first": [[1, 2], [1, 2, 3, 4, 5]],
    "unrelated": [[1, 2, 3], [4, 5, 6], [7]],
    "deep": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7], [1, 2, 3, 9], [1, 2, 8], [1, 10], [1, 2, 3, 4, 11, 12]],
    "split_twice": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 9, 9], [1, 2, 7, 7, 7], [1, 2, 3, 4, 5, 6, 0]],
    "same_first_token": [[3, 1], [3, 2], [3, 3], [3]],
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("share", [False, True])
def test_layout_recovers_every_sequence_and_its_context(name, share):
    seqs = CASES[name]
    row = PackedRow.build(seqs, share)
    _check_layout(row, seqs)
    if not share:
        assert row.flat and row.num_tokens == sum(map(len, seqs))
        assert row.kv_index("cpu").tolist() == list(range(row.num_tokens))


@pytest.mark.parametrize("min_saved", [1, 3, 6, 12, 1000])
def test_runs_that_save_little_are_copied_into_their_branches(min_saved):
    """PackedRow.build(min_saved): the invariants hold for every threshold; fewer levels and segments, each
    remaining shared segment saving at least the threshold, and nothing shared when nothing is worth it."""
    rng = random.Random(min_saved)
    cases = [list(seqs) for seqs in CASES.values()]
    for _ in range(25):
        cases.append([[rng.randrange(1, 4) for _ in range(rng.randrange(1, 12))] for _ in range(rng.randrange(1, 9))])
    for seqs in cases:
        exact = PackedRow.build(seqs, share=True)
        row = PackedRow.build(seqs, share=True, min_saved=min_saved)
        _check_layout(row, seqs)
        assert len(row.levels) <= len(exact.levels) and row.num_segments <= exact.num_segments
        assert exact.num_tokens <= row.num_tokens <= sum(map(len, seqs))
        # a shared segment saves its length for every sequence through it but one, at least the threshold
        through = row.multiplicity()[row.seg_starts[:-1]]
        lengths = np.diff(row.seg_starts)
        for segment in set(row.seg_parents[row.seg_parents >= 0].tolist()):
            assert lengths[segment] * (through[segment] - 1) >= min_saved
        if min_saved == 1000:
            assert row.flat


def test_the_trainer_packs_only_the_runs_worth_a_tree_level():
    from palingenesis.tree_packing import TREE_MIN_SAVED, pack

    rng = random.Random(0)
    template = [rng.randrange(1, 90) for _ in range(6)]  # a chat template's opening tokens, shared by every group
    short, long = [], []
    for _ in range(4):
        prompt = template + [rng.randrange(1, 90) for _ in range(40)]
        document = template + [rng.randrange(1, 90) for _ in range(TREE_MIN_SAVED // 7 + 10)]
        for i in range(8):
            agreed = [91, 92] if i < 4 else [91, 93]  # rollouts agree on their first tokens
            tail = agreed + [rng.randrange(1, 90) for _ in range(20)]
            short.append(prompt + tail)
            long.append(document + tail)
    row = pack(short, share=True)  # 46-token prompts x 8 rollouts save too little: a flat row, as varlen
    assert row.flat and row.num_tokens == sum(map(len, short))
    assert pack(short, share=False).num_tokens == row.num_tokens
    row = pack(long, share=True)  # long prompts: one segment per prompt, the rollouts below it (two levels)
    assert len(row.levels) == 2 and row.num_segments == 4 + 32
    exact = PackedRow.build(long, share=True)
    assert len(exact.levels) > 2 and row.num_tokens - exact.num_tokens < 4 * 64
    _check_layout(row, long)


def test_trie_shares_exactly_the_common_prefixes():
    assert PackedRow.build(CASES["identical"], True).num_tokens == 4
    assert PackedRow.build(CASES["prefixes"], True).num_tokens == 5
    deep = PackedRow.build(CASES["deep"], True)
    # unique tokens: the trie's nodes
    nodes = {tuple(s[: j + 1]) for s in CASES["deep"] for j in range(len(s))}
    assert deep.num_tokens == len(nodes)
    # [1] -> [2] [10]; [2] -> [3] [8]; [3] -> [4] [9]; [4] -> [5] [11 12]; [5] -> [6] [7]: one branch per level
    assert len(deep.levels) == 6 and deep.num_segments == 11


def test_random_tries_keep_the_invariants():
    rng = random.Random(3)
    for _ in range(30):
        seqs = []
        for _ in range(rng.randint(1, 9)):
            if seqs and rng.random() < 0.7:  # continue a prefix of an earlier sequence
                base = rng.choice(seqs)
                seqs.append(base[: rng.randint(1, len(base))] + [rng.randrange(4) for _ in range(rng.randint(0, 6))])
            else:
                seqs.append([rng.randrange(4) for _ in range(rng.randint(1, 10))])
        for share in (False, True):
            row = PackedRow.build(seqs, share)
            _check_layout(row, seqs)
            if share:
                assert row.num_tokens == len({tuple(s[: j + 1]) for s in seqs for j in range(len(s))})


def test_build_refuses_empty_input():
    with pytest.raises(PackingError):
        PackedRow.build([], True)
    with pytest.raises(PackingError):
        PackedRow.build([[1, 2], []], True)


def test_conv_halo_reads_the_ancestors_last_inputs():
    seqs = [[1, 2, 3, 4, 5], [1, 2, 3, 6], [1, 7]]
    row = PackedRow.build(seqs, share=True)
    src, real = row.conv_index(4, "cpu")
    n = row.num_tokens
    x = torch.arange(n + 1, dtype=torch.float64)
    x[n] = -1.0  # the zero column
    extended = x[src]
    for i, s in enumerate(seqs):  # the window ending at each token: its sequence's previous 3 inputs (or padding)
        pos = row.positions(i)
        for j in range(len(s)):
            at = int(real[pos[j]])  # where packed token pos[j] sits in the extended sequence
            window = extended[at - 3 : at + 1].tolist()
            want = [-1.0] * max(0, 3 - j) + [float(p) for p in pos[max(0, j - 3) : j + 1]]
            assert window == want, (i, j, window, want)


def test_routing_of_a_shared_prefix_must_agree():
    seqs = [[1, 2, 3], [1, 2, 4]]
    row = PackedRow.build(seqs, share=True)
    r0 = np.array([[[0]], [[1]], [[2]]])
    same = np.array([[[0]], [[1]], [[5]]])
    routing = row.routing([r0, same], (1, 1))[0, :, 0, 0]
    assert routing[row.positions(0)].tolist() == [0, 1, 2]
    assert routing[row.positions(1)].tolist() == [0, 1, 5]
    other = np.array([[[9]], [[9]], [[5]]])  # the engine routed the second rollout's prompt differently
    with pytest.raises(PackingError, match="recorded different experts"):
        row.routing([r0, other], (1, 1))
    assert row.routing([None, other[:2]], (1, 1))[0, :, 0, 0][row.positions(1)].tolist() == [9, 9, -1]
    flat = PackedRow.build(seqs, share=False)  # nothing shared: every sequence replays its own routing
    routing = flat.routing([r0, other], (1, 1))[0, :, 0, 0]
    assert routing[flat.positions(1)].tolist() == [9, 9, 5]


def test_multiplicity_counts_the_sequences_through_a_token():
    seqs = [[1, 2, 3, 4], [1, 2, 5], [1, 2, 5], [7]]
    row = PackedRow.build(seqs, share=True)
    counts = row.multiplicity()
    assert counts.sum() == sum(map(len, seqs)) and len(counts) == row.num_tokens
    assert counts[row.positions(0)].tolist() == [3, 3, 1, 1]
    assert counts[row.positions(1)].tolist() == [3, 3, 2]
    assert PackedRow.build(seqs, share=False).multiplicity().tolist() == [1] * 11


def test_the_trie_counts_the_rows_its_attention_gathers():
    rng = random.Random(3)
    for _ in range(30):
        seqs = [[rng.randrange(1, 4) for _ in range(rng.randrange(1, 9))] for _ in range(rng.randrange(1, 9))]
        arrays = [np.asarray(q, dtype=np.int64) for q in seqs]
        for share in (False, True):
            trie = _Trie(arrays, share)
            for i, array in enumerate(arrays):
                match, gathered = trie.probe(array)
                unique, before = trie.unique, trie.gathered
                trie.insert(i)
                assert trie.unique - unique == len(array) - match  # what probe promised
                assert trie.gathered - before == gathered
            row = PackedRow.build(seqs, share)
            assert trie.gathered == row.gathered_tokens and trie.unique == row.num_tokens


# ── planning ───────────────────────────────────────────────────────────────────


def test_varlen_plan_fills_the_budget_and_isolates_long_sequences():
    lengths = [50, 40, 30, 30, 20, 10, 150]
    seqs = [[1] * n for n in lengths]
    bins = plan(seqs, list(range(len(seqs))), [1] * len(seqs), 100, share=False)
    assert sorted(i for b in bins for i in b) == list(range(len(seqs)))
    for b in bins:
        assert sum(lengths[i] for i in b) <= 100 or len(b) == 1
    assert [6] in bins  # longer than the budget: alone


def test_tree_plan_counts_unique_tokens_and_keeps_groups_together():
    rng = random.Random(0)
    seqs, groups = [], []
    for g in range(4):
        prompt = [rng.randrange(1, 90) for _ in range(60)]
        for _ in range(8):
            seqs.append(prompt + [rng.randrange(1, 90) for _ in range(10)])
            groups.append(g)
    trained = [10] * len(seqs)
    bins = plan(seqs, groups, trained, 160, share=True)
    assert sorted(i for b in bins for i in b) == list(range(len(seqs)))
    for b in bins:
        row = PackedRow.build([seqs[i] for i in b], share=True)
        assert row.num_tokens <= 160 and sum(trained[i] for i in b) <= 160
    # a group (60 + 8 x 10 = 140 unique tokens) fits one micro-batch: it is never split
    assert len(bins) == 4 and all(len({groups[i] for i in b}) == 1 for b in bins)
    # a raw-length budget would need 70 tokens per sequence: 32 x 70 / 160 = 14+ micro-batches
    assert len(plan(seqs, groups, trained, 160, share=False)) >= 14


def test_tree_plan_bounds_the_gathered_rows():
    rng = random.Random(1)
    prompt = [rng.randrange(1, 90) for _ in range(200)]
    seqs = [prompt + [90 + i] + [rng.randrange(1, 90) for _ in range(9)] for i in range(32)]
    groups, trained = [0] * 32, [10] * 32
    # 200 + 32 x 10 = 520 packed tokens fit one micro-batch, whose attention would gather 200 + 32 x 210 rows
    (whole,) = plan(seqs, groups, trained, 600, share=True)
    assert PackedRow.build([seqs[i] for i in whole], share=True).gathered_tokens == 200 + 32 * 210
    bins = plan(seqs, groups, trained, 600, share=True, gather_budget=2400)
    assert sorted(i for b in bins for i in b) == list(range(32)) and len(bins) == 4
    for b in bins:
        row = PackedRow.build([seqs[i] for i in b], share=True)
        assert row.gathered_tokens <= 2400 and row.num_tokens <= 600
    # a sequence that alone gathers more than the bound still gets a micro-batch
    assert plan(seqs[:2], [0, 0], [10, 10], 600, share=True, gather_budget=100) == [[0], [1]]
    # varlen gathers nothing: the bound does not apply
    assert plan(seqs, groups, trained, 600, share=False, gather_budget=100) == plan(
        seqs, groups, trained, 600, share=False
    )


def test_the_trainer_bounds_gathered_rows_by_default_and_by_option():
    from palingenesis.rl.config import packing_gather_tokens

    trainer = _trainer("gpt2", "tree", **{"train.micro_tokens": 64})
    assert packing_gather_tokens(trainer.config) == 8 * 64
    prompt = list(range(3, 43))
    group = []
    for i in range(16):
        t = Trajectory({}, 0, list(prompt), [])
        t.append_generated([50 + i, 60, 61, 2], [-1.0] * 4, 0)
        group.append(t)
    # 40 + 16 x 3 = 88 unique tokens: split by train.micro_tokens first (the default bound is 8 x 64 rows)
    parts = trainer._micro_batches(group)
    assert sorted(id(t) for part in parts for t in part) == sorted(id(t) for t in group)
    for part in parts:
        sequences = [t.prompt_ids + t.tokens[:-1] for t in part]
        row = PackedRow.build(sequences, share=True)
        assert row.num_tokens <= 64 and row.gathered_tokens <= 512
    trainer.config.train.packing_gather_tokens = 130
    parts = trainer._micro_batches(group)
    assert max(len(part) for part in parts) == 2  # 40 + 2 x 43 = 126 rows


def test_tree_plan_bounds_trained_tokens():
    seqs = [[1, 2, 3, 4]] * 6  # identical: 4 unique tokens, but 6 x 3 loss terms
    bins = plan(seqs, [0] * 6, [3] * 6, 7, share=True)
    assert all(3 * len(b) <= 7 for b in bins)


# ── exactness of packed forwards ───────────────────────────────────────────────


def _f64_delta_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                    use_qk_l2norm_in_kernel=False, **kwargs):  # fmt: skip
    """The gated delta rule step by step in float64 (transformers' torch kernels compute in float32): a
    reference whose packed and unpacked results agree to float64 rounding."""
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    q, k, v, b, gg = (x.transpose(1, 2).double() for x in (query, key, value, beta, g))
    batch, heads, length, dk = k.shape
    q = q * dk**-0.5
    state = q.new_zeros(batch, heads, dk, v.shape[-1]) if initial_state is None else initial_state.double()
    out = []
    for t in range(length):
        state = state * gg[:, :, t].exp()[..., None, None]
        delta = (v[:, :, t] - (state * k[:, :, t, :, None]).sum(-2)) * b[:, :, t, None]
        state = state + k[:, :, t, :, None] * delta[:, :, None, :]
        out.append((state * q[:, :, t, :, None]).sum(-2))
    out = torch.stack(out, 2).transpose(1, 2).to(value.dtype)
    return out, (state if output_final_state else None)


def _model(arch: str, f64_linear: bool = False):
    from test_seco import MODELS

    if arch == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(0)
        cfg = Qwen3Config(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=512,
        )
        model = Qwen3ForCausalLM(cfg).double().train()
    else:
        model = MODELS[arch]()
    if f64_linear:
        for m in model.modules():
            if "chunk_gated_delta_rule" in m.__dict__:
                m.chunk_gated_delta_rule = _f64_delta_rule
    return model


def _sequences(seed=0):
    rng = random.Random(seed)
    prompt = [rng.randrange(3, 90) for _ in range(20)]
    seqs = [prompt + [rng.randrange(3, 90) for _ in range(rng.randint(3, 12))] for _ in range(4)]
    base = prompt + [5, 6, 7, 8, 9, 10, 11]
    seqs += [base + [12, 13, 14], base + [12, 15], base, [40, 41, 42], list(seqs[0])]
    # a long branch: the recurrent state crosses several 64-token kernel chunks after the fork
    seqs.append(prompt + [rng.randrange(3, 90) for _ in range(150)])
    return seqs


def _packed_hidden(model, support, seqs, share):
    row = PackedRow.build(seqs, share)
    kwargs = row.forward_kwargs("cpu")
    row.begin_forward()
    token = PACKED_VARLEN.set(True)
    try:
        hidden = final_hidden_states(model, kwargs.pop("input_ids"), None, **kwargs)[0]
    finally:
        PACKED_VARLEN.reset(token)
    support.check(row)
    return row, hidden


@pytest.mark.parametrize(
    "arch,f64_linear,tol,grad_tol",
    [
        ("gpt2", False, 1e-12, 1e-12),
        # RMSNorm (and the Gated DeltaNet gate) compute in float32 inside the float64 models: the gradient of a
        # shared token, summed over its sequences before those ops, is rounded once instead of once per copy
        ("qwen3", False, 1e-12, 1e-6),
        ("qwen35_hybrid", True, 1e-11, 1e-6),
        # transformers' torch delta rule computes in float32 and cuts a branch into other 64-token chunks
        ("qwen35_hybrid", False, 2e-5, 2e-4),
    ],
)
@pytest.mark.parametrize("share", [False, True])
def test_packed_forward_and_backward_match_unpacked_sequences(arch, f64_linear, tol, grad_tol, share):
    """Hidden states at every token and the gradients of a loss over them: one packed row against each sequence
    forwarded alone (float64 models; the delta rule in float64 too, or transformers' float32 torch kernel)."""
    seqs = _sequences()
    model = _model(arch, f64_linear)
    support = prepare_model(model)
    weights = [torch.randn(len(s), model.config.get_text_config().hidden_size, dtype=torch.float64) for s in seqs]

    model.zero_grad()
    ref_hidden, ref_loss = [], 0.0
    for s, w in zip(seqs, weights):
        h = final_hidden_states(model, torch.tensor([s]), None)[0]
        ref_hidden.append(h.detach())
        ref_loss = ref_loss + (h * w).sum()
    ref_loss.backward()
    ref_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad()
    row, hidden = _packed_hidden(model, support, seqs, share)
    loss = sum((hidden[torch.from_numpy(row.positions(i))] * w).sum() for i, w in enumerate(weights))
    loss.backward()
    for i in range(len(seqs)):
        got = hidden[torch.from_numpy(row.positions(i))].detach()
        assert torch.allclose(got, ref_hidden[i], rtol=tol, atol=tol), (i, (got - ref_hidden[i]).abs().max())
    assert abs(float(loss.detach()) - float(ref_loss.detach())) <= tol * max(1.0, abs(float(ref_loss.detach())))
    params = dict(model.named_parameters())
    for name, ref in ref_grads.items():
        error = (params[name].grad - ref).abs().max() / ref.abs().max().clamp_min(1e-30)
        assert error <= grad_tol, (name, float(error))
    if share:
        assert row.num_tokens < sum(map(len, seqs))


def test_prepare_model_refuses_what_it_cannot_pack_exactly():
    from seco_archs import ARCHS, COMMON
    from transformers import AutoConfig, AutoModelForCausalLM

    kind, overrides = ARCHS["mistral_sliding"]
    cfg = AutoConfig.for_model(kind, **{**COMMON, **overrides})
    with pytest.raises(PackingError, match="sliding"):
        prepare_model(AutoModelForCausalLM.from_config(cfg))
    eager = AutoModelForCausalLM.from_config(
        AutoConfig.for_model("qwen3", **COMMON), attn_implementation="eager"
    )  # attention not on the routed sdpa path: the packed layout would be ignored
    with pytest.raises(PackingError, match="sdpa"):
        prepare_model(eager)


def test_head_dims_the_flash_kernel_cannot_take_are_refused_on_a_gpu():
    """No silent fallback to the [N, N] mask at training sizes: refused when the model is prepared for CUDA (from
    its config) and again at the attention itself (CUDA tensors; here only its shape check)."""
    from seco_archs import COMMON
    from transformers import AutoConfig, AutoModelForCausalLM

    from palingenesis.tree_packing import _config_head_dims, _flash_capable, _flash_shape_error

    assert _flash_shape_error(128, 128, 128) is None and _flash_shape_error(256, 256, 256) is None
    assert "head dim 320" in _flash_shape_error(320, 320, 320)
    assert "head dim 12" in _flash_shape_error(12, 12, 12)
    assert "differ" in _flash_shape_error(192, 192, 128)  # multi-head latent attention
    cfg = AutoConfig.for_model("qwen3", **{**COMMON, "head_dim": 12})
    assert _config_head_dims(cfg) == (12, 12, 12)
    assert _config_head_dims(AutoConfig.for_model("gpt2", n_embd=64, n_head=4)) == (16, 16, 16)

    class MLA:
        qk_head_dim, v_head_dim = 192, 128

    assert _config_head_dims(MLA()) == (192, 192, 128)
    model = AutoModelForCausalLM.from_config(cfg)
    with pytest.raises(PackingError, match="head dim 12"):
        prepare_model(model, "cuda:0")
    prepare_model(model)  # the CPU reference path takes any head dim
    q = torch.zeros(3, 2, 12)
    assert _flash_capable(q, q, q) is False  # CPU / fp32: the reference path, as before


def test_check_catches_a_layer_that_ignored_the_layout():
    model = _model("qwen3")
    support = prepare_model(model)
    row = PackedRow.build([[1, 2, 3], [1, 2, 4]], share=True)
    row.begin_forward()
    with pytest.raises(PackingError, match="without the packed layout"):
        support.check(row)


# ── the RL trainer step ────────────────────────────────────────────────────────


def _groups(seed: int, multi_turn: bool = True, candidates: bool = False):
    """Three groups of four with a common system prompt; in each, two rollouts share a first turn and a tool
    result before their second turns differ (branching turns), and observation targets in the tool results.
    `candidates`: each sampled token comes with a sampler support of 2-5 tokens holding it (top-k / top-p)."""
    from palingenesis.opd.rollout import Candidates

    rng = random.Random(seed)

    def support(ids):
        if not candidates:
            return None
        rows = []
        for token in ids:
            others = rng.sample([i for i in range(1, 90) if i != token], rng.randint(1, 4))
            rows.append(([token, *others], [-0.5 - rng.random() for _ in range(len(others) + 1)]))
        return Candidates.from_rows(rows)

    system = [rng.randrange(1, 90) for _ in range(6)]
    out = []
    for g in range(3):
        prompt = system + [rng.randrange(1, 90) for _ in range(rng.randint(5, 30))]
        first = [rng.randrange(1, 90) for _ in range(5)]
        result = [rng.randrange(1, 90) for _ in range(4)]
        group = []
        for k in range(4):
            t = Trajectory({}, g, prompt, [])
            if multi_turn and k < 2:
                t.append_generated(first, [-4.5 + 0.1 * j for j in range(len(first))], 0, candidates=support(first))
                start = len(t.tokens)
                t.append_context(result)
                t.append_observations(start, [(1, 1), (2, 1), (3, 2)])
            n = rng.randint(3, 25)
            ids = [rng.randrange(1, 90) for _ in range(n)]
            t.append_generated(ids, [-4.5 + rng.random() for _ in range(n)], 0, candidates=support(ids))
            t.reward = rng.random()
            group.append(t)
        out.append(group)
    return out


def _trainer(arch: str, packing: str, **options):
    from palingenesis.rl.trainer import RLTrainer

    trainer = object.__new__(RLTrainer)
    trainer.config = RLConfig()
    for key, value in {
        "train.packing": packing,
        "train.prompt_sharing": "off",
        "train.micro_tokens": 96,
        "loss.seq_mask": 0.0,
        "loss.observation_weight": 0.05,
        **options,
    }.items():
        trainer.config.set(key, value)
    trainer.parallel, trainer.device, trainer.pad_id = Parallel(fsdp=False), "cpu", 0
    trainer.model = _model(arch, f64_linear="qwen35" in arch)
    trainer.head, trainer.head_weight = output_head(trainer.model), LowPrecisionWeight()
    return trainer


@pytest.mark.parametrize("arch", ["gpt2", "qwen3", "qwen35_hybrid"])
def test_rl_step_is_the_same_padded_varlen_and_tree(arch):
    """One RL step on multi-turn groups (shared prompts, branching turns, observation loss): the same loss
    statistics and parameter gradients whichever way the micro-batches are laid out."""
    results = {}
    for packing in ("padded", "varlen", "tree"):
        trainer = _trainer(arch, packing)
        metrics = trainer._train_step(_groups(0))
        grads = {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}
        results[packing] = (metrics, grads, len(trainer._micro_batches([t for g in _groups(0) for t in g])))
    from palingenesis.tree_packing import pack

    rows = [pack([t.prompt_ids + t.tokens[:-1] for t in g], share=True) for g in _groups(0)]
    assert all(len(row.levels) > 1 for row in rows)  # the tree step ran real trees
    base, base_grads, padded_count = results["padded"]
    assert base["policy/seq_masked"] == 0.0 and base["observation_tokens"] > 0
    for packing in ("varlen", "tree"):
        metrics, grads, _ = results[packing]
        for key in ("loss", "policy/abs_ratio_dev", "policy/mismatch_k3", "trained_tokens", "policy/entropy",
                    "loss/observation"):  # fmt: skip
            assert abs(metrics[key] - base[key]) <= 1e-6 * max(1.0, abs(base[key])), (packing, key)
        assert grads.keys() == base_grads.keys()
        for name, ref in base_grads.items():
            assert torch.allclose(grads[name], ref, rtol=1e-6, atol=1e-6 * float(ref.abs().max())), (packing, name)
    # micro_tokens bounds unique tokens: the tree needs no more micro-batches than varlen
    assert results["tree"][2] <= results["varlen"][2]


@pytest.mark.parametrize("centering", ["off", "topk"])
def test_rl_step_with_top_k_support_is_the_same_padded_varlen_and_tree(centering):
    """Truncated sampling (log-probs renormalised over each token's sampler support) and score centering read the
    packed hidden states in the same order as the padded ones: the same statistics and gradients."""
    results = {}
    for packing in ("padded", "varlen", "tree"):
        trainer = _trainer(
            "gpt2", packing, **{"loss.score_centering": centering, "loss.score_centering_k": 3, "loss.type": "cispo"}
        )
        trainer.candidates, trainer.renormalise = 13, True
        metrics = trainer._train_step(_groups(1, candidates=True))
        grads = {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}
        results[packing] = (metrics, grads)
    base, base_grads = results["padded"]
    assert base["sampling/support_incomplete"] == 0.0 and base["sampling/support_size"] > 1
    for packing in ("varlen", "tree"):
        metrics, grads = results[packing]
        for key in ("loss", "policy/mismatch_k3", "sampling/support_size", "policy/entropy"):
            assert abs(metrics[key] - base[key]) <= 1e-6 * max(1.0, abs(base[key])), (packing, key)
        for name, ref in base_grads.items():
            assert torch.allclose(grads[name], ref, rtol=1e-6, atol=1e-6 * float(ref.abs().max())), (packing, name)


def test_rl_step_routes_long_prompt_groups_through_the_tree_not_prompt_sharing():
    trainer = _trainer("gpt2", "tree", **{"train.prompt_sharing": "auto"})
    assert trainer._share_prompts(_groups(0)) is False


def test_packing_options_are_validated():
    config = RLConfig()
    config.set("model.policy", "x")
    config.set("data.dataset", "x.jsonl")
    config.set("rewards.r.fn", "math")
    config.validate()
    config.set("train.packing", "packed")
    with pytest.raises(RLConfigError, match="train.packing must be one of"):
        config.validate()
    config.set("train.packing", "tree")
    config.validate()
    config.set("train.prompt_sharing", "on")
    with pytest.raises(RLConfigError, match="prompt_sharing on with train.packing tree"):
        config.validate()
    config.set("train.prompt_sharing", "auto")
    config.set("rollout.routing_replay", "auto")  # a shared prefix is forwarded once: one routing to replay
    with pytest.raises(RLConfigError, match="train.packing tree with rollout.routing_replay"):
        config.validate()
    config.set("train.packing", "varlen")
    config.validate()
    config.set("rollout.routing_replay", "off")
    config.set("train.packing_gather_tokens", 4096)
    with pytest.raises(RLConfigError, match="only used with train.packing tree"):
        config.validate()
    config.set("train.packing", "tree")
    with pytest.raises(RLConfigError, match="must be >= train.micro_tokens"):
        config.validate()
    config.set("train.packing_gather_tokens", 65536)
    config.validate()
    config.train.packing_gather_tokens = -1
    with pytest.raises(RLConfigError, match="packing_gather_tokens must be an integer >= 0"):
        config.validate()
    config.train.packing_gather_tokens = 0
    config.set("model.multimodal", True)
    config.set("rollout.backend", "vllm")
    with pytest.raises(RLConfigError, match="model.multimodal"):
        config.validate()


def test_trainer_turns_a_packing_error_into_a_config_error():
    trainer = _trainer("qwen3", "varlen")
    trainer.model.config._attn_implementation_internal = "eager"
    with pytest.raises(RLConfigError, match="train.packing varlen"):
        trainer._train_step(_groups(0))


def test_math_attention_reference_matches_explicit_per_sequence_attention():
    """The [N, N] ancestor-mask attention (the path without a flash kernel) over a tree equals plain causal
    attention over each sequence."""
    seqs = CASES["deep"]
    row = PackedRow.build(seqs, share=True)
    gen = torch.Generator().manual_seed(0)
    n = row.num_tokens
    q, k, v = (torch.randn(n, h, 8, generator=gen, dtype=torch.float64) for h in (4, 2, 2))
    from palingenesis.tree_packing import _math_attention

    out = _math_attention(q, k, v, row.attention_mask("cpu"), 0.3)
    for i in range(len(seqs)):
        pos = torch.from_numpy(row.positions(i))
        qs, ks, vs = (
            q[pos].transpose(0, 1),
            k[pos].repeat_interleave(2, 1).transpose(0, 1),
            v[pos].repeat_interleave(2, 1).transpose(0, 1),
        )
        want = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=0.3).transpose(0, 1)
        assert torch.allclose(out[pos], want, atol=1e-12)


@pytest.mark.parametrize("packing", ["varlen", "tree"])
@pytest.mark.parametrize("fsdp", [False, True], ids=["single", "fsdp2"])
def test_trainer_end_to_end_with_packing(tmp_path, packing, fsdp):
    """A whole run (hf rollouts, two steps) with packed micro-batches: the model is checked and routed at start-up,
    and FSDP2 with full activation checkpointing (the layers recomputed in the backward) trains through them."""
    import json

    from test_rl import tiny_policy

    from palingenesis.rl.trainer import RLTrainer

    rows = [{"prompt": f"What is {i} plus {i + 1}?", "answer": str(2 * i + 1)} for i in range(30)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = RLConfig()
    for key, value in {
        "model.policy": tiny_policy(tmp_path),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "model.gradient_checkpointing": "full" if fsdp else "none",
        "data.dataset": str(tmp_path / "data.jsonl"),
        "rollout.backend": "hf",
        "rollout.batch_prompts": 3,
        "rollout.group_size": 3,
        "rollout.max_new_tokens": 8,
        "rewards.m.fn": "math",
        "rewards.m.weight": 0.0,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "train.save_final": False,
        "train.fsdp": fsdp,
        "train.packing": packing,
        "train.micro_tokens": 64,
    }.items():
        config.set(key, value)

    def length(completion):  # dense, so a random policy's groups disagree
        return len(completion) / 40

    def local(p):  # FSDP2 parameters are DTensors; on one rank the local shard is the whole tensor
        p = p.detach()
        return p.to_local() if hasattr(p, "to_local") else p

    trainer = RLTrainer(config, rewards=[length])
    assert trainer._packing is not None and trainer._packing.attention
    before = [local(p).clone() for p in trainer.model.parameters()]
    trainer.train()
    assert any(not torch.equal(local(p), b) for p, b in zip(trainer.model.parameters(), before))
