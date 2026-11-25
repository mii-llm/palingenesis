"""dpo.packing varlen / tree (palingenesis.dpo.PackedPairs over palingenesis.tree_packing): a preference
micro-batch as one packed row, every pair a two-leaf prefix trie.

The central property: per-token and per-sequence log-probs, the loss and the parameter gradients equal the
padded [2P, S] path's for every objective (float64 tiny Qwen3 and Qwen3.5 / Gated DeltaNet models, random
init), on pairs with the edge cases a dataset holds: identical sides, no shared prefix, one side a prefix of
the other, a truncated rejected side, scored assistant turns inside the shared prompt, answers that start alike.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import distill_helpers as H  # noqa: E402
from test_tree_packing import _model  # noqa: E402

from palingenesis.config import Config, ConfigError  # noqa: E402
from palingenesis.data import IGNORE_INDEX  # noqa: E402
from palingenesis.dpo import (  # noqa: E402
    LOSS_TYPES,
    PackedPairs,
    PairBatches,
    collate_preferences,
    packed_reference_logps,
    preference_step,
    reference_logps,
    score_weights,
    token_logps,
)
from palingenesis.loss import shift_labels  # noqa: E402
from palingenesis.train import _get_hidden_states, _get_lm_head  # noqa: E402
from palingenesis.tree_packing import (  # noqa: E402
    TREE_MIN_SAVED,
    PackedRow,
    PackingError,
    forward_packed,
    group_cost,
    pack_groups,
    prepare_model,
)

VOCAB = 97


@pytest.fixture(autouse=True)
def small_trees(request, monkeypatch):
    """A row of pairs becomes a tree when it saves a tree level's cost (TREE_MIN_SAVED forwarded tokens), far more
    than these tests' pairs hold: lowered, so their rows are trees."""
    if "worth_a_tree_level" not in request.node.name:
        monkeypatch.setattr("palingenesis.tree_packing.TREE_MIN_SAVED", 1)


# ── pairs ──────────────────────────────────────────────────────────────────────


def _side(prompt, answer, scored_prompt=()):
    """One side of a pair as the data pipeline emits it: tokens, and labels on the answer (and on the prompt
    positions of an earlier assistant turn)."""
    ids = torch.tensor(list(prompt) + list(answer))
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[len(prompt) :] = ids[len(prompt) :]
    for i in scored_prompt:
        labels[i] = ids[i]
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _pairs(case: str, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)

    def tokens(n):
        return rng.integers(3, 90, n).tolist()

    def plain(prompt_len=12, chosen_len=7, rejected_len=5):
        prompt = tokens(prompt_len)
        return {
            "chosen": _side(prompt, [91] + tokens(chosen_len)),
            "rejected": _side(prompt, [92] + tokens(rejected_len)),
        }

    if case == "plain":
        return [plain(12, 7, 5), plain(20, 3, 9), plain(5, 11, 11)]
    if case == "identical":  # the dataset drops such pairs; the layout must still score both sides
        pair = plain()
        pair["rejected"] = copy.deepcopy(pair["chosen"])
        return [plain(), pair]
    if case == "empty_prefix":  # implicit prompts that differ from their first token: nothing shared
        return [
            {"chosen": _side([1] + tokens(8), tokens(6)), "rejected": _side([2] + tokens(10), tokens(4))},
            plain(),
        ]
    if case == "multi_turn":
        # an earlier assistant turn inside the shared prompt is scored on both sides (data.last_turn_only off),
        # and the answers start with the same tokens (minimal-edit pairs): the trie branches inside the answer
        out = []
        for turns in (1, 2):
            prompt, scored = [], []
            for _ in range(turns):
                prompt += tokens(6)
                scored += list(range(len(prompt), len(prompt) + 4))
                prompt += tokens(4)
            prompt += tokens(5)
            start = tokens(3)
            out.append(
                {
                    "chosen": _side(prompt, start + [91] + tokens(5), scored),
                    "rejected": _side(prompt, start + [92] + tokens(8), scored),
                }
            )
        return out
    if case == "truncation":
        # a rejected side cut at the length limit (its last scored token is the row's last), a rejected side that
        # is a strict prefix of the chosen one, and a chosen side that is a prefix of the rejected one (a loop)
        prompt, answer = tokens(10), tokens(12)
        cut = {"chosen": _side(prompt, tokens(6)), "rejected": _side(prompt, tokens(30))}
        cut["rejected"] = {k: v[:24] for k, v in cut["rejected"].items()}
        return [
            cut,
            {"chosen": _side(prompt, answer), "rejected": _side(prompt, answer[:5])},
            {"chosen": _side(prompt, answer[:7]), "rejected": _side(prompt, answer)},
        ]
    if case == "mixed":
        return [p for name in ("plain", "identical", "empty_prefix", "multi_turn", "truncation") for p in _pairs(name)]
    raise ValueError(case)


CASES = ["plain", "identical", "empty_prefix", "multi_turn", "truncation", "mixed"]


def _batch(case: str, seed: int = 0):
    return collate_preferences(_pairs(case, seed), pad_id=0)


def _policy_and_reference(arch: str):
    """Float64 models (the Gated DeltaNet delta rule in float64 too); the policy moved off the reference."""
    reference = _model(arch, f64_linear=True).eval().requires_grad_(False)
    policy = _model(arch, f64_linear=True)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return policy.train(), reference


SETTINGS = {  # every objective, with the modifiers spread over them
    "sigmoid": {},
    "hinge": {"beta": 0.5, "sft_weight": 0.3},
    "ipo": {"ld_alpha": 0.5},
    "robust": {"label_smoothing": 0.1, "sft_weight": 0.2, "sft_scope": "edits", "sft_edit_window": 1},
    "sigmoid_norm": {"sft_weight": 0.5, "ld_alpha": 0.3},
}


def _padded_step(policy, reference, batch, num_chunks=2, **kw):
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    num_pairs = ids.shape[0] // 2
    # right-padded rows run without a mask, as in the trainer
    ref = reference_logps(reference, lambda m, i, a: _get_hidden_states(m, i, None), _get_lm_head, ids, mask, labels)
    hidden = _get_hidden_states(policy, ids, None)
    logps = token_logps(hidden.detach(), shift_labels(labels), _get_lm_head(policy))
    step = preference_step(
        hidden,
        labels,
        _get_lm_head(policy),
        ref,
        num_pairs,
        num_chunks=num_chunks,
        sft_pair_weight=batch["sft_pair_weight"],
        **kw,
    )
    return step, logps, ref


def _packed_step(policy, reference, batch, supports, share=True, num_chunks=2, **kw):
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    packed = PackedPairs.build(ids, mask, labels, share=share)
    ref = packed_reference_logps(reference, supports[1], packed, _get_lm_head(reference), num_chunks)
    hidden = forward_packed(policy, supports[0], packed.row, "cpu")
    logps = packed.token_logps(hidden.detach(), _get_lm_head(policy), num_chunks)
    step = preference_step(
        hidden,
        labels,
        _get_lm_head(policy),
        ref,
        ids.shape[0] // 2,
        num_chunks=num_chunks,
        sft_pair_weight=batch["sft_pair_weight"],
        packed=packed,
        **kw,
    )
    return step, logps, ref, packed


def _grads(model):
    """Gradients by parameter name (without the activation-checkpoint wrapper's prefix; FSDP2: the local shard)."""

    def local(t):
        return t.to_local() if hasattr(t, "to_local") else t

    return {
        n.replace("_checkpoint_wrapped_module.", ""): local(p.grad).clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def _assert_same_step(policy, padded, packed, grads_padded, grads_packed, tol=1e-10, grad_tol=1e-6):
    step_a, logps_a, ref_a = padded
    step_b, logps_b, ref_b = packed[:3]
    assert (logps_a - logps_b).abs().max() <= tol  # per-token policy log-probs, [2P, S]
    assert (ref_a - ref_b).abs().max() <= tol  # per-token reference log-probs
    assert (logps_a.sum(1) - logps_b.sum(1)).abs().max() <= tol  # per-sequence log-probs
    loss_a, loss_b = float(step_a.loss.detach()), float(step_b.loss.detach())
    assert abs(loss_a - loss_b) <= tol * max(1.0, abs(loss_a))
    for key, value in step_a.metrics.items():
        assert abs(value - step_b.metrics[key]) <= tol * max(1.0, abs(value)), key
    assert grads_padded.keys() == grads_packed.keys() and grads_padded
    for name, ref in grads_padded.items():
        error = (grads_packed[name] - ref).abs().max() / ref.abs().max().clamp_min(1e-30)
        assert error <= grad_tol, (name, float(error))


# ── exactness against the padded path ──────────────────────────────────────────


@pytest.mark.parametrize("arch", ["qwen3", "qwen35_hybrid"])
@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_packed_step_equals_the_padded_step_for_every_objective(arch, loss_type):
    """Scores, loss, metrics and gradients of one micro-batch holding every edge case, tree and varlen against
    the padded [2P, S] rows, for each objective with its modifiers (SFT term and scope, LD-DPO, pair weights)."""
    policy, reference = _policy_and_reference(arch)
    supports = prepare_model(policy), prepare_model(reference)
    batch = _batch("mixed")
    batch["sft_pair_weight"] = torch.linspace(0.0, 1.0, int(batch["num_pairs"]))
    kw = {"loss_type": loss_type, **SETTINGS[loss_type]}
    policy.zero_grad()
    padded = _padded_step(policy, reference, batch, **kw)
    padded[0].loss.backward()
    grads_padded = _grads(policy)
    for share in (True, False):
        policy.zero_grad()
        packed = _packed_step(policy, reference, batch, supports, share=share, **kw)
        packed[0].loss.backward()
        _assert_same_step(policy, padded, packed, grads_padded, _grads(policy))
        row = packed[3].row
        assert row.flat == (not share)
        assert packed[0].metrics["dpo/forwarded_share"] == row.num_tokens / int(batch["attention_mask"].sum())


@pytest.mark.parametrize("arch", ["qwen3", "qwen35_hybrid"])
@pytest.mark.parametrize("case", [c for c in CASES if c != "mixed"])
def test_edge_cases_one_at_a_time(arch, case):
    policy, reference = _policy_and_reference(arch)
    supports = prepare_model(policy), prepare_model(reference)
    batch = _batch(case, seed=3)
    kw = {"loss_type": "sigmoid_norm", "sft_weight": 0.4, "ld_alpha": 0.5, "num_chunks": 3}
    policy.zero_grad()
    padded = _padded_step(policy, reference, batch, **kw)
    padded[0].loss.backward()
    grads_padded = _grads(policy)
    policy.zero_grad()
    packed = _packed_step(policy, reference, batch, supports, **kw)
    packed[0].loss.backward()
    _assert_same_step(policy, padded, packed, grads_padded, _grads(policy))


def test_identical_sides_have_zero_margin_and_one_segment():
    policy, reference = _policy_and_reference("qwen3")
    supports = prepare_model(policy), prepare_model(reference)
    batch = collate_preferences([_pairs("identical")[1]], pad_id=0)
    step, logps, _, packed = _packed_step(policy, reference, batch, supports)
    assert packed.row.num_segments == 1 and packed.row.num_tokens == batch["input_ids"].shape[1]
    assert torch.equal(logps[0], logps[1])
    assert step.metrics["rewards/margins"] == 0.0 and abs(step.metrics["dpo/loss"] - np.log(2)) < 1e-12


@pytest.mark.parametrize("mode", ["full", "selective"])
@pytest.mark.parametrize("arch", ["qwen3", "qwen35_hybrid"])
def test_exact_under_activation_checkpointing(arch, mode):
    """train.gradient_checkpointing: the layers recomputed in the backward follow the packed layout too."""
    from palingenesis.kernels import apply_activation_checkpointing

    policy, reference = _policy_and_reference(arch)
    batch = _batch("mixed", seed=5)
    kw = {"loss_type": "sigmoid", "sft_weight": 0.2}
    padded = _padded_step(policy, reference, batch, **kw)
    padded[0].loss.backward()
    grads_padded = _grads(policy)
    apply_activation_checkpointing(policy, mode=mode)
    supports = prepare_model(policy), prepare_model(reference)
    policy.zero_grad()
    packed = _packed_step(policy, reference, batch, supports, **kw)
    packed[0].loss.backward()
    _assert_same_step(policy, padded, packed, grads_padded, _grads(policy))


def test_exact_with_lora_adapters():
    """LoRA: the frozen base and the adapters' gradients through a packed row equal the padded rows'."""
    from palingenesis.lora import LoRALinear

    policy, reference = _policy_and_reference("qwen35_hybrid")
    policy.requires_grad_(False)
    torch.manual_seed(2)
    for parent in list(policy.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear) and name in ("q_proj", "v_proj", "in_proj_qkv"):
                adapted = LoRALinear(child, rank=2, alpha=4.0).double()
                torch.nn.init.normal_(adapted.lora_B, std=0.05)
                setattr(parent, name, adapted)
    supports = prepare_model(policy), prepare_model(reference)
    batch = _batch("mixed", seed=6)
    padded = _padded_step(policy, reference, batch, loss_type="sigmoid")
    padded[0].loss.backward()
    grads_padded = _grads(policy)
    assert grads_padded and all("lora_" in n for n in grads_padded)
    policy.zero_grad()
    packed = _packed_step(policy, reference, batch, supports, loss_type="sigmoid")
    packed[0].loss.backward()
    _assert_same_step(policy, padded, packed, grads_padded, _grads(policy))


def test_fsdp2_with_full_checkpointing_trains_through_packed_rows():
    """FSDP2 (one rank) with the layers recomputed in the backward: the packed step's loss and gradients equal
    the plain model's padded step."""
    from palingenesis.config import ParallelConfig
    from palingenesis.distributed import apply_fsdp
    from palingenesis.kernels import apply_activation_checkpointing
    from palingenesis.rl.parallel import Parallel

    policy, reference = _policy_and_reference("qwen35_hybrid")
    batch = _batch("mixed", seed=7)
    kw = {"loss_type": "ipo", "sft_weight": 0.2}
    padded = _padded_step(policy, reference, batch, **kw)
    padded[0].loss.backward()
    grads_padded = _grads(policy)
    policy.zero_grad()
    apply_activation_checkpointing(policy, mode="full")
    supports = prepare_model(policy), prepare_model(reference)
    policy = apply_fsdp(policy, Parallel(fsdp=True).mesh, ParallelConfig(), bf16=False)  # (float64 kept)
    packed = _packed_step(policy, reference, batch, supports, **kw)
    packed[0].loss.backward()
    _assert_same_step(policy, padded, packed, grads_padded, _grads(policy))


def test_accumulated_packed_micro_batches_equal_one_padded_batch():
    """Micro-batches of different pair counts (dpo.micro_tokens) with the trainer's denominators (the window's
    pairs and chosen tokens) accumulate to the gradient of one padded batch over all their pairs."""
    policy, reference = _policy_and_reference("qwen3")
    supports = prepare_model(policy), prepare_model(reference)
    pairs = _pairs("mixed", seed=8)
    whole = collate_preferences(pairs, pad_id=0)
    chosen_tokens = float((shift_labels(whole["labels"][: len(pairs)]) != IGNORE_INDEX).sum())
    kw = {"loss_type": "sigmoid", "sft_weight": 0.3, "pair_denom": len(pairs), "chosen_token_denom": chosen_tokens}
    padded = _padded_step(policy, reference, whole, **kw)
    padded[0].loss.backward()
    grads_padded = _grads(policy)
    policy.zero_grad()
    for part in (pairs[:1], pairs[1:5], pairs[5:]):
        _packed_step(policy, reference, collate_preferences(part, pad_id=0), supports, **kw)[0].loss.backward()
    for name, ref in grads_padded.items():
        error = (policy.get_parameter(name).grad - ref).abs().max() / ref.abs().max().clamp_min(1e-30)
        assert error <= 1e-6, (name, float(error))


# ── the layout of a row of pairs ───────────────────────────────────────────────


def _sequences(batch):
    lengths = batch["attention_mask"].sum(1).tolist()
    return [batch["input_ids"][b, :n].tolist() for b, n in enumerate(lengths)]


def _common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


@pytest.mark.parametrize("case", CASES)
def test_every_pair_is_a_two_leaf_trie_and_pairs_share_nothing(case):
    batch = _batch(case)
    packed = PackedPairs.build(batch["input_ids"], batch["attention_mask"], batch["labels"])
    row, sequences = packed.row, _sequences(batch)
    num_pairs = len(sequences) // 2
    for b, s in enumerate(sequences):  # every sequence is recovered, at its own positions
        assert row.tokens[row.positions(b)].tolist() == s
        assert row.position_ids[row.positions(b)].tolist() == list(range(len(s)))
    shared = [_common(sequences[p], sequences[p + num_pairs]) for p in range(num_pairs)]
    # the shared prefix of each pair once, whatever the other pairs hold
    assert row.num_tokens == sum(map(len, sequences)) - sum(shared)
    assert len(row.levels) <= 2 and packed.sequence_tokens == sum(map(len, sequences))
    for p in range(num_pairs):
        chosen, rejected = row.positions(p), row.positions(p + num_pairs)
        assert np.array_equal(chosen[: shared[p]], rejected[: shared[p]])
        assert not set(chosen[shared[p] :]) & set(rejected[shared[p] :])
        others = {q for b in range(len(sequences)) if b % num_pairs != p for q in row.positions(b).tolist()}
        assert not others & (set(chosen.tolist()) | set(rejected.tolist()))
    assert (
        tuple(group_cost([sequences[0], sequences[num_pairs]]))[0]
        == len(sequences[0]) + len(sequences[num_pairs]) - shared[0]
    )


def test_pairs_with_the_same_prompt_stay_apart():
    """Sharing is per pair: two pairs on one prompt are two tries, so a row of pairs has two levels (one
    Gated DeltaNet kernel call per level), whatever its pairs have in common."""
    prompt = list(range(10, 30))
    pairs = [
        {"chosen": _side(prompt, [91, 5, 6]), "rejected": _side(prompt, [92, 7])},
        {"chosen": _side(prompt, [91, 5, 8]), "rejected": _side(prompt, [93])},
    ]
    batch = collate_preferences(pairs, pad_id=0)
    row = PackedPairs.build(batch["input_ids"], batch["attention_mask"], batch["labels"]).row
    assert len(row.levels) == 2 and row.num_segments == 6 and row.num_tokens == 2 * len(prompt) + 3 + 2 + 3 + 1
    exact = PackedRow.build(_sequences(batch), share=True)  # the unrestricted trie would share across the pairs
    assert exact.num_tokens < row.num_tokens and len(exact.levels) > 2


def test_a_row_becomes_a_tree_only_when_worth_a_tree_level():
    """The real threshold: the row's pairs together must save a tree level's cost (TREE_MIN_SAVED tokens)."""
    rng = np.random.default_rng(0)

    def rows(prompt_len, count):
        out = []
        for _ in range(count):
            prompt = rng.integers(3, 90, prompt_len).tolist()
            out += [prompt + [91, 1], prompt + [92, 2, 3]]
        return out[0::2] + out[1::2]  # chosen rows first

    few = rows(TREE_MIN_SAVED // 8, 4)  # saves half a level's cost
    groups = [b % 4 for b in range(8)]
    assert pack_groups(few, groups).flat and pack_groups(few, groups).num_tokens == sum(map(len, few))
    many = rows(TREE_MIN_SAVED // 8, 8)  # the same pairs, enough of them
    row = pack_groups(many, [b % 8 for b in range(16)])
    assert len(row.levels) == 2 and row.num_tokens == sum(map(len, many)) - TREE_MIN_SAVED
    assert pack_groups(many, [b % 8 for b in range(16)], share=False).flat


def test_scored_tokens_inside_a_shared_prefix_are_computed_once():
    batch = _batch("multi_turn")
    packed = PackedPairs.build(batch["input_ids"], batch["attention_mask"], batch["labels"])
    scored = int(packed.valid.sum())
    # pair 0: 4 scored prompt tokens and a 3-token common start of the answers; pair 1: 8 and 3. The token that
    # ends the shared prefix predicts each side's own next token: two terms.
    assert packed.num_terms == scored - (4 + 3) - (8 + 3)
    varlen = PackedPairs.build(batch["input_ids"], batch["attention_mask"], batch["labels"], share=False)
    assert varlen.num_terms == scored


def test_build_refuses_what_is_not_a_right_padded_pair_batch():
    batch = _batch("plain")
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    with pytest.raises(PackingError, match="chosen and rejected"):
        PackedPairs.build(ids[:3], mask[:3], labels[:3])
    left = mask.clone()
    left[0, 0] = 0
    with pytest.raises(PackingError, match="right-padded"):
        PackedPairs.build(ids, left, labels)
    short = mask.clone()
    short[0, -3:] = 0  # the row's last tokens are scored but masked out
    scored = labels.clone()
    scored[0, -3:] = ids[0, -3:]
    with pytest.raises(PackingError, match="past the end"):
        PackedPairs.build(ids, short, scored)
    packed = PackedPairs.build(ids, mask, labels)
    with pytest.raises(PackingError, match="packed hidden states"):
        packed.token_logps(torch.zeros(ids.shape[0], ids.shape[1], 8), torch.nn.Linear(8, VOCAB))


# ── micro-batches under a token budget ─────────────────────────────────────────


def test_pair_batches_fill_the_budget_in_order():
    pairs = [p for seed in range(4) for p in _pairs("mixed", seed)]
    costs = [group_cost([p["chosen"]["input_ids"].numpy(), p["rejected"]["input_ids"].numpy()])[0] for p in pairs]
    budget = 90
    batches = list(PairBatches(pairs, budget, lambda b: b))
    assert [id(p) for b in batches for p in b] == [id(p) for p in pairs]  # every pair once, in the data's order
    cursor = 0
    for i, b in enumerate(batches):
        load = sum(costs[cursor : cursor + len(b)])
        assert load <= budget
        if i < len(batches) - 1:  # the next pair did not fit
            assert load + costs[cursor + len(b)] > budget
        cursor += len(b)
    assert len({len(b) for b in batches}) > 1  # pairs per micro-batch follow their lengths
    # tree: a shared prefix counts once; varlen: both sides' tokens
    flat = PairBatches(pairs, budget, lambda b: b, share=False)
    assert flat.cost(pairs[0]) == pairs[0]["chosen"]["input_ids"].numel() + pairs[0]["rejected"]["input_ids"].numel()
    assert flat.cost(pairs[0]) - costs[0] == 12 and len(list(flat)) > len(batches)
    alone = list(PairBatches(pairs, 5, lambda b: b))  # pairs above the budget: one micro-batch each
    assert [len(b) for b in alone] == [1] * len(pairs)


def test_dataloader_packs_by_tokens_and_collates_without_padding_multiple():
    from test_dpo import PAIR, QWEN35_TEMPLATE

    from palingenesis.dpo import build_preference_dataloader

    tok = H.tokenizer()
    tok.chat_template = QWEN35_TEMPLATE
    rows = [
        {**PAIR, "prompt": [{"role": "user", "content": "expand: cheap flights " + "to Rome " * (7 * (i % 4))}]}
        for i in range(12)
    ]
    data, dpo = Config().data, Config().dpo
    data.max_seq_length, data.num_workers = 256, 0
    dpo.packing, dpo.micro_tokens = "tree", 10**6
    (everything,) = list(build_preference_dataloader(rows, tok, data, dpo, 0, 1, batch_size=2))
    assert int(everything["num_pairs"]) == 12  # the budget, not per_device_batch_size, sets the pairs
    sizes = everything["attention_mask"].sum(1)
    dpo.micro_tokens = budget = int(sizes.max()) * 3
    batches = list(build_preference_dataloader(rows, tok, data, dpo, 0, 1, batch_size=2))
    assert sum(int(b["num_pairs"]) for b in batches) == 12 and len(batches) > 1
    for b in batches:
        packed = PackedPairs.build(b["input_ids"], b["attention_mask"], b["labels"])
        assert len(packed.row.levels) == 2 and packed.row.num_tokens <= budget
        assert packed.row.num_tokens < int(b["attention_mask"].sum())  # the prompts once
        assert b["input_ids"].shape[1] == int(b["attention_mask"].sum(1).max())  # no padding to a multiple
    dpo.micro_tokens = 0  # a fixed number of pairs, as padded; rows padded to the longest only
    fixed = list(build_preference_dataloader(rows, tok, data, dpo, 0, 1, batch_size=2))
    assert [int(b["num_pairs"]) for b in fixed] == [2] * 6
    assert all(b["input_ids"].shape[1] == int(b["attention_mask"].sum(1).max()) for b in fixed)


# ── configuration ──────────────────────────────────────────────────────────────


def _config(**dpo):
    config = Config()
    config.dpo.enabled = True
    config.model.compile = False
    for key, value in dpo.items():
        setattr(config.dpo, key, value)
    return config


def test_packing_options_are_validated():
    assert Config().dpo.packing == "padded" and Config().dpo.micro_tokens == 0
    for packing in ("padded", "varlen", "tree"):
        _config(packing=packing).validate()
    _config(packing="tree", micro_tokens=16384).validate()
    for dpo, match in [
        ({"packing": "trie"}, "dpo.packing must be one of"),
        ({"micro_tokens": 4096}, "only used with dpo.packing varlen or tree"),
        ({"packing": "tree", "micro_tokens": -1}, "dpo.micro_tokens must be an integer >= 0"),
        ({"packing": "tree", "micro_tokens": True}, "dpo.micro_tokens must be an integer >= 0"),
    ]:
        with pytest.raises(ConfigError, match=match):
            _config(**dpo).validate()
    for section, field, value, match in [
        ("model", "compile", True, "model.compile"),
        ("model", "attn_implementation", "flash_attention_2", "attn_implementation: sdpa"),
        ("multimodal", "enabled", True, "multimodal.enabled"),
        ("memory", "activation_offload", True, "memory.activation_offload"),
    ]:
        config = _config(packing="tree")
        setattr(getattr(config, section), field, value)
        with pytest.raises(ConfigError, match=match):
            config.validate()
        config.dpo.packing = "padded"  # the same settings without packing: not this error
        try:
            config.validate()
        except ConfigError as error:
            assert "dpo.packing" not in str(error)


# ── the trainer ────────────────────────────────────────────────────────────────


def _preference_rows(n=24):
    return [
        {
            "prompt": [{"role": "user", "content": f"What is {i} plus {i + 1}?" + " Think." * (i % 4)}],
            "chosen": [{"role": "assistant", "content": f"It is {2 * i + 1}."}],
            "rejected": [{"role": "assistant", "content": f"It is {2 * i + 7}, I think."}],
        }
        for i in range(n)
    ]


def _run(tmp_path, name, plugins=None, **dpo):
    tok = H.tokenizer()
    if not (tmp_path / "student").exists():
        H.save_model(tmp_path / "student", tok, seed=0)
        H.write_rows(tmp_path / "pairs.jsonl", _preference_rows())
    c = H.cpu_config(tmp_path, tmp_path / "student", str(tmp_path / "pairs.jsonl"), steps=3)
    c.train.output_dir = str(tmp_path / name)
    c.train.learning_rate = 1e-3
    c.train.logging_steps = 1
    c.dpo.enabled = True
    c.dpo.sft_weight = 0.2
    for key, value in dpo.items():
        setattr(c.dpo, key, value)
    return H.run_cpu(c, plugins=plugins)


def test_training_run_with_tree_packing_follows_the_padded_run(tmp_path):
    """Whole runs of the trainer (three optimizer steps, accumulation 2): tree packing sees the same
    micro-batches as the padded run and logs the same loss and rewards (fp32)."""
    padded = _run(tmp_path, "padded")
    tree = _run(tmp_path, "tree", packing="tree")
    assert tree.global_step == padded.global_step == 3
    assert tree.metrics["train/dpo/forwarded_share"] < 0.8 and "train/dpo/forwarded_share" not in padded.metrics
    for key in ("train/dpo/loss", "train/dpo/sft_nll", "train/rewards/margins", "train/logps/chosen"):
        assert abs(tree.metrics[key] - padded.metrics[key]) <= 1e-3 * max(1.0, abs(padded.metrics[key])), key
    varlen = _run(tmp_path, "varlen", packing="varlen")
    assert varlen.metrics["train/dpo/forwarded_share"] == 1.0
    assert abs(varlen.metrics["train/dpo/loss"] - padded.metrics["train/dpo/loss"]) <= 1e-3


def test_training_run_with_a_token_budget_and_lora(tmp_path):
    from palingenesis.lora import LoRA

    result = _run(tmp_path, "budget", plugins=[LoRA(rank=4)], packing="tree", micro_tokens=120)
    assert result.global_step == 3
    assert result.metrics["plugins/lora/delta_rel"] > 0
    assert 0 < result.metrics["train/dpo/forwarded_share"] < 1
    assert np.isfinite(result.metrics["train/dpo/loss"])


def test_trainer_refuses_a_model_it_cannot_pack(tmp_path, monkeypatch):
    def refuse(model, device=None):
        raise PackingError("packing: the model uses sliding-window attention, which packed rows do not implement.")

    monkeypatch.setattr("palingenesis.tree_packing.prepare_model", refuse)
    with pytest.raises(ConfigError, match="dpo.packing tree: packing: the model uses sliding-window"):
        _run(tmp_path, "refused", packing="tree")


def test_score_weights_and_objectives_read_the_padded_layout():
    """The packed path changes where hidden states come from, nothing else: labels, LD-DPO weights and the SFT
    mask are those of the collated rows."""
    batch = _batch("truncation")
    packed = PackedPairs.build(batch["input_ids"], batch["attention_mask"], batch["labels"])
    shifted = shift_labels(batch["labels"])
    assert torch.equal(packed.valid, shifted != IGNORE_INDEX)
    assert torch.equal(score_weights(shifted, 3) > 0, packed.valid)
    assert torch.equal(packed.targets[packed.inverse], shifted[packed.valid])
