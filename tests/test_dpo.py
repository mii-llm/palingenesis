"""Tests for DPO (palingenesis.dpo).

The central property: the two-pass chunked implementation — which never
materialises the full [B, S, V] logits — must produce the SAME loss and the SAME
parameter gradients as a naive implementation that builds the full logits and
backpropagates end to end. Checked for every loss type, with and without the SFT
anchor and LD-DPO weighting, across chunk counts.

Data tests use GPT-2's tokenizer (always cached, network-free) with real chat
templates: one with a `{% generation %}` span (fast masking path) and one
without (fallback path, which is what Qwen3.5-style templates hit).
"""

import copy
import math
import pathlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.data import IGNORE_INDEX  # noqa: E402
from palingenesis.dpo import (  # noqa: E402
    LOSS_TYPES,
    PreferenceDataset,
    collate_preferences,
    preference_loss,
    preference_step,
    score_weights,
    token_logps,
)
from palingenesis.loss import shift_labels  # noqa: E402

# Same shape as the production template: a {% generation %} span around the
# assistant turn, reasoning in <think>, and enable_thinking honoured.
TEMPLATE_WITH_GENERATION = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{%- set content = message['content'] %}{%- set reasoning = '' %}"
    "{%- if message.reasoning_content is defined and message.reasoning_content is string %}"
    "{%- set reasoning = message.reasoning_content %}{%- endif %}"
    "{{- '<|im_start|>assistant\\n' }}{%- generation %}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' + content }}"
    "{%- else %}{{- '<think>\\n' + reasoning + '\\n</think>\\n\\n' + content }}{%- endif %}"
    "{{- '<|im_end|>\\n' }}{%- endgeneration %}"
    "{%- endif %}{%- endfor %}"
)
# The same rendering without a generation span: forces the fallback masking path.
TEMPLATE_FALLBACK = TEMPLATE_WITH_GENERATION.replace("{%- generation %}", "").replace("{%- endgeneration %}", "")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def _dataset(tok, template, **kwargs):
    tok = copy.copy(tok)
    tok.chat_template = template
    return PreferenceDataset([], tok, max_seq_length=kwargs.pop("max_seq_length", 512), **kwargs), tok


def _decode_scored(tok, side):
    ids = side["input_ids"][side["labels"] != IGNORE_INDEX]
    return tok.decode(ids)


PAIR = {
    "prompt": [{"role": "user", "content": "expand: cheap flights"}],
    "chosen": [{"role": "assistant", "content": "cheap flights budget airfare deals"}],
    "rejected": [{"role": "assistant", "content": "cheap flights cheap flights cheap"}],
}


# ── data ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("template", [TEMPLATE_WITH_GENERATION, TEMPLATE_FALLBACK], ids=["fast", "fallback"])
def test_scores_only_the_answers(tokenizer, template):
    ds, tok = _dataset(tokenizer, template)
    pair = ds.process(PAIR)
    assert pair is not None
    chosen, rejected = _decode_scored(tok, pair["chosen"]), _decode_scored(tok, pair["rejected"])
    assert "budget airfare deals" in chosen and "expand:" not in chosen
    assert "cheap flights cheap" in rejected and "expand:" not in rejected


def test_explicit_and_implicit_prompt_are_equivalent(tokenizer):
    ds, _ = _dataset(tokenizer, TEMPLATE_WITH_GENERATION)
    explicit = ds.process(PAIR)
    implicit = ds.process({"chosen": PAIR["prompt"] + PAIR["chosen"], "rejected": PAIR["prompt"] + PAIR["rejected"]})
    for side in ("chosen", "rejected"):
        assert torch.equal(explicit[side]["input_ids"], implicit[side]["input_ids"])
        assert torch.equal(explicit[side]["labels"], implicit[side]["labels"])


def test_string_completions_become_assistant_turns(tokenizer):
    ds, _ = _dataset(tokenizer, TEMPLATE_WITH_GENERATION)
    as_messages = ds.process(PAIR)
    as_strings = ds.process(
        {
            "prompt": "expand: cheap flights",
            "chosen": "cheap flights budget airfare deals",
            "rejected": "cheap flights cheap flights cheap",
        }
    )
    assert torch.equal(as_messages["chosen"]["input_ids"], as_strings["chosen"]["input_ids"])


def test_per_row_chat_template_kwargs_reach_the_render(tokenizer):
    ds, tok = _dataset(tokenizer, TEMPLATE_WITH_GENERATION)
    thinking = {**PAIR, "chosen": [{"role": "assistant", "content": "ans", "reasoning": "because"}]}
    with_trace = ds.process(thinking)
    no_trace = ds.process({**thinking, "chat_template_kwargs": {"enable_thinking": False}})
    assert "because" in tok.decode(with_trace["chosen"]["input_ids"])
    assert "because" not in tok.decode(no_trace["chosen"]["input_ids"])


# A template that reads the modern `reasoning` key instead of `reasoning_content`.
TEMPLATE_READS_REASONING = TEMPLATE_WITH_GENERATION.replace(
    "message.reasoning_content is defined and message.reasoning_content is string",
    "message.reasoning is defined and message.reasoning is string",
).replace("set reasoning = message.reasoning_content", "set reasoning = message.reasoning")


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize(
    "template",
    [TEMPLATE_WITH_GENERATION, TEMPLATE_READS_REASONING],
    ids=["template_reads_reasoning_content", "template_reads_reasoning"],
)
def test_reasoning_field_renders_whichever_key_the_template_reads(tokenizer, field, template):
    """`reasoning` (current convention) and `reasoning_content` (older) are
    interchangeable: the trace must render and be scored on either template."""
    ds, tok = _dataset(tokenizer, template)
    row = {**PAIR, "chosen": [{"role": "assistant", "content": "final answer", field: "the trace"}]}
    pair = ds.process(row)
    assert "the trace" in tok.decode(pair["chosen"]["input_ids"])
    assert "the trace" in _decode_scored(tok, pair["chosen"])


def test_reasoning_takes_precedence_over_reasoning_content(tokenizer):
    ds, tok = _dataset(tokenizer, TEMPLATE_WITH_GENERATION)
    row = {
        **PAIR,
        "chosen": [{"role": "assistant", "content": "a", "reasoning": "modern", "reasoning_content": "legacy"}],
    }
    rendered = tok.decode(ds.process(row)["chosen"]["input_ids"])
    assert "modern" in rendered and "legacy" not in rendered


# The official Qwen/Qwen3.5-0.8B template (Apache-2.0), vendored verbatim. Its Jinja
# reads `message.reasoning_content`; callers send `reasoning`, which vLLM (like
# normalize_messages) mirrors into both keys before rendering.
QWEN35_TEMPLATE = (pathlib.Path(__file__).parent / "fixtures" / "qwen3_5_chat_template.jinja").read_text()


@pytest.mark.parametrize("last_turn_only", [False, True])
def test_reasoning_field_on_the_real_qwen35_template(tokenizer, last_turn_only):
    ds, tok = _dataset(tokenizer, QWEN35_TEMPLATE, last_turn_only=last_turn_only)
    row = {
        "prompt": [{"role": "user", "content": "expand: cheap flights"}],
        "chosen": [
            {
                "role": "assistant",
                "content": "cheap flights budget airfare",
                "reasoning": "the user wants cheaper synonyms",
            }
        ],
        "rejected": [{"role": "assistant", "content": "cheap flights cheap flights", "reasoning": "repeat it"}],
    }
    pair = ds.process(row)
    for side, trace, answer in (
        ("chosen", "the user wants cheaper synonyms", "cheap flights budget airfare"),
        ("rejected", "repeat it", "cheap flights cheap flights"),
    ):
        rendered = tok.decode(pair[side]["input_ids"])
        assert f"<think>\n{trace}\n</think>\n\n{answer}" in rendered
        scored = _decode_scored(tok, pair[side])
        assert trace in scored and answer in scored
        assert "expand:" not in scored


def test_train_on_reasoning_false_excludes_the_trace(tokenizer):
    row = {**PAIR, "chosen": [{"role": "assistant", "content": "final answer", "reasoning": "long reasoning"}]}
    on, tok = _dataset(tokenizer, TEMPLATE_WITH_GENERATION, train_on_reasoning=True)
    off, _ = _dataset(tokenizer, TEMPLATE_WITH_GENERATION, train_on_reasoning=False)
    assert "long reasoning" in _decode_scored(tok, on.process(row)["chosen"])
    scored_off = _decode_scored(tok, off.process(row)["chosen"])
    assert "long reasoning" not in scored_off and "final answer" in scored_off


def test_last_turn_only_scores_the_final_assistant_turn(tokenizer):
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "q2"},
    ]
    row = {"prompt": history, "chosen": "good final", "rejected": "bad final"}
    every, tok = _dataset(tokenizer, TEMPLATE_WITH_GENERATION, last_turn_only=False)
    last, _ = _dataset(tokenizer, TEMPLATE_WITH_GENERATION, last_turn_only=True)
    assert "earlier answer" in _decode_scored(tok, every.process(row)["chosen"])
    scored = _decode_scored(tok, last.process(row)["chosen"])
    assert "earlier answer" not in scored and "good final" in scored


def test_chosen_is_never_truncated_rejected_may_be(tokenizer):
    loop = "again " * 200
    ds, _ = _dataset(tokenizer, TEMPLATE_WITH_GENERATION, max_seq_length=64)
    assert ds.process({**PAIR, "rejected": loop}) is not None  # rejected truncated
    assert ds.stats["rejected_truncated"] == 1
    assert ds.process({**PAIR, "chosen": loop}) is None  # chosen too long: dropped
    assert ds.stats["dropped_chosen_too_long"] == 1
    strict, _ = _dataset(tokenizer, TEMPLATE_WITH_GENERATION, max_seq_length=64, truncate_rejected=False)
    assert strict.process({**PAIR, "rejected": loop}) is None


def test_identical_pairs_are_dropped(tokenizer):
    ds, _ = _dataset(tokenizer, TEMPLATE_WITH_GENERATION)
    assert ds.process({**PAIR, "rejected": PAIR["chosen"]}) is None
    assert ds.stats["dropped_identical"] == 1


def test_collate_puts_chosen_first(tokenizer):
    ds, tok = _dataset(tokenizer, TEMPLATE_WITH_GENERATION)
    pairs = [ds.process(PAIR), ds.process({**PAIR, "chosen": "x y z"})]
    batch = collate_preferences(pairs, tok.pad_token_id)
    assert batch["input_ids"].shape[0] == 4 and int(batch["num_pairs"]) == 2
    assert torch.equal(
        batch["input_ids"][0, : pairs[0]["chosen"]["input_ids"].numel()], pairs[0]["chosen"]["input_ids"]
    )
    assert torch.equal(
        batch["input_ids"][2, : pairs[0]["rejected"]["input_ids"].numel()], pairs[0]["rejected"]["input_ids"]
    )


# ── loss definitions ─────────────────────────────────────────────────────────


def test_loss_formulas_match_their_definitions():
    beta, eps = 0.1, 0.2
    policy = torch.tensor([-10.0, -30.0])  # chosen, rejected
    reference = torch.tensor([-12.0, -25.0])
    lengths = torch.tensor([4.0, 8.0])
    delta = (-10 + 12) - (-30 + 25)  # = 7
    expect = {
        "sigmoid": -F.logsigmoid(torch.tensor(beta * delta)),
        "hinge": torch.relu(torch.tensor(1 - beta * delta)),
        "robust": (
            -(1 - eps) * F.logsigmoid(torch.tensor(beta * delta)) + eps * F.logsigmoid(torch.tensor(-beta * delta))
        )
        / (1 - 2 * eps),
    }
    delta_avg = (-10 + 12) / 4 - (-30 + 25) / 8
    expect["ipo"] = torch.tensor((delta_avg - 1 / (2 * beta)) ** 2)
    expect["sigmoid_norm"] = -F.logsigmoid(torch.tensor(beta * delta_avg))
    for lt in LOSS_TYPES:
        got = preference_loss(policy, reference, lengths, loss_type=lt, beta=beta, label_smoothing=eps)
        assert torch.allclose(got, expect[lt].reshape(1), atol=1e-6), lt


def test_ld_alpha_downweights_only_the_longer_tail():
    labels = torch.full((2, 8), IGNORE_INDEX)
    labels[0, 1:4] = 1  # chosen: 3 scored
    labels[1, 1:7] = 1  # rejected: 6 scored
    a = score_weights(labels, num_pairs=1, ld_alpha=0.5)
    assert a[0].sum() == 3  # the shorter answer is untouched
    assert a[1].sum() == 3 + 0.5 * 3  # shared 3 at 1.0, tail 3 at 0.5


# ── the core property: chunked two-pass == naive full-logits autograd ─────────


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=128, vocab_size=97)
    return GPT2LMHeadModel(cfg).eval().double()  # eval: no dropout; double: tight tolerances


def _toy_batch(num_pairs=3, seq=20, vocab=97):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, vocab, (2 * num_pairs, seq), generator=g)
    labels = torch.full_like(ids, IGNORE_INDEX)
    for b in range(2 * num_pairs):
        start = 5 + b % 3
        end = seq - (b % 4)  # different lengths per row
        labels[b, start:end] = ids[b, start:end]
    return ids, torch.ones_like(ids), labels


def _naive(model, ids, mask, labels, ref_logps, P, **kw):
    logits = model(input_ids=ids, attention_mask=mask).logits.double()
    shifted = shift_labels(labels)
    logp = torch.log_softmax(logits, -1).gather(-1, shifted.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    logp = torch.where(shifted != IGNORE_INDEX, logp, 0.0)
    a = score_weights(shifted, P, kw["ld_alpha"]).double()
    scores, ref_scores = (logp * a).sum(1), (ref_logps.double() * a).sum(1)
    loss = (
        preference_loss(
            scores,
            ref_scores,
            a.sum(1),
            loss_type=kw["loss_type"],
            beta=kw["beta"],
            label_smoothing=kw["label_smoothing"],
        ).sum()
        / P
    )
    if kw["sft_weight"]:
        chosen = (shifted[:P] != IGNORE_INDEX).double()
        loss = loss + kw["sft_weight"] * -(logp[:P] * chosen).sum() / chosen.sum()
    return loss


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("sft_weight,ld_alpha", [(0.0, None), (0.3, None), (0.0, 0.5), (0.2, 0.5)])
@pytest.mark.parametrize("num_chunks", [1, 3])
def test_chunked_gradients_equal_naive(loss_type, sft_weight, ld_alpha, num_chunks):
    model = _tiny_model()
    ref = copy.deepcopy(model)
    with torch.no_grad():  # a reference that differs from the policy
        for p in ref.parameters():
            p.add_(0.01 * torch.randn_like(p))
    ids, mask, labels = _toy_batch()
    P = ids.shape[0] // 2
    with torch.no_grad():
        ref_logps = token_logps(
            ref.transformer(input_ids=ids, attention_mask=mask)[0], shift_labels(labels), ref.lm_head, num_chunks=1
        )
    kw = dict(
        loss_type=loss_type,
        beta=0.1,
        label_smoothing=0.1 if loss_type == "robust" else 0.0,
        ld_alpha=ld_alpha,
        sft_weight=sft_weight,
    )

    naive = _naive(model, ids, mask, labels, ref_logps, P, **kw)
    naive.backward()
    naive_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()

    hidden = model.transformer(input_ids=ids, attention_mask=mask)[0]
    step = preference_step(hidden, labels, model.lm_head, ref_logps, P, num_chunks=num_chunks, **kw)
    step.loss.backward()

    assert torch.allclose(step.loss, naive, atol=1e-12, rtol=1e-10)
    for n, p in model.named_parameters():
        # Relative to the gradient's scale: the two implementations reduce in a
        # different order, so near-zero entries differ in the last float64 digits
        # (measured worst case ~1e-8 relative). A real bug is orders larger.
        scale = naive_grads[n].abs().max().clamp(min=1e-30)
        assert (p.grad - naive_grads[n]).abs().max() / scale < 1e-6, n


def test_metrics_are_reported():
    model = _tiny_model()
    ids, mask, labels = _toy_batch()
    P = ids.shape[0] // 2
    hidden = model.transformer(input_ids=ids, attention_mask=mask)[0]
    ref_logps = token_logps(hidden.detach(), shift_labels(labels), model.lm_head)
    step = preference_step(hidden, labels, model.lm_head, ref_logps, P)
    # policy == reference: zero rewards, loss log(2), no preference yet
    assert abs(step.metrics["dpo/loss"] - math.log(2)) < 1e-12
    assert step.metrics["rewards/margins"] == 0.0
    for key in ("rewards/chosen", "rewards/rejected", "rewards/accuracies", "logps/chosen", "logps/rejected"):
        assert key in step.metrics


def test_prompt_not_repeated_when_completions_are_full_conversations():
    """ultrafeedback_binarized-style rows: a prompt string AND chosen/rejected that
    already start with that user turn. The prompt must appear once."""
    from palingenesis.dpo import _to_conversations

    row = {
        "prompt": "Hi?",
        "chosen": [{"role": "user", "content": "Hi?"}, {"role": "assistant", "content": "Hello"}],
        "rejected": [{"role": "user", "content": "Hi?"}, {"role": "assistant", "content": "Go away"}],
    }
    chosen, rejected = _to_conversations(row)
    assert [m["role"] for m in chosen] == ["user", "assistant"]
    assert [m["role"] for m in rejected] == ["user", "assistant"]
    # a prompt followed by bare completions is still prepended
    chosen, _ = _to_conversations({"prompt": "Hi?", "chosen": "Hello", "rejected": "Go away"})
    assert [m["role"] for m in chosen] == ["user", "assistant"]
