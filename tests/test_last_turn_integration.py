"""End-to-end integration tests for `last_turn_only` + eval `mode`.

Unlike the mock-tokenizer unit tests, these exercise the code with the project's
REAL chat template — the one baked by the zagreus export script, which uses a
`{% generation %}` span (so `return_assistant_tokens_mask` fires → the FAST masking
path in ChatDataset, which is what the production model actually hits).

Coverage:
  * FAST path masking with the real template: default (all turns), last_turn_only
    (final turn only), single-turn no-op, think-scaffold inclusion, reasoning_content.
  * The real `build_dataloader` pipeline for each single source mode (sft + last_turn,
    sft default, pretrain) — proves the config→ChatDataset/PretrainDataset wiring.
  * MultiEvaluator scoring: pretrain (all-token CE), sft, sft+last_turn (final only),
    each single and combined.

A GPT-2 base (always cached) hosts the template so the tests are network-free and
deterministic; GPT-2 + the `{% generation %}` template yields real assistant masks.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.data import IGNORE_INDEX, ChatDataset  # noqa: E402

# ── the REAL baked template (verbatim shape from the zagreus export script) ─────
DEFAULT_SYSTEM_MESSAGE = "Sei un assistente utile."
CHAT_TEMPLATE = (
    "{%- if not messages %}{{- raise_exception('No messages provided.') }}{%- endif %}"
    "{%- if messages[0]['role'] == 'system' %}{%- set loop_messages = messages %}"
    "{%- else %}{%- set loop_messages = [{'role':'system','content':'" + DEFAULT_SYSTEM_MESSAGE + "'}] + messages %}{%- endif %}"
    "{%- for message in loop_messages %}"
    "{%- if message['role'] == 'system' %}{{- '<|im_start|>system\\n' }}{{- message['content'] | trim }}{{- '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}{{- '<|im_start|>user\\n' }}{{- message['content'] | trim }}{{- '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{%- set content = message['content'] | trim %}{%- set reasoning = '' %}"
    "{%- if message.reasoning_content is defined and message.reasoning_content is string %}{%- set reasoning = message.reasoning_content | trim %}"
    "{%- elif '</think>' in content %}{%- set reasoning = content.split('</think>')[0].split('<think>')[-1] | trim %}{%- set content = content.split('</think>')[-1] | trim %}{%- endif %}"
    "{{- '<|im_start|>assistant\\n' }}{%- generation %}{{- '<think>\\n' + reasoning + '\\n</think>\\n\\n' + content }}{{- '<|im_end|>\\n' }}{%- endgeneration %}"
    "{%- else %}{{- raise_exception('Unexpected role: ' + message['role']) }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}"
)


def _make_tokenizer():
    """GPT-2 + the real baked template + ChatML special tokens. Returns None if the
    tokenizer isn't cached (offline CI without the model)."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return None
    tok.add_special_tokens(
        {"additional_special_tokens": [x for x in ("<|im_start|>", "<|im_end|>", "<think>", "</think>") if x not in tok.get_vocab()]}
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.eos_token = "<|im_end|>"
    tok.pad_token = tok.eos_token
    return tok


TOK = _make_tokenizer()
needs_tok = pytest.mark.skipif(TOK is None, reason="gpt2 tokenizer not cached (offline)")

FIVE_SHOT = [
    {"role": "system", "content": "Sei un assistente utile."},
    {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "ZEBRA"},
    {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "QUOKKA"},
    {"role": "user", "content": "Qreal"}, {"role": "assistant", "content": "FINALX"},
]


def _trained_text(tok, res):
    ids, lab = res["input_ids"], res["labels"]
    return tok.decode([int(ids[i]) for i in range(len(ids)) if int(lab[i]) != IGNORE_INDEX])


# ── FAST path fires with the real template ──────────────────────────────────────
@needs_tok
def test_fast_path_is_actually_used():
    enc = TOK.apply_chat_template(
        FIVE_SHOT, tokenize=True, add_generation_prompt=False,
        return_assistant_tokens_mask=True, return_dict=True,
    )
    mk = "assistant_masks" if "assistant_masks" in enc else "assistant_tokens_mask"
    assert sum(enc[mk]) > 0, "real template must produce a non-empty assistant mask (fast path)"


@needs_tok
def test_fast_default_trains_all_assistant_turns():
    ds = ChatDataset(None, TOK, 4096, last_turn_only=False)
    trained = _trained_text(TOK, ds._process({"messages": FIVE_SHOT}))
    assert "ZEBRA" in trained and "QUOKKA" in trained and "FINALX" in trained


@needs_tok
def test_fast_last_turn_only_trains_final_only():
    ds = ChatDataset(None, TOK, 4096, last_turn_only=True)
    r = ds._process({"messages": FIVE_SHOT})
    trained = _trained_text(TOK, r)
    # only the final answer is trained; the exemplar answers are masked
    assert "FINALX" in trained
    assert "ZEBRA" not in trained and "QUOKKA" not in trained
    # and it's strictly fewer scored tokens than the default
    default = ChatDataset(None, TOK, 4096, last_turn_only=False)._process({"messages": FIVE_SHOT})
    assert int((r["labels"] != IGNORE_INDEX).sum()) < int((default["labels"] != IGNORE_INDEX).sum())


@needs_tok
def test_fast_think_scaffold_is_trained():
    # the model must learn to emit the empty-think scaffold before the answer,
    # matching enable_thinking=False at inference
    ds = ChatDataset(None, TOK, 4096, last_turn_only=True)
    trained = _trained_text(TOK, ds._process({"messages": FIVE_SHOT}))
    assert "</think>" in trained


@needs_tok
def test_fast_train_on_reasoning_false_strips_think():
    """FAST path (template's {% generation %} span encloses <think>): train_on_reasoning
    must be honored here too -- False strips the reasoning, True keeps it. Regression for
    the bug where the flag was silently ignored on the fast path."""
    msgs = [{"role": "user", "content": "Q"},
            {"role": "assistant", "content": "<think>because 6*7=42</think>The answer is 42"}]
    on = _trained_text(TOK, ChatDataset(None, TOK, 4096, train_on_reasoning=True)._process({"messages": msgs}))
    off = _trained_text(TOK, ChatDataset(None, TOK, 4096, train_on_reasoning=False)._process({"messages": msgs}))
    assert on == "<think>\nbecause 6*7=42\n</think>\n\nThe answer is 42<|im_end|>\n", on
    assert off == "The answer is 42<|im_end|>\n", off


@needs_tok
def test_fast_train_on_reasoning_false_multiturn_and_last_turn():
    """The strip only touches trained tokens, so it composes with last_turn_only and
    leaves each answer intact across turns."""
    msgs = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "first", "reasoning_content": "r1"},
            {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "final", "reasoning_content": "r2"}]
    default = _trained_text(TOK, ChatDataset(None, TOK, 4096, train_on_reasoning=False)._process({"messages": msgs}))
    last = _trained_text(TOK, ChatDataset(None, TOK, 4096, train_on_reasoning=False, last_turn_only=True)._process({"messages": msgs}))
    assert default == "first<|im_end|>\nfinal<|im_end|>\n", default
    assert last == "final<|im_end|>\n", last
    assert "r1" not in default and "r2" not in default


@needs_tok
def test_fast_single_turn_is_noop():
    single = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "ONLY"}]
    a = _trained_text(TOK, ChatDataset(None, TOK, 4096, last_turn_only=False)._process({"messages": single}))
    b = _trained_text(TOK, ChatDataset(None, TOK, 4096, last_turn_only=True)._process({"messages": single}))
    assert a == b and "ONLY" in b


@needs_tok
def test_fast_reasoning_content_kept_on_last_turn():
    msgs = [
        {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "SHOT"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "ANSWER", "reasoning_content": "BECAUSE_REASON"},
    ]
    ds = ChatDataset(None, TOK, 4096, last_turn_only=True)
    trained = _trained_text(TOK, ds._process({"messages": msgs}))
    assert "ANSWER" in trained and "BECAUSE_REASON" in trained and "SHOT" not in trained


# ── the REAL build_dataloader pipeline, per single source mode ──────────────────
def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def _one_batch(cfg):
    from palingenesis.data import build_dataloader

    dl = build_dataloader(cfg, TOK, cfg, rank=0, world_size=1, batch_size=1)
    return next(iter(dl))


def _batch_trained_text(batch):
    ids, lab = batch["input_ids"][0], batch["labels"][0]
    return TOK.decode([int(ids[i]) for i in range(len(ids)) if int(lab[i]) != IGNORE_INDEX])


@needs_tok
def test_pipeline_sft_last_turn_only(tmp_path):
    from palingenesis.config import DataConfig

    p = tmp_path / "sft.jsonl"
    _write(p, [{"messages": FIVE_SHOT}] * 3)
    cfg = DataConfig(
        sources=[{"dataset": str(p), "split": "train", "mode": "sft", "messages_field": "messages", "last_turn_only": True}],
        max_seq_length=512, packing=False, num_workers=0, length_group_buffer=0, streaming=True,
    )
    trained = _batch_trained_text(_one_batch(cfg))
    assert "FINALX" in trained and "ZEBRA" not in trained and "QUOKKA" not in trained


@needs_tok
def test_pipeline_sft_default_all_turns(tmp_path):
    from palingenesis.config import DataConfig

    p = tmp_path / "sft.jsonl"
    _write(p, [{"messages": FIVE_SHOT}] * 3)
    cfg = DataConfig(
        sources=[{"dataset": str(p), "split": "train", "mode": "sft", "messages_field": "messages"}],
        max_seq_length=512, packing=False, num_workers=0, length_group_buffer=0, streaming=True,
    )
    trained = _batch_trained_text(_one_batch(cfg))
    assert "ZEBRA" in trained and "QUOKKA" in trained and "FINALX" in trained


@needs_tok
def test_pipeline_pretrain_mode_all_tokens(tmp_path):
    from palingenesis.config import DataConfig

    p = tmp_path / "pt.jsonl"
    _write(p, [{"text": "Roma è la capitale d'Italia."}] * 3)
    cfg = DataConfig(
        sources=[{"dataset": str(p), "split": "train", "mode": "pretrain", "text_field": "text"}],
        max_seq_length=512, packing=False, num_workers=0, length_group_buffer=0, streaming=True,
    )
    batch = _one_batch(cfg)
    lab = batch["attention_mask"][0]
    # pretrain = all real (non-pad) tokens get loss
    n_labeled = int((batch["labels"][0] != IGNORE_INDEX).sum())
    assert n_labeled == int(lab.sum()) and n_labeled > 0


# ── MultiEvaluator: modes + last_turn scoring ───────────────────────────────────
class _TinyModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, 16)
        self.head = nn.Linear(16, vocab)

    def forward(self, input_ids, attention_mask=None, **kw):
        return type("O", (), {"logits": self.head(self.embed(input_ids))})()


@needs_tok
def test_eval_sft_last_turn_only_scores_final_only(tmp_path):
    from palingenesis.multi_eval import MultiEvaluator

    p = tmp_path / "eval.jsonl"
    _write(p, [{"messages": FIVE_SHOT}] * 2)

    def scored_tokens(last_turn_only):
        ev = MultiEvaluator(
            [{"name": "mcqa", "dataset": str(p), "split": "train", "mode": "sft",
              "messages_field": "messages", "last_turn_only": last_turn_only, "weight": 1.0, "samples": 2}],
            TOK, max_seq_length=512, device=torch.device("cpu"),
        )
        assert ev.sources[0]["batches"], "no eval batches built"
        return sum(int((b["labels"] != IGNORE_INDEX).sum()) for b in ev.sources[0]["batches"])

    assert scored_tokens(True) < scored_tokens(False)


@needs_tok
def test_eval_combined_pretrain_and_sft(tmp_path):
    from palingenesis.multi_eval import MultiEvaluator

    lm = tmp_path / "lm.jsonl"
    mcq = tmp_path / "mcq.jsonl"
    _write(lm, [{"text": "La fisica descrive la natura."}] * 3)
    _write(mcq, [{"messages": FIVE_SHOT}] * 2)

    sources = [
        {"name": "lm", "dataset": str(lm), "split": "train", "mode": "pretrain", "text_field": "text", "weight": 0.4, "samples": 3},
        {"name": "mcqa", "dataset": str(mcq), "split": "train", "mode": "sft", "messages_field": "messages", "last_turn_only": True, "weight": 0.6, "samples": 2},
    ]
    ev = MultiEvaluator(sources, TOK, max_seq_length=512, device=torch.device("cpu"))
    assert len(ev.sources) == 2 and all(s["batches"] for s in ev.sources)

    result = ev.evaluate(_TinyModel(len(TOK)), dtype=torch.float32)
    assert "lm" in result.per_source and "mcqa" in result.per_source
    assert result.score > 0 and result.tokens_total > 0
    # composite is the weighted mean of the per-source losses
    expected = 0.4 * result.per_source["lm"] + 0.6 * result.per_source["mcqa"]
    assert abs(result.score - expected) < 1e-4
