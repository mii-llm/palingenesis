"""Regression tests for the robust offset-based fallback masker (`_fallback_offsets`).

The nasty real-world case is a chat template that REWRITES history -- e.g. Qwen3.x drops
the `<think>` block from every assistant turn except the last one. That makes
render(messages[:k]) NOT a token-prefix of render(messages), which silently breaks the
legacy progressive masker (it emits garbage or drops the sample). The offset masker
locates each turn's text by forward-searching the FINAL render, so it is immune.

These tests are network-free and deterministic: a GPT-2 fast tokenizer (byte-level BPE,
so leading spaces merge into the first content token -- the same edge that bit Mistral)
hosts a synthetic history-stripping ChatML template that has NO `{% generation %}` span,
which forces ChatDataset onto the fallback path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.data import IGNORE_INDEX, ChatDataset  # noqa: E402

# History-stripping ChatML: only the LAST assistant turn keeps its <think> block; earlier
# ones render content only. No {% generation %} -> return_assistant_tokens_mask is empty
# -> ChatDataset falls back. Reproduces the Qwen3.x non-prefix-consistency property.
HISTORY_STRIP_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{%- set content = message['content'] %}{%- set reasoning = '' %}"
    "{%- if message.reasoning_content is defined and message.reasoning_content is string %}{%- set reasoning = message.reasoning_content %}"
    "{%- elif '</think>' in content %}{%- set reasoning = content.split('</think>')[0].split('<think>')[-1] %}{%- set content = content.split('</think>')[-1] %}{%- endif %}"
    "{%- if loop.last %}{{- '<|im_start|>assistant\\n<think>\\n' + reasoning + '\\n</think>\\n\\n' + content + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>assistant\\n' + content + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endif %}{%- endfor %}"
)


def _make_tok():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return None
    tok.add_special_tokens(
        {"additional_special_tokens": [t for t in ("<|im_start|>", "<|im_end|>", "<think>", "</think>") if t not in tok.get_vocab()]}
    )
    tok.chat_template = HISTORY_STRIP_TEMPLATE
    tok.eos_token = "<|im_end|>"
    tok.pad_token = tok.eos_token
    return tok


TOK = _make_tok()
needs_tok = pytest.mark.skipif(TOK is None, reason="gpt2 tokenizer not cached (offline)")


def _trained(res):
    ids, lab = res["input_ids"], res["labels"]
    return TOK.decode([int(ids[i]) for i in range(len(ids)) if int(lab[i]) != IGNORE_INDEX])


MULTI = [
    {"role": "system", "content": "Sei un assistente utile."},
    {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "ZEBRA"},
    {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "QUOKKA"},
    {"role": "user", "content": "Qreal"}, {"role": "assistant", "content": "FINALX"},
]


@needs_tok
def test_uses_fallback_not_fast_path():
    """The template has no {% generation %} -> the fast assistant mask must be empty."""
    enc = TOK.apply_chat_template(MULTI, tokenize=True, add_generation_prompt=False,
                                  return_assistant_tokens_mask=True, return_dict=True)
    mk = "assistant_masks" if "assistant_masks" in enc else "assistant_tokens_mask"
    assert sum(enc.get(mk, []) or []) == 0


@needs_tok
def test_offsets_path_is_taken():
    assert TOK.is_fast
    ds = ChatDataset(None, TOK, 4096)
    # the offset masker must succeed on its own (not defer to progressive)
    assert ds._fallback_offsets(MULTI) is not None


@needs_tok
def test_history_stripping_default_trains_all_turns():
    """The case the progressive masker gets WRONG: all three answers + terminators, and
    crucially NO user/system text or headers leak in."""
    ds = ChatDataset(None, TOK, 4096, last_turn_only=False)
    trained = _trained(ds._process({"messages": MULTI}))
    assert trained == "ZEBRA<|im_end|>\nQUOKKA<|im_end|>\nFINALX<|im_end|>\n", trained
    assert "user" not in trained and "system" not in trained and "Q1" not in trained


@needs_tok
def test_history_stripping_last_turn_only():
    ds = ChatDataset(None, TOK, 4096, last_turn_only=True)
    trained = _trained(ds._process({"messages": MULTI}))
    assert trained == "FINALX<|im_end|>\n", trained


@needs_tok
def test_reasoning_included_only_when_enabled():
    msgs = [
        {"role": "user", "content": "Qr"},
        {"role": "assistant", "content": "the answer is 42", "reasoning_content": "deep thought here"},
    ]
    on = _trained(ChatDataset(None, TOK, 4096, train_on_reasoning=True)._process({"messages": msgs}))
    off = _trained(ChatDataset(None, TOK, 4096, train_on_reasoning=False)._process({"messages": msgs}))
    assert on == "<think>\ndeep thought here\n</think>\n\nthe answer is 42<|im_end|>\n", on
    assert off == "the answer is 42<|im_end|>\n", off


@needs_tok
def test_leading_space_first_token_not_dropped():
    """Byte-level BPE merges a leading space into the first content token (o0 == c0-1);
    overlap-based membership must still capture the full answer (the bug that dropped
    Mistral's first letter)."""
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello world foobar"}]
    trained = _trained(ChatDataset(None, TOK, 4096)._process({"messages": msgs}))
    assert "Hello world foobar" in trained, trained


@needs_tok
def test_embedded_think_in_content():
    """Interleaved-thinking style: <think>..</think> embedded directly in content (as
    MiniMax-M2 ships it), preserved across turns. Reasoning gated by train_on_reasoning."""
    msgs = [
        {"role": "user", "content": "Qr"},
        {"role": "assistant", "content": "<think>reasoning trace</think>the final reply"},
    ]
    on = _trained(ChatDataset(None, TOK, 4096, train_on_reasoning=True)._process({"messages": msgs}))
    off = _trained(ChatDataset(None, TOK, 4096, train_on_reasoning=False)._process({"messages": msgs}))
    assert "reasoning trace" in on and "the final reply" in on
    assert "reasoning trace" not in off and "the final reply" in off


@needs_tok
def test_progressive_would_fail_here():
    """Documents WHY the offset path exists: the legacy progressive masker mangles the
    history-stripping template (proves the two paths are not equivalent on this input)."""
    ds = ChatDataset(None, TOK, 4096, last_turn_only=False)
    prog = ds._fallback_progressive(MULTI)
    prog_txt = _trained(prog) if prog is not None else ""
    # progressive leaks headers / drops answers; offsets is clean
    assert prog_txt != "ZEBRA<|im_end|>\nQUOKKA<|im_end|>\nFINALX<|im_end|>\n"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("ALL FALLBACK-OFFSET TESTS PASSED")
