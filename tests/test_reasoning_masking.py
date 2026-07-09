"""Tests for reasoning-trace masking in ChatDataset's fallback path.

Reproduces the bug found on Qwen3.5 + reasoning datasets: the chat template
has no {% generation %} markers, so masking uses the progressive-tokenization
fallback. That fallback located "content start" by rendering a stub message
with content="" — but the stub kept reasoning_content, so the <think> block
was rendered in the stub and the trained region started AFTER </think>.
Result: reasoning traces silently received no loss.

Uses a fake Qwen-style tokenizer (char-level ids) that renders
reasoning_content as a <think>...</think> block, like Qwen3.5 does.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.data import ChatDataset, IGNORE_INDEX  # noqa: E402


class FakeReasoningTokenizer:
    """Minimal chat-template tokenizer: char-level ids, Qwen-style rendering.

    Renders:  <|role|>{<think>REASONING</think>}CONTENT<|end|>
    No {% generation %} support → ChatDataset must use the fallback path
    (we call _fallback directly to pin the code path under test).
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        parts = []
        for m in messages:
            seg = f"<|{m['role']}|>"
            if m["role"] == "assistant" and m.get("reasoning_content"):
                seg += f"<think>{m['reasoning_content']}</think>"
            seg += (m.get("content") or "") + "<|end|>"
            parts.append(seg)
        text = "".join(parts)
        if add_generation_prompt:
            text += "<|assistant|>"
        assert not tokenize, "test tokenizer only renders text"
        return text

    def __call__(self, text, truncation=False, max_length=None, return_tensors=None):
        ids = [ord(c) for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4.", "reasoning_content": "Let me add: 2+2=4."},
]


def _make_ds(train_on_reasoning: bool) -> ChatDataset:
    return ChatDataset(
        dataset=None,  # not used: we call _fallback directly
        tokenizer=FakeReasoningTokenizer(),
        max_seq_length=4096,
        train_on_reasoning=train_on_reasoning,
    )


def _trained_text(result: dict) -> str:
    """Reconstruct the text of positions that receive loss."""
    ids = result["input_ids"]
    labels = result["labels"]
    return "".join(chr(ids[i].item()) for i in range(len(ids)) if labels[i].item() != IGNORE_INDEX)


def test_reasoning_trained_by_default():
    """train_on_reasoning=True (default): the <think> block gets loss."""
    result = _make_ds(train_on_reasoning=True)._fallback(MESSAGES)
    assert result is not None
    trained = _trained_text(result)

    assert "Let me add: 2+2=4." in trained, f"Reasoning must be trained, got: {trained!r}"
    assert "The answer is 4." in trained, f"Response must be trained, got: {trained!r}"
    assert "What is 2+2?" not in trained, "User turn must stay masked"
    assert "You are helpful." not in trained, "System turn must stay masked"
    print(f"  trained: {trained!r}")
    print("✓ test_reasoning_trained_by_default PASSED")


def test_reasoning_masked_when_disabled():
    """train_on_reasoning=False: only the post-</think> response gets loss."""
    result = _make_ds(train_on_reasoning=False)._fallback(MESSAGES)
    assert result is not None
    trained = _trained_text(result)

    assert "Let me add" not in trained, f"Reasoning must be masked, got: {trained!r}"
    assert "The answer is 4." in trained, f"Response must still be trained, got: {trained!r}"
    print(f"  trained: {trained!r}")
    print("✓ test_reasoning_masked_when_disabled PASSED")


def test_reasoning_only_message():
    """Assistant turn with reasoning but empty content: trained when enabled,
    dropped (no valid tokens) when disabled."""
    msgs = [
        {"role": "user", "content": "Think about it."},
        {"role": "assistant", "content": "", "reasoning_content": "Deep thoughts here."},
    ]

    result_on = _make_ds(train_on_reasoning=True)._fallback(msgs)
    assert result_on is not None
    assert "Deep thoughts here." in _trained_text(result_on)

    result_off = _make_ds(train_on_reasoning=False)._fallback(msgs)
    # Nothing left to train → sample is dropped entirely
    assert result_off is None or "Deep thoughts" not in _trained_text(result_off)
    print("✓ test_reasoning_only_message PASSED")


def test_plain_messages_unaffected():
    """No reasoning_content anywhere: both settings behave identically."""
    msgs = [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello there."},
    ]
    on = _make_ds(train_on_reasoning=True)._fallback(msgs)
    off = _make_ds(train_on_reasoning=False)._fallback(msgs)
    assert on is not None and off is not None
    assert torch.equal(on["labels"], off["labels"]), "Setting must be a no-op without reasoning"
    assert "Hello there." in _trained_text(on)
    print("✓ test_plain_messages_unaffected PASSED")


def test_config_field_default():
    from palingenesis.config import Config

    cfg = Config()
    assert cfg.data.train_on_reasoning is True, "Training on reasoning must be the default"
    print("✓ test_config_field_default PASSED")


if __name__ == "__main__":
    test_reasoning_trained_by_default()
    test_reasoning_masked_when_disabled()
    test_reasoning_only_message()
    test_plain_messages_unaffected()
    test_config_field_default()
    print("\nALL REASONING-MASKING TESTS PASSED ✓")
