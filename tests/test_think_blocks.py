"""Assistant content with its <think> block baked in: one block rendered and trained, never
an injected empty block before it; <think> tags later in an answer are text, trained as such;
templates that do not render reasoning keep the baked content verbatim."""

import sys

import pytest

sys.path.insert(0, "src")

from palingenesis.validate_data import normalize_messages, restore_baked_think, split_leading_think  # noqa: E402

transformers = pytest.importorskip("transformers")


def test_split_leading_think():
    assert split_leading_think("<think>\nplan\n</think>\n\nIt is 4.") == ("plan", "It is 4.")
    assert split_leading_think("  <think></think>\n\nok") == ("", "ok")
    assert split_leading_think("You write <think> then </think>.") == (None, "You write <think> then </think>.")
    assert split_leading_think("<think>unclosed") == (None, "<think>unclosed")


def assistant(content, **extra):
    return normalize_messages({"messages": [{"role": "user", "content": "q"},
                                            {"role": "assistant", "content": content, **extra}]})[1]


def test_normalization():
    m = assistant("<think>\nplan\n</think>\n\nIt is 4.")
    assert (m["reasoning_content"], m["content"]) == ("plan", "It is 4.")
    m = assistant("<think>\nbaked\n</think>\n\nIt is 4.", reasoning_content="explicit")
    assert (m["reasoning_content"], m["content"]) == ("explicit", "It is 4.")        # explicit wins, no second block
    m = assistant("You write <think> then </think>.")
    assert m["content"] == "You write <think> then </think>." and m["reasoning_content"] == ""
    m = assistant("<think>\n\n</think>\n\nIt is 4.")
    assert m["content"] == "It is 4." and not m.get("reasoning_content")
    restored = restore_baked_think([assistant("<think>\nplan\n</think>\n\nIt is 4.")])[0]
    assert restored["content"] == "<think>\nplan\n</think>\n\nIt is 4." and "reasoning_content" not in restored


ROWS = {
    "baked": ([{"role": "user", "content": "2+2?"},
               {"role": "assistant", "content": "<think>\nadd them\n</think>\n\nIt is 4."}],
              "<think>\nadd them\n</think>\n\nIt is 4."),
    "literal": ([{"role": "user", "content": "how?"},
                 {"role": "assistant", "content": "You write <think> then </think>."}],
                "You write <think> then </think>."),
}


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM2-135M-Instruct"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("row", ROWS)
def test_rendered_and_trained(model, thinking, row):
    from palingenesis.data import ChatDataset

    try:
        tok = transformers.AutoTokenizer.from_pretrained(model)
    except Exception as e:  # noqa: BLE001 — offline
        pytest.skip(f"tokenizer unavailable: {e}")
    messages, answer = ROWS[row]
    ex = next(iter(ChatDataset([{"messages": messages, "chat_template_kwargs": {"enable_thinking": thinking}}],
                               tok, 4096)))
    text = tok.decode(ex["input_ids"])
    trained = tok.decode([t for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100])
    assert "</think>\n\n<think>" not in text and "<think>\n\n</think>\n\n<think>" not in text   # never two blocks
    assert trained.startswith(answer) or answer in trained                                       # all of the answer
    if row == "literal":
        assert "You write <think> then </think>." in trained                                     # tags are text
