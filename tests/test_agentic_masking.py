"""Masking of agentic conversations: tool calls, tool definitions, per-turn flags.

An assistant turn is everything the chat template writes after its assistant header
up to the end-of-turn token, and all of it is what the model generates at inference:
text, reasoning, and tool calls, which templates render from `tool_calls` rather than
from `content`. These tests pin that the whole turn is trained, including the tool-call
markup and the end-of-turn token that teaches the model to stop after calling a tool.

Network-free: a GPT-2 fast tokenizer hosts a Qwen3-style template (tools in the system
prompt, `<tool_call>` blocks rendered from `tool_calls`, tool results wrapped in
`<tool_response>` inside a user turn, arguments iterated as a mapping). Tests against the
real Qwen templates run when those tokenizers are in the local Hugging Face cache.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.data import IGNORE_INDEX, ChatDataset, TurnMarkers, derive_turn_markers  # noqa: E402
from palingenesis.validate_data import normalize_messages  # noqa: E402

QWEN_LIKE_TEMPLATE = (
    "{%- if tools %}{{- '<|im_start|>system\\n# Tools\\n' }}"
    "{%- for tool in tools %}{{- tool | tojson }}{{- '\\n' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' %}{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if message.reasoning_content is defined and message.reasoning_content %}"
    "{{- '<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' }}{%- endif %}"
    "{{- message.content }}"
    "{%- if message.tool_calls is defined and message.tool_calls %}{%- for tc in message.tool_calls %}"
    "{{- '\\n<tool_call>\\n<function=' + tc.function.name + '>\\n' }}"
    "{%- for name, value in tc.function.arguments|items %}"
    "{{- '<parameter=' + name + '>\\n' + value|string + '\\n</parameter>\\n' }}{%- endfor %}"
    "{{- '</function>\\n</tool_call>' }}{%- endfor %}{%- endif %}"
    "{{- '<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


def _make_tok():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return None
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tok.add_tokens(["<think>", "</think>", "<tool_call>", "</tool_call>"])
    tok.chat_template = QWEN_LIKE_TEMPLATE
    tok.eos_token = "<|im_end|>"
    tok.pad_token = tok.eos_token
    return tok


TOK = _make_tok()
needs_tok = pytest.mark.skipif(TOK is None, reason="gpt2 tokenizer not cached (offline)")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

AGENT = [
    {"role": "user", "content": "Weather in Rome?"},
    {
        "role": "assistant",
        "content": "Let me check.",
        "reasoning_content": "Need the tool.",
        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}],
    },
    {"role": "tool", "content": "SUNNY_24C"},
    {"role": "assistant", "content": "It is sunny, 24C."},
]


def _trained(tok, res):
    ids, lab = res["input_ids"], res["labels"]
    return tok.decode([int(ids[i]) for i in range(len(ids)) if int(lab[i]) != IGNORE_INDEX])


def _render(tok, row):
    return tok.decode(ChatDataset(None, tok, 4096)._process(row)["input_ids"])


@needs_tok
def test_markers_derived_from_template():
    ds = ChatDataset(None, TOK, 4096)
    assert ds._turn_markers() == TurnMarkers(
        header="<|im_start|>assistant\n", end="<|im_end|>", turn_open="<|im_start|>"
    )


@needs_tok
def test_tool_call_and_end_of_turn_are_trained():
    trained = _trained(TOK, ChatDataset(None, TOK, 4096)._process({"messages": AGENT}))
    assert trained == (
        "<think>\nNeed the tool.\n</think>\n\nLet me check.\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call><|im_end|>"
        "It is sunny, 24C.<|im_end|>"
    ), trained
    assert "SUNNY" not in trained and "Weather in Rome" not in trained


@needs_tok
def test_tool_call_without_text_is_trained():
    msgs = [AGENT[0], {**AGENT[1], "content": "", "reasoning_content": ""}, AGENT[2], AGENT[3]]
    trained = _trained(TOK, ChatDataset(None, TOK, 4096)._process({"messages": msgs}))
    # (GPT-2 merges the header's "\n" with the next one into one token, which the span
    # overlaps: hence the containment check rather than startswith.)
    assert "\n<tool_call>\n<function=get_weather>" in trained and "assistant" not in trained, trained
    assert "</tool_call><|im_end|>" in trained


@needs_tok
def test_string_arguments_are_parsed_for_the_template():
    """The OpenAI wire format stores arguments as a JSON string; the template iterates
    them as a mapping and would raise on the string."""
    msgs = normalize_messages({"messages": AGENT})
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Rome"}
    assert "<parameter=city>\nRome" in _render(TOK, {"messages": AGENT})


@needs_tok
def test_tools_reach_the_system_prompt_untrained():
    for tools in (TOOLS, json.dumps(TOOLS)):  # list, or JSON string as Arrow often stores it
        res = ChatDataset(None, TOK, 4096)._process({"messages": AGENT, "tools": tools})
        assert "get_weather" in TOK.decode(res["input_ids"][:40])
        assert "# Tools" not in _trained(TOK, res)


@needs_tok
def test_tools_field_is_configurable():
    res = ChatDataset(None, TOK, 4096, tools_field="functions")._process({"messages": AGENT, "functions": TOOLS})
    assert "# Tools" in TOK.decode(res["input_ids"])


@needs_tok
def test_loss_false_turn_is_context_only():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "GREETING", "loss": False},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ANSWER", "loss": True},
    ]
    res = ChatDataset(None, TOK, 4096)._process({"messages": msgs})
    assert _trained(TOK, res) == "ANSWER<|im_end|>"
    assert "GREETING" in TOK.decode(res["input_ids"])
    # Arrow fills `loss: None` into messages without the key: that means "train".
    msgs[3]["loss"] = None
    assert _trained(TOK, ChatDataset(None, TOK, 4096)._process({"messages": msgs})) == "ANSWER<|im_end|>"


@needs_tok
def test_all_turns_flagged_off_drops_the_row():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "loss": False}]
    assert ChatDataset(None, TOK, 4096)._process({"messages": msgs}) is None


@needs_tok
def test_messages_stored_as_json_string():
    res = ChatDataset(None, TOK, 4096)._process({"messages": json.dumps(AGENT)})
    assert "It is sunny" in _trained(TOK, res)


@needs_tok
def test_observations_trained_with_echo():
    trained = _trained(TOK, ChatDataset(None, TOK, 4096, include_observations=True)._process({"messages": AGENT}))
    assert "SUNNY_24C" in trained and "<tool_call>" in trained


@needs_tok
def test_last_turn_only_with_tool_calls():
    trained = _trained(TOK, ChatDataset(None, TOK, 4096, last_turn_only=True)._process({"messages": AGENT}))
    assert trained == "It is sunny, 24C.<|im_end|>", trained


@needs_tok
def test_reasoning_masked_when_disabled_keeps_tool_call():
    trained = _trained(TOK, ChatDataset(None, TOK, 4096, train_on_reasoning=False)._process({"messages": AGENT}))
    assert "Need the tool" not in trained and "<think>" not in trained
    assert trained.startswith("Let me check.\n<tool_call>"), trained


@needs_tok
def test_truncation_keeps_whole_turns_and_ends_on_an_answer():
    long_answer = " ".join(["word"] * 300)
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": long_answer},
        {"role": "user", "content": "q3"},
    ]
    ds = ChatDataset(None, TOK, 64)
    res = ds._process({"messages": msgs})
    assert _trained(TOK, res) == "FIRST<|im_end|>"
    assert "q2" not in TOK.decode(res["input_ids"])
    assert ds.stats["truncated"] == 1


@needs_tok
def test_row_whose_first_answer_does_not_fit_is_dropped():
    msgs = [{"role": "user", "content": " ".join(["word"] * 300)}, {"role": "assistant", "content": "A"}]
    ds = ChatDataset(None, TOK, 64)
    assert ds._process({"messages": msgs}) is None
    assert ds.stats["dropped_too_long"] == 1


def _cached_tokenizer(name):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return None
    return tok if tok.chat_template else None  # a partial cache can lack the template


@pytest.mark.parametrize("name", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"])
def test_real_qwen_templates_train_tool_calls(name):
    tok = _cached_tokenizer(name)
    if tok is None:
        pytest.skip(f"{name} tokenizer not cached")
    ds = ChatDataset(None, tok, 4096)
    assert ds._turn_markers() == TurnMarkers("<|im_start|>assistant\n", "<|im_end|>", "<|im_start|>")
    res = ds._process({"messages": AGENT, "tools": TOOLS})
    trained = _trained(tok, res)
    assert "get_weather" in trained and "Rome" in trained, trained
    assert trained.count("<|im_end|>") == 2
    assert "SUNNY_24C" not in trained and "Weather in Rome" not in trained
    assert trained.endswith("It is sunny, 24C.<|im_end|>")


def test_derive_turn_markers_rejects_templates_without_end_token():
    """A plain-text template has no special end-of-turn token: no markers, so turns are
    located by their text instead."""
    if TOK is None:
        pytest.skip("gpt2 tokenizer not cached")

    def render(messages, add_generation_prompt=False, **_):
        text = "".join(f"### {m['role']}:\n{m['content']}\n\n" for m in messages)
        return text + ("### assistant:\n" if add_generation_prompt else "")

    assert derive_turn_markers(render, TOK) is None
