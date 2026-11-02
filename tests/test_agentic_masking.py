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


def test_inject_loop_pastes_an_untrained_repeat_before_the_next_gold_turn():
    from palingenesis.data import inject_loop

    assert inject_loop(AGENT, 0.0) is AGENT and inject_loop(AGENT, 1.0, seed=1) == inject_loop(AGENT, 1.0, seed=1)
    for seed in range(8):
        out = inject_loop(AGENT, 1.0, seed=seed)
        copies = (len(out) - len(AGENT)) // 2
        assert copies in (1, 2, 3) and out[:3] == AGENT[:3] and out[-1] == AGENT[-1]
        assert out[3:-1] == [{**AGENT[1], "loss": False}, AGENT[2]] * copies
    assert inject_loop(AGENT[:2], 1.0) == AGENT[:2]  # no observation and next turn: unchanged
    assert inject_loop(AGENT, 1.0)[1] is AGENT[1] and "loss" not in AGENT[1]  # the input is not modified


@needs_tok
def test_loop_recovery_trains_the_original_call_and_the_escape_only():
    res = ChatDataset(None, TOK, 4096, loop_recovery=1.0)._process({"messages": AGENT, "tools": TOOLS})
    text = TOK.decode(res["input_ids"])
    assert text.count("SUNNY_24C") >= 2  # the repeated observation is in the context
    trained = _trained(TOK, res)
    assert trained.count("<function=get_weather>") == 1 and trained.endswith("It is sunny, 24C.<|im_end|>")


REASONING_ROW = [
    {"role": "user", "content": "Weather in Rome?"},
    {
        "role": "assistant",
        "content": "Let me check.",
        "reasoning_content": "The user wants the weather in Rome. I should call the weather tool. Then I answer briefly.",
        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}],
    },
    {"role": "tool", "content": "SUNNY_24C"},
    {"role": "assistant", "content": "It is sunny, 24C."},
]
SENTENCES = ["The user wants the weather in Rome.", "I should call the weather tool."]


def test_inject_loop_repeats_a_reasoning_sentence_and_marks_the_copies():
    from palingenesis.data import UNTRAINED_REASONING, inject_loop

    for seed in range(6):
        out = inject_loop(REASONING_ROW, 0.0, seed=seed, reasoning_share=1.0)
        text = out[1]["reasoning_content"]
        [(start, end, stop)] = out[1][UNTRAINED_REASONING]
        assert text[start:end] in SENTENCES and text[end:stop] in [(" " + text[start:end]) * n for n in range(1, 5)]
        assert text[:end] + text[stop:] == REASONING_ROW[1]["reasoning_content"]  # without the copies: the gold
    assert UNTRAINED_REASONING not in REASONING_ROW[1]
    padded = [dict(m) for m in REASONING_ROW]
    padded[1]["reasoning_content"] = "\n\n  " + padded[1]["reasoning_content"]  # templates trim: never start in blanks
    for seed in range(6):
        [(start, _, _)] = inject_loop(padded, 0.0, seed=seed, reasoning_share=1.0)[1][UNTRAINED_REASONING]
        assert not padded[1]["reasoning_content"][start].isspace()


@needs_tok
def test_loop_recovery_leaves_a_row_unaugmented_when_the_copies_would_not_fit():
    row = {"messages": AGENT, "tools": TOOLS}
    n = len(ChatDataset(None, TOK, 4096)._process(row)["input_ids"])
    ds = ChatDataset(None, TOK, n + 2, loop_recovery=1.0)
    res = ds._process(row)
    assert len(res["input_ids"]) == n and ds.stats["loop_recovery_skipped"] == 1 and not ds.stats["truncated"]


@needs_tok
def test_reasoning_loop_recovery_trains_around_the_copies_only():
    res = ChatDataset(None, TOK, 4096, reasoning_loop_recovery=1.0)._process({"messages": REASONING_ROW})
    rendered, trained = TOK.decode(res["input_ids"]), _trained(TOK, res)
    assert max(rendered.count(x) for x in SENTENCES) >= 2  # the repetition is in the context
    assert all(trained.count(x) == 1 for x in SENTENCES)  # the original sentence is trained, its copies are not
    assert "Then I answer briefly." in trained and trained.endswith("It is sunny, 24C.<|im_end|>")


def _call(name, city, **extra):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": {"city": city}}}],
        **extra,
    }


def _targets(msgs, kind="repeat_targets", tok=None):
    tok = tok or TOK
    res = ChatDataset(None, tok, 4096, repeat_targets=True)._process({"messages": msgs})
    ids, rt = res["input_ids"], res[kind]
    pos = (rt != IGNORE_INDEX).nonzero().flatten().tolist()
    return [(tok.decode(ids[:p]), tok.decode([int(ids[p])]), tok.decode([int(rt[p])])) for p in pos], res


@needs_tok
def test_repeat_targets_mark_the_escape_after_a_repeated_call():
    obs = {"role": "tool", "content": "ERROR"}
    looped = [{"role": "user", "content": "Weather?"}, _call("get_weather", "Rome"), obs]
    looped += [_call("get_weather", "Rome", loss=False), obs, _call("get_time", "Milan")]
    targets, res = _targets(looped)
    [(before, gold, repeat)] = targets  # inside a loop the first divergence is the target, even the tool name
    assert before.count("get_weather") == 2 and gold != repeat and repeat.strip() in "get_weather"
    assert res["labels"][len(TOK(before)["input_ids"])] != IGNORE_INDEX  # a trained position

    # no loop: another tool right away is ordinary variety, not a repetition
    plain = [{"role": "user", "content": "Weather?"}, _call("get_weather", "Rome"), obs, _call("get_time", "Milan")]
    assert _targets(plain)[0] == []


@needs_tok
def test_repeat_targets_mark_the_escape_after_a_repeated_sentence():
    s = "The weather tool needs the city name."
    row = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "Ok.", "reasoning_content": f"{s} {s} Then I call it for Rome now."},
    ]
    [(before, gold, repeat)] = _targets(row)[0]
    assert before.endswith(f"{s} {s}") and gold.strip() == "Then" and repeat.strip() == "The"
    # different sentences: none
    row[1]["reasoning_content"] = f"{s} Then I call it for Rome now. Rome is in Italy."
    assert _targets(row)[0] == []


@needs_tok
def test_stop_targets_cover_the_ways_a_call_could_start_after_the_final_reply():
    row = [dict(m) for m in AGENT]
    row[-1] = {"role": "assistant", "content": "It is sunny, 24C.\nEnjoy the day."}
    targets, res = _targets(row, "stop_targets")
    assert _targets(row)[0] == []  # no repeat targets here
    # derived from the template (this one writes "\n<tool_call>" after text): the reply's first token, the token
    # after the separator, and the separator at the end of the turn
    assert [(gold, repeat) for _, gold, repeat in targets] == [
        ("It", "<tool_call>"),
        ("Enjoy", "<tool_call>"),
        ("<|im_end|>", "\n"),
    ]
    # chat (no tool calls before), and a row cut by truncation (its last kept turn is not the end): no targets
    assert (
        _targets([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there."}], "stop_targets")[
            0
        ]
        == []
    )
    n = len(ChatDataset(None, TOK, 4096)._process({"messages": row[:4]})["input_ids"])
    long = row[:4] + [{"role": "user", "content": "more " * 50}, {"role": "assistant", "content": "x " * 50}]
    cut = ChatDataset(None, TOK, n + 5, repeat_targets=True)._process({"messages": long})
    assert int((cut["stop_targets"] != IGNORE_INDEX).sum()) == 0


@pytest.mark.parametrize("name", ["Qwen/Qwen3.5-0.8B"])
def test_repeat_targets_ignore_call_scaffolding_on_the_qwen_template(name):
    tok = _cached_tokenizer(name)
    if tok is None:
        pytest.skip(f"{name} tokenizer not cached")

    def bash(cmd, **extra):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": cmd}}}],
            **extra,
        }

    obs = {"role": "tool", "content": "out"}
    for a, b in [("ls", "cat a"), ("cat a", "cat b"), ("cat b", "pwd")]:  # ordinary variety: no target
        assert _targets([{"role": "user", "content": "go"}, bash(a), obs, bash(b)], tok=tok)[0] == []
    # a late divergence in a long near-repeat is one
    a, b = (
        "cd /testbed && python -m pytest tests/test_parser.py -x -q",
        "cd /testbed && python -m pytest tests/test_lexer.py -x -q",
    )
    [(_, gold, repeat)] = _targets([{"role": "user", "content": "go"}, bash(a), obs, bash(b)], tok=tok)[0]
    assert gold != repeat
    # after a repeated call, the escape is a target even when it differs at once
    [(_, gold, repeat)] = _targets(
        [{"role": "user", "content": "go"}, bash("ls"), obs, bash("ls", loss=False), obs, bash("pwd")], tok=tok
    )[0]
    assert (gold, repeat) == ("pwd", "ls")


def _toy_tok(template):
    tok = _make_tok()
    tok.chat_template = template
    return tok


@needs_tok
def test_reasoning_loop_recovery_with_a_template_that_reads_message_reasoning():
    tok = _toy_tok(QWEN_LIKE_TEMPLATE.replace("reasoning_content", "reasoning"))
    row = [dict(m) for m in REASONING_ROW]
    row[1] = {k: v for k, v in row[1].items() if k != "reasoning_content"} | {
        "reasoning": REASONING_ROW[1]["reasoning_content"]
    }
    res = ChatDataset(None, tok, 4096, reasoning_loop_recovery=1.0)._process({"messages": row})
    rendered, trained = tok.decode(res["input_ids"]), _trained(tok, res)
    assert max(rendered.count(x) for x in SENTENCES) >= 2 and all(trained.count(x) == 1 for x in SENTENCES)
    assert "<function=get_weather>" in trained  # the turn keeps its loss


@needs_tok
def test_call_opener_is_derived_from_the_template_or_refused(caplog):
    from palingenesis.data import derive_call_opener

    def opener(tok):
        return derive_call_opener(lambda m, **kw: tok.apply_chat_template(m, tokenize=False, **kw), tok)

    c = opener(TOK)
    assert TOK.decode([c.opener]) == "<tool_call>" and c.separator_text == "\n"
    # calls that start with ordinary text ("{"): pushing that down would suppress replies, so no opener
    json_calls = QWEN_LIKE_TEMPLATE.replace(
        "'\\n<tool_call>\\n<function=' + tc.function.name + '>\\n'", "'\\n{\"name\": \"' + tc.function.name + '\"'"
    )
    assert opener(_toy_tok(json_calls)) is None
    # and the dataset says so once, instead of silently training without stop targets
    with caplog.at_level("WARNING"):
        ChatDataset(None, _toy_tok(json_calls), 4096, repeat_targets=True)._process({"messages": AGENT})
    assert "opener could not be derived" in caplog.text


@pytest.mark.parametrize("name", ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"])
def test_stop_targets_on_real_templates(name):
    tok = _cached_tokenizer(name)
    if tok is None:
        pytest.skip(f"{name} tokenizer not cached")
    row = [dict(m) for m in AGENT]
    row[-1] = {"role": "assistant", "content": "Done. The fix works."}
    targets = [(gold, repeat) for _, gold, repeat in _targets(row, "stop_targets", tok)[0]]
    assert targets[0] == ("Done", "<tool_call>")
    assert ("<|im_end|>", "\n\n" if "3.5" in name else "\n") in targets
    if "3.5" not in name:  # Qwen3 merges "." with the newline: that path is a target too
        assert any(gold == "." and repeat.startswith(".") for gold, repeat in targets)


@pytest.mark.parametrize("name", ["Qwen/Qwen3.5-0.8B"])
def test_repeat_target_found_when_the_text_quotes_the_command(name):
    tok = _cached_tokenizer(name)
    if tok is None:
        pytest.skip(f"{name} tokenizer not cached")
    a = "cd /testbed && python -m pytest tests/test_parser.py -x -q"
    b = "cd /testbed && python -m pytest tests/test_lexer.py -x -q"

    def bash(cmd, content=""):
        call = {"type": "function", "function": {"name": "bash", "arguments": {"command": cmd}}}
        return {"role": "assistant", "content": content, "tool_calls": [call]}

    msgs = [{"role": "user", "content": "go"}, bash(a, f"Run {a}"), {"role": "tool", "content": "out"}, bash(b)]
    assert len(_targets(msgs, tok=tok)[0]) == 1


@needs_tok
def test_repeat_targets_stay_fast_on_long_reasoning():
    import time

    long = " ".join(f"Step {i} checks the value of item number {i} in the list." for i in range(400))
    row = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok", "reasoning_content": long}]
    t0 = time.perf_counter()
    ChatDataset(None, TOK, 65536, repeat_targets=True)._process({"messages": row})
    assert time.perf_counter() - t0 < 2.0


def test_repeat_unlikelihood_needs_seco():
    from palingenesis.config import Config, ConfigError

    cfg = Config()
    cfg.data.repeat_unlikelihood = 5.0
    with pytest.raises(ConfigError, match="repeat_unlikelihood"):
        cfg.validate()
    cfg.memory.seco = True
    try:
        cfg.validate()
    except ConfigError as e:  # other settings of a default config may object; this one must not
        assert "repeat_unlikelihood" not in str(e)


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
