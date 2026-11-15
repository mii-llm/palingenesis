"""ChatDataset truncation: deciding from a prefix of the render that a conversation is too long, and
reusing the chosen cut's encoding in the masker, gives every row exactly what tokenizing the whole
render gave (the previous algorithm, kept below as the reference), across masking modes, reasoning,
tools, per-turn loss flags, repeat and stop targets, images and lengths.

Network-free: GPT-2's tokenizer under the vendored Qwen3.5 template (it drops the reasoning of turns
before the last user turn, so rows with two user turns are not rendered as prefixes), under a
prefix-consistent ChatML template, and under one that shows reasoning on the last turn only; the
real Qwen3.5 tokenizer too when it is cached.
"""

import copy
import json
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from palingenesis.data import ChatDataset, renders_turns_as_prefixes

CHATML_TOOLS = (
    "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' }}"
    "{%- if m.reasoning_content %}{{- '<think>\\n' + m.reasoning_content + '\\n</think>\\n\\n' }}{%- endif %}"
    "{{- m['content'] or '' }}"
    "{%- for c in m.tool_calls or [] %}{{- '\\n<tool_call>\\n' + (c.function | tojson) + '\\n</tool_call>' }}{%- endfor %}"
    "{{- '<|im_end|>\\n' }}{%- endfor %}"
)


def _gpt2(template: str | None = None):
    try:
        from multimodal_helpers import tiny_tokenizer

        tok = tiny_tokenizer()  # GPT-2 with the Qwen3.5 template and its special tokens
    except Exception:
        return None
    if template is not None:
        tok.chat_template = template
    return tok


def _qwen35():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
    except Exception:
        return None
    return tok if tok.chat_template else None


class Reference(ChatDataset):
    """The previous truncation: the whole render tokenized to count it, the candidate cuts tokenized
    without offsets, and only the last encoding kept for the masker."""

    def _encode(self, text):
        cached = self.__dict__.get("_last")
        if cached is not None and cached[0] == text:
            return cached[1]
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        self._last = (text, enc)
        return enc

    def _smart_truncate(self, messages):
        from palingenesis.validate_data import is_trained_message

        payload = len(json.dumps(messages, ensure_ascii=False, default=str).encode())
        tools = self._template_kwargs.get("tools")
        if tools:
            payload += len(json.dumps(tools, ensure_ascii=False, default=str).encode())
        payload += sum(image.tokens for image in self._row_images)
        if payload + 64 * len(messages) + 2048 <= self.max_seq_length:
            return messages

        def render(k):
            return self._render_chat(messages[:k], tokenize=False, add_generation_prompt=False)

        try:
            full = render(len(messages))
            n_full = len(self._encode(full)["input_ids"])
        except Exception:
            return messages
        if n_full <= self.max_seq_length:
            return messages
        chars_per_token = len(full) / max(n_full, 1)

        def fits(k):
            try:
                return len(self.tokenizer(render(k), add_special_tokens=False)["input_ids"]) <= self.max_seq_length
            except Exception:
                return False

        ends = [
            k
            for k in range(1, len(messages) + 1)
            if messages[k - 1].get("role") == "assistant" and is_trained_message(messages[k - 1])
        ]
        if not ends:
            self.stats["dropped_no_target"] += 1
            return None
        lo, hi, guess = 0, len(ends) - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                estimate = len(render(ends[mid])) / chars_per_token
            except Exception:
                estimate = float("inf")
            if estimate <= self.max_seq_length:
                guess, lo = mid, mid + 1
            else:
                hi = mid - 1
        i = max(guess, 0)
        while i >= 0 and not fits(ends[i]):
            i -= 1
        while 0 <= i < len(ends) - 1 and fits(ends[i + 1]):
            i += 1
        if i < 0:
            self.stats["dropped_too_long"] += 1
            return None
        self.stats["truncated"] += 1
        return messages[: ends[i]]


WORDS = (
    "the quick brown fox jumps over lazy dog parse value return list index error module import "
    "function class self test assert result file path read write open close data model token"
).split()
CODE = [
    "def f(x):\n    return x + 1\n",
    "for i in range(10):\n        print(i)\n",
    "class A:\n    pass\n\n\n",
    "    if a and b:\n        raise ValueError('bad')\n",
    "x = {'a': 1, 'b': [1, 2, 3]}\n",
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]},
        },
    }
    for name in ("bash", "read_file", "search")
]


def _prose(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _agent_row(rng, target_chars, variant):
    msgs = [{"role": "system", "content": "You are an agent. " + _prose(rng, 20)}, {"role": "user", "content": _prose(rng, 30)}]
    size = 0
    while size < target_chars:
        call = {"type": "function", "function": {"name": rng.choice(("bash", "read_file")), "arguments": {"arg": _prose(rng, 3)}}}
        turn = {"role": "assistant", "content": _prose(rng, rng.randint(0, 20)), "tool_calls": [call]}
        if rng.random() < 0.7:
            turn["reasoning_content"] = _prose(rng, rng.randint(5, 80))
        obs = "".join(rng.choice(CODE) for _ in range(rng.randint(2, 40)))
        msgs += [turn, {"role": "tool", "content": obs}]
        size += len(obs) + len(json.dumps(turn))
    msgs.append({"role": "assistant", "content": _prose(rng, 40), "reasoning_content": _prose(rng, 30)})
    if variant == "two_users":  # the template rewrites earlier turns once a later user turn arrives
        k = rng.randrange(3, len(msgs) - 1)
        msgs.insert(k, {"role": "user", "content": "And check the tests. " * rng.randint(1, 20)})
    elif variant == "untrained":
        for m in msgs:
            if m["role"] == "assistant" and rng.random() < 0.3:
                m["loss"] = False
    elif variant == "ends_on_tool":
        msgs = msgs[:-1]
    elif variant == "baked":
        for m in msgs:
            if m["role"] == "assistant" and m.get("reasoning_content"):
                m["content"] = "<think>\n" + m.pop("reasoning_content") + "\n</think>\n\n" + m["content"]
    return {"messages": msgs, "tools": TOOLS}


def _chat_row(rng, turns):
    msgs = []
    for _ in range(turns):
        msgs.append({"role": "user", "content": _prose(rng, rng.randint(5, 200))})
        reply = {"role": "assistant", "content": _prose(rng, rng.randint(20, 400))}
        if rng.random() < 0.6:
            reply["reasoning_content"] = _prose(rng, rng.randint(20, 400))
        msgs.append(reply)
    return {"messages": msgs, "chat_template_kwargs": {"enable_thinking": rng.random() < 0.5}}


def _rows(seed=0, n=24):
    rng = random.Random(seed)
    variants = ("plain", "two_users", "untrained", "ends_on_tool", "baked", "plain")
    rows = [_agent_row(rng, rng.choice((3000, 8000, 16000, 30000)), variants[i % len(variants)]) for i in range(n)]
    rows += [_chat_row(rng, rng.randint(2, 4)) for _ in range(n // 2)]
    return rows


SETTINGS = [
    (256, {}),
    (512, {}),
    (1024, {}),
    (2048, {}),
    (512, {"repeat_targets": True}),
    (1024, {"train_on_reasoning": False}),
    (1024, {"include_observations": True}),
    (512, {"last_turn_only": True}),
    (1024, {"turn_scaling": "sqrt"}),
    (1024, {"loop_recovery": 0.5, "reasoning_loop_recovery": 0.5, "repeat_targets": True}),
    (768, {"think_tags": ["<think>", "</think>"]}),
]


def _same(a, b):
    if a is None or b is None:
        return a is b
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def _compare(tok, rows, settings, monkeypatch, **common):
    shortcuts = []
    original = ChatDataset._chars_per_token_over_limit

    def spy(self, messages, full):
        result = original(self, messages, full)
        shortcuts.append(result is not None)
        return result

    monkeypatch.setattr(ChatDataset, "_chars_per_token_over_limit", spy)
    for max_len, options in settings:
        new = ChatDataset(None, tok, max_len, **common, **options)
        ref = Reference(None, tok, max_len, **common, **options)
        for i, row in enumerate(rows):
            got, want = new._process(copy.deepcopy(row)), ref._process(copy.deepcopy(row))
            assert _same(got, want), f"row {i} differs at max_seq_length={max_len} {options}"
        assert new.stats == ref.stats, (max_len, options)
    return sum(shortcuts)


@pytest.mark.parametrize("template", ["qwen3_5", "chatml"])
def test_truncation_from_a_prefix_matches_the_full_count(template, monkeypatch):
    tok = _gpt2(None if template == "qwen3_5" else CHATML_TOOLS)
    if tok is None:
        pytest.skip("gpt2 tokenizer not cached")
    assert _compare(tok, _rows(), SETTINGS, monkeypatch) > 20  # the prefix decided often


def test_templates_that_rewrite_history_keep_the_full_count(monkeypatch):
    from test_fallback_offsets import HISTORY_STRIP_TEMPLATE

    tok = _gpt2(HISTORY_STRIP_TEMPLATE)
    if tok is None:
        pytest.skip("gpt2 tokenizer not cached")
    assert not renders_turns_as_prefixes(lambda m: tok.apply_chat_template(m, tokenize=False))
    assert _compare(tok, _rows(1, 12), SETTINGS[:5], monkeypatch) == 0


def _mixed_density_rows(seed: int, n: int) -> list[tuple[int, dict]]:
    """(max_seq_length, row): several user turns, turns of dense words or of sparse digits."""
    rng = random.Random(seed)
    words = "alpha beta gamma delta epsilon zeta theta kappa lambda omicron sigma".split()

    def text(n, dense):
        if dense:
            return " ".join(rng.choice(words) for _ in range(n))
        return " ".join(str(rng.randrange(10)) + "," + str(rng.randrange(10)) for _ in range(n))

    rows = []
    for _ in range(n):
        max_len = rng.choice([128, 192, 256, 384])
        msgs = [{"role": "system", "content": "sys"}]
        for _ in range(rng.randint(2, 5)):
            msgs.append({"role": "user", "content": text(rng.randint(2, 120), rng.random() < 0.5)})
            for a in range(rng.randint(1, 2)):
                turn = {"role": "assistant", "content": text(rng.randint(1, 40), rng.random() < 0.5)}
                if rng.random() < 0.8:
                    turn["reasoning_content"] = text(rng.randint(1, 400), rng.random() < 0.5)
                msgs.append(turn)
                if a == 0 and rng.random() < 0.3:
                    turn["tool_calls"] = [{"type": "function", "function": {"name": "f", "arguments": {"a": "b"}}}]
                    msgs.append({"role": "tool", "content": text(rng.randint(1, 300), rng.random() < 0.5)})
        rows.append((max_len, {"messages": msgs}))
    return rows


def test_rows_whose_earlier_turns_are_rewritten_keep_the_full_count():
    """With two user turns the Qwen3.5 template drops earlier reasoning, so a longer conversation can
    have fewer tokens and the cut depends on where the search starts: on these rows an estimate from
    the prefix would pick another cut than the full count. Such rows are counted whole."""

    class Ungated(ChatDataset):  # the prefix estimate applied regardless of the user turns
        def _chars_per_token_over_limit(self, messages, full):
            return super()._chars_per_token_over_limit([m for m in messages if m["role"] != "user"], full)

    tok = _gpt2()
    if tok is None:
        pytest.skip("gpt2 tokenizer not cached")
    rows = _mixed_density_rows(1, 1308)
    ungated_differs = 0
    for max_len, row in (rows[948], rows[1234], rows[1307]):
        want = Reference(None, tok, max_len)._process(copy.deepcopy(row))
        assert _same(ChatDataset(None, tok, max_len)._process(copy.deepcopy(row)), want)
        ungated_differs += not _same(Ungated(None, tok, max_len)._process(copy.deepcopy(row)), want)
    assert ungated_differs  # the case is real


def test_real_qwen35_tokenizer(monkeypatch):
    tok = _qwen35()
    if tok is None:
        pytest.skip("Qwen/Qwen3.5-0.8B tokenizer not cached")
    assert _compare(tok, _rows(2, 18), SETTINGS[:6], monkeypatch) > 10


def test_rows_with_images(monkeypatch):
    try:
        from multimodal_helpers import tiny_image, tiny_image_processor, tiny_tokenizer

        from palingenesis.multimodal import VisionProcessor

        tok = tiny_tokenizer()
    except Exception:
        pytest.skip("gpt2 tokenizer not cached")
    vision = VisionProcessor(tiny_image_processor(), tok, "qwen3_5")
    rng = random.Random(3)
    rows = []
    for i in range(8):
        row = _agent_row(rng, 6000, "plain")
        image = {"type": "image", "image": tiny_image(24, 40, i)}
        row["messages"][1]["content"] = [{"type": "text", "text": row["messages"][1]["content"]}, image]
        tool = next(k for k in range(len(row["messages"]) - 1, 0, -1) if row["messages"][k]["role"] == "tool")
        row["messages"][tool]["content"] = [{"type": "text", "text": "screen"}, {"type": "image", "image": tiny_image(32, 16, i)}]
        rows.append(row)
    assert _compare(tok, rows, [(256, {}), (512, {}), (1024, {"repeat_targets": True})], monkeypatch, vision=vision) > 0


def test_probe_tells_prefix_consistent_templates_apart():
    from test_fallback_offsets import HISTORY_STRIP_TEMPLATE

    tok = _gpt2(CHATML_TOOLS)
    if tok is None:
        pytest.skip("gpt2 tokenizer not cached")
    qwen = _gpt2()
    for template, expected in ((CHATML_TOOLS, True), (HISTORY_STRIP_TEMPLATE, False), (qwen.chat_template, True)):
        tok.chat_template = template
        assert renders_turns_as_prefixes(lambda m: tok.apply_chat_template(m, tokenize=False)) is expected
