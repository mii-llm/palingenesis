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
    return normalize_messages(
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": content, **extra}]}
    )[1]


def test_normalization():
    m = assistant("<think>\nplan\n</think>\n\nIt is 4.")
    assert (m["reasoning_content"], m["content"]) == ("plan", "It is 4.")
    m = assistant("<think>\nbaked\n</think>\n\nIt is 4.", reasoning_content="explicit")
    assert (m["reasoning_content"], m["content"]) == ("explicit", "It is 4.")  # explicit wins, no second block
    m = assistant("You write <think> then </think>.")
    assert m["content"] == "You write <think> then </think>." and m["reasoning_content"] == ""
    m = assistant("<think>\n\n</think>\n\nIt is 4.")
    assert m["content"] == "It is 4." and not m.get("reasoning_content")
    restored = restore_baked_think([assistant("<think>\nplan\n</think>\n\nIt is 4.")])[0]
    assert restored["content"] == "<think>\nplan\n</think>\n\nIt is 4." and "reasoning_content" not in restored


ROWS = {
    "baked": (
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>\nadd them\n</think>\n\nIt is 4."},
        ],
        "<think>\nadd them\n</think>\n\nIt is 4.",
    ),
    "literal": (
        [{"role": "user", "content": "how?"}, {"role": "assistant", "content": "You write <think> then </think>."}],
        "You write <think> then </think>.",
    ),
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
    ex = next(
        iter(ChatDataset([{"messages": messages, "chat_template_kwargs": {"enable_thinking": thinking}}], tok, 4096))
    )
    text = tok.decode(ex["input_ids"])
    trained = tok.decode([t for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100])
    assert "</think>\n\n<think>" not in text and "<think>\n\n</think>\n\n<think>" not in text  # never two blocks
    assert trained.startswith(answer) or answer in trained  # all of the answer
    if row == "literal":
        assert "You write <think> then </think>." in trained  # tags are text


# A template whose reasoning delimiters are not <think></think> (Magistral-style).
BRACKET_TEMPLATE = (
    "{% for m in messages %}{% if m.role == 'user' %}<|im_start|>user\n{{ m.content }}<|im_end|>\n"
    "{% else %}<|im_start|>assistant\n{% if m.reasoning_content %}[THINK]{{ m.reasoning_content }}[/THINK]"
    "{% endif %}{{ m.content }}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _tokenizer(model="Qwen/Qwen3-0.6B", template=None):
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model)
    except Exception as e:  # noqa: BLE001 — offline
        pytest.skip(f"tokenizer unavailable: {e}")
    if template is not None:
        tok.chat_template = template
    return tok


def _trained(tok, rows, **kwargs):
    from palingenesis.data import ChatDataset

    out = []
    for ex in ChatDataset(rows, tok, 4096, **kwargs):
        ids, labels = ex["input_ids"], ex["labels"]
        out.append((tok.decode(ids), tok.decode([t for t, lab in zip(ids, labels) if lab != -100])))
    return out


def test_detect_think_tags():
    from palingenesis.data import detect_think_tags

    def render_with(tok):
        return lambda m, **kw: tok.apply_chat_template(m, tokenize=False, **kw)

    assert detect_think_tags(render_with(_tokenizer())) == ("<think>", "</think>")
    assert detect_think_tags(render_with(_tokenizer(template=BRACKET_TEMPLATE))) == ("[THINK]", "[/THINK]")
    assert detect_think_tags(render_with(_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct"))) is None


def test_custom_tags_in_data_and_template():
    """Data baked with the template's own [THINK] tags: one block, parsed without configuration."""
    tok = _tokenizer(template=BRACKET_TEMPLATE)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "[THINK]add them[/THINK]It is 4."},
            ]
        }
    ]
    [(text, trained)] = _trained(tok, rows)
    assert text.count("[THINK]") == 1 and "[THINK]add them[/THINK]It is 4.<|im_end|>" in text
    assert trained.startswith("[THINK]add them[/THINK]It is 4.")
    [(_, trained)] = _trained(tok, rows, train_on_reasoning=False)
    assert "add them" not in trained and "It is 4." in trained


def test_data_tags_converted_to_template_format():
    """Data written with another model's tags (think_tags) renders in this template's format."""
    tok = _tokenizer()  # Qwen3: <think></think>
    rows = [
        {
            "messages": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "◁think▷add them◁/think▷It is 4."},
            ]
        }
    ]
    [(text, trained)] = _trained(tok, rows, think_tags=["◁think▷", "◁/think▷"])
    assert "◁think▷" not in text and "<think>\nadd them\n</think>\n\nIt is 4." in text
    assert "add them" in trained
    [(text, _)] = _trained(tok, rows)  # unconfigured: text, verbatim
    assert "◁think▷add them◁/think▷It is 4." in text


def test_mixed_thinking_rows():
    """Dataset-level chat_template_kwargs apply to every row; a row's own override them."""
    tok = _tokenizer("Qwen/Qwen3.5-0.8B")
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "<think>\nplan\n</think>\n\nA."},
            ]
        },
        {
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "B."}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ]
    ds_kwargs = {"chat_template_kwargs": {"enable_thinking": True}}
    from palingenesis.data import ChatDataset

    ds = ChatDataset(rows, tok, 4096, **ds_kwargs)
    seen = []
    orig = ds._render_chat
    ds._render_chat = lambda m, **kw: (seen.append(dict(ds._template_kwargs)), orig(m, **kw))[1]
    out = list(ds)
    assert len(out) == 2
    assert {"enable_thinking": True} in seen and {"enable_thinking": False} in seen
    texts = [tok.decode(ex["input_ids"]) for ex in out]
    assert texts[0].count("<think>") == 1 and "plan" in texts[0]


def test_think_tags_validation():
    from palingenesis.validate_data import valid_think_tags

    assert valid_think_tags(["[THINK]", "[/THINK]"])
    assert not valid_think_tags(["<think>"]) and not valid_think_tags(["x", "x"]) and not valid_think_tags(["", "y"])
