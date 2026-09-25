"""OPD alignment: token bytes, byte-chunk alignment across real tokenizer pairs.

Real tokenizers (Qwen3, Qwen3.5, SmolLM2: byte-level BPE with 151k/248k/49k
vocabularies; Mistral v0.3: SentencePiece with byte fallback) are loaded from the
Hugging Face cache or hub; a pair that cannot be loaded is skipped.
"""

import functools
import sys

import pytest

sys.path.insert(0, "src")

from palingenesis.opd.align import ByteChunkAligner, _is_byte_level, align_chunks, token_bytes, vocab_map  # noqa: E402

QWEN3 = "Qwen/Qwen3-0.6B"
QWEN35 = "Qwen/Qwen3.5-0.8B"
SMOL = "HuggingFaceTB/SmolLM2-360M-Instruct"
MISTRAL = "unsloth/mistral-7b-instruct-v0.3"

TEXTS = [
    "The answer is 42.",
    "Perché città, così — l'aquila d'oro? È già qui: naïve café. 日本語のテキスト、中文。",
    "Emoji: 🙂👍🏽 and flags 🇮🇹, a family 👨‍👩‍👧 and a snake 🐍!",
    "def f(x):\n\tif x:\n        return [n**2 for n in range(10)]  # comment\n\n\n    pass",
    "Digits 1234567890 and 3.14159, -2e-10, 1,000,000 and $12.50.",
    "  leading and trailing spaces   \n\n",
    "Answer: <think>not special in Qwen3</think> done",
]


@functools.cache
def tokenizer(name):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception as e:  # noqa: BLE001 — offline or not cached
        pytest.skip(f"tokenizer {name} unavailable: {e}")


@functools.cache
def aligner(student, teacher, mask_whitespace=True):
    s, t = tokenizer(student), tokenizer(teacher)
    stop = s.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in s.get_vocab() else s.eos_token_id
    t_eot = t.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in t.get_vocab() else t.eos_token_id
    return ByteChunkAligner(s, t, (stop,), t_eot, mask_whitespace=mask_whitespace)


def check_chunks(al, completion, view):
    """Every chunk spells the same bytes on both sides, and chunks tile the text.

    A SentencePiece teacher spells its first token with the prefix space it adds
    to every text ("▁The" for "The"); the offsets still place it on "The".
    """
    ch = view.chunks
    teacher_ids = view.input_ids[view.prompt_len :]
    assert len(ch.student) == len(completion) and len(ch.teacher) == len(teacher_ids)
    s_text = {c: b"" for c in range(ch.n_chunks)}
    t_text = {c: b"" for c in range(ch.n_chunks)}
    for token, c in zip(completion, ch.student):
        if c >= 0:
            s_text[c] += al.student_bytes[token] or b""
    for token, c in zip(teacher_ids, ch.teacher):
        assert c >= 0, "every teacher completion token belongs to a chunk"
        t_text[c] += al.teacher_bytes[token] or b""
    for c in range(ch.n_chunks):
        prefix_space = c == 0 and not _is_byte_level(al.teacher_tok)
        assert s_text[c] == (t_text[c][1:] if prefix_space and not s_text[c].startswith(b" ") else t_text[c]), (
            c,
            s_text[c],
            t_text[c],
        )
        if ch.keep[c] and s_text[c]:
            assert not s_text[c].isspace()
    # chunk ids increase along both sequences
    for seq in (ch.student, ch.teacher):
        ids = [c for c in seq if c >= 0]
        assert ids == sorted(ids)
    for s_pos, t_pos in ch.one_to_one:
        assert ch.student[s_pos] == ch.teacher[t_pos]
        assert ch.student.count(ch.student[s_pos]) == 1 and ch.teacher.count(ch.teacher[t_pos]) == 1
    return s_text


def test_align_chunks_two_pointer():
    # "ab|c|de" vs "a|bc|d|e": common ends at 3 and 5
    assert align_chunks([2, 3, 5], [1, 3, 4, 5]) == [(range(0, 2), range(0, 2)), (range(2, 3), range(2, 4))]
    assert align_chunks([1, 2], [1, 2]) == [(range(0, 1), range(0, 1)), (range(1, 2), range(1, 2))]
    assert align_chunks([], []) == []
    # fractional ends (a teacher token inside a character) never close a chunk
    assert align_chunks([4], [2.5, 4]) == [(range(0, 1), range(0, 2))]


@pytest.mark.parametrize("name", [QWEN3, QWEN35, SMOL, MISTRAL])
def test_token_bytes_spell_the_text(name):
    tok = tokenizer(name)
    table = token_bytes(tok)
    assert len(table) == len(tok)
    for text in TEXTS:
        ids = tok.encode(text, add_special_tokens=False)
        spelled = b"".join(table[i] for i in ids)
        if name == MISTRAL:  # SentencePiece prefixes a space to the text
            assert spelled.lstrip(b" ") == text.encode().lstrip(b" ")
        else:
            assert spelled == text.encode()
    assert all(table[i] is None for i in tok.all_special_ids)


def test_vocab_map_matches_spellings():
    s, t = tokenizer(QWEN3), tokenizer(SMOL)
    s_bytes, t_bytes = token_bytes(s), token_bytes(t)
    mapping = vocab_map(s_bytes, t_bytes)
    assert len(mapping) == len(t)
    mapped = [i for i, m in enumerate(mapping) if m >= 0]
    assert len(mapped) > 0.5 * len(t)  # most of SmolLM2's pieces exist in Qwen3's vocabulary
    for i in mapped[:: max(1, len(mapped) // 500)]:
        assert s_bytes[mapping[i]] == t_bytes[i]
    the = t.encode(" the", add_special_tokens=False)
    assert len(the) == 1 and s_bytes[mapping[the[0]]] == b" the"


PAIRS = [(QWEN3, QWEN35), (QWEN3, SMOL), (SMOL, QWEN3), (QWEN35, QWEN3), (QWEN3, MISTRAL)]


@pytest.mark.parametrize("student,teacher", PAIRS)
@pytest.mark.parametrize("text", TEXTS)
def test_byte_chunks_on_real_pairs(student, teacher, text):
    al = aligner(student, teacher)
    s = tokenizer(student)
    completion = s.encode(text, add_special_tokens=False) + [al.stop_ids[0]]
    view = al.view([1, 2, 3], completion)
    assert view.input_ids[:3] == [1, 2, 3] and view.prompt_len == 3
    s_text = check_chunks(al, completion, view)
    assert b"".join(s_text[c] for c in range(view.chunks.n_chunks)).lstrip(b" ") == text.encode().lstrip(b" ")
    # the stop token pairs with the teacher's end of turn as the last one-to-one chunk
    assert view.input_ids[-1] == al.teacher_eot
    assert view.chunks.one_to_one[-1] == (len(completion) - 1, view.completion_len - 1)
    assert view.chunks.keep[-1]


def test_non_canonical_student_tokenization():
    """Sampled ids that differ from re-encoding the text: byte offsets come from the ids."""
    al = aligner(QWEN3, SMOL)
    s = tokenizer(QWEN3)
    text = "hello world, unbelievable 🙂"
    canonical = s.encode(text, add_special_tokens=False)
    by_char = [t for ch in text for t in s.encode(ch, add_special_tokens=False)]  # one piece per character
    assert by_char != canonical and s.decode(by_char) == text
    view = al.view([], by_char)
    check_chunks(al, by_char, view)
    t = tokenizer(SMOL)
    assert view.input_ids == t.encode(text, add_special_tokens=False)  # the teacher sees its own canonical ids
    assert all(c >= 0 for c in view.chunks.student)
    # the student sampled " " and "🙂" separately; SmolLM2's first piece of " 🙂" spans the space and
    # half the emoji, so both student tokens fall in one chunk
    assert view.chunks.student[-2] == view.chunks.student[-1]
    assert view.chunks.n_chunks < len(by_char)


def test_truncated_utf8_and_mid_completion_specials():
    al = aligner(QWEN3, QWEN35)
    s = tokenizer(QWEN3)
    head = s.encode("ok ", add_special_tokens=False)
    emoji = [s.convert_tokens_to_ids(p) for p in ["ð", "Ł", "Ļ", "Ĥ"]]  # 🙂 byte by byte
    im_start = s.convert_tokens_to_ids("<|im_start|>")
    # completion hit the length limit in the middle of the emoji: its bytes are left out
    completion = head + [im_start] + emoji[:2]
    view = al.view([], completion)
    check_chunks(al, completion, view)
    assert view.chunks.student[len(head)] == -1  # the special token carries no text
    assert view.chunks.student[-2:] == [-1, -1]  # incomplete UTF-8 at the end
    assert tokenizer(QWEN35).decode(view.input_ids) == "ok "
    # the complete emoji aligns
    view = al.view([], head + emoji)
    check_chunks(al, head + emoji, view)
    assert tokenizer(QWEN35).decode(view.input_ids) == "ok 🙂"


def test_clean_cuts_at_stop_and_before_ids_outside_the_tokenizer():
    al = aligner(QWEN3, SMOL)
    stop = al.stop_ids[0]
    assert al.clean([5, 6, stop, 7]) == [5, 6, stop]
    assert al.clean([5, 6]) == [5, 6]
    # Qwen3's embedding has 151936 rows for 151669 tokens: a sampled padding row ends the text
    assert al.clean([5, 151700, 6, stop]) == [5]


def test_whitespace_chunks_are_masked():
    al = aligner(QWEN3, SMOL)
    s = tokenizer(QWEN3)
    text = "a\n\n    b"
    completion = s.encode(text, add_special_tokens=False)
    view = al.view([], completion)
    s_text = check_chunks(al, completion, view)
    for c, kept in enumerate(view.chunks.keep):
        assert kept == (not s_text[c].isspace())
    assert not all(view.chunks.keep)
    unmasked = aligner(QWEN3, SMOL, mask_whitespace=False).view([], completion)
    assert all(unmasked.chunks.keep)


def test_identical_tokenizers_align_one_to_one():
    al = aligner(QWEN3, QWEN3)
    s = tokenizer(QWEN3)
    for text in TEXTS:
        completion = s.encode(text, add_special_tokens=False)
        view = al.view([], completion)
        assert view.input_ids == completion
        assert view.chunks.student == view.chunks.teacher == list(range(len(completion)))
