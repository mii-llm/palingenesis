"""Test on-policy distillation: token bridge, prompt pool, formatting, config."""

import json
import random
import sys

import pytest

sys.path.insert(0, "src")

# ---------------------------------------------------------------------------
# Token bridge
# ---------------------------------------------------------------------------

SHARED = 128256
IM_END = 128256   # student-only ChatML terminator
EOT_ID = 128009   # teacher end-of-turn
END_OF_TEXT = 128001  # shared end-of-text


class FakeTokenizer:
    """Just enough tokenizer surface for TokenBridge/check_compatible."""

    def __init__(self, vocab_size, eos_token, tokens, byte_offset=0):
        self._vocab_size = vocab_size
        self._tokens = tokens  # name -> id
        self.eos_token = eos_token
        self.eos_token_id = tokens.get(eos_token)
        self.pad_token_id = None
        self._byte_offset = byte_offset  # shift ids to simulate a diverging tokenizer

    def __len__(self):
        return self._vocab_size

    def convert_tokens_to_ids(self, name):
        return self._tokens.get(name)

    def encode(self, text, add_special_tokens=False):
        return [(b + self._byte_offset) % 256 for b in text.encode()]


def make_pair(**student_kwargs):
    student = FakeTokenizer(
        vocab_size=SHARED + 6,
        eos_token="<|im_end|>",
        tokens={"<|im_end|>": IM_END, "<|end_of_text|>": END_OF_TEXT, "<|eot_id|>": EOT_ID},
        **student_kwargs,
    )
    teacher = FakeTokenizer(
        vocab_size=SHARED,
        eos_token="<|eot_id|>",
        tokens={"<|eot_id|>": EOT_ID, "<|end_of_text|>": END_OF_TEXT},
    )
    return student, teacher


def test_bridge_from_tokenizers_explicit_map():
    from palingenesis.opd.token_bridge import TokenBridge

    student, teacher = make_pair()
    bridge = TokenBridge.from_tokenizers(
        student, teacher,
        eos_map={"<|im_end|>": "<|eot_id|>"},
        extra_stop_tokens=("<|end_of_text|>",),
    )
    assert bridge.shared_vocab_size == SHARED
    assert bridge.swap == {IM_END: EOT_ID}
    assert set(bridge.stop_ids) == {IM_END, END_OF_TEXT}


def test_bridge_auto_eos_map():
    """Empty eos_map: student eos (outside shared vocab) maps to teacher eos."""
    from palingenesis.opd.token_bridge import TokenBridge

    student, teacher = make_pair()
    bridge = TokenBridge.from_tokenizers(student, teacher)
    assert bridge.swap == {IM_END: EOT_ID}


def test_bridge_clean_completion():
    from palingenesis.opd.token_bridge import TokenBridge

    bridge = TokenBridge(shared_vocab_size=SHARED, swap={IM_END: EOT_ID},
                         stop_ids=(IM_END, END_OF_TEXT))
    # cut at first stop token, inclusive (stopping is supervised too)
    assert bridge.clean_completion([1, 2, IM_END, 3, 4]) == [1, 2, IM_END]
    assert bridge.clean_completion([1, 2, END_OF_TEXT]) == [1, 2, END_OF_TEXT]
    # unmapped out-of-shared-vocab token truncates BEFORE it (teacher can't score it)
    assert bridge.clean_completion([1, 2, SHARED + 3, 4]) == [1, 2]
    assert bridge.clean_completion([5, 6, 7]) == [5, 6, 7]


def test_bridge_to_teacher():
    from palingenesis.opd.token_bridge import TokenBridge

    bridge = TokenBridge(shared_vocab_size=SHARED, swap={IM_END: EOT_ID},
                         stop_ids=(IM_END, END_OF_TEXT))
    assert bridge.to_teacher([1, 2, IM_END]) == [1, 2, EOT_ID]
    assert bridge.to_teacher([1, 2, END_OF_TEXT]) == [1, 2, END_OF_TEXT]


def test_check_compatible_passes_and_rejects_divergence():
    from palingenesis.opd.token_bridge import TokenBridge, TokenBridgeError, check_compatible

    student, teacher = make_pair()
    bridge = TokenBridge.from_tokenizers(student, teacher, eos_map={"<|im_end|>": "<|eot_id|>"})
    check_compatible(student, teacher, bridge)  # must not raise

    diverging, teacher = make_pair(byte_offset=1)
    with pytest.raises(TokenBridgeError, match="diverge"):
        check_compatible(diverging, teacher, bridge)


def test_check_compatible_rejects_bad_swap():
    from palingenesis.opd.token_bridge import TokenBridge, TokenBridgeError, check_compatible

    student, teacher = make_pair()
    inside_shared = TokenBridge(shared_vocab_size=SHARED, swap={100: EOT_ID}, stop_ids=(IM_END,))
    with pytest.raises(TokenBridgeError, match="inside the shared vocab"):
        check_compatible(student, teacher, inside_shared)

    no_stops = TokenBridge(shared_vocab_size=SHARED, swap={}, stop_ids=())
    with pytest.raises(TokenBridgeError, match="stop"):
        check_compatible(student, teacher, no_stops)


def test_bridge_rejects_student_smaller_than_teacher():
    from palingenesis.opd.token_bridge import TokenBridge, TokenBridgeError

    student, teacher = make_pair()
    with pytest.raises(TokenBridgeError, match="smaller"):
        TokenBridge.from_tokenizers(teacher, student)  # swapped roles


# ---------------------------------------------------------------------------
# Prompt pool
# ---------------------------------------------------------------------------

def test_question_hash_normalizes_accents_case_punctuation():
    from palingenesis.opd.pool import question_hash

    assert question_hash("Perché l'uovo?") == question_hash("perche luovo")
    assert question_hash("A") != question_hash("B")


def test_split_pool_deterministic_and_disjoint():
    from palingenesis.opd.pool import question_hash, split_pool

    rows = [{"question": f"q{i}", "options": [("A", "x"), ("B", "y")], "answer": "A"}
            for i in range(50)]
    train1, dev1 = split_pool(rows, dev_size=10, seed=0)
    train2, dev2 = split_pool(list(reversed(rows)), dev_size=10, seed=0)

    assert len(dev1) == 10 and len(train1) == 40
    # hash-ranked: same dev set regardless of input order or seed
    assert {r["question"] for r in dev1} == {r["question"] for r in dev2}
    dev_hashes = {question_hash(r["question"]) for r in dev1}
    assert all(question_hash(r["question"]) not in dev_hashes for r in train1)


def test_split_pool_with_duplicated_rows():
    """Upweighted (duplicated) pools: dev stays unique, no dev question leaks into train."""
    from palingenesis.opd.pool import question_hash, split_pool

    rows = [{"question": f"q{i}", "options": [("A", "x"), ("B", "y")], "answer": "A"}
            for i in range(30)]
    duplicated = rows + rows[:15] * 3  # upweight the first 15 questions x4
    train, dev = split_pool(duplicated, dev_size=10)

    dev_questions = [r["question"] for r in dev]
    assert len(dev_questions) == len(set(dev_questions)) == 10
    dev_hashes = {question_hash(q) for q in dev_questions}
    assert all(question_hash(r["question"]) not in dev_hashes for r in train)
    # train keeps the duplicates of non-dev questions (that's the upweighting)
    assert len(train) > 30 - 10


def test_pool_roundtrip(tmp_path):
    from palingenesis.opd.pool import load_pool, write_pool

    rows = [{"question": "q", "options": [["A", "sì"], ["B", "no"]], "answer": "A",
             "category": "storia", "source": "test"}]
    path = tmp_path / "pool.jsonl"
    assert write_pool(rows, str(path)) == 1
    loaded = load_pool(str(path))
    assert loaded[0]["options"] == [("A", "sì"), ("B", "no")]


def test_normalize_mmlu_italian():
    from palingenesis.opd.pool import normalize_mmlu_italian

    row = normalize_mmlu_italian({
        "input_translation": "Quale pianeta è il più vicino al Sole?",
        "choices_translation": ["Mercurio", "Venere", "Terra", "Marte"],
        "label": 0,
        "metadata": {"subject": "Astronomia"},
    })
    assert row["answer"] == "A"
    assert row["options"][0] == ("A", "Mercurio")
    assert row["category"] == "astronomia"
    # invalid label -> rejected
    assert normalize_mmlu_italian({"input_translation": "q", "choices_translation": ["a"], "label": 5}) is None


def test_valid_row_rejects_malformed():
    from palingenesis.opd.pool import valid_row

    opts = [("A", "x"), ("B", "y")]
    assert valid_row("q", opts, "A")
    assert not valid_row("", opts, "A")            # no question
    assert not valid_row("q", opts, "C")           # answer not among options
    assert not valid_row("q", [("A", "x")], "A")   # single option
    assert not valid_row("q", [("A", " "), ("B", "y")], "A")  # blank option text


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

ROW = {"question": "Chi scrisse la Divina Commedia?",
       "options": [("A", "Dante"), ("B", "Petrarca")],
       "answer": "A", "category": "letteratura"}


def test_fast_template_matches_benchmark_format():
    from palingenesis.opd.formatting import build_user_query

    q = build_user_query(ROW, fast=True)
    assert q.startswith("Rispondi alla seguente domanda a scelta multipla sull'argomento 'letteratura'.")
    assert "A) Dante\nB) Petrarca" in q
    assert "una tra AB" in q
    assert q.endswith("Risposta:")

    cot = build_user_query(ROW, fast=False)
    assert "'Risposta: LETTERA'" in cot
    assert not cot.endswith("Risposta:")


def test_build_messages_structure():
    from palingenesis.opd.formatting import DEFAULT_SYSTEM_MESSAGE, build_messages

    shot = dict(ROW, question="Altro quesito?")
    messages = build_messages(ROW, few_shots=[shot], fast=True)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == DEFAULT_SYSTEM_MESSAGE
    assert messages[2]["content"] == "A"  # shots answer with the bare letter

    custom = build_messages(ROW, system_message="Sei un esperto.")
    assert custom[0]["content"] == "Sei un esperto."


def test_renderer_regimes():
    from palingenesis.opd.formatting import PromptRenderer

    pool = [dict(ROW, question=f"q{i}") for i in range(30)]
    shots = [dict(ROW, question="shot")]
    renderer = PromptRenderer(pool, shots, p_reference_shots=1.0, p_pool_shots=0.0,
                              rng=random.Random(0))
    messages, row, fast = renderer.sample()
    assert fast is True  # cot_fraction=0
    assert messages[1]["content"].count("shot") == 1  # the reference shot turn

    zero = PromptRenderer(pool, [], p_reference_shots=0.0, p_pool_shots=0.0,
                          rng=random.Random(0))
    messages, _, _ = zero.sample()
    assert [m["role"] for m in messages] == ["system", "user"]

    pooled = PromptRenderer(pool, [], p_reference_shots=0.0, p_pool_shots=1.0,
                            pool_shots_max_k=3, rng=random.Random(0))
    messages, row, _ = pooled.sample()
    n_shots = sum(1 for m in messages if m["role"] == "assistant")
    assert 1 <= n_shots <= 3
    # the target row is never one of its own shots
    assert all(row["question"] not in m["content"] for m in messages[1:-1])


def test_extract_letter():
    from palingenesis.opd.formatting import extract_letter

    assert extract_letter("Risposta: B") == "B"
    assert extract_letter("A") == "A"
    assert extract_letter("nessuna lettera") is None


def test_letter_token_ids():
    from palingenesis.opd.formatting import letter_token_ids

    class SingleTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(text)]

    ids = letter_token_ids(SingleTokenizer(), letters="ABC")
    assert ids == {"A": 65, "B": 66, "C": 67}

    class MultiTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(ValueError, match="single-token"):
        letter_token_ids(MultiTokenizer(), letters="A")


def test_load_reference_shots_both_layouts(tmp_path):
    from palingenesis.opd.formatting import load_reference_shots

    path = tmp_path / "shots.jsonl"
    path.write_text(
        json.dumps({"question": "q1", "options": [{"A": "x"}, {"B": "y"}],
                    "answer": "A", "category": "c"}) + "\n" +
        json.dumps({"question": "q2", "options": [["A", "x"], ["B", "y"]],
                    "answer": "B", "category": "c"}) + "\n"
    )
    shots = load_reference_shots(str(path))
    assert shots[0]["options"] == [("A", "x"), ("B", "y")]
    assert shots[1]["options"] == [("A", "x"), ("B", "y")]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_opd_config_from_yaml_and_cli(tmp_path):
    from palingenesis.opd.config import OPDConfig

    path = tmp_path / "opd.yaml"
    path.write_text(
        "model:\n  student: my/student\n  teacher: my/teacher\n"
        "bridge:\n  eos_map:\n    '<|im_end|>': '<|eot_id|>'\n"
        "train:\n  learning_rate: 5.0e-6\n  steps: 100\n"
    )
    config = OPDConfig.from_cli(["--config", str(path), "--train.steps", "250",
                                 "--sampling.cot_fraction", "0.3",
                                 "--model.gradient_checkpointing", "true"])
    assert config.model.student == "my/student"
    assert config.bridge.eos_map == {"<|im_end|>": "<|eot_id|>"}
    assert config.train.learning_rate == 5.0e-6
    assert config.train.steps == 250  # CLI wins over YAML
    assert config.sampling.cot_fraction == 0.3
    assert config.model.gradient_checkpointing is True


def test_opd_config_validate():
    import pytest

    from palingenesis.opd.config import OPDConfig, OPDConfigError

    config = OPDConfig()
    config.data.shots_path = "shots.jsonl"
    assert config.validate() == []

    config.train.loss_fn = "nonsense"
    with pytest.raises(OPDConfigError, match="loss_fn"):
        config.validate()

    config = OPDConfig()
    config.data.p_reference_shots = 0.8
    config.data.p_pool_shots = 0.5
    with pytest.raises(OPDConfigError, match="p_reference_shots"):
        config.validate()

    config = OPDConfig()
    config.data.shots_path = "shots.jsonl"
    config.sampling.group_size = 4  # legal but useless with cot_fraction=0
    warnings = config.validate()
    assert len(warnings) == 1 and "group_size" in warnings[0]
