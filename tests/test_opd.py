"""Test on-policy distillation: token bridge, prompt pool, formatting, sources, config."""

import collections
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

CONFIGS = Path(__file__).parent.parent / "configs"

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
    bridge = TokenBridge.from_tokenizers(student, teacher, eos_map={"<|im_end|>": "<|eot_id|>"})
    assert bridge.shared_vocab_size == SHARED
    assert bridge.swap == {IM_END: EOT_ID}


def test_bridge_auto_eos_map():
    """Empty eos_map: student eos (outside shared vocab) maps to teacher eos."""
    from palingenesis.opd.token_bridge import TokenBridge

    student, teacher = make_pair()
    bridge = TokenBridge.from_tokenizers(student, teacher)
    assert bridge.swap == {IM_END: EOT_ID}


def test_bridge_clean_completion():
    from palingenesis.opd.token_bridge import TokenBridge

    bridge = TokenBridge(shared_vocab_size=SHARED, swap={IM_END: EOT_ID})
    stops = (IM_END, END_OF_TEXT)
    # cut at first stop token, inclusive (stopping is supervised too)
    assert bridge.clean_completion([1, 2, IM_END, 3, 4], stops) == [1, 2, IM_END]
    assert bridge.clean_completion([1, 2, END_OF_TEXT], stops) == [1, 2, END_OF_TEXT]
    # unmapped out-of-shared-vocab token truncates BEFORE it (teacher can't score it)
    assert bridge.clean_completion([1, 2, SHARED + 3, 4], stops) == [1, 2]
    assert bridge.clean_completion([5, 6, 7], stops) == [5, 6, 7]
    # a stop token the teacher cannot embed ends the completion without being scored
    assert bridge.clean_completion([1, SHARED + 4], stops + (SHARED + 4,)) == [1]


def test_bridge_to_teacher():
    from palingenesis.opd.token_bridge import TokenBridge

    bridge = TokenBridge(shared_vocab_size=SHARED, swap={IM_END: EOT_ID})
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
    inside_shared = TokenBridge(shared_vocab_size=SHARED, swap={100: EOT_ID})
    with pytest.raises(TokenBridgeError, match="inside the shared vocab"):
        check_compatible(student, teacher, inside_shared)
    outside_teacher = TokenBridge(shared_vocab_size=SHARED, swap={IM_END: SHARED + 1})
    with pytest.raises(TokenBridgeError, match="outside the teacher vocab"):
        check_compatible(student, teacher, outside_teacher)


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


def test_default_templates_are_neutral_english():
    from palingenesis.opd.formatting import build_user_query

    q = build_user_query(ROW, fast=True)
    assert q.startswith("Answer the following multiple-choice question about 'letteratura'.")
    assert "A) Dante\nB) Petrarca" in q
    assert "one of AB" in q
    assert q.endswith("Answer:")

    cot = build_user_query(ROW, fast=False)
    assert "'Answer: LETTER'" in cot
    assert not cot.endswith("Answer:")


def test_italic_config_templates_render_verbatim():
    """The example config carries ITALIC's exact prompt bytes — the benchmark
    policy lives in the config, and this locks the reproduction path."""
    from palingenesis.opd.config import OPDConfig
    from palingenesis.opd.formatting import build_user_query

    config = OPDConfig.from_yaml(CONFIGS / "distill_opd.yaml")
    assert config.validate() == []
    source = config.sources["italic"]
    assert source.system_message == "Sei un assistente utile."

    assert build_user_query(ROW, fast=True, template=source.fast_template) == (
        "Rispondi alla seguente domanda a scelta multipla sull'argomento 'letteratura'. "
        "La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) "
        "dove LETTERA è una tra AB. Scrivi solo la lettera corrispondente alla tua "
        "risposta senza spiegazioni.\n\nChi scrisse la Divina Commedia?\n\n"
        "A) Dante\nB) Petrarca\n\nRisposta:"
    )
    assert build_user_query(ROW, fast=False, template=source.cot_template) == (
        "Rispondi alla seguente domanda a scelta multipla sull'argomento 'letteratura'. "
        "L'ultima riga della tua risposta deve essere nel seguente formato: "
        "'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra AB. "
        "Ragiona brevemente prima di rispondere.\n\nChi scrisse la Divina Commedia?\n\n"
        "A) Dante\nB) Petrarca"
    )


def test_template_placeholder_validation():
    from palingenesis.opd.config import OPDConfigError

    config = _valid_base_config()
    source = config.sources["pool"]
    source.format = "mcqa"
    source.shots_path = "shots.jsonl"
    source.fast_template = "{question}\n{options}\n{merged_letters}"
    assert config.validate() == []

    source.fast_template = "{question}\n{options}\n{answer_key}"  # unknown field
    with pytest.raises(OPDConfigError, match="unknown placeholders"):
        config.validate()

    source.fast_template = "{question} only"  # missing {options}
    with pytest.raises(OPDConfigError, match="missing required"):
        config.validate()


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
    # CoT: incidental capitals early, answer at the end
    cot = "A causa della regola X, la risposta corretta è B"
    assert extract_letter(cot) == "A"          # first (wrong for CoT)
    assert extract_letter(cot, last=True) == "B"


def test_extract_number():
    from palingenesis.opd.formatting import extract_number

    assert extract_number("3 apples + 4 = 7\nAnswer: 7") == "7"
    assert extract_number("Answer: $1,250.50 total, 3 items") == "1250.50"    # after "Answer:", not the last
    assert extract_number("so it is -12") == "-12"                             # no "Answer:": the last number
    assert extract_number("no digits") is None


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
# Prompt sources
# ---------------------------------------------------------------------------

class FakeEngine:
    """Engine stub: greedy answers 'A' (or "Answer: 7") to everything; dev_kl returns a constant."""

    def __init__(self, answer="A"):
        self.calls = []
        self.answer = answer

    def greedy_generate(self, messages_list, max_new_tokens):
        self.calls.append(("greedy", len(messages_list), max_new_tokens))
        return [self.answer] * len(messages_list)

    def dev_kl(self, messages_list, max_new_tokens):
        self.calls.append(("dev_kl", len(messages_list), max_new_tokens))
        return {"dev_kl": 0.5, "dev_len": 3.0}


def write_mcqa_pool(tmp_path, n=20):
    rows = [{"question": f"Domanda {i}?", "options": [["A", "sì"], ["B", "no"]],
             "answer": "A" if i % 2 == 0 else "B", "category": "storia", "source": "t"}
            for i in range(n)]
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def test_mcqa_source(tmp_path):
    from palingenesis.opd.config import SourceConfig
    from palingenesis.opd.sources import McqaPoolSource

    source = McqaPoolSource(SourceConfig(format="mcqa", path=str(write_mcqa_pool(tmp_path)), dev_size=4,
                                         max_new_tokens=16), eval_samples=4, seed=0, rng=random.Random(0))
    messages, mnt, meta = source.sample()
    assert messages[-1]["role"] == "user"
    assert mnt == 16  # cot_fraction=0 -> always fast
    assert "row" in meta and meta["fast"] is True

    engine = FakeEngine()
    metrics = source.evaluate(engine)
    # FakeEngine answers 'A'; half the dev rows have answer 'A'
    assert engine.calls == [("greedy", 4, 8)]
    assert 0.0 <= metrics["dev_acc"] <= 1.0
    assert "dev_acc_cot" not in metrics  # cot_fraction=0: fast eval only

    stats = source.batch_stats([(meta, "A"), (meta, "boh niente lettera")])
    assert stats["format_ok"] == 0.5


def test_mcqa_source_cot_eval(tmp_path):
    """With cot_fraction > 0 the dev metric covers both modes."""
    from palingenesis.opd.config import SourceConfig
    from palingenesis.opd.sources import McqaPoolSource

    config = SourceConfig(format="mcqa", path=str(write_mcqa_pool(tmp_path)), dev_size=4, cot_fraction=0.3,
                          cot_max_new_tokens=300)
    source = McqaPoolSource(config, eval_samples=4, seed=0, rng=random.Random(0))
    engine = FakeEngine()
    metrics = source.evaluate(engine)
    assert engine.calls == [("greedy", 4, 8), ("greedy", 4, 300)]
    assert set(metrics) == {"dev_acc", "dev_acc_cot"}


def write_chat(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_messages_source(tmp_path):
    from palingenesis.opd.config import SourceConfig
    from palingenesis.opd.sources import ChatMessagesSource

    rows = [{"messages": [{"role": "user", "content": f"Ciao {i}"}]} for i in range(12)]
    rows.append({"messages": [{"role": "user", "content": "x"},
                              {"role": "assistant", "content": "ends wrong"}]})  # skipped
    config = SourceConfig(path=write_chat(tmp_path / "chat.jsonl", rows), dev_size=3, max_new_tokens=64)
    source = ChatMessagesSource(config, eval_samples=3, seed=0, rng=random.Random(0))
    assert len(source.dev_rows) == 3 and len(source.train_rows) == 9  # 12 usable - 3 dev

    messages, mnt, meta = source.sample()
    assert messages[-1]["role"] == "user" and mnt == 64 and meta == {}

    engine = FakeEngine()
    assert source.evaluate(engine) == {"dev_kl": 0.5, "dev_len": 3.0}   # no answers: no accuracy
    assert engine.calls == [("dev_kl", 3, 64)]
    assert source.batch_stats([({}, "whatever")]) == {}


def test_messages_source_dev_path_and_answers(tmp_path):
    """dev_path: the held-out set is another file (a test split); rows with answers get dev_acc."""
    from palingenesis.opd.config import SourceConfig
    from palingenesis.opd.sources import ChatMessagesSource

    train = [{"messages": [{"role": "user", "content": f"q{i}"}], "answer": "1"} for i in range(5)]
    dev = [{"messages": [{"role": "user", "content": f"t{i}"}], "answer": a} for i, a in enumerate(["7", "8", "7.0"])]
    config = SourceConfig(path=write_chat(tmp_path / "train.jsonl", train),
                          dev_path=write_chat(tmp_path / "dev.jsonl", dev), max_new_tokens=32)
    source = ChatMessagesSource(config, eval_samples=10, seed=0, rng=random.Random(0))
    assert len(source.train_rows) == 5 and len(source.dev_rows) == 3
    engine = FakeEngine(answer="so 3 + 4\nAnswer: 7")
    metrics = source.evaluate(engine)
    assert metrics["dev_acc"] == pytest.approx(2 / 3)
    assert engine.calls == [("dev_kl", 3, 32), ("greedy", 3, 32)]


def test_build_source_mixes_the_configured_sources(tmp_path):
    from palingenesis.opd.config import OPDConfig
    from palingenesis.opd.sources import ChatMessagesSource, McqaPoolSource, MixedSource, build_source

    config = OPDConfig()
    config.set("sources.pool.format", "mcqa", "test")
    config.set("sources.pool.path", str(write_mcqa_pool(tmp_path)), "test")
    config.set("sources.pool.dev_size", 4, "test")
    config.set("sources.chat.path", write_chat(tmp_path / "chat.jsonl",
                                               [{"messages": [{"role": "user", "content": f"c{i}"}]}
                                                for i in range(8)]), "test")
    config.set("sources.chat.dev_size", 2, "test")
    source = build_source(config, rng=random.Random(0))
    assert isinstance(source, MixedSource)
    assert isinstance(source.subs["pool"], McqaPoolSource) and isinstance(source.subs["chat"], ChatMessagesSource)
    assert {source.sample()[2]["_src"] for _ in range(50)} == {"pool", "chat"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_opd_config_from_yaml_and_cli(tmp_path):
    from palingenesis.opd.config import OPDConfig

    path = tmp_path / "opd.yaml"
    path.write_text(
        "model:\n  student: my/student\n  chat_template_kwargs: {enable_thinking: false}\n"
        "teachers:\n  big:\n    model: my/teacher\n    eos_map:\n      '<|im_end|>': '<|eot_id|>'\n"
        "sources:\n  chat:\n    path: chat.jsonl\n    teacher: big\n"
        "train:\n  learning_rate: 5.0e-6\n  steps: 100\n"
    )
    config = OPDConfig.from_cli(["--config", str(path), "--train.steps", "250",
                                 "--teachers.big.backend", "vllm",
                                 "--teachers.small.model", "my/small",
                                 "--sources.chat.max_new_tokens", "128",
                                 "--model.stop_tokens", "['<|end_of_text|>']",
                                 "--model.gradient_checkpointing", "true"])
    assert config.model.student == "my/student"
    assert config.model.chat_template_kwargs == {"enable_thinking": False}
    assert config.model.stop_tokens == ["<|end_of_text|>"]
    assert config.teachers["big"].eos_map == {"<|im_end|>": "<|eot_id|>"}
    assert config.teachers["big"].backend == "vllm"
    assert config.teachers["small"].model == "my/small"            # a new teacher from the command line
    assert config.sources["chat"].max_new_tokens == 128
    assert config.train.learning_rate == 5.0e-6
    assert config.train.steps == 250  # CLI wins over YAML
    assert config.model.gradient_checkpointing is True
    assert config.teacher_of("chat") == "big"


def test_opd_config_rejects_unknown_and_moved_options(tmp_path):
    from palingenesis.config import ConfigError
    from palingenesis.opd.config import OPDConfig

    def load(text):
        path = tmp_path / "c.yaml"
        path.write_text(text)
        return OPDConfig.from_yaml(path)

    with pytest.raises(ConfigError, match="did you mean teachers.t.model"):
        load("teachers:\n  t:\n    modle: x\n")
    with pytest.raises(ConfigError, match="rollout.batch_prompts"):
        OPDConfig.from_cli(["--rollout.batch_prompt", "3"])
    # the first OPD format's options say where they went
    with pytest.raises(ConfigError, match="moved to `teachers:"):
        load("model:\n  student: s\n  teacher: t\n")
    with pytest.raises(ConfigError, match="replaced by `sources:"):
        load("data:\n  format: messages\n")
    with pytest.raises(ConfigError, match="full_kl is now full_rkl"):
        load("train:\n  loss_fn: full_kl\n")
    with pytest.raises(ConfigError, match="cannot contain dots"):
        load("teachers:\n  qwen3.5:\n    model: x\n")


def _valid_base_config():
    from palingenesis.opd.config import OPDConfig

    config = OPDConfig()
    config.set("model.student", "org/student", "test")
    config.set("teachers.big.model", "org/teacher", "test")
    config.set("sources.pool.path", "pool.jsonl", "test")
    return config


def test_opd_config_validate():
    from palingenesis.opd.config import OPDConfig, OPDConfigError

    assert _valid_base_config().validate() == []

    # student, teacher and source have no defaults — the pair is the experiment's decision
    with pytest.raises(OPDConfigError, match="model.student") as e:
        OPDConfig().validate()
    assert "teacher is required" in str(e.value) and "source is required" in str(e.value)

    def invalid(match, **changes):
        config = _valid_base_config()
        for key, value in changes.items():
            config.set(key.replace("__", "."), value, "test")
        with pytest.raises(OPDConfigError, match=match):
            config.validate()

    invalid("full_rkl needs the teacher's full distribution", teachers__big__backend="vllm",
            teachers__big__loss="full_rkl")
    invalid("loss must be one of", teachers__big__loss="full_kl")
    invalid("hf backend cannot", rollout__max_staleness=1)
    invalid("not one of the teachers", sources__pool__teacher="nobody")
    invalid("rollout.backend", rollout__backend="sglang")
    invalid("device and offload apply to hf teachers", teachers__big__backend="vllm", teachers__big__offload=True)
    invalid("xtok_spread", loss__xtok_spread="evenly")
    invalid("rs_kd needs the teacher's full distribution", teachers__big__backend="vllm", teachers__big__loss="rs_kd")
    invalid("rs_rounds", loss__rs_rounds=0)
    invalid("rs_temperature", loss__rs_temperature=0.0)
    invalid("token_weighting must be one of", loss__token_weighting="forking")
    invalid("entropy_keep", loss__entropy_keep=0.0)
    invalid("sure_alpha", loss__sure_alpha=-1.0)
    invalid("p_reference_shots", sources__pool__p_reference_shots=0.8, sources__pool__p_pool_shots=0.5)

    config = _valid_base_config()
    config.set("rollout.backend", "vllm", "test")
    config.set("rollout.max_staleness", 1, "test")
    config.set("teachers.big.backend", "vllm", "test")      # auto loss: topk_kl
    assert config.validate() == []

    config.set("sources.pool.cot_fraction", 0.2, "test")     # legal but meaningless for messages
    warnings = config.validate()
    assert len(warnings) == 1 and "cot_fraction" in warnings[0]


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("distill_*.yaml")), ids=lambda p: p.name)
def test_example_configs_load_and_validate(path):
    from palingenesis.opd.config import OPDConfig

    config = OPDConfig.from_yaml(path)
    assert config.validate() == []
    for source in config.sources:
        assert config.teacher_of(source) in config.teachers


# ---------------------------------------------------------------------------
# MixedSource (compose sub-sources into one OPD run)
# ---------------------------------------------------------------------------

class _FakeSub:
    """Minimal PromptSource: samples a fixed mnt, reports a per-source metric."""

    def __init__(self, name, mnt, dev):
        self.name, self.mnt, self.dev = name, mnt, dev

    def sample(self):
        return [{"role": "user", "content": self.name}], self.mnt, {"tag": self.name}

    def evaluate(self, engine):
        return {"dev": self.dev if engine is None else engine}

    def batch_stats(self, rollouts):
        return {"n": float(len(rollouts))}


def test_mixed_source_routes_samples_and_merges_metrics():
    from palingenesis.opd.sources import MixedSource

    a, b = _FakeSub("mcqa", 8, 0.4), _FakeSub("chat", 1024, 0.8)
    m = MixedSource([("mcqa", 1.0, a), ("chat", 1.0, b)], random.Random(0))

    # sample() tags meta with _src and preserves each sub's max_new_tokens
    seen = {}
    for _ in range(300):
        _, mnt, meta = m.sample()
        seen[meta["_src"]] = mnt
    assert seen == {"mcqa": 8, "chat": 1024}

    # evaluate() merges under metric/<name>; a function of the name gives per-source engines
    assert m.evaluate(engine=None) == {"dev/mcqa": 0.4, "dev/chat": 0.8}
    assert m.evaluate(engine=lambda name: len(name)) == {"dev/mcqa": 4, "dev/chat": 4}

    # batch_stats() routes rollouts back to the right sub by _src
    rolls = [({"_src": "mcqa"}, "x"), ({"_src": "mcqa"}, "y"), ({"_src": "chat"}, "z")]
    assert m.batch_stats(rolls) == {"n/mcqa": 2.0, "n/chat": 1.0}


def test_mixed_source_respects_weights():
    from palingenesis.opd.sources import MixedSource

    a, b = _FakeSub("rare", 8, 0.0), _FakeSub("common", 8, 0.0)
    m = MixedSource([("rare", 1.0, a), ("common", 9.0, b)], random.Random(0))
    counts = collections.Counter(m.sample()[2]["_src"] for _ in range(2000))
    assert counts["common"] > counts["rare"] * 4  # ~9:1, generous margin


def test_mixed_source_rejects_bad_construction():
    from palingenesis.opd.sources import MixedSource

    a = _FakeSub("a", 8, 0.0)
    with pytest.raises(ValueError, match="at least one"):
        MixedSource([], random.Random(0))
    with pytest.raises(ValueError, match="duplicate"):
        MixedSource([("a", 1.0, a), ("a", 1.0, a)], random.Random(0))
    with pytest.raises(ValueError, match="weights"):
        MixedSource([("a", 0.0, a)], random.Random(0))
