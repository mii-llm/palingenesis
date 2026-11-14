"""`pgs prepare`: model-free cleaning (data_clean), the report, and the scoring it shares a tokenization with.

Network-free: a GPT-2 tokenizer under a ChatML template, and a table-lookup model on CPU.
"""

import hashlib
import json
import math
import os
import sys

import pytest
import torch

from palingenesis import data_clean
from palingenesis.data_clean import (
    LENGTH,
    ROW_INDEX,
    CleanReport,
    cap_groups,
    conversation_key,
    drop_exact_duplicates,
    prompt_key,
    split_holdout,
    take_token_budget,
)
from palingenesis.prepare import PREPARED_META, Scorer, prepare_data

CHATML = (
    "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{%- endfor %}"
)


def _make_tok():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return None
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tok.chat_template = CHATML
    tok.eos_token = "<|im_end|>"
    tok.pad_token = tok.eos_token
    return tok


TOK = _make_tok()
needs_tok = pytest.mark.skipif(TOK is None, reason="gpt2 tokenizer not cached (offline)")


class TableModel:
    """Logits depend on the token alone: scores do not depend on batch layout."""

    device = torch.device("cpu")

    def __init__(self, vocab: int):
        g = torch.Generator().manual_seed(0)
        self.embed, self.head = torch.randn(vocab, 8, generator=g), torch.randn(8, vocab, generator=g)
        self.forwards = 0

    def __call__(self, input_ids, attention_mask=None, use_cache=None):
        self.forwards += 1

        class Out:
            logits = self.embed[input_ids] @ self.head

        return Out()


class FakeScorer(Scorer):
    """A Scorer over the test tokenizer; counts how often the model is asked for."""

    def __init__(self, allow_model: bool = True):
        self.model_name = "fake"
        self.tokenizer = TOK
        self.allow_model = allow_model
        self.loads = 0
        self._model = None

    @property
    def model(self):
        assert self.allow_model, "the model must not be loaded when nothing needs a score"
        if self._model is None:
            self.loads += 1
            self._model = TableModel(len(TOK))
        return self._model


def _row(question: str, answer: str, **extra) -> dict:
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}], **extra}


def _write(tmp_path, rows) -> str:
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _rows(n: int) -> list[dict]:
    return [_row(f"question number {i}", f"this is answer number {i}, with a few more words") for i in range(n)]


# ── keys ──────────────────────────────────────────────────────────────────────


def test_conversation_key_ignores_whitespace_runs_and_schema_but_not_content_or_tools():
    a = _row("what is  2+2?", "four")
    sharegpt = {"conversations": [{"from": "human", "value": "what is 2+2?"}, {"from": "gpt", "value": "four"}]}
    assert conversation_key(a) == conversation_key(sharegpt, "conversations")
    assert conversation_key(a) != conversation_key(_row("what is 2+2?", "five"))
    with_tools = dict(a, tools=[{"type": "function", "function": {"name": "calc"}}])
    assert conversation_key(a) != conversation_key(with_tools)


def _reference_digest(obj) -> str:
    """The definition the fast paths must match: compact sorted JSON, str.split() whitespace runs."""
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha1(" ".join(text.split()).encode()).hexdigest()


def _key_objects() -> list:
    import datetime
    import enum

    class Color(enum.Enum):
        RED = 1

    class Tag(str):
        pass

    every_char = "".join(chr(c) for c in range(0x110000) if not 0xD800 <= c < 0xE000)
    nested: list = ["leaf"]
    for _ in range(300):  # deeper than orjson writes
        nested = [nested]
    return [
        [every_char[i : i + 4096] for i in range(0, len(every_char), 4096)],
        {"b": 1, "a": [1, 2.5, None, True, False], "é": "x", "\u00e9a": "y", "z\u3000": "\u3000 a  b\xa0\xa0c"},
        [2**63 - 1, -(2**63), 2**64 - 1, 2**64, -(2**63) - 1, 10**30],
        [0.1, 1.0, -0.0, 1e15, 1e16, 1e-4, 1e-5, 1e300, 5e-324, 123456.789, float("nan"), float("inf"), -float("inf")],
        ("tuple", ["in", ("tuples",)]),
        {1: "int key", 2: "another"},
        [datetime.datetime(2024, 1, 2, 3, 4, 5), Color.RED, Tag("sub  class")],
        nested,
        {"messages": [{"role": "user", "content": "  leading and trailing  "}], "tools": None},
        [" ", "  ", "a" + " " * 33 + "b", "\t\n\r tabs \t and\n\nnewlines"],
    ]


def test_dedupe_keys_match_the_reference_with_and_without_orjson(monkeypatch):
    """orjson (when installed) only speeds the keys up: both paths give the reference digest, so
    environments with and without it dedupe identically."""
    objects = _key_objects()
    want = [_reference_digest(o) for o in objects]
    assert [data_clean._digest(o) for o in objects] == want
    monkeypatch.setattr(data_clean, "orjson", None)
    assert [data_clean._digest(o) for o in objects] == want


def test_dedupe_keys_collapse_every_unicode_whitespace_like_str_split(monkeypatch):
    spaces = [chr(c) for c in range(0x110000) if chr(c).isspace()]
    # JSON escapes what is below U+0020; the rest is written as is and must be collapsed
    assert set(data_clean._UNICODE_SPACES) == {c for c in spaces if c > " "}
    for orjson in (data_clean.orjson, None):
        monkeypatch.setattr(data_clean, "orjson", orjson)
        for c in spaces:
            row = [{"content": f"a{c}{c} b{c}"}, f"{c}x"]
            assert data_clean._digest(row) == _reference_digest(row), (orjson, hex(ord(c)))
        assert data_clean._digest(["a \u3000\xa0 b"]) == data_clean._digest(["a b"]) != data_clean._digest(["a\nb"])


def test_prompt_key_is_the_conversation_up_to_the_first_user_turn():
    a, b = _row("fix the bug", "attempt one"), _row("fix the bug", "attempt two")
    assert prompt_key(a) == prompt_key(b) != prompt_key(_row("fix the other bug", "attempt one"))
    system = {"messages": [{"role": "system", "content": "be brief"}] + a["messages"]}
    assert prompt_key(system) != prompt_key(a)


# ── model-free steps ──────────────────────────────────────────────────────────


def test_exact_duplicates_keep_the_first_row_and_name_it():
    rows = [_row("q", "a"), _row("q", "b"), _row("q", "a")]
    for i, r in enumerate(rows):
        r[ROW_INDEX] = i
    report = CleanReport(loaded=3)
    kept = drop_exact_duplicates(rows, report)
    assert [r[ROW_INDEX] for r in kept] == [0, 1]
    assert report.dropped == [{"row": 2, "reason": "duplicate", "duplicate_of": 0}]


def test_holdout_never_splits_a_prompt():
    # 30 tasks x 4 trajectories: a row-level split would put most tasks on both sides
    rows = [_row(f"task {t}", f"trajectory {k} of task {t}") for t in range(30) for k in range(4)]
    train, held = split_holdout(rows, 20)
    assert len(held) == 20 and len(train) == 100
    assert not {prompt_key(r) for r in train} & {prompt_key(r) for r in held}
    again = split_holdout(rows, 20)[1]
    assert [r["messages"] for r in held] == [r["messages"] for r in again]
    # one prompt holding every row cannot be split: no holdout, not a leaking one
    same = [_row("one task", f"trajectory {k}") for k in range(10)]
    assert split_holdout(same, 3) == (same, [])


def test_group_caps_bound_rows_and_tokens_and_report_what_they_removed():
    rows = [{"source": "a" if i < 30 else "b", LENGTH: 100, ROW_INDEX: i} for i in range(40)]
    report = CleanReport()
    kept = cap_groups(rows, lambda r: r["source"], report, "group_cap", max_rows=12)
    assert sum(r["source"] == "a" for r in kept) == 12 and sum(r["source"] == "b" for r in kept) == 10
    assert report.reasons["group_cap"] == 18
    assert [r[ROW_INDEX] for r in kept] == sorted(r[ROW_INDEX] for r in kept)  # input order kept
    assert {r[ROW_INDEX] for r in kept if r["source"] == "a"} != set(range(12))  # a draw, not the first rows
    assert kept == cap_groups(rows, lambda r: r["source"], CleanReport(), "group_cap", max_rows=12)

    by_tokens = cap_groups(rows, lambda r: r["source"], CleanReport(), "group_cap", max_tokens=450)
    assert sum(r["source"] == "a" for r in by_tokens) == 4 and sum(r["source"] == "b" for r in by_tokens) == 4
    # a group whose single row exceeds the cap is not emptied
    assert (
        len(cap_groups([{"source": "a", LENGTH: 900}], lambda r: r["source"], CleanReport(), "x", max_tokens=450)) == 1
    )


def test_token_budget_is_a_random_subset_within_the_budget():
    rows = [{LENGTH: 50 + (i % 7) * 100, ROW_INDEX: i} for i in range(200)]
    kept = take_token_budget(rows, 10_000)
    total = sum(r[LENGTH] for r in kept)
    assert 9_950 <= total <= 10_000
    assert kept == take_token_budget(rows, 10_000) and kept != rows[: len(kept)]
    assert take_token_budget(rows, 0) is rows


# ── the pipeline ──────────────────────────────────────────────────────────────


@needs_tok
def test_prepare_cleans_without_loading_the_model_and_reports_every_removed_row(tmp_path):
    rows = _rows(12)
    rows.append(_row("question number 3", "this is answer number 3, with a few more words"))  # 12: duplicate of 3
    rows.append({"messages": [{"role": "user", "content": "nobody answered this"}]})  # 13: nothing to train on
    rows.append({"messages": []})  # 14: no messages
    rows.append(_row("short question", "long answer " * 400))  # 15: the only trained turn does not fit
    rows.append(  # 16: fits once its last turn is cut
        {"messages": _row("first", "short answer")["messages"] + _row("second", "long answer " * 400)["messages"]}
    )
    out = prepare_data(
        "fake", _write(tmp_path, rows), str(tmp_path / "out"), max_seq_length=256, scorer=FakeScorer(allow_model=False)
    )

    meta = json.loads((tmp_path / "out" / PREPARED_META).read_text())
    assert meta["scored"] is False and meta["num_samples"] == 13
    assert meta["cleaning"] == {
        "loaded": 17,
        "dropped": {"duplicate": 1, "no_trained_turn_fits": 1, "nothing_to_train_on": 2},
        "truncated_to_max_seq_length": 1,
    }
    dropped = [json.loads(line) for line in (tmp_path / "out" / data_clean.DROPPED_BASENAME).read_text().splitlines()]
    assert {(d["row"], d["reason"]) for d in dropped} == {
        (12, "duplicate"),
        (13, "nothing_to_train_on"),
        (14, "nothing_to_train_on"),
        (15, "no_trained_turn_fits"),
    }
    assert next(d for d in dropped if d["reason"] == "duplicate")["duplicate_of"] == 3

    from palingenesis.data import _load_dataset_source

    kept = list(_load_dataset_source(str(out), "train", streaming=False))
    assert sorted(r[ROW_INDEX] for r in kept) == [*range(12), 16]
    assert "_score_response_ppl" not in kept[0] and "_score_familiarity" not in kept[0]
    assert meta["tokens"]["total"] == sum(r[LENGTH] for r in kept) and 0 < meta["tokens"]["trained_share"] < 1
    assert meta["tokens"]["length_max"] <= 256


@needs_tok
def test_prepared_lengths_are_the_lengths_training_sees(tmp_path):
    from palingenesis.data import IGNORE_INDEX, ChatDataset, _load_dataset_source

    out = prepare_data("fake", _write(tmp_path, _rows(5)), str(tmp_path / "out"), scorer=FakeScorer(allow_model=False))
    kept = list(_load_dataset_source(str(out), "train", streaming=False))
    for row, trained in zip(kept, ChatDataset(kept, TOK, 8192)):
        assert row[LENGTH] == trained["input_ids"].numel()
        assert row["_score_response_token_count"] == int((trained["labels"] != IGNORE_INDEX).sum())


@needs_tok
def test_holdout_caps_and_budgets_in_one_run(tmp_path):
    rows = [
        _row(f"task {t}", f"trajectory {k} of task {t} " * 3, source="swe" if t < 30 else "chat")
        for t in range(40)
        for k in range(3)
    ]
    prepare_data(
        "fake",
        _write(tmp_path, rows),
        str(tmp_path / "out"),
        eval_holdout=12,
        max_per_prompt=2,
        group_field="source",
        max_per_group=30,
        budget=40,
        scorer=FakeScorer(allow_model=False),
    )
    from palingenesis.data import _load_dataset_source

    train = list(_load_dataset_source(str(tmp_path / "out" / "scored_data.parquet"), "train", streaming=False))
    held = list(_load_dataset_source(str(tmp_path / "out" / "eval_data.parquet"), "train", streaming=False))
    assert len(held) == 12 and len(train) == 40
    assert not {prompt_key(r) for r in train} & {prompt_key(r) for r in held}
    per_prompt = {}
    for r in train:
        per_prompt[prompt_key(r)] = per_prompt.get(prompt_key(r), 0) + 1
    assert max(per_prompt.values()) <= 2

    meta = json.loads((tmp_path / "out" / PREPARED_META).read_text())
    assert meta["cleaning"]["dropped"]["prompt_cap"] == 36  # 36 tasks left after the holdout, one of three cut
    assert meta["groups_scored_pool"]["swe"]["count"] == 90
    assert meta["selection"]["selected"] == 40 and meta["selection"]["pool"] <= 30 + 20
    assert sum(g["count"] for g in meta["groups_selected"].values()) == 40


@needs_tok
def test_a_perplexity_option_turns_scoring_on_with_one_model_load(tmp_path):
    scorer = FakeScorer()
    out = prepare_data(
        "fake",
        _write(tmp_path, _rows(9)),
        str(tmp_path / "out"),
        max_ppl=1e9,
        compute_hes=True,
        hes_top_k_pct=25.0,
        batch_size=4,
        scorer=scorer,
    )
    assert scorer.loads == 1
    assert scorer.model.forwards == 3  # 9 rows in batches of 4: the entropies come from the same forwards
    from palingenesis.data import _load_dataset_source

    kept = list(_load_dataset_source(str(out), "train", streaming=False))
    assert all(math.isfinite(r["_score_response_ppl"]) and r["_score_hes"] > 0 for r in kept)
    meta = json.loads((tmp_path / "out" / PREPARED_META).read_text())
    assert meta["scored"] is True and sum(meta["familiarity_distribution"].values()) == 9


def test_hes_is_the_top_entropies_of_the_response_tokens():
    """Batched HES equals a per-sample reference over the trained positions only; padding and prompt tokens
    never enter it."""
    from palingenesis.prepare import _score_padded_batch

    vocab = 50
    model = TableModel(vocab)
    torch.manual_seed(0)
    batch, samples = [], []
    for i, n in enumerate([6, 11, 40]):
        ids = torch.randint(1, vocab, (n,))
        mask = torch.zeros(n, dtype=torch.bool)
        mask[n // 2 :] = True
        batch.append((i, ids, mask))
        samples.append({})
    _score_padded_batch(model, batch, samples, 0, hes_top_k_pct=20.0)

    for (_, ids, mask), sample in zip(batch, samples):
        log_probs = torch.log_softmax((model.embed[ids] @ model.head)[:-1], dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)[mask[1:]]
        k = max(1, int(len(entropy) * 20.0 / 100))
        top = entropy.topk(k).values
        assert sample["_score_hes"] == pytest.approx(top.sum().item(), abs=1e-3)
        assert sample["_score_hes_mean"] == pytest.approx(top.mean().item(), abs=1e-3)


def test_filter_keeps_unscored_rows_and_rows_up_to_the_sequence_limit():
    from palingenesis.prepare import filter_samples

    unscored = [{LENGTH: 100}, {LENGTH: 60_000}, {LENGTH: 3}]
    seen = []
    kept = filter_samples(unscored, min_ppl=1.5, max_ppl=500, max_length=65_536, on_drop=lambda s, r: seen.append(r))
    assert kept == unscored[:2] and seen == ["too_short"]
    seen.clear()
    unscoreable = {"_score_response_ppl": float("inf"), "_score_ppl": float("inf"), LENGTH: 0}
    assert filter_samples([unscoreable], max_ppl=0, on_drop=lambda s, r: seen.append(r)) == []
    assert seen == ["unscoreable"]


def test_settings_that_cannot_work_are_refused_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="Unknown selection strategy"):
        prepare_data("fake", "missing.jsonl", str(tmp_path), strategy="optimall")
    with pytest.raises(ValueError, match="token_budget"):
        prepare_data("fake", "missing.jsonl", str(tmp_path), token_budget=10, strategy="curriculum")
    with pytest.raises(ValueError, match="group_field"):
        prepare_data("fake", "missing.jsonl", str(tmp_path), max_per_group=10)


def test_config_carries_the_render_options_and_defaults_to_no_scoring(monkeypatch):
    from palingenesis import prepare
    from palingenesis.config import Config, ConfigError, PreprocessConfig

    pre = PreprocessConfig()
    assert (pre.score, pre.min_ppl, pre.max_ppl, pre.dedupe, pre.strategy) == (False, 0.0, 0.0, True, "random")

    config = Config()
    config.data.think_tags = ["<reason>", "</reason>"]
    config.data.chat_template_kwargs = {"enable_thinking": False}
    config.preprocess.max_per_prompt = 2
    seen = {}
    monkeypatch.setattr(prepare, "prepare_data", lambda **kw: seen.update(kw))
    prepare.prepare_from_config(config)
    assert seen["chat_options"]["think_tags"] == ["<reason>", "</reason>"]
    assert seen["chat_options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["max_per_prompt"] == 2 and seen["score"] is False and seen["dedupe"] is True

    config.preprocess.token_budget = 1000
    config.preprocess.strategy = "curriculum"
    config.preprocess.max_per_group = 5
    with pytest.raises(ConfigError) as refused:
        config.validate()
    assert "token_budget" in str(refused.value) and "group_field" in str(refused.value)


@needs_tok
def test_prepare_multi_reports_row_and_token_shares(tmp_path, monkeypatch):
    from datasets import Dataset

    from palingenesis import prepare

    monkeypatch.setattr(prepare, "Scorer", lambda name: FakeScorer(allow_model=False))
    short = tmp_path / "short.parquet"
    Dataset.from_list(_rows(6)).to_parquet(str(short))
    long_rows = [_row(f"long question {i}", "a much longer answer " * 30) for i in range(6)]
    sources = [
        {"name": "short", "dataset": str(short), "weight": 1.0},
        {"name": "long", "dataset": _write(tmp_path, long_rows), "weight": 1.0, "budget": 4},
        {"name": "text", "dataset": "unused", "mode": "pretrain"},
    ]
    prepare.prepare_multi_source("fake", sources, str(tmp_path / "out"))

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    by_name = {s["name"]: s for s in manifest["sources"]}
    assert set(by_name) == {"short", "long"}
    assert by_name["short"]["num_samples"] == 6 and by_name["long"]["num_samples"] == 4
    assert by_name["short"]["normalized_weight"] == by_name["long"]["normalized_weight"] == 0.5
    assert by_name["long"]["token_share"] > 0.8  # equal weights, far more tokens
    assert by_name["short"]["token_share"] + by_name["long"]["token_share"] == pytest.approx(1.0, abs=1e-3)
    assert "estimated_overfit_risk" not in by_name["short"] and "recommended_epoch_multiplier" not in by_name["short"]
    assert (tmp_path / "out" / "long" / "scored_data.jsonl").exists()


# ── parallel tokenization ─────────────────────────────────────────────────────


def _messy_rows() -> list[dict]:
    """Rows of every outcome: kept, duplicate, nothing to train on, too long, cut, unrenderable."""
    rows = []
    for i in range(70):
        kind = i % 7
        if kind == 0:
            rows.append(_row(f"question {i}", f"answer {i} " * (1 + i % 5)))
        elif kind == 1:
            rows.append(_row("question 0", "answer 0 "))  # duplicate of row 0
        elif kind == 2:
            rows.append({"messages": [{"role": "user", "content": f"unanswered {i}"}]})
        elif kind == 3:
            rows.append(_row(f"short {i}", "long answer " * 400))  # no trained turn fits
        elif kind == 4:
            first, second = _row(f"first {i}", "short answer")["messages"], _row("second", "long " * 400)["messages"]
            rows.append({"messages": first + second})  # cut to its first turn
        elif kind == 5:
            rows.append({"messages": [{"role": "user", "content": f"q {i}"}, {"role": "assistant", "content": i}]})
        else:
            rows.append(_row(f"task {i % 3}", f"trajectory {i} " * 3, source="a" if i % 2 else "b"))
    return rows


def _outputs(out) -> dict:
    files = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
    meta = json.loads(files.pop(PREPARED_META))
    for key in ("output_file", "dropped_file", "eval_file"):
        if meta.get(key):
            meta[key] = os.path.basename(meta[key])
    return {**files, PREPARED_META: meta}


@needs_tok
@pytest.mark.parametrize("output_format", ["jsonl", "parquet"])
def test_parallel_tokenization_writes_the_same_files(tmp_path, monkeypatch, output_format):
    monkeypatch.setattr(data_clean, "MIN_ROWS_PER_WORKER", 4)
    monkeypatch.setattr(data_clean, "MAX_SHARD_ROWS", 3)  # many shards: their order must hold
    data = _write(tmp_path, _messy_rows())
    runs = {}
    for workers in (1, 3):
        prepare_data(
            "fake",
            data,
            str(tmp_path / f"out{workers}"),
            max_seq_length=256,
            output_format=output_format,
            max_ppl=1e9,
            compute_hes=True,
            eval_holdout=5,
            group_field="source",
            budget=30,
            scorer=FakeScorer(),
            num_workers=workers,
        )
        runs[workers] = _outputs(tmp_path / f"out{workers}")
    assert runs[1] == runs[3]
    cleaning = runs[1][PREPARED_META]["cleaning"]
    assert set(cleaning["dropped"]) == {"duplicate", "no_trained_turn_fits", "nothing_to_train_on", "unrenderable"}
    assert cleaning["truncated_to_max_seq_length"] == 10


@needs_tok
@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_parallel_tokenize_rows_returns_and_counts_the_same(monkeypatch, method):
    from palingenesis.data import ChatDataset

    if method == "fork" and not sys.platform.startswith("linux"):
        pytest.skip("fork workers are used on Linux only")
    monkeypatch.setattr(data_clean, "MIN_ROWS_PER_WORKER", 4)
    monkeypatch.setattr(data_clean, "_start_method", lambda: method)
    results = {}
    for workers in (1, 2):
        rows = _messy_rows()
        for i, r in enumerate(rows):
            r[ROW_INDEX] = i
        chat, report = ChatDataset(None, TOK, 256), CleanReport(loaded=len(rows))
        kept, entries = data_clean.tokenize_rows(rows, chat, report, keep_tokens=True, num_workers=workers)
        results[workers] = (kept, [(i, ids.tolist(), mask.tolist(), ids.dtype, mask.dtype) for i, ids, mask in entries],
                            report.summary(), report.dropped, dict(chat.stats))
    assert results[1] == results[2]
    assert results[1][1][0][3] == torch.int32 and results[1][1][0][4] == torch.bool


@needs_tok
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork workers are used on Linux only")
def test_a_worker_that_dies_fails_the_run_cleanly(monkeypatch):
    from palingenesis.data import ChatDataset

    parent = os.getpid()
    process = ChatDataset._process

    def dies_in_a_worker(self, sample):
        if os.getpid() != parent and sample.get("die"):
            os._exit(1)
        return process(self, sample)

    monkeypatch.setattr(ChatDataset, "_process", dies_in_a_worker)
    monkeypatch.setattr(data_clean, "MIN_ROWS_PER_WORKER", 4)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    rows = _rows(20)
    rows[13]["die"] = True
    with pytest.raises(RuntimeError, match="worker died"):
        data_clean.tokenize_rows(rows, ChatDataset(None, TOK, 256), CleanReport(), num_workers=2)
    assert "TOKENIZERS_PARALLELISM" not in os.environ  # the caller's environment is restored


def test_worker_count_defaults_and_small_inputs_stay_in_process(monkeypatch):
    assert 1 <= data_clean.default_workers() <= 8
    calls = []
    monkeypatch.setattr(data_clean, "_tokenize_parallel", lambda *a: calls.append(a) or [])
    monkeypatch.setattr(data_clean, "_tokenize_one", lambda *a: ("drop", "nothing_to_train_on", {}))
    rows = [{} for _ in range(2 * data_clean.MIN_ROWS_PER_WORKER - 1)]
    data_clean.tokenize_rows(rows, None, CleanReport(), num_workers=8)
    assert not calls  # fewer than two workers' worth of rows: no processes
    data_clean.tokenize_rows(rows + [{}], None, CleanReport(), num_workers=8)
    assert calls[0][3] == 2
