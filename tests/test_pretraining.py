"""Raw-text training (mode: pretrain): epochs over every chunk, the end-of-document token, mixing by
token share, random initialisation, and raw-text cleaning in `pgs prepare`."""

import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest
import torch

from palingenesis import data_size
from palingenesis.config import ConfigError, DataConfig
from palingenesis.data import (
    MixedDataset,
    PackedDataset,
    PretrainDataset,
    build_dataset,
    document_end_id,
    estimate_run,
    mix_weight_unit,
    pretokenize_fingerprint,
)
from palingenesis.loss import shift_labels
from tests import distill_helpers as H
from tests.test_last_turn_integration import TOK, needs_tok


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _doc(tag: str, i: int, words: int) -> dict:
    """A document whose words carry the first letter of `tag` (so every chunk of it can be told apart)."""
    return {"text": f"{tag}{i} " + " ".join(f"{tag[0]}{(i * 7 + j) % 89}" for j in range(words))}


def _letter(sample) -> str:
    """The first letter of the document's tag, read from any of its chunks."""
    return re.search(r"\b([a-z])\d+\b", TOK.decode(sample["input_ids"])).group(1)


def _chunks(text: str, max_len: int) -> int:
    return len(PretrainDataset([], TOK, max_len)._process({"text": text}))


# ── epochs cover every chunk ──────────────────────────────────────────────────


@needs_tok
def test_an_epoch_of_one_raw_text_source_covers_every_chunk_of_every_document(tmp_path):
    """mix_epoch total (the default) counted rows: 10 documents of 3 chunks gave 10 sequences an epoch."""
    rows = [_doc("long", i, 150) for i in range(10)]
    per_doc = [_chunks(r["text"], 64) for r in rows]
    assert min(per_doc) >= 3
    config = DataConfig(
        sources=[{"dataset": _write_jsonl(tmp_path / "t.jsonl", rows), "mode": "pretrain"}],
        max_seq_length=64,
        streaming=False,
    )
    for mix_epoch in ("total", "first_exhausted"):
        config.mix_epoch = mix_epoch
        dataset = build_dataset(config, TOK, config, 0, 1, 1)
        for epoch in range(2):
            dataset.set_epoch(epoch)
            samples = list(dataset)
            assert len(samples) == sum(per_doc), (mix_epoch, len(samples))
            # every document ends once: its last chunk closes with the end-of-document token
            assert sum(int(s["input_ids"][-1]) == TOK.eos_token_id for s in samples) == len(rows)


@needs_tok
def test_unweighted_sources_mix_like_one_shuffled_concatenation(tmp_path, monkeypatch):
    """Long documents (several chunks) next to short ones, no weights: each source gets about one pass of
    its chunks, and the size estimate counts the same epoch."""
    monkeypatch.setenv("PALINGENESIS_SIZE_SAMPLE", "400")
    long_rows = [_doc("long", i, 150) for i in range(200)]
    short_rows = [_doc("short", i, 12) for i in range(200)]
    total_long = sum(_chunks(r["text"], 64) for r in long_rows)
    assert all(_chunks(r["text"], 64) == 1 for r in short_rows) and total_long >= 600
    config = DataConfig(
        sources=[
            {"dataset": _write_jsonl(tmp_path / "long.jsonl", long_rows), "mode": "pretrain", "name": "long"},
            {"dataset": _write_jsonl(tmp_path / "short.jsonl", short_rows), "mode": "pretrain", "name": "short"},
        ],
        max_seq_length=64,
        streaming=False,
    )
    assert mix_weight_unit(config) == "rows"
    samples = list(build_dataset(config, TOK, config, 0, 1, 1))
    ends = [int(s["input_ids"][-1]) == TOK.eos_token_id for s in samples]
    assert sum(ends) == 400  # the epoch is the 400 rows of the two sources
    short = sum(_letter(s) == "s" for s in samples)
    assert abs(short - 200) < 45, short  # about one pass of the short documents...
    assert abs((len(samples) - short) - total_long) < 0.2 * total_long  # ...and of the long ones' chunks
    estimate = estimate_run(config, TOK, 1, 1)
    assert abs(estimate.examples_per_epoch - len(samples)) / len(samples) < 0.08, estimate.examples_per_epoch


def test_mix_counts_rows_and_keeps_sequence_draws_for_single_sequence_rows():
    """Sources whose rows are one sequence each (chat data) are drawn as before: `epoch_draws` sequences."""
    a = [{"input_ids": torch.zeros(4, dtype=torch.long), "s": "a"} for _ in range(30)]
    b = [{"input_ids": torch.zeros(4, dtype=torch.long), "s": "b"} for _ in range(30)]
    for unit in ("sequences", "rows"):
        out = [x["s"] for x in MixedDataset([a, b], [0.5, 0.5], seed=3, epoch_draws=40, weight_unit=unit)]
        assert len(out) == 40
        reference = random.Random(3)
        assert out == ["ab"[reference.choices([0, 1], weights=[0.5, 0.5], k=1)[0]] for _ in range(40)]
    with pytest.raises(ValueError, match="weight_unit"):
        MixedDataset([a, b], [0.5, 0.5], weight_unit="documents")


@needs_tok
def test_replay_epoch_is_one_pass_over_the_target_rows(tmp_path):
    """Replay text is drawn alongside: the epoch ends when the target's rows are done, every chunk of a
    raw-text target included."""
    rows = [_doc("target", i, 150) for i in range(40)]
    replay = [_doc("replay", i, 30) for i in range(500)]
    chunks = sum(_chunks(r["text"], 64) for r in rows)
    config = DataConfig(
        sources=[{"dataset": _write_jsonl(tmp_path / "t.jsonl", rows), "mode": "pretrain"}],
        pretrain_replay_dataset=_write_jsonl(tmp_path / "replay.jsonl", replay),
        pretrain_replay_weight=0.25,
        max_seq_length=64,
        streaming=False,
    )
    samples = list(build_dataset(config, TOK, config, 0, 1, 1))
    target = [s for s in samples if _letter(s) == "t"]
    others = len(samples) - len(target)
    assert len(target) == chunks
    assert 0.15 < others / len(samples) < 0.35  # about the replay weight, per sequence
    # replay text is trained as it always was: no end-of-document token (the EOS of a chat tokenizer is
    # its end-of-turn token), while the raw-text source's documents each end with one
    eos = TOK.eos_token_id
    assert not any((s["input_ids"] == eos).any() for s in samples if _letter(s) == "r")
    assert sum(int(s["input_ids"][-1]) == eos for s in target) == len(rows)
    estimate = estimate_run(config, TOK, 1, 1)
    assert abs(estimate.examples_per_epoch - len(samples)) / len(samples) < 0.12


class _Rows(torch.utils.data.IterableDataset):
    def __init__(self, n: int, tag: str):
        self.n, self.tag = n, tag

    def __iter__(self):
        for _ in range(self.n):
            yield {"input_ids": torch.zeros(4, dtype=torch.long), "s": self.tag}


def test_epoch_ends_when_no_counted_source_is_left(caplog):
    """The target yields nothing and is dropped: only the uncounted replay is left, and the epoch must end
    instead of drawing replay for ever."""
    import itertools

    mix = MixedDataset(
        [_Rows(0, "target"), _Rows(3, "replay")], [0.9, 0.1], epoch_draws=5, counted=[True, False], names=["t", "r"]
    )
    with caplog.at_level("WARNING"):
        out = list(itertools.islice(mix, 200))
    assert len(out) <= 3, len(out)
    assert "No data source that counts towards the epoch is left" in caplog.text
    # with a target that yields, replay restarts as often as needed and the epoch is the target's rows
    mix = MixedDataset([_Rows(4, "target"), _Rows(3, "replay")], [0.5, 0.5], epoch_draws=8, counted=[True, False])
    out = [x["s"] for x in itertools.islice(mix, 500)]
    assert out.count("target") == 8 and len(out) < 100


@needs_tok
def test_default_eos_that_is_the_end_of_turn_token_warns(caplog):
    from palingenesis import data

    assert TOK.eos_token in TOK.chat_template  # the test tokenizer is a chat tokenizer
    data._END_OF_TURN_WARNED.clear()
    with caplog.at_level("WARNING"):
        PretrainDataset([], TOK, 64)
        PretrainDataset([], TOK, 64)
    assert caplog.text.count("chat template's end-of-turn token") == 1  # once, and it names the fix
    assert "<|endoftext|>" in caplog.text
    caplog.clear()
    data._END_OF_TURN_WARNED.clear()
    with caplog.at_level("WARNING"):
        PretrainDataset([], TOK, 64, append_eos="<|endoftext|>")
        PretrainDataset([], TOK, 64, append_eos=False)
        PretrainDataset([], H.tokenizer(), 64)  # EOS <|endoftext|>, not written in its template
    assert "end-of-turn" not in caplog.text


# ── end-of-document token ─────────────────────────────────────────────────────


@needs_tok
def test_append_eos_default_string_and_off():
    text = "Il treno parte alle otto dal binario tre."
    ids = TOK(text)["input_ids"]
    eos = TOK.eos_token_id

    default = PretrainDataset([], TOK, 64)._process({"text": text})[0]
    assert default["input_ids"].tolist() == ids + [eos]
    assert default["labels"].tolist() == ids + [eos]  # the end of the document is a target too
    # a document that already ends with the token keeps one
    again = PretrainDataset([], TOK, 64)._process({"text": text + TOK.eos_token})[0]
    assert again["input_ids"].tolist() == ids + [eos]
    off = PretrainDataset([], TOK, 64, append_eos=False)._process({"text": text})[0]
    assert off["input_ids"].tolist() == ids
    named = PretrainDataset([], TOK, 64, append_eos="!")._process({"text": text})[0]
    assert named["input_ids"].tolist() == ids + TOK("!")["input_ids"]

    assert document_end_id(TOK, True) == eos and document_end_id(TOK, False) is None
    with pytest.raises(ValueError, match="not a token of this tokenizer"):
        document_end_id(TOK, "<|no_such_token|>")
    with pytest.raises(ValueError, match="true, false or a token string"):
        document_end_id(TOK, 3)

    class NoEos:
        eos_token_id = None

    assert document_end_id(NoEos(), True) is None  # nothing to append: a warning, not an error


@needs_tok
def test_long_document_gets_eos_on_its_last_chunk_only():
    chunks = PretrainDataset([], TOK, 64)._process(_doc("long", 0, 150))
    assert len(chunks) >= 3
    eos = TOK.eos_token_id
    assert [int(c["input_ids"][-1]) == eos for c in chunks] == [False] * (len(chunks) - 1) + [True]
    assert all((c["input_ids"] == eos).sum() == 0 for c in chunks[:-1])
    marks = [end for _, end in PretrainDataset([_doc("long", 0, 150), _doc("s", 1, 5)], TOK, 64).iter_marked()]
    assert marks == [False] * (len(chunks) - 1) + [True, True]


@needs_tok
def test_exact_multiple_of_max_seq_length_keeps_its_eos():
    """A document of exactly k x max_seq_length tokens: the EOS would be a 1-token tail. It is kept, in a
    last chunk that reaches back into the previous one; nothing of the document is lost."""
    eos = TOK.eos_token_id
    ids = TOK(" ".join(f"w{i}" for i in range(400)))["input_ids"][:128]
    assert TOK(TOK.decode(ids))["input_ids"] == ids
    for extra in (0, 1, 14):  # tails of 1, 2 and 15 tokens with the EOS
        tokens = ids + ids[:extra]
        assert TOK(TOK.decode(tokens))["input_ids"] == tokens
        chunks = PretrainDataset([], TOK, 64)._process({"text": TOK.decode(tokens)})
        assert [c["input_ids"].numel() for c in chunks] == [64, 64, 16]
        last = chunks[-1]
        assert int(last["input_ids"][-1]) == eos and last["labels"].tolist() == last["input_ids"].tolist()
        assert last["input_ids"].tolist() == (tokens + [eos])[-16:]
        assert sum(int((c["input_ids"] == eos).sum()) for c in chunks) == 1
        # without the token the short tail is dropped, as before
        plain = PretrainDataset([], TOK, 64, append_eos=False)._process({"text": TOK.decode(tokens)})
        assert [c["input_ids"].numel() for c in plain] == [64, 64]
    single = PretrainDataset([], TOK, 64)._process({"text": TOK.decode(ids[:64])})
    assert [c["input_ids"].numel() for c in single] == [64, 16] and int(single[-1]["input_ids"][-1]) == eos


@needs_tok
def test_packed_documents_are_separated_by_eos_and_the_boundary_is_not_trained():
    """In a packed block every document ends with EOS, EOS is predicted from the document's own last
    token, and no position is asked to predict the first token of the next document."""
    docs = [{"text": f"Documento {i}: " + "parola " * (4 + i)} for i in range(4)]
    block = next(iter(PackedDataset(PretrainDataset(docs, TOK, 64), 64, sort_buffer=0)))
    ids, positions = block["input_ids"], block["position_ids"]
    starts = (positions == 0).nonzero().flatten().tolist()
    assert len(starts) >= 2
    eos = TOK.eos_token_id
    ends = [s - 1 for s in starts[1:]] + [ids.numel() - 1]
    assert all(int(ids[e]) == eos for e in ends)  # each document closes with the token

    shifted = shift_labels(block["labels"][None], positions[None])[0]
    for end in ends:
        assert int(shifted[end - 1]) == eos  # the token before predicts the end of its document
        assert int(shifted[end]) == -100  # EOS predicts nothing: not the next document's first token
    trained = (shifted != -100).sum().item()
    assert trained == ids.numel() - len(starts)  # every token but each document's first is a target


# ── weights by token share ────────────────────────────────────────────────────


def _token_shares(samples, tag_of) -> dict:
    tokens = Counter()
    for s in samples:
        tokens[tag_of(s)] += s["input_ids"].numel()
    total = sum(tokens.values())
    return {k: v / total for k, v in tokens.items()}


@needs_tok
def test_weights_by_token_share(tmp_path, monkeypatch):
    """Equal weights on a source of long sequences and one of short ones: by sequence the long source
    takes most of the tokens, by token each takes its weight; the size estimate follows either."""
    monkeypatch.setenv("PALINGENESIS_SIZE_SAMPLE", "400")
    long_rows = [_doc("long", i, 55) for i in range(400)]
    short_rows = [_doc("short", i, 5) for i in range(400)]
    assert all(_chunks(r["text"], 128) == 1 for r in long_rows)  # one sequence per row in both sources
    config = DataConfig(
        sources=[
            {"dataset": _write_jsonl(tmp_path / "long.jsonl", long_rows), "mode": "pretrain", "weight": 0.5},
            {"dataset": _write_jsonl(tmp_path / "short.jsonl", short_rows), "mode": "pretrain", "weight": 0.5},
        ],
        max_seq_length=128,
        streaming=False,
    )
    tag = _letter

    by_sequence = list(build_dataset(config, TOK, config, 0, 1, 1))
    assert _token_shares(by_sequence, tag)["l"] > 0.8
    sequences = Counter(tag(s) for s in by_sequence)
    assert abs(sequences["l"] / len(by_sequence) - 0.5) < 0.06

    config.mix_weights = "tokens"
    assert mix_weight_unit(config) == "tokens"
    by_token = list(build_dataset(config, TOK, config, 0, 1, 1))
    shares = _token_shares(by_token, tag)
    assert abs(shares["l"] - 0.5) < 0.05, shares
    assert Counter(tag(s) for s in by_token)["s"] > 3 * Counter(tag(s) for s in by_token)["l"]
    # the epoch is still the sources' rows together
    assert len(by_token) == len(by_sequence) == 800
    estimate = estimate_run(config, TOK, 1, 1)
    assert abs(estimate.examples_per_epoch - 800) < 1
    probabilities = data_size.mixture_epoch(estimate.sources, 1, 800)[1]
    assert probabilities[1] > 3 * probabilities[0]

    # unequal shares, and a cache built for one unit is not read for the other
    config.sources[0]["weight"], config.sources[1]["weight"] = 0.8, 0.2
    shares = _token_shares(list(build_dataset(config, TOK, config, 0, 1, 1)), tag)
    assert abs(shares["l"] - 0.8) < 0.05, shares

    class Cfg:
        data = config

    tokens_print = pretokenize_fingerprint(Cfg, TOK)
    config.mix_weights = "sequences"
    assert pretokenize_fingerprint(Cfg, TOK) != tokens_print


def test_sequence_weight_units():
    source = data_size.SourceEstimate("s", 100, 0.5, examples_per_row=3.0, tokens_per_row=600.0, yield_share=0.75)
    assert source.sequences_per_row == 4.0  # the rows that give anything give 4 sequences
    assert data_size.sequence_weight(source, "sequences") == 0.5
    assert data_size.sequence_weight(source, "tokens") == pytest.approx(0.5 / 200)
    assert data_size.sequence_weight(source, "rows") == 2.0
    # an epoch of 100 rows of this source alone: 400 sequences
    assert data_size.mixture_epoch([source], 1, 100)[0] == pytest.approx(400)
    # replay drawn alongside does not count towards the epoch's rows
    chat = data_size.SourceEstimate("chat", 100, 0.75, examples_per_row=1.0, tokens_per_row=50.0)
    replay = data_size.SourceEstimate("replay", 1000, 0.25, examples_per_row=1.0, tokens_per_row=50.0, counted=False)
    assert data_size.mixture_epoch([chat, replay], 1, 100)[0] == pytest.approx(100 / 0.75)


# ── random initialisation ─────────────────────────────────────────────────────


def _config_only(tmp_path, tok) -> Path:
    """A model directory with config.json and the tokenizer, and no weights."""
    path = tmp_path / "arch"
    H.tiny_llama(len(tok)).config.save_pretrained(path)
    tok.save_pretrained(path)
    assert not list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
    return path


def test_random_init_is_seeded_and_leaves_the_generator_alone(tmp_path):
    from palingenesis import model_init

    tok = H.tokenizer()
    path = _config_only(tmp_path, tok)
    torch.manual_seed(123)
    expected_next = torch.rand(3)
    torch.manual_seed(123)
    a = model_init.random_init(str(path), seed=7, tokenizer=tok)
    assert torch.equal(torch.rand(3), expected_next)  # the run's own seeding is untouched
    b = model_init.random_init(str(path), seed=7)
    c = model_init.random_init(str(path), seed=8)
    state_a, state_b, state_c = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)  # every rank draws the same weights
    assert any(not torch.equal(state_a[k], state_c[k]) for k in state_a)
    assert next(a.parameters()).dtype == torch.float32
    assert next(model_init.random_init(str(path), seed=7, dtype=torch.bfloat16).parameters()).dtype == torch.bfloat16

    # weights at the path are ignored
    saved = H.save_model(tmp_path / "trained", tok, seed=5)
    fresh = model_init.random_init(str(tmp_path / "trained"), seed=7)
    assert any(not torch.equal(v, saved.state_dict()[k]) for k, v in fresh.state_dict().items())
    assert all(torch.equal(v, state_a[k]) for k, v in fresh.state_dict().items())

    small = json.loads((path / "config.json").read_text())
    small["vocab_size"] = 100
    (path / "config.json").write_text(json.dumps(small))
    with pytest.raises(ConfigError, match="vocab_size 100"):
        model_init.random_init(str(path), seed=7, tokenizer=tok)


def test_random_init_in_low_precision_matches_a_loaded_checkpoint(tmp_path, monkeypatch):
    """bf16: built directly in bf16 (one copy in memory, as from_pretrained), parameters bf16 and the
    float buffers (rotary frequencies) float32, as in a loaded checkpoint."""
    import transformers

    from palingenesis import model_init

    tok = H.tokenizer()
    path = _config_only(tmp_path, tok)
    H.save_model(tmp_path / "trained", tok)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "trained", dtype=torch.bfloat16)
    reference = {name: b.dtype for name, b in loaded.named_buffers()}
    assert any(d == torch.float32 for d in reference.values())  # the architecture has float buffers

    calls = []
    real = transformers.AutoModelForCausalLM.from_config

    def spy(config, **kwargs):
        calls.append(kwargs.get("dtype"))
        return real(config, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", spy)
    model = model_init.random_init(str(path), seed=3, dtype=torch.bfloat16)
    assert calls == [torch.bfloat16]  # never the float32 model first
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert {name: b.dtype for name, b in model.named_buffers()} == reference

    # an initialiser that cannot draw in bf16: float32, then the parameters only are cast
    def failing(config, **kwargs):
        calls.append(kwargs.get("dtype"))
        if kwargs.get("dtype") != torch.float32:
            raise RuntimeError("normal_ is not implemented for this dtype")
        return real(config, **kwargs)

    calls.clear()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", failing)
    fallback = model_init.random_init(str(path), seed=3, dtype=torch.bfloat16)
    assert calls == [torch.bfloat16, torch.float32]
    assert all(p.dtype == torch.bfloat16 for p in fallback.parameters())
    assert {name: b.dtype for name, b in fallback.named_buffers()} == reference


def test_random_init_forks_every_cuda_generator(tmp_path, monkeypatch):
    """torch.manual_seed seeds CUDA too: the initialisation runs inside fork_rng over every CUDA device,
    so each rank keeps the CUDA state its own seed + rank gave it."""
    import contextlib

    from palingenesis import model_init

    path = _config_only(tmp_path, H.tokenizer())
    forked = []
    real = torch.random.fork_rng

    @contextlib.contextmanager
    def fork(devices=None, **kwargs):
        forked.append(list(devices))
        with real(devices=[]):
            yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.random, "fork_rng", fork)
    model_init.random_init(str(path), seed=7)
    assert forked == [[0, 1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_random_init_leaves_the_cuda_generator_alone(tmp_path):
    from palingenesis import model_init

    path = _config_only(tmp_path, H.tokenizer())
    torch.cuda.manual_seed_all(123)
    expected = torch.rand(3, device="cuda")
    torch.cuda.manual_seed_all(123)
    model_init.random_init(str(path), seed=7)
    assert torch.equal(torch.rand(3, device="cuda"), expected)


def test_random_init_is_refused_for_preference_training():
    from palingenesis.config import Config

    config = Config()
    config.model.init, config.model.torch_dtype, config.dpo.enabled = "random", "float32", True
    with pytest.raises(ConfigError, match="model.init: random does not support dpo.enabled"):
        config.validate()


def test_training_from_random_weights_runs_and_exports(tmp_path):
    """model.init: random on a directory without weights: raw-text training, loss falls, `final` loads."""
    from transformers import AutoModelForCausalLM

    from palingenesis import model_init

    tok = H.tokenizer()
    path = _config_only(tmp_path, tok)
    rows = [{"text": "uno due tre quattro cinque sei sette otto nove dieci " * 4} for _ in range(64)]
    c = H.cpu_config(tmp_path, path, "unused", steps=12)
    c.data.sources = [{"dataset": _write_jsonl(tmp_path / "text.jsonl", rows), "mode": "pretrain"}]
    c.model.init = "random"
    c.model.init_seed = 11
    c.train.learning_rate = 1e-2
    c.train.epochs = 100
    assert c.validate() == []  # no warning at a from-scratch learning rate
    start = model_init.random_init(str(path), seed=11)
    ids = torch.tensor([tok(rows[0]["text"])["input_ids"]])
    with torch.no_grad():
        before = start(input_ids=ids, labels=ids).loss.item()
    result = H.run_cpu(c)
    assert result.global_step == 12
    final = AutoModelForCausalLM.from_pretrained(Path(c.train.output_dir) / "final")
    with torch.no_grad():
        after = final(input_ids=ids, labels=ids).loss.item()
    assert after < before - 1.0, (before, after)


# ── raw-text cleaning ─────────────────────────────────────────────────────────

BENCH_ITEM = (
    "A train leaves the station at eight in the morning and travels north at sixty kilometres per hour "
    "while a second train leaves two hours later"
)


def _corpus() -> list[dict]:
    rng = random.Random(0)
    vocabulary = [f"term{i}" for i in range(400)]

    def article(i: int) -> str:
        return f"Article {i}. " + " ".join(rng.choice(vocabulary) for _ in range(120))

    docs = [article(i) for i in range(20)]
    rows = [{"text": d, "id": f"d{i}"} for i, d in enumerate(docs)]
    rows.append({"text": docs[3].upper().replace(" ", "  "), "id": "exact"})  # same text up to case / spaces
    near = docs[5].split()
    near[10:13] = ["edited", "words", "here"]
    rows.append({"text": " ".join(near), "id": "near"})  # a lightly edited copy
    rows.append({"text": f"{docs[7][:200]} As the exercise says: {BENCH_ITEM}, so what then?", "id": "leak"})
    rows.append({"text": "   ", "id": "empty"})
    rows.append({"id": "missing"})
    rows.append({"text": "Too short.", "id": "short"})
    return rows


def test_text_clean_checks():
    from palingenesis.data_clean import ROW_INDEX, CleanReport
    from palingenesis.text_clean import NearDuplicates, NgramIndex, clean_documents, lsh_bands, text_key

    assert text_key("Hello   World\n") == text_key("hello world") != text_key("hello, world")
    bench = NgramIndex(n=13)
    bench.add(BENCH_ITEM)
    bench.add("too short to give a thirteen word n-gram")
    assert bench.too_short == 1 and bench.hashes
    assert bench.contaminated(f"Intro text. {BENCH_ITEM.upper()}!")  # case and punctuation do not hide it
    assert not bench.contaminated("A train leaves the station at eight in the morning and goes home.")

    bands, rows = lsh_bands(0.8)
    assert bands * rows <= 128 and 0.6 < (1 / bands) ** (1 / rows) < 0.85  # the S-curve turns near the threshold
    near = NearDuplicates(0.8)
    base = " ".join(f"w{i}" for i in range(300))
    assert near.check(base, "first") is None
    assert near.check(base.replace("w150", "changed"), "second") == "first"
    assert near.check(" ".join(f"x{i}" for i in range(300)), "third") is None
    assert near.check("", "fourth") is None  # no words: nothing to compare

    samples = _corpus()
    for i, s in enumerate(samples):
        s[ROW_INDEX] = i
    report = CleanReport(loaded=len(samples))
    kept = clean_documents(samples, report, dedupe=True, near_dedupe=0.8, bench=bench, min_chars=50)
    assert [s["id"] for s in kept] == [f"d{i}" for i in range(20)]
    assert dict(report.reasons) == {
        "duplicate": 1,
        "near_duplicate": 1,
        "contaminated": 1,
        "empty_text": 2,
        "too_short": 1,
    }
    by_reason = {d["reason"]: d for d in report.dropped}
    assert by_reason["duplicate"]["duplicate_of"] == 3 and by_reason["near_duplicate"]["duplicate_of"] == 5
    # the checks can be turned off one by one
    report = CleanReport(loaded=len(samples))
    assert len(clean_documents(samples, report, dedupe=False)) == 24 and set(report.reasons) == {"empty_text"}


def test_decontamination_reads_fields_and_ignores_shared_instructions(tmp_path):
    """A benchmark whose every row repeats one instruction: the instruction must not remove the corpus
    documents that quote it; only the items do. `fields` limits what is read."""
    from palingenesis.text_clean import NgramIndex

    instruction = (
        "Answer the following multiple choice question by writing only the letter of the single correct option"
    )
    rng = random.Random(1)
    stems = ["alpha", "beta", "gamma", "delta", "kappa"]

    def item(i: int) -> str:
        return f"Question {i}: " + " ".join(rng.choice(stems) + str(rng.randrange(50)) for _ in range(20))

    items = [item(i) for i in range(30)]
    rows = [
        {"id": f"bench-{i:04d}", "system": instruction, "question": q, "meta": {"note": instruction}}
        for i, q in enumerate(items)
    ]
    bench = _write_jsonl(tmp_path / "bench.jsonl", rows)
    quoting = f"A study guide. {instruction}. That is how most exams phrase it, and it says nothing about the content."
    leaked = f"Forum post: {items[4]} Does anyone know?"

    index = NgramIndex.from_files([bench])
    assert index.shared > 0 and index.summary()["shared_ngrams_ignored"] == index.shared
    assert not index.contaminated(quoting) and index.contaminated(leaked)
    # without the rule the instruction removes every document that quotes it
    assert NgramIndex.from_files([bench], shared_row_share=1.0).contaminated(quoting)

    by_field = NgramIndex.from_files([bench], fields=["question"], shared_row_share=1.0)
    assert by_field.strings == 30 and by_field.summary()["fields"] == ["question"]
    assert not by_field.contaminated(quoting) and by_field.contaminated(leaked)
    with pytest.raises(ValueError, match="no row of .*bench.jsonl has any of the fields"):
        NgramIndex.from_files([bench], fields=["prompt"])
    # a small file (a handful of rows) is indexed whole: a repeated string there may be the item itself
    small = _write_jsonl(tmp_path / "small.jsonl", rows[:3])
    assert NgramIndex.from_files([small]).contaminated(quoting)


def test_eval_reader_and_resume_warnings(caplog):
    from palingenesis.config import Config
    from palingenesis.train import _warn_stream_change, eval_reader

    config = Config()
    config.data.sources = [
        {"dataset": "a.jsonl", "mode": "pretrain"},
        {"dataset": "b.jsonl", "mode": "pretrain", "append_eos": False},
    ]
    with caplog.at_level("WARNING"):
        eval_reader(config, H.tokenizer(), [])
    assert "first source's text_field and append_eos" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert _warn_stream_change(config, {"step": 10}) is True  # a checkpoint from before the change
        assert _warn_stream_change(config, {"step": 10, "data_stream": 2}) is False
        config.data.sources = [{"dataset": "chat.jsonl"}]
        assert _warn_stream_change(config, {"step": 10}) is False  # no raw text: nothing changed
    assert caplog.text.count("no end-of-document token") == 1


def test_checkpoint_meta_records_the_stream_version(tmp_path):
    from palingenesis.checkpoint import DATA_STREAM_VERSION, TRAINING_META_FILE, _save_meta

    _save_meta(None, 3, 0, 5, tmp_path)
    assert json.loads((tmp_path / TRAINING_META_FILE).read_text())["data_stream"] == DATA_STREAM_VERSION == 2


def test_prepare_text_writes_data_report_and_holdout(tmp_path):
    from palingenesis.data import _load_dataset_source
    from palingenesis.prepare import PREPARED_META, find_prepared_eval, prepare_text

    data = _write_jsonl(tmp_path / "corpus.jsonl", _corpus())
    bench = _write_jsonl(tmp_path / "bench.jsonl", [{"question": BENCH_ITEM, "answer": "At noon."}])
    out = prepare_text(
        data,
        str(tmp_path / "out"),
        near_dedupe=0.8,
        decontaminate=[bench],
        min_chars=50,
        eval_holdout=4,
        tokenizer=H.tokenizer(),
    )
    meta = json.loads((tmp_path / "out" / PREPARED_META).read_text())
    assert meta["mode"] == "pretrain" and meta["num_samples"] == 16 and meta["eval_holdout"] == 4
    assert meta["cleaning"]["loaded"] == 26
    assert meta["cleaning"]["dropped"] == {
        "contaminated": 1,
        "duplicate": 1,
        "empty_text": 2,
        "near_duplicate": 1,
        "too_short": 1,
    }
    assert meta["decontamination"]["ngrams"] > 0 and meta["decontamination"]["strings"] == 2
    assert meta["tokens"]["total"] > meta["text"]["words"] > 0
    dropped = [json.loads(line) for line in (tmp_path / "out" / "dropped.jsonl").read_text().splitlines()]
    assert len(dropped) == 6 and {d["reason"] for d in dropped} == set(meta["cleaning"]["dropped"])

    train_ids = {r["id"] for r in _load_dataset_source(str(tmp_path / "out"), "train", streaming=False)}
    held_ids = {r["id"] for r in _load_dataset_source(str(find_prepared_eval(tmp_path / "out")), "train", False)}
    assert out.suffix == ".parquet" and len(train_ids) == 16 and len(held_ids) == 4
    assert not train_ids & held_ids and train_ids | held_ids == {f"d{i}" for i in range(20)}

    with pytest.raises(ValueError, match="0 documents after cleaning"):
        prepare_text(data, str(tmp_path / "wrong"), text_field="body")
    with pytest.raises(FileNotFoundError, match="decontamination file not found"):
        prepare_text(data, str(tmp_path / "wrong"), decontaminate=[str(tmp_path / "nope.jsonl")])


def test_prepare_config_and_cli_clean_raw_text_sources(tmp_path, monkeypatch, caplog):
    """`pgs prepare --config` with mode: pretrain sources, and `pgs prepare --mode pretrain` without a model."""
    import sys

    import yaml

    from palingenesis import prepare

    data = _write_jsonl(tmp_path / "corpus.jsonl", _corpus())
    bench = _write_jsonl(tmp_path / "bench.jsonl", [{"question": BENCH_ITEM}])
    config = {
        "data": {"sources": [{"dataset": data, "name": "domain", "mode": "pretrain"}]},
        "preprocess": {
            "output_dir": str(tmp_path / "prepared"),
            "format": "jsonl",
            "near_dedupe": 0.8,
            "decontaminate": [bench],
            "min_chars": 50,
        },
    }
    (tmp_path / "cpt.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["pgs", "--config", str(tmp_path / "cpt.yaml")])
    prepare.main()
    lines = (tmp_path / "prepared" / "domain" / "scored_data.jsonl").read_text().splitlines()
    assert len(lines) == 20

    monkeypatch.setattr(
        sys,
        "argv",
        ["pgs", "--data", data, "--output", str(tmp_path / "cli"), "--mode", "pretrain", "--format", "jsonl"]
        + ["--decontaminate", bench, "--near_dedupe", "0.8"],
    )
    prepare.main()
    meta = json.loads((tmp_path / "cli" / prepare.PREPARED_META).read_text())
    assert meta["num_samples"] == 21 and meta["tokens"] == {}  # min_chars off: the short document stays
    assert meta["cleaning"]["dropped"] == {"contaminated": 1, "duplicate": 1, "empty_text": 2, "near_duplicate": 1}

    # raw-text sources in prepare-multi: cleaned next to the chat sources, with their token counts
    class FakeScorer:
        tokenizer = H.tokenizer()

    monkeypatch.setattr(prepare, "Scorer", lambda name: FakeScorer())
    prepare.prepare_multi_source(
        "fake",
        [{"name": "text", "dataset": data, "mode": "pretrain", "min_chars": 50}],
        str(tmp_path / "multi"),
        decontaminate=[bench],
        near_dedupe=0.8,
    )
    manifest = json.loads((tmp_path / "multi" / "manifest.json").read_text())
    (source,) = manifest["sources"]
    assert source["mode"] == "pretrain" and source["num_samples"] == 20 and source["token_share"] == 1.0
    assert source["cleaning"]["dropped"]["near_duplicate"] == 1
    # per-source dedupe / max_samples reach the raw-text cleaning
    prepare.prepare_multi_source(
        "fake",
        [{"name": "text", "dataset": data, "mode": "pretrain", "dedupe": False, "max_samples": 26}],
        str(tmp_path / "multi_keep"),
    )
    (source,) = json.loads((tmp_path / "multi_keep" / "manifest.json").read_text())["sources"]
    assert "duplicate" not in source["cleaning"]["dropped"] and source["num_samples"] == 24
