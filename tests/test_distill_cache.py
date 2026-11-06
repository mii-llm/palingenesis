"""The top-k teacher cache: format round trip, resume, refusals (settings, vocabulary), collate-time
lookup on padded and packed batches with routing, the HF builder against the teacher's own
distribution, and the vLLM prompt-logprob parser."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

from palingenesis.distill.cache import (  # noqa: E402
    CacheError,
    CacheLookup,
    CacheMeta,
    CacheWriter,
    TopKCache,
    doc_spans,
    sequence_key,
    trained_positions,
    vocab_fingerprint,
)

IGNORE = -100


def _meta(k=3, vocab_hash="abc"):
    return CacheMeta(top_k=k, vocab_hash=vocab_hash, vocab_size=50, teacher="t")


def _doc(n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 50, (n,), generator=g)
    labels = ids.clone()
    labels[: n // 2] = IGNORE
    return ids, labels


def _targets(labels, k, seed):
    pos = trained_positions(labels.numpy())
    rng = np.random.default_rng(seed)
    return pos, rng.integers(0, 50, (len(pos), k)), -rng.random((len(pos), k)).astype(np.float32), -rng.random(len(pos))


def test_key_depends_on_tokens_and_mask():
    ids, labels = _doc(10, 0)
    k1 = sequence_key(ids, labels)
    assert k1 == sequence_key(ids.numpy(), labels.numpy())
    other = labels.clone()
    other[-1] = IGNORE
    assert sequence_key(ids, other) != k1
    assert sequence_key(ids[:-1], labels[:-1]) != k1


def test_write_read_roundtrip_and_resume(tmp_path):
    w = CacheWriter(tmp_path, _meta(), shard_rows=8)
    docs = [_doc(12, s) for s in range(5)]
    written = {}
    for s, (ids, labels) in enumerate(docs[:3]):
        key, check = sequence_key(ids, labels)
        t = _targets(labels, 3, s)
        w.add(key, check, len(ids), *t)
        written[s] = t
    w.close()
    cache = TopKCache(tmp_path)
    assert len(cache) == 3
    for s, (ids, labels) in enumerate(docs[:3]):
        got = cache.get(*sequence_key(ids, labels), len(ids))
        pos, ids_k, lp, gold = written[s]
        np.testing.assert_array_equal(got.positions, pos)
        np.testing.assert_array_equal(got.ids, ids_k)
        np.testing.assert_allclose(got.lp, lp, atol=1e-3)  # float16
        np.testing.assert_allclose(got.gold_lp, gold, rtol=1e-6)
    assert cache.get(*sequence_key(*docs[4]), 12) is None
    assert cache.get(*sequence_key(*docs[0]), 13) is None  # length is checked too

    # an interrupted shard (no index) is discarded; finished documents are skipped
    (tmp_path / "shard-r000-00099.ids.bin.tmp").write_bytes(b"x")
    w2 = CacheWriter(tmp_path, _meta(), shard_rows=8)
    assert not (tmp_path / "shard-r000-00099.ids.bin.tmp").exists()
    assert w2.resumed == 3 and w2.done(sequence_key(*docs[0])[0])
    for s, (ids, labels) in enumerate(docs):
        w2.add(*sequence_key(ids, labels), len(ids), *_targets(labels, 3, s))
    w2.close()
    assert w2.written_docs == 2
    assert len(TopKCache(tmp_path)) == 5


def test_other_settings_are_refused(tmp_path):
    CacheWriter(tmp_path, _meta(k=3)).close()
    with pytest.raises(CacheError, match="top_k"):
        CacheWriter(tmp_path, _meta(k=4))
    with pytest.raises(CacheError, match="not a distillation cache"):
        TopKCache(tmp_path / "missing")


def test_vocabulary_fingerprint_is_verified(tmp_path):
    tok = H.tokenizer()
    CacheWriter(tmp_path, _meta(vocab_hash=vocab_fingerprint(tok))).close()
    TopKCache(tmp_path).verify_tokenizer(tok)
    tok.add_tokens(["<extra_token_for_test>"])
    with pytest.raises(CacheError, match="another vocabulary"):
        TopKCache(tmp_path).verify_tokenizer(tok)


def test_doc_spans():
    attn = torch.tensor([1, 1, 1, 1, 1, 0, 0])
    assert doc_spans(attn, None) == [(0, 5)]
    pos = torch.tensor([0, 1, 2, 0, 1, 0, 1])
    assert doc_spans(torch.ones(7), pos) == [(0, 3), (3, 2), (5, 2)]


def _cache_with(tmp_path, docs, k=3, name="c"):
    w = CacheWriter(tmp_path / name, _meta(k=k))
    truth = {}
    for s, (ids, labels) in enumerate(docs):
        t = _targets(labels, k, s)
        w.add(*sequence_key(ids, labels), len(ids), *t)
        truth[s] = t
    w.close()
    return TopKCache(tmp_path / name), truth


def test_lookup_places_targets_in_padded_and_packed_batches(tmp_path):
    docs = [_doc(n, s) for s, n in enumerate((8, 6, 10))]
    cache, truth = _cache_with(tmp_path, docs)
    # padded: one doc per row
    width = 10
    ids = torch.zeros(2, width, dtype=torch.long)
    labels = torch.full((2, width), IGNORE)
    attn = torch.zeros(2, width, dtype=torch.long)
    for b, (d_ids, d_lab) in enumerate(docs[:2]):
        ids[b, : len(d_ids)], labels[b, : len(d_ids)], attn[b, : len(d_ids)] = d_ids, d_lab, 1
    out = CacheLookup({"c": cache})({"input_ids": ids, "labels": labels, "attention_mask": attn})
    for b in range(2):
        pos, k_ids, _, gold = truth[b]
        assert out["kd.c.hit"][b].nonzero().flatten().tolist() == pos.tolist()
        np.testing.assert_array_equal(out["kd.c.ids"][b, pos].numpy(), k_ids)
        np.testing.assert_allclose(out["kd.c.gold"][b, pos].numpy(), gold, rtol=1e-6)
    # packed: docs 0 and 2 in one row, then the collator's padding
    p_ids = torch.cat([docs[0][0], docs[2][0], torch.zeros(4, dtype=torch.long)])[None]
    p_lab = torch.cat([docs[0][1], docs[2][1], torch.full((4,), IGNORE)])[None]
    p_pos = torch.cat([torch.arange(8), torch.arange(10), torch.arange(4)])[None]
    out = CacheLookup({"c": cache})(
        {"input_ids": p_ids, "labels": p_lab, "attention_mask": torch.ones_like(p_ids), "position_ids": p_pos}
    )
    expected = truth[0][0].tolist() + (truth[2][0] + 8).tolist()
    assert out["kd.c.hit"][0].nonzero().flatten().tolist() == expected


def test_lookup_routing_by_source(tmp_path):
    docs = [_doc(8, 0)]
    cache, _ = _cache_with(tmp_path, docs)
    batch = {
        "input_ids": docs[0][0][None],
        "labels": docs[0][1][None],
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "source_ids": torch.full((1, 8), 1),
    }
    assert not CacheLookup({"c": cache}, {"c": {0}})(dict(batch))["kd.c.hit"].any()
    assert CacheLookup({"c": cache}, {"c": {1}})(dict(batch))["kd.c.hit"].any()


def test_parse_prompt_logprobs():
    from palingenesis.distill.build import parse_prompt_logprobs

    lp = lambda v, r: SimpleNamespace(logprob=v, rank=r)  # noqa: E731
    teacher_ids = [5, 7, 9, 2]
    entries = [
        None,
        {7: lp(-0.1, 1), 3: lp(-2.5, 2)},
        {4: lp(-0.2, 1), 9: lp(-3.0, 5), 1: lp(-1.9, 2)},
        {2: lp(-0.5, 1)},
    ]
    ids, lps, gold = parse_prompt_logprobs(entries, teacher_ids, [0, 1, 2], k=2)
    assert ids.tolist() == [[7, 3], [4, 1], [2, -1]]
    np.testing.assert_allclose(lps[:, 0], [-0.1, -0.2, -0.5])
    np.testing.assert_allclose(gold, [-0.1, -3.0, -0.5])  # the actual token, even outside the top-k


# ── building with an HF teacher ─────────────────────────────────────────────


def _build_setup(tmp_path, **teacher):
    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    teacher_model = H.tiny_llama(len(tok), hidden=48, seed=3)
    data = H.write_rows(tmp_path / "rows.jsonl")
    config = H.cpu_config(tmp_path, tmp_path / "student", data)
    config.distill.enabled = True
    config.distill.teachers = {
        "big": {"model": "unused-in-test", "cache": str(tmp_path / "cache"), "top_k": 5, **teacher}
    }
    return tok, teacher_model, config


def test_build_cache_matches_the_teachers_distribution(tmp_path):
    from palingenesis.data import _load_dataset_source, source_stage
    from palingenesis.distill.build import build_cache

    tok, teacher, config = _build_setup(tmp_path)
    stats = build_cache(
        config, "big", teacher_model=teacher, teacher_tokenizer=tok, student_tokenizer=tok, max_batch_tokens=256
    )
    assert stats["documents"] == len(H.ROWS)
    cache = TopKCache(tmp_path / "cache")
    cache.verify_tokenizer(tok)
    assert json.loads((tmp_path / "cache" / "meta.json").read_text())["top_k"] == 5
    stage = source_stage(_load_dataset_source(config.data.dataset, "train", False), tok, config.data, None, 0, 1)
    sample = next(iter(stage))
    doc = cache.get(*sequence_key(sample["input_ids"], sample["labels"]), int(sample["input_ids"].numel()))
    with torch.no_grad():
        device = next(teacher.parameters()).device  # the builder may have moved the teacher to a GPU
        lp = torch.log_softmax(teacher(input_ids=sample["input_ids"][None].to(device)).logits[0].float(), -1).cpu()
    pos = torch.from_numpy(np.array(doc.positions)).long()
    top = lp[pos].topk(5, -1)
    np.testing.assert_array_equal(np.asarray(doc.ids), top.indices.numpy())
    np.testing.assert_allclose(np.asarray(doc.lp, dtype=np.float32), top.values.numpy(), rtol=1e-3)  # float16
    gold = sample["input_ids"][pos + 1]
    np.testing.assert_allclose(np.asarray(doc.gold_lp), lp[pos, gold].numpy(), rtol=1e-4, atol=1e-5)

    again = build_cache(config, "big", teacher_model=teacher, teacher_tokenizer=tok, student_tokenizer=tok)
    assert again["documents"] == 0  # resumed: everything already there


def test_training_on_a_cache_and_missing_documents(tmp_path):
    from palingenesis.distill.build import build_cache
    from palingenesis.distill.distiller import MissingTeacherError

    tok, teacher, config = _build_setup(tmp_path)
    build_cache(config, "big", teacher_model=teacher, teacher_tokenizer=tok, student_tokenizer=tok, max_docs=24)
    config.distill.ce_weight = 0.1
    result = H.run_cpu(config)
    assert result.global_step == config.train.max_steps
    assert result.metrics["train/kd_coverage"] == 1.0
    assert 0 < result.metrics["train/topk_mass/big"] <= 1.0

    # a cache built from part of the data: the trainer stops at the first uncovered token
    other = tmp_path / "partial"
    config.distill.teachers["big"]["cache"] = str(other)
    build_cache(config, "big", teacher_model=teacher, teacher_tokenizer=tok, student_tokenizer=tok, max_docs=3)
    config.train.output_dir = str(tmp_path / "out2")
    with pytest.raises(MissingTeacherError, match="no teacher target"):
        H.run_cpu(config)
    config.distill.missing = "skip"
    result = H.run_cpu(config)
    assert result.metrics["train/kd_coverage"] < 1.0


def test_cache_of_a_teacher_view_is_in_student_positions(tmp_path):
    """A context-distillation cache: the teacher read the hint, the targets sit at the student's positions."""
    from palingenesis.data import _load_dataset_source, source_stage
    from palingenesis.distill.build import build_cache
    from palingenesis.distill.config import parse_teachers
    from palingenesis.distill.views import TeacherView

    tok, teacher, config = _build_setup(tmp_path, privileged_field="hint")
    build_cache(config, "big", teacher_model=teacher, teacher_tokenizer=tok, student_tokenizer=tok)
    cache = TopKCache(tmp_path / "cache")
    assert cache.meta.view["privileged_field"] == "hint"
    spec = parse_teachers(config.distill.teachers)[0]
    views = [(TeacherView(spec), tok, 256)]
    stage = source_stage(
        _load_dataset_source(config.data.dataset, "train", False), tok, config.data, None, 0, 1, teacher_views=views
    )
    sample = next(iter(stage))
    t_ids = sample["teacher_input_ids.big"]
    assert t_ids.numel() > sample["input_ids"].numel()  # the hint is in the teacher's copy only
    doc = cache.get(*sequence_key(sample["input_ids"], sample["labels"]), int(sample["input_ids"].numel()))
    with torch.no_grad():
        device = next(teacher.parameters()).device  # the builder may have moved the teacher to a GPU
        lp = torch.log_softmax(teacher(input_ids=t_ids[None].to(device)).logits[0].float(), -1).cpu()
    q = sample["teacher_positions.big"][torch.from_numpy(np.array(doc.positions)).long()]
    np.testing.assert_array_equal(np.asarray(doc.ids), lp[q].topk(5, -1).indices.numpy())
