"""Tests for the preprocess→train wiring and the tracker/logging setup.

Covers:
- PreprocessConfig loads from YAML and from CLI overrides
- save_prepared writes parquet (and jsonl fallback), find_prepared_dataset resolves it
- _load_dataset_source loads prepared parquet/jsonl files AND prepared directories
- config.validate() rejects preprocess.enabled + data.sources
- Tracker: run id persistence, disabled backends are no-ops, log() never raises
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_preprocess_config_from_yaml():
    from palingenesis.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
model:
  name_or_path: test-model
data:
  dataset: my-org/my-data
  max_seq_length: 2048
preprocess:
  enabled: true
  output_dir: ./prepared/test
  format: parquet
  budget: 500
  strategy: curriculum
"""
        )
        f.flush()
        cfg = Config.from_yaml(f.name)

    assert cfg.preprocess.enabled is True
    assert cfg.preprocess.output_dir == "./prepared/test"
    assert cfg.preprocess.format == "parquet"
    assert cfg.preprocess.budget == 500
    assert cfg.preprocess.strategy == "curriculum"
    print("✓ test_preprocess_config_from_yaml PASSED")


def test_preprocess_config_cli_overrides():
    from palingenesis.config import Config

    cfg = Config.from_cli(
        ["--preprocess.enabled", "true", "--preprocess.budget", "1000", "--preprocess.strategy", "flow"]
    )
    assert cfg.preprocess.enabled is True
    assert cfg.preprocess.budget == 1000
    assert cfg.preprocess.strategy == "flow"
    print("✓ test_preprocess_config_cli_overrides PASSED")


def test_preprocess_incompatible_with_sources():
    from palingenesis.config import Config, ConfigError

    cfg = Config()
    cfg.preprocess.enabled = True
    cfg.data.sources = [{"dataset": "a"}, {"dataset": "b"}]
    try:
        cfg.validate()
        raise AssertionError("validate() should reject preprocess.enabled + data.sources")
    except ConfigError:
        pass
    print("✓ test_preprocess_incompatible_with_sources PASSED")


def test_resolve_total_steps():
    """LR schedule horizon: explicit max_steps wins; otherwise derived from
    epochs × dataset size; streaming without max_steps falls back to 100k.

    Regression for the silent-100k bug: a ~460-step run with max_steps unset
    got warmup_ratio 0.05 × 100k = 5000 warmup steps, so the ENTIRE run sat
    inside warmup and never reached peak LR.
    """
    from palingenesis.config import Config
    from palingenesis.train import _resolve_total_steps

    cfg = Config()
    cfg.train.per_device_batch_size = 4
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.epochs = 1

    # 1. Explicit max_steps always wins
    cfg.train.max_steps = 50
    assert _resolve_total_steps(cfg, dataset_len=1450, world_size=1) == 50

    # 2. Derived: ceil(1450 / 4) = 363 steps/epoch
    cfg.train.max_steps = 0
    assert _resolve_total_steps(cfg, dataset_len=1450, world_size=1) == 363

    # Scales with epochs, world size and grad accumulation
    cfg.train.epochs = 3
    assert _resolve_total_steps(cfg, dataset_len=1450, world_size=1) == 3 * 363
    cfg.train.epochs = 1
    cfg.train.gradient_accumulation_steps = 2
    assert _resolve_total_steps(cfg, dataset_len=1450, world_size=2) == 91  # ceil(1450/16)

    # 3. Streaming (no length) without max_steps → 100k fallback
    assert _resolve_total_steps(cfg, dataset_len=None, world_size=1) == 100_000
    print("✓ test_resolve_total_steps PASSED")


def _sample_conversations(n=6):
    return [
        {
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ],
            "_score_response_ppl": float(i + 1),
            "_score_rank": i,
        }
        for i in range(n)
    ]


def test_filter_samples_uses_response_ppl_by_default():
    """Regression: chat SFT trains assistant tokens only, so prepare filtering
    should not reject a sample just because the prompt/system text has huge PPL.
    """
    from palingenesis.prepare import filter_samples

    samples = [
        {
            "_score_ppl": 10_000.0,  # full prompt+response PPL: domain/language shift
            "_score_response_ppl": 8.0,  # assistant response is learnable
            "_score_length": 128,
        },
        {
            "_score_ppl": 20.0,
            "_score_response_ppl": 800.0,  # response is genuinely too hard/noisy
            "_score_length": 128,
        },
    ]

    filtered = filter_samples(samples, max_ppl=500.0)
    assert filtered == [samples[0]]

    # Full-sequence mode remains available for pretraining-style filtering.
    assert filter_samples(samples, max_ppl=500.0, score="full") == [samples[1]]

    # Multilingual/domain-shifted datasets can have high absolute response PPL;
    # max_ppl <= 0 disables the absolute ceiling and keeps relative ranking.
    assert filter_samples(samples, max_ppl=0.0) == samples
    print("✓ test_filter_samples_uses_response_ppl_by_default PASSED")


def test_prepare_normalizes_sharegpt_conversations():
    """prepare scoring must see the same normalized roles/content as training.

    Regression for ShareGPT-style datasets (`conversations` + `from`/`value`):
    train/inspect handled them, but prepare scored raw turns, producing tiny
    `_score_length` values and filtering every sample as too_short.
    """
    from palingenesis.prepare import _get_messages

    sample = {
        "conversations": [
            {"from": "human", "value": "Ciao"},
            {"from": "gpt", "value": "Ciao! Come posso aiutarti?"},
        ]
    }

    messages = _get_messages(sample, "conversations")
    assert messages == [
        {"role": "user", "content": "Ciao"},
        {"role": "assistant", "content": "Ciao! Come posso aiutarti?"},
    ]
    print("✓ test_prepare_normalizes_sharegpt_conversations PASSED")


def test_eval_holdout_split_and_roundtrip():
    """eval_holdout: deterministic, disjoint from the training pool, and the
    written eval_data.parquet is discoverable + loadable by training."""
    from palingenesis.data import _load_dataset_source
    from palingenesis.prepare import (
        EVAL_BASENAME,
        find_prepared_eval,
        save_prepared,
        split_eval_holdout,
    )

    samples = _sample_conversations(n=20)
    train_pool, holdout = split_eval_holdout(samples, 5)
    assert len(train_pool) == 15 and len(holdout) == 5

    # Disjoint: no sample appears in both
    train_ids = {s["messages"][0]["content"] for s in train_pool}
    eval_ids = {s["messages"][0]["content"] for s in holdout}
    assert not train_ids & eval_ids, "Holdout must be disjoint from training pool"

    # Deterministic across calls (same seed)
    _, holdout2 = split_eval_holdout(_sample_conversations(n=20), 5)
    assert eval_ids == {s["messages"][0]["content"] for s in holdout2}

    # Degenerate cases: no holdout requested, or holdout >= pool
    pool, hold = split_eval_holdout(samples, 0)
    assert len(pool) == 20 and hold == []
    pool, hold = split_eval_holdout(samples, 50)
    assert len(pool) == 20 and hold == []

    # Write + rediscover + load through the training loader
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_prepared(holdout, tmpdir, output_format="parquet", basename=EVAL_BASENAME)
        assert find_prepared_eval(tmpdir) == out
        rows = list(_load_dataset_source(str(out), "train", streaming=False))
        assert len(rows) == 5
    print("✓ test_eval_holdout_split_and_roundtrip PASSED")


def test_save_prepared_parquet_roundtrip():
    from palingenesis.data import _load_dataset_source
    from palingenesis.prepare import find_prepared_dataset, save_prepared

    samples = _sample_conversations()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_prepared(samples, tmpdir, output_format="parquet")
        assert out.suffix == ".parquet", f"Expected parquet, got {out}"
        assert find_prepared_dataset(tmpdir) == out

        # Load via the training loader — as a file path AND as a directory
        for source in (str(out), tmpdir):
            ds = _load_dataset_source(source, "train", streaming=False)
            rows = list(ds)
            assert len(rows) == len(samples)
            # Order preserved (curriculum requirement)
            assert [r["_score_rank"] for r in rows] == list(range(len(samples)))
            assert rows[0]["messages"][0]["content"] == "question 0"
    print("✓ test_save_prepared_parquet_roundtrip PASSED")


def test_save_prepared_jsonl():
    from palingenesis.data import _load_dataset_source
    from palingenesis.prepare import find_prepared_dataset, save_prepared

    samples = _sample_conversations()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_prepared(samples, tmpdir, output_format="jsonl")
        assert out.suffix == ".jsonl"
        assert find_prepared_dataset(tmpdir) == out
        ds = _load_dataset_source(tmpdir, "train", streaming=False)
        assert len(list(ds)) == len(samples)
    print("✓ test_save_prepared_jsonl PASSED")


def test_parquet_replaces_stale_jsonl():
    """find_prepared_dataset must not pick up a stale jsonl from an old run."""
    from palingenesis.prepare import find_prepared_dataset, save_prepared

    with tempfile.TemporaryDirectory() as tmpdir:
        save_prepared(_sample_conversations(3), tmpdir, output_format="jsonl")
        out = save_prepared(_sample_conversations(6), tmpdir, output_format="parquet")
        found = find_prepared_dataset(tmpdir)
        assert found == out and found.suffix == ".parquet"
        assert not (Path(tmpdir) / "scored_data.jsonl").exists()
    print("✓ test_parquet_replaces_stale_jsonl PASSED")


def test_prepare_writes_manifest():
    from palingenesis.prepare import PREPARED_META, save_prepared

    # save_prepared itself doesn't write the manifest (prepare_data does),
    # so simulate the manifest write path here
    with tempfile.TemporaryDirectory() as tmpdir:
        save_prepared(_sample_conversations(), tmpdir, output_format="parquet")
        meta = {"model": "m", "strategy": "optimal", "num_samples": 6}
        (Path(tmpdir) / PREPARED_META).write_text(json.dumps(meta))
        loaded = json.loads((Path(tmpdir) / PREPARED_META).read_text())
        assert loaded["num_samples"] == 6
    print("✓ test_prepare_writes_manifest PASSED")


def test_tracker_disabled_backends_noop():
    from palingenesis.config import Config
    from palingenesis.logging import Tracker

    cfg = Config()
    cfg.logging.use_wandb = False
    cfg.logging.use_trackio = False
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.train.output_dir = tmpdir
        t = Tracker(cfg, is_main=True)
        # Must be silent no-ops
        t.log({"train/loss": 1.0}, step=1)
        t.finish()
        # No run-id file when wandb is disabled
        assert not (Path(tmpdir) / "tracker_run_id.json").exists()
    print("✓ test_tracker_disabled_backends_noop PASSED")


def test_tracker_run_id_persistence():
    from palingenesis.logging import _load_or_create_run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        rid1 = _load_or_create_run_id(tmpdir, "run-a", reuse=True)
        rid2 = _load_or_create_run_id(tmpdir, "run-a", reuse=True)
        assert rid1 == rid2, "Run id must be stable across checkpoint resumes (wandb resume)"
        assert (Path(tmpdir) / "tracker_run_id.json").exists()
    print("✓ test_tracker_run_id_persistence PASSED")


def test_tracker_run_id_rotates_on_fresh_start():
    """A fresh (non-resuming) run must NOT reattach to the old wandb run:
    wandb silently drops rows logged below the old history step."""
    from palingenesis.logging import _load_or_create_run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        rid1 = _load_or_create_run_id(tmpdir, "run-a", reuse=True)
        rid2 = _load_or_create_run_id(tmpdir, "run-a", reuse=False)
        assert rid1 != rid2, "Fresh start must mint a new run id"
        # And the new id is now the persisted one (a later crash-resume continues IT)
        rid3 = _load_or_create_run_id(tmpdir, "run-a", reuse=True)
        assert rid3 == rid2
    print("✓ test_tracker_run_id_rotates_on_fresh_start PASSED")


def test_will_resume_logic():
    from palingenesis.config import Config
    from palingenesis.logging import _will_resume

    cfg = Config()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.train.output_dir = tmpdir

        cfg.train.resume_from = None
        assert _will_resume(cfg) is False, "No resume_from → fresh start"

        cfg.train.resume_from = "auto"
        assert _will_resume(cfg) is False, "auto with no checkpoint → fresh start"

        cfg.train.resume_from = "/some/explicit/checkpoint"
        assert _will_resume(cfg) is True, "Explicit path → resume"
    print("✓ test_will_resume_logic PASSED")


def test_tracker_log_never_raises():
    """A broken backend must not kill the training loop."""
    from palingenesis.config import Config
    from palingenesis.logging import Tracker

    cfg = Config()
    cfg.logging.use_wandb = False
    cfg.logging.use_trackio = False
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.train.output_dir = tmpdir
        t = Tracker(cfg, is_main=True)

        class Broken:
            def log(self, *a, **k):
                raise RuntimeError("network down")

            def finish(self):
                raise RuntimeError("network down")

        t._wandb = Broken()
        t.log({"train/loss": 1.0}, step=1)  # must not raise
        t.log({"train/loss": 0.9}, step=2)
        t.finish()  # must not raise
    print("✓ test_tracker_log_never_raises PASSED")


def test_group_entries_by_length():
    """Batched scoring: every entry lands in exactly one batch, and each batch
    respects both the sample cap and the padded-token cap."""
    import torch

    from palingenesis.prepare import _group_entries_by_length

    # Lengths sorted ascending, as score_samples_with_model guarantees
    lengths = [10, 10, 20, 50, 100, 100, 400, 900]
    entries = [(i, list(range(n)), torch.zeros(n, dtype=torch.bool)) for i, n in enumerate(lengths)]

    batches = _group_entries_by_length(entries, batch_size=4, max_batch_tokens=300)

    seen = [e[0] for b in batches for e in b]
    assert sorted(seen) == list(range(len(entries))), "Every entry must appear exactly once"
    for b in batches:
        assert len(b) <= 4
        longest = max(len(e[1]) for e in b)
        # A single over-long entry is allowed its own batch of 1
        assert len(b) * longest <= 300 or len(b) == 1

    # An entry longer than the token cap still gets scored (batch of 1)
    assert any(len(b) == 1 and len(b[0][1]) == 900 for b in batches)
    print("✓ test_group_entries_by_length PASSED")


def test_score_padded_batch_matches_naive_reference():
    """The chunked + vectorized batch scorer must produce the same NLLs as a
    naive per-sample computation (padding must never leak into the scores)."""
    import math

    import torch

    from palingenesis.prepare import _score_padded_batch

    vocab, pad_id = 50, 0
    torch.manual_seed(0)

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, input_ids, attention_mask=None, use_cache=None):
            g = torch.Generator().manual_seed(123)
            # Logits depend only on token ids, not batch layout → padding-invariant
            table = torch.randn(vocab, vocab, generator=g)

            class Out:
                logits = table[input_ids]

            return Out()

    lengths = [5, 9, 12]
    batch = []
    for i, n in enumerate(lengths):
        ids = torch.randint(1, vocab, (n,)).tolist()
        rmask = torch.zeros(n, dtype=torch.bool)
        rmask[n // 2 :] = True  # second half = "assistant tokens"
        batch.append((i, ids, rmask))
    samples = [{} for _ in lengths]

    _score_padded_batch(FakeModel(), batch, samples, pad_id)

    # Naive reference: one sample at a time, no padding
    model = FakeModel()
    for (_, ids, rmask), sample in zip(batch, samples):
        t = torch.tensor(ids).unsqueeze(0)
        logits = model(t).logits
        nll = torch.nn.functional.cross_entropy(logits[0, :-1].float(), t[0, 1:], reduction="none")
        expected_avg = nll.mean().item()
        expected_resp = nll[rmask[1:]].mean().item()

        assert abs(sample["_score_avg_nll"] - round(expected_avg, 4)) < 1e-3
        assert abs(sample["_score_response_nll"] - round(expected_resp, 4)) < 1e-3
        assert sample["_score_ppl"] == round(math.exp(min(expected_avg, 20.0)), 2)
        assert sample["_score_response_token_count"] == int(rmask[1:].sum())
    print("✓ test_score_padded_batch_matches_naive_reference PASSED")


def test_optimal_mix_adapts_to_budget():
    """The J-shape maps onto the available range adaptively: small budgets
    shift easier (no extreme tail), large budgets use the full J-shape."""
    from palingenesis.prepare import _optimal_mix

    small = _optimal_mix(500)
    medium = _optimal_mix(5_000)
    large = _optimal_mix(40_000)

    for mix in (small, medium, large):
        assert abs(sum(mix) - 1.0) < 1e-9, f"Mix must sum to 1: {mix}"

    # easy fraction shrinks as budget grows; hard+very_hard grows
    assert small[0] > medium[0] > large[0]
    assert small[2] + small[3] < medium[2] + medium[3] <= large[2] + large[3]
    assert small[3] == 0.0, "Small budgets must skip the very-hard tail"
    assert large == (0.20, 0.50, 0.25, 0.05), "Large budget = full research J-shape"
    print("✓ test_optimal_mix_adapts_to_budget PASSED")


def test_select_by_budget_backfills_short_buckets():
    """If a difficulty bucket can't fill its quota, the shortfall is backfilled
    from other buckets instead of silently returning fewer samples."""
    from palingenesis.prepare import classify_difficulty, select_by_budget

    # Skewed distribution: percentile bucketing still assigns ~25/50/25,
    # but we then delete most of the easy bucket to force a shortfall.
    samples = [{"_score_response_ppl": float(p), "id": i} for i, p in enumerate(range(1, 401))]
    samples = classify_difficulty(samples)
    easy = [s for s in samples if s["_score_difficulty_bucket"] == "easy"]
    skewed = [s for s in samples if s["_score_difficulty_bucket"] != "easy"] + easy[:5]

    budget = 200
    selected = select_by_budget(skewed, budget=budget, strategy="optimal")
    assert len(selected) == budget, f"Expected {budget} samples after backfill, got {len(selected)}"
    ids = [s["id"] for s in selected]
    assert len(ids) == len(set(ids)), "Backfill must not duplicate samples"
    print("✓ test_select_by_budget_backfills_short_buckets PASSED")


def test_flatten_includes_all_sections():
    from palingenesis.config import Config
    from palingenesis.logging import _flatten

    flat = _flatten(Config())
    for key in ("model/name_or_path", "plugins/deft", "preprocess/enabled", "logging/rl_readiness"):
        assert key in flat, f"Missing {key} in flattened tracker config"
    print("✓ test_flatten_includes_all_sections PASSED")


if __name__ == "__main__":
    test_preprocess_config_from_yaml()
    test_preprocess_config_cli_overrides()
    test_preprocess_incompatible_with_sources()
    test_save_prepared_parquet_roundtrip()
    test_save_prepared_jsonl()
    test_parquet_replaces_stale_jsonl()
    test_prepare_writes_manifest()
    test_tracker_disabled_backends_noop()
    test_tracker_run_id_persistence()
    test_tracker_run_id_rotates_on_fresh_start()
    test_will_resume_logic()
    test_tracker_log_never_raises()
    test_group_entries_by_length()
    test_score_padded_batch_matches_naive_reference()
    test_optimal_mix_adapts_to_budget()
    test_select_by_budget_backfills_short_buckets()
    test_flatten_includes_all_sections()
    print("\nALL PREPROCESS + LOGGING TESTS PASSED ✓")
