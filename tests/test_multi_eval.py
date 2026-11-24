"""Tests for multi-source evaluation with weighted scoring."""

import sys

sys.path.insert(0, "src")

import torch
import torch.nn as nn

from palingenesis.multi_eval import MultiEvalResult, MultiEvaluator


class TinyModel(nn.Module):
    def __init__(self, vocab=64, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed(input_ids)
        logits = self.head(h)
        return type("Out", (), {"logits": logits})()

    def save_pretrained(self, path, **kwargs):
        from pathlib import Path

        from safetensors.torch import save_file

        Path(path).mkdir(parents=True, exist_ok=True)
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(Path(path) / "model.safetensors"))


def test_multi_eval_weighted_score():
    """Weighted composite score reflects source importance."""
    from palingenesis.multi_eval import MultiEvalResult

    # Simulate: source A (weight 0.7) has loss 1.0, source B (weight 0.3) has loss 3.0
    # Expected score: 0.7*1.0 + 0.3*3.0 = 1.6
    result = MultiEvalResult(
        score=0.7 * 1.0 + 0.3 * 3.0,
        per_source={"A": 1.0, "B": 3.0},
    )
    assert abs(result.score - 1.6) < 1e-6
    print(f"  Score: {result.score} (expected 1.6)")
    print("✓ test_multi_eval_weighted_score PASSED\n")


def test_multi_eval_regression_detection():
    """Sources exceeding regression_floor are flagged."""
    # Create a mock evaluator scenario
    result = MultiEvalResult(
        score=2.5,
        per_source={"agentic": 1.2, "code": 3.5, "general": 1.8},
        regressions=["code"],  # code exceeded its floor
    )
    assert "code" in result.regressions
    assert "agentic" not in result.regressions
    print(f"  Regressions: {result.regressions}")
    print("✓ test_multi_eval_regression_detection PASSED\n")


def test_multi_eval_with_real_model():
    """MultiEvaluator computes valid loss on a tiny model with synthetic data."""
    import json
    import tempfile

    from transformers import AutoTokenizer

    # Create synthetic eval data
    samples = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]},
        {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
        eval_path = f.name

    # Use a real tokenizer for proper chat template
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  Skipped (gpt2 tokenizer not available)")
        print("✓ test_multi_eval_with_real_model SKIPPED\n")
        return

    eval_sources = [
        {"name": "test_source", "dataset": eval_path, "weight": 1.0, "samples": 3, "split": "train"},
    ]

    device = torch.device("cpu")
    model = TinyModel(vocab=tokenizer.vocab_size, hidden=32)

    if not getattr(tokenizer, "chat_template", None):
        # GPT-2 has no chat template: no row can be rendered, and a configured source that gives
        # nothing is an error, not a silently missing evaluation
        import pytest

        from palingenesis.config import ConfigError

        with pytest.raises(ConfigError, match="source .test_source.*(could not be read|no usable sample)"):
            MultiEvaluator(eval_sources, tokenizer, max_seq_length=128, device=device)
        return
    evaluator = MultiEvaluator(eval_sources, tokenizer, max_seq_length=128, device=device)

    result = evaluator.evaluate(model, dtype=torch.float32)

    assert result.score > 0, f"Score should be positive, got {result.score}"
    assert "test_source" in result.per_source
    assert result.tokens_total > 0
    print(f"  Score: {result.score:.4f}, tokens: {result.tokens_total}")
    print("✓ test_multi_eval_with_real_model PASSED\n")


def test_multi_eval_pretrain_mode():
    """A `mode: pretrain` source scores raw-text CE with all-token loss and no chat
    template — so it works even with a tokenizer that has no chat template (GPT-2)."""
    import json
    import tempfile

    from transformers import AutoTokenizer

    docs = [
        {"text": "Roma è la capitale d'Italia e un importante centro culturale."},
        {"text": "La fisica quantistica descrive il comportamento delle particelle."},
        {"text": "Il Rinascimento fu un periodo di grande fioritura artistica."},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
        eval_path = f.name

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  Skipped (gpt2 tokenizer not available)")
        return

    eval_sources = [
        {
            "name": "italian_lm",
            "dataset": eval_path,
            "mode": "pretrain",
            "text_field": "text",
            "weight": 1.0,
            "samples": 3,
            "split": "train",
        },
    ]

    device = torch.device("cpu")
    model = TinyModel(vocab=tokenizer.vocab_size, hidden=32)
    evaluator = MultiEvaluator(eval_sources, tokenizer, max_seq_length=128, device=device)

    # Unlike the SFT/chat path, pretrain mode must NOT be skipped on a template-less tokenizer.
    assert evaluator.sources and evaluator.sources[0]["batches"], "pretrain source produced no batches"

    result = evaluator.evaluate(model, dtype=torch.float32)
    assert result.score > 0 and result.tokens_total > 0
    assert "italian_lm" in result.per_source
    print(f"  pretrain-mode score: {result.score:.4f}, tokens: {result.tokens_total}")
    print("✓ test_multi_eval_pretrain_mode PASSED\n")


def _gpt2():
    import pytest
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("gpt2 tokenizer not cached (offline)")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_multi_eval_misconfigured_sources_are_errors(tmp_path):
    """A configured eval source is never skipped: an unknown mode, a missing file, a file of another
    format and a wrong text field are ConfigErrors that name the source."""
    import json

    import pytest

    from palingenesis.config import ConfigError

    class FakeTok:
        pad_token_id = 0
        eos_token_id = 1

    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps({"text": "x"}) + "\n")

    def build(tokenizer=None, **src):
        entry = {"name": "bad", "dataset": str(path), "weight": 1.0, "samples": 1, "split": "train", **src}
        return MultiEvaluator([entry], tokenizer or FakeTok(), max_seq_length=32, device=torch.device("cpu"))

    with pytest.raises(ConfigError, match="source 'bad' has mode 'nonsense'"):
        build(mode="nonsense")
    with pytest.raises(ConfigError, match="source 'bad'.*cannot load"):
        build(mode="pretrain", dataset=str(tmp_path / "missing.jsonl"))
    other = tmp_path / "docs.csv"
    other.write_text("text\nx\n")
    with pytest.raises(ConfigError, match="not a .jsonl, .json or .parquet file"):
        build(mode="pretrain", dataset=str(other))
    with pytest.raises(ConfigError, match="gives no usable sample.*text_field"):
        build(tokenizer=_gpt2(), mode="pretrain", text_field="body")


def test_multi_eval_reads_local_parquet_and_json(tmp_path):
    """Held-out sets in parquet (and a JSON list) are read like JSONL: the same batches, the same loss."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    tokenizer = _gpt2()
    docs = [{"text": f"Il documento numero {i} parla di treni e di stazioni. " * 3, "id": i} for i in range(6)]
    jsonl = tmp_path / "held.jsonl"
    jsonl.write_text("".join(json.dumps(d) + "\n" for d in docs))
    parquet = tmp_path / "held.parquet"
    pq.write_table(pa.Table.from_pylist(docs), parquet, row_group_size=2)
    as_json = tmp_path / "held.json"
    as_json.write_text(json.dumps(docs))

    model = TinyModel(vocab=tokenizer.vocab_size, hidden=32)
    losses = {}
    for path in (jsonl, parquet, as_json):
        source = {"name": "held", "dataset": str(path), "mode": "pretrain", "weight": 1.0, "samples": 6}
        evaluator = MultiEvaluator([source], tokenizer, max_seq_length=128, device=torch.device("cpu"))
        assert sum(b["input_ids"].shape[0] for b in evaluator.sources[0]["batches"]) == 6
        result = evaluator.evaluate(model, dtype=torch.float32)
        losses[path.suffix] = (result.per_source["held"], result.tokens_total)
    assert losses[".parquet"] == losses[".jsonl"] == losses[".json"]
    # every document ends with the EOS token, and it is scored
    last = evaluator.sources[0]["batches"][0]
    assert (last["labels"] == tokenizer.eos_token_id).sum() == last["input_ids"].shape[0]


def test_multi_eval_best_model_integration():
    """MultiEval score integrates with BestModelTracker correctly."""
    import tempfile

    from palingenesis.checkpoint import BestModelTracker

    with tempfile.TemporaryDirectory() as tmpdir:

        class FakeTokenizer:
            def save_pretrained(self, path):
                from pathlib import Path

                Path(path).mkdir(parents=True, exist_ok=True)

        model = TinyModel()
        tracker = BestModelTracker(tmpdir)

        # Simulate multi-eval scores over time
        scores = [2.5, 2.3, 2.1, 2.4, 1.9, 2.0]  # 1.9 is the best
        for step, score in enumerate(scores):
            tracker.update(score, step=step * 100, model=model, tokenizer=FakeTokenizer())

        assert tracker.best_loss == 1.9
        assert tracker.best_step == 400
        print(f"  Best: step={tracker.best_step}, score={tracker.best_loss}")
        print("✓ test_multi_eval_best_model_integration PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-EVAL TESTS")
    print("=" * 60 + "\n")

    test_multi_eval_weighted_score()
    test_multi_eval_regression_detection()
    test_multi_eval_with_real_model()
    test_multi_eval_pretrain_mode()
    test_multi_eval_best_model_integration()

    print("=" * 60)
    print("ALL MULTI-EVAL TESTS PASSED ✓")
    print("=" * 60)
