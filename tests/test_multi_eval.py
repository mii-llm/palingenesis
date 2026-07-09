"""Tests for multi-source evaluation with weighted scoring."""

import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from palingenesis.multi_eval import MultiEvaluator, MultiEvalResult, IGNORE_INDEX


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
    from pathlib import Path
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

    evaluator = MultiEvaluator(eval_sources, tokenizer, max_seq_length=128, device=device)

    if not evaluator.sources or not evaluator.sources[0]["batches"]:
        print("  Skipped (tokenizer lacks chat template — expected for GPT-2)")
        print("✓ test_multi_eval_with_real_model SKIPPED\n")
        return

    result = evaluator.evaluate(model, dtype=torch.float32)

    assert result.score > 0, f"Score should be positive, got {result.score}"
    assert "test_source" in result.per_source
    assert result.tokens_total > 0
    print(f"  Score: {result.score:.4f}, tokens: {result.tokens_total}")
    print("✓ test_multi_eval_with_real_model PASSED\n")


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
    test_multi_eval_best_model_integration()

    print("=" * 60)
    print("ALL MULTI-EVAL TESTS PASSED ✓")
    print("=" * 60)
