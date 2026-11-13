"""Gradient accumulation in the trainer's loop (on the CPU): what runs once per optimizer step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402


def test_mona_augments_once_per_optimizer_step(tmp_path, monkeypatch):
    # MONA differences successive gradients: under accumulation those are the accumulated ones, not the partial sums
    from palingenesis.optim import MONAAcceleration

    applied = []
    apply = MONAAcceleration.apply
    monkeypatch.setattr(MONAAcceleration, "apply", lambda self: (applied.append(1), apply(self))[1])
    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    c = H.cpu_config(tmp_path, tmp_path / "student", H.write_rows(tmp_path / "rows.jsonl"), steps=3)
    c.train.gradient_accumulation_steps = 2
    c.train.mona = True
    c.train.save_final = False
    result = H.run_cpu(c)
    assert result.global_step == 3
    assert len(applied) == 3
