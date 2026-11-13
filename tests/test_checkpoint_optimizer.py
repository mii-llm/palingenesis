"""Optimizer checkpoint shard round-trip.

Regression for a crash at save time: bitsandbytes 8-bit optimizers (Lion8bit,
AdamW8bit) keep NON-tensor state entries that are nested dicts *containing*
tensors. The old code dumped all non-tensor state to JSON, which raised
`TypeError: Object of type Tensor is not JSON serializable` — killing training
exactly at the first checkpoint. Non-tensor state now goes through torch.save.
"""

import json
import tempfile
from pathlib import Path

import torch

from palingenesis.checkpoint import _load_optimizer_sharded, _save_optimizer_sharded


def _stepped_adamw(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    return opt


def test_optimizer_shard_roundtrip_plain():
    model = torch.nn.Linear(4, 4)
    opt = _stepped_adamw(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "optimizer"
        _save_optimizer_sharded(opt, path)
        assert list(path.glob("shard_*.safetensors")), "tensor shards must be written"

        model2 = torch.nn.Linear(4, 4)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        _load_optimizer_sharded(opt2, path, None)

        orig = opt.state_dict()["state"]
        loaded = opt2.state_dict()["state"]
        assert orig.keys() == loaded.keys()
        for pid in orig:
            torch.testing.assert_close(orig[pid]["exp_avg"], loaded[pid]["exp_avg"])
    print("✓ test_optimizer_shard_roundtrip_plain PASSED")


def test_optimizer_shard_roundtrip_bnb_like_nested_state():
    """bnb 8-bit state: nested dicts holding tensors must survive save+load."""
    model = torch.nn.Linear(4, 4)
    opt = _stepped_adamw(model)

    # Inject bnb-style state: a nested dict containing a tensor (this is what
    # crashed json.dump), plus a plain int step counter.
    qmap = torch.linspace(-1, 1, 16)
    for p in model.parameters():
        opt.state[p]["qmap1"] = {"code": qmap.clone(), "signed": True}
        opt.state[p]["bnb_step"] = 7

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "optimizer"
        _save_optimizer_sharded(opt, path)  # must NOT raise
        assert list(path.glob("shard_*_meta.pt")), "non-tensor state goes to .pt"

        model2 = torch.nn.Linear(4, 4)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        _load_optimizer_sharded(opt2, path, None)

        loaded = opt2.state_dict()["state"]
        for pid in loaded:
            torch.testing.assert_close(loaded[pid]["qmap1"]["code"], qmap)
            assert loaded[pid]["qmap1"]["signed"] is True
            assert loaded[pid]["bnb_step"] == 7
    print("✓ test_optimizer_shard_roundtrip_bnb_like_nested_state PASSED")


def test_optimizer_shard_loads_legacy_json_meta():
    """Checkpoints written before the fix used _meta.json for step counts."""
    model = torch.nn.Linear(4, 4)
    opt = _stepped_adamw(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "optimizer"
        _save_optimizer_sharded(opt, path)

        # Rewrite meta in the legacy JSON format (step as plain number)
        for meta_pt in path.glob("shard_*_meta.pt"):
            meta = torch.load(meta_pt, weights_only=False)
            legacy = {k: (float(v) if torch.is_tensor(v) else v) for k, v in meta.items()}
            meta_pt.with_suffix(".json").write_text(json.dumps(legacy))
            meta_pt.unlink()
        # AdamW stores step AS a tensor, so the .pt meta may be empty for it —
        # ensure at least one legacy step entry exists to exercise the path.
        json_files = list(path.glob("shard_*_meta.json"))
        if not json_files:
            (path / "shard_0000_meta.json").write_text(json.dumps({"0.step": 1.0}))

        model2 = torch.nn.Linear(4, 4)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        _load_optimizer_sharded(opt2, path, None)

        loaded = opt2.state_dict()["state"]
        for pid in loaded:
            step = loaded[pid]["step"]
            assert torch.is_tensor(step), "legacy JSON step must be converted to tensor"
    print("✓ test_optimizer_shard_loads_legacy_json_meta PASSED")


if __name__ == "__main__":
    test_optimizer_shard_roundtrip_plain()
    test_optimizer_shard_roundtrip_bnb_like_nested_state()
    test_optimizer_shard_loads_legacy_json_meta()
    print("\nALL CHECKPOINT OPTIMIZER TESTS PASSED ✓")


def test_hyperball_moves_weights_after_a_zero_lr_warmup_step():
    """With LR warmup the first step has lr 0: calibrating the angular step there
    fixed it at 0 and froze every constrained matrix for the whole run."""
    import torch

    from palingenesis.optim import Hyperball, build_scheduler

    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 16, bias=False)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=0.0)
    sched = build_scheduler(opt, "cosine", 100, 0.05, 0.1)
    hb = Hyperball(opt, [lin.weight])
    w0 = lin.weight.detach().clone()
    for _ in range(10):
        lin(torch.randn(4, 16)).pow(2).sum().backward()
        hb.step()
        sched.step()
        opt.zero_grad()
    assert (lin.weight - w0).norm() > 0.1
    torch.testing.assert_close(lin.weight.norm(), w0.norm())


def _tracker(tmp_path, monkeypatch, **kwargs):
    """A BestModelTracker whose model save only records the call (the bookkeeping is under test)."""
    from palingenesis import checkpoint

    saves = []

    def save(model, tokenizer, path, is_fsdp, export=None):
        path.mkdir(parents=True, exist_ok=True)
        saves.append(path)

    monkeypatch.setattr(checkpoint, "_save_gathered_or_local", save)
    return checkpoint.BestModelTracker(str(tmp_path), **kwargs), saves


def test_best_tracker_never_takes_a_non_finite_loss(tmp_path, monkeypatch):
    """A NaN (or infinite) evaluation is not a new best: it compares False with the current best,
    which replaced best/ and made every later evaluation replace it again."""
    tracker, saves = _tracker(tmp_path, monkeypatch)
    assert tracker.update(1.0, 10, None, None)
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert not tracker.update(bad, 20, None, None)
    assert not tracker.update(5.0, 30, None, None)  # worse than the best, still refused
    assert tracker.update(0.5, 40, None, None)
    assert (tracker.best_loss, tracker.best_step, len(saves)) == (0.5, 40, 2)
    assert json.loads((tmp_path / "best" / "best_meta.json").read_text()) == {"step": 40, "eval_loss": 0.5}

    first, _ = _tracker(tmp_path / "fresh", monkeypatch)
    assert not first.update(float("nan"), 1, None, None) and not first.has_best


def test_best_tracker_resume_continues_from_the_recorded_best(tmp_path, monkeypatch):
    """A resumed run keeps the best it had: a worse evaluation after the resume does not replace
    best/. A fresh run in the same output_dir ignores what an earlier run left there."""
    tracker, _ = _tracker(tmp_path, monkeypatch)
    assert tracker.update(1.25, 100, None, None)

    resumed, saves = _tracker(tmp_path, monkeypatch, resume=True)
    assert (resumed.best_loss, resumed.best_step, resumed.has_best) == (1.25, 100, True)
    assert not resumed.update(2.0, 200, None, None) and not saves
    assert resumed.update(1.0, 300, None, None) and resumed.best_step == 300

    fresh, saves = _tracker(tmp_path, monkeypatch)
    assert not fresh.has_best
    assert fresh.update(2.0, 10, None, None) and len(saves) == 1

    # nothing recorded, or an unreadable record: start without a best
    assert not _tracker(tmp_path / "empty", monkeypatch, resume=True)[0].has_best
    (tmp_path / "best" / "best_meta.json").write_text("{not json")
    assert not _tracker(tmp_path, monkeypatch, resume=True)[0].has_best
