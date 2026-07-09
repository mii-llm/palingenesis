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
