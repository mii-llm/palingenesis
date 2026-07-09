"""Autopilot orchestration wiring.

Regressions covered:
- _build_config used to set eval_dataset but leave eval_split at the default
  "test", which most datasets don't have → every sweep trial failed at eval
  loading (counted as diverged) and Phase 3 crashed.
- Trial configs didn't copy the preprocess section and ran redundant in-train
  eval (trials are scored by a separate evaluate() pass).
"""

from palingenesis.autopilot.ablate import _make_ablation_config
from palingenesis.autopilot.run import _build_config
from palingenesis.autopilot.sweep import _make_trial_config
from palingenesis.config import Config

RECOMMENDED = {
    "train.per_device_batch_size": 2,
    "train.gradient_accumulation_steps": 8,
    "memory.loss_num_chunks": 8,
    "memory.chunked_loss": True,
    "train.bf16": True,
}


def _base_cfg(**kwargs) -> Config:
    return _build_config(
        model="Qwen/Qwen3.5-4B",
        dataset="org/my-data",
        dataset_split="train",
        seq_length=4096,
        trust_remote_code=True,
        recommended=RECOMMENDED,
        lr=2e-5,
        max_steps=1000,
        output_dir="/tmp/autopilot-test",
        **kwargs,
    )


def test_build_config_eval_falls_back_to_train_split():
    """Without a val dataset, eval must use the TRAIN dataset + split — the
    old default eval_split='test' crashed on datasets without that split."""
    cfg = _base_cfg()
    assert cfg.data.eval_dataset == "org/my-data"
    assert cfg.data.eval_split == "train", f"Expected train split, got {cfg.data.eval_split}"
    print("✓ test_build_config_eval_falls_back_to_train_split PASSED")


def test_build_config_eval_uses_explicit_val_dataset():
    cfg = _base_cfg(val_dataset="org/my-val", val_split="validation")
    assert cfg.data.eval_dataset == "org/my-val"
    assert cfg.data.eval_split == "validation"
    print("✓ test_build_config_eval_uses_explicit_val_dataset PASSED")


def test_build_config_applies_recommended():
    cfg = _base_cfg()
    assert cfg.train.per_device_batch_size == 2
    assert cfg.train.gradient_accumulation_steps == 8
    assert cfg.preprocess.enabled is False, "Autopilot must not require a prepared dataset"
    assert cfg.logging.use_wandb is False, "Tracking off by default (enabled only for Phase 3)"
    print("✓ test_build_config_applies_recommended PASSED")


def test_sweep_trial_config_isolated():
    base = _base_cfg()
    trial = _make_trial_config(base, lr=5e-5, max_steps=50, output_dir="/tmp/trial")

    assert trial.train.learning_rate == 5e-5
    assert trial.train.max_steps == 50
    assert trial.logging.use_wandb is False and trial.logging.use_trackio is False
    assert trial.data.eval_dataset == "", "Trials must not run in-train eval (scored separately)"
    assert trial.preprocess.enabled == base.preprocess.enabled

    # Mutating the trial must not leak into the base config
    trial.train.learning_rate = 1e-3
    assert base.train.learning_rate == 2e-5
    print("✓ test_sweep_trial_config_isolated PASSED")


def test_ablation_config_applies_overrides():
    base = _base_cfg()
    trial = _make_ablation_config(
        base,
        name="3-epochs",
        overrides={"train.epochs": 3, "plugins.cadft": True},
        max_steps=100,
        output_dir="/tmp/ablation-trial",
    )

    assert trial.train.epochs == 3
    assert trial.plugins.cadft is True
    assert trial.data.eval_dataset == "", "Ablation trials must not run in-train eval"
    assert base.plugins.cadft is False, "Override leaked into base config"
    print("✓ test_ablation_config_applies_overrides PASSED")


if __name__ == "__main__":
    test_build_config_eval_falls_back_to_train_split()
    test_build_config_eval_uses_explicit_val_dataset()
    test_build_config_applies_recommended()
    test_sweep_trial_config_isolated()
    test_ablation_config_applies_overrides()
    print("\nALL AUTOPILOT WIRING TESTS PASSED ✓")
