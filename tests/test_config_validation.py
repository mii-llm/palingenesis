"""Tests for config compatibility validation.

Verifies that:
- Hard incompatibilities raise ConfigError
- Untested combinations produce warnings
- Valid configs pass cleanly
- Validation is callable after from_yaml / from_cli
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from palingenesis.config import Config, ConfigError

# ══════════════════════════════════════════════════════════════════════════════
# HARD INCOMPATIBILITIES (must raise)
# ══════════════════════════════════════════════════════════════════════════════


def test_gradient_release_requires_ga_1():
    """gradient_release + GA > 1 is a hard error."""
    cfg = Config()
    cfg.memory.gradient_release = True
    cfg.train.gradient_accumulation_steps = 4
    with pytest.raises(ConfigError, match="gradient_release.*gradient_accumulation_steps"):
        cfg.validate()


def test_gradient_release_incompatible_with_muon():
    """gradient_release + muon optimizer is a hard error."""
    cfg = Config()
    cfg.memory.gradient_release = True
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.optimizer = "muon"
    with pytest.raises(ConfigError, match="gradient_release.*Muon"):
        cfg.validate()


def test_gradient_release_incompatible_with_ga_ramp():
    """gradient_release + ga_ramp is a hard error."""
    cfg = Config()
    cfg.memory.gradient_release = True
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.ga_ramp_start = 4
    with pytest.raises(ConfigError, match="gradient_release.*ga_ramp"):
        cfg.validate()


def test_packing_incompatible_with_context_parallel():
    """packing + context_parallel is a hard error."""
    cfg = Config()
    cfg.data.packing = True
    cfg.parallel.context_parallel = True
    with pytest.raises(ConfigError, match="packing.*context_parallel"):
        cfg.validate()


def test_mona_incompatible_with_schedule_free():
    """mona + schedule_free is a hard error."""
    cfg = Config()
    cfg.train.mona = True
    cfg.plugins.schedule_free = True
    with pytest.raises(ConfigError, match="mona.*schedule_free"):
        cfg.validate()


def test_hyperball_incompatible_with_schedule_free():
    """hyperball + schedule_free is a hard error."""
    cfg = Config()
    cfg.train.hyperball = True
    cfg.plugins.schedule_free = True
    with pytest.raises(ConfigError, match="hyperball.*schedule_free"):
        cfg.validate()


def test_multiple_loss_functions_exclusive():
    """Only one token-weighting loss can be active."""
    cfg = Config()
    cfg.plugins.deft = True
    cfg.plugins.dft = True
    with pytest.raises(ConfigError, match="one training objective plugin"):
        cfg.validate()

    cfg2 = Config()
    cfg2.plugins.cadft = True
    cfg2.plugins.info_sft = True
    with pytest.raises(ConfigError, match="one training objective plugin"):
        cfg2.validate()


# ══════════════════════════════════════════════════════════════════════════════
# WARNINGS (untested combos, should NOT raise)
# ══════════════════════════════════════════════════════════════════════════════


def test_gradient_release_plus_hyperball_is_an_error():
    """gradient_release steps the optimizer inside backward, so Hyperball (which
    wraps the optimizer step) would silently never run."""
    cfg = Config()
    cfg.memory.gradient_release = True
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.hyperball = True
    with pytest.raises(ConfigError, match="Hyperball"):
        cfg.validate()


def test_ema_plus_base_merge_warns():
    """ema + base_merge warns about interaction."""
    cfg = Config()
    cfg.train.ema = True
    cfg.train.base_merge = True
    warnings = cfg.validate()
    assert any("ema" in w and "base_merge" in w for w in warnings)


def test_adagc_plus_spike_detection_warns():
    """adagc + spike_detection is redundant, warns."""
    cfg = Config()
    cfg.train.adagc = True
    cfg.train.spike_detection = True
    warnings = cfg.validate()
    assert any("adagc" in w and "spike_detection" in w for w in warnings)


def test_pre_rl_is_exclusive_and_not_shadowed_by_chunked_loss():
    """pre_rl with another objective is an error; with chunked_loss it runs (the
    trainer computes full logits for it instead of chunking)."""
    cfg = Config()
    cfg.plugins.deft = True
    cfg.plugins.pre_rl = True
    with pytest.raises(ConfigError, match="one training objective plugin"):
        cfg.validate()

    cfg2 = Config()
    cfg2.plugins.pre_rl = True
    cfg2.memory.chunked_loss = True
    cfg2.validate()


# ══════════════════════════════════════════════════════════════════════════════
# VALID CONFIGS (no errors, minimal warnings)
# ══════════════════════════════════════════════════════════════════════════════


def test_default_config_valid():
    """Default config has no errors and no warnings."""
    cfg = Config()
    warnings = cfg.validate()
    assert warnings == []


def test_quickstart_style_config_valid():
    """A typical quickstart config validates cleanly."""
    cfg = Config()
    cfg.plugins.deft = True
    cfg.memory.chunked_loss = True
    cfg.train.gradient_checkpointing = "selective"
    cfg.data.packing = True
    warnings = cfg.validate()
    assert not any("incompatible" in w.lower() for w in warnings)


def test_flagship_config_validates():
    """The flagship A100 config (many features) should validate with warnings.
    (Hyperball is left out: it cannot run under gradient_release, see above.)"""
    cfg = Config()
    cfg.memory.gradient_release = True
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.adagc = True
    cfg.train.ema = True
    cfg.train.base_merge = True
    cfg.plugins.deft = True
    cfg.train.spike_detection = True
    # Should not raise — these are compatible, just untested together
    warnings = cfg.validate()
    # Should have some warnings about untested combos
    assert len(warnings) >= 2


def test_validate_works_after_from_yaml():
    """validate() is callable on config loaded from YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("train:\n  learning_rate: 3e-5\nplugins:\n  deft: true\n")
        f.flush()
        cfg = Config.from_yaml(f.name)

    warnings = cfg.validate()
    # Should pass without error
    assert isinstance(warnings, list)


def test_packing_epoch_horizon_warns():
    """packing + epochs-based schedule: total_steps counts rows, packing compresses
    the epoch, so the LR never fully decays — must warn and point at max_steps/wsd."""
    cfg = Config()
    cfg.data.packing = True
    cfg.train.max_steps = -1
    warnings = cfg.validate()
    assert any("packing" in w and "max_steps" in w for w in warnings)

    # Setting max_steps resolves it
    cfg.train.max_steps = 5000
    warnings = cfg.validate()
    assert not any("epochs-based LR horizon" in w for w in warnings)


# ── Loading: unknown options are errors, values are coerced to the option's type ──
def _yaml(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_yaml_unknown_option_is_an_error_with_a_hint(tmp_path):
    import pytest

    from palingenesis.config import Config, ConfigError

    with pytest.raises(ConfigError, match=r"train\.learnig_rate.*did you mean train\.learning_rate"):
        Config.from_yaml(_yaml(tmp_path, "train:\n  learnig_rate: 1.0e-5\n"))
    with pytest.raises(ConfigError, match="unknown config section `trian`.*`train`"):
        Config.from_yaml(_yaml(tmp_path, "trian:\n  epochs: 2\n"))


def test_yaml_removed_option_explains_why(tmp_path):
    import pytest

    from palingenesis.config import Config, ConfigError

    with pytest.raises(ConfigError, match="seq_len_curriculum was removed: it never took effect"):
        Config.from_yaml(_yaml(tmp_path, "data:\n  seq_len_curriculum: true\n"))


def test_yaml_values_are_coerced(tmp_path):
    from palingenesis.config import Config

    c = Config.from_yaml(
        _yaml(
            tmp_path,
            "train:\n  learning_rate: 2e-5\n  max_grad_norm: 1\ndata:\n  packing: 'true'\n  max_seq_length: '4096'\n",
        )
    )
    assert c.train.learning_rate == 2e-5 and isinstance(c.train.max_grad_norm, float)
    assert c.data.packing is True and c.data.max_seq_length == 4096


def test_cli_overrides_are_checked():
    import pytest

    from palingenesis.config import Config, ConfigError

    c = Config.from_cli(["--train.learning_rate", "3e-5", "--data.packing", "false", "--train.resume_from", "auto"])
    assert c.train.learning_rate == 3e-5 and c.data.packing is False and c.train.resume_from == "auto"
    with pytest.raises(ConfigError, match="did you mean train.epochs"):
        Config.from_cli(["--train.epoch", "2"])
    with pytest.raises(ConfigError, match="not a valid bool"):
        Config.from_cli(["--data.packing", "maybe"])


def test_every_shipped_config_loads():
    from pathlib import Path

    from palingenesis.config import Config
    from palingenesis.opd.config import OPDConfig
    from palingenesis.rl.config import RLConfig

    configs = sorted((Path(__file__).parent.parent / "configs").rglob("*.yaml"))
    assert configs
    for path in configs:
        # on-policy distillation and RL configs have their own schemas
        if path.name.startswith("rl_"):
            RLConfig.from_yaml(path).validate()
        else:
            (OPDConfig if path.name.startswith("distill_") else Config).from_yaml(path)
