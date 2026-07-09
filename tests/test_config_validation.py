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
    with pytest.raises(ConfigError, match="one token-weighting loss"):
        cfg.validate()

    cfg2 = Config()
    cfg2.plugins.cadft = True
    cfg2.plugins.info_sft = True
    with pytest.raises(ConfigError, match="one token-weighting loss"):
        cfg2.validate()


# ══════════════════════════════════════════════════════════════════════════════
# WARNINGS (untested combos, should NOT raise)
# ══════════════════════════════════════════════════════════════════════════════


def test_gradient_release_plus_hyperball_warns():
    """gradient_release + hyperball is untested, warns but doesn't raise."""
    cfg = Config()
    cfg.memory.gradient_release = True
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.hyperball = True
    warnings = cfg.validate()  # should NOT raise
    assert any("hyperball" in w for w in warnings)


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


def test_pre_rl_shadowed_warns():
    """pre_rl is silently shadowed by earlier loss branches — validate() must warn.

    The trainer picks ONE loss objective per run (chunked DEFT > chunked CE >
    CADFT > DEFT > DFT > InfoSFT > pre_rl > CE). Enabling pre_rl alongside deft
    or chunked_loss means it never runs.
    """
    cfg = Config()
    cfg.plugins.deft = True
    cfg.plugins.pre_rl = True
    cfg.memory.chunked_loss = True
    warnings = cfg.validate()
    shadow_warnings = [w for w in warnings if "IGNORED" in w and "pre_rl" in w]
    assert shadow_warnings, f"Expected a pre_rl-shadowed warning, got: {warnings}"
    assert "plugins.deft" in shadow_warnings[0]
    assert "memory.chunked_loss" in shadow_warnings[0]

    # A correctly-configured pre_rl run does not warn
    cfg2 = Config()
    cfg2.plugins.pre_rl = True
    cfg2.memory.chunked_loss = False
    warnings2 = cfg2.validate()
    assert not any("IGNORED" in w for w in warnings2), warnings2


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
    """The flagship A100 config (many features) should validate with warnings."""
    cfg = Config()
    cfg.memory.gradient_release = True
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.hyperball = True
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
