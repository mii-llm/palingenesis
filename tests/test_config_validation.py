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


def test_packing_with_an_epoch_horizon_no_longer_warns():
    """The run's length is estimated through the real packer (palingenesis.data_size), so an
    epoch-based schedule with packing decays over the packed epoch: nothing to warn about."""
    cfg = Config()
    cfg.data.packing = True
    cfg.train.max_steps = -1
    assert not any("epochs-based LR horizon" in w for w in cfg.validate())


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
        if path.parent.name == "eval":  # evaluation tasks (pgs eval)
            from palingenesis.eval.config import MODES, load_task, task_errors

            assert task_errors(load_task(str(path)), set(MODES)) == []
        elif path.name.startswith("rl_"):
            RLConfig.from_yaml(path).validate()
        else:
            (OPDConfig if path.name.startswith("distill_") else Config).from_yaml(path)


def test_keep_checkpoints_must_be_non_negative():
    cfg = Config()
    assert cfg.train.keep_checkpoints == 5
    cfg.train.keep_checkpoints = -1
    with pytest.raises(ConfigError, match="train.keep_checkpoints"):
        cfg.validate()


# ══════════════════════════════════════════════════════════════════════════════
# RAW-TEXT TRAINING: sources, token-share weights, random initialisation, cleaning
# ══════════════════════════════════════════════════════════════════════════════


def _sources(*sources, **data):
    cfg = Config()
    cfg.data.sources = list(sources)
    for key, value in data.items():
        setattr(cfg.data, key, value)
    return cfg


def test_source_entries_are_checked():
    text = {"dataset": "corpus.jsonl", "mode": "pretrain"}
    assert _sources(text, {"dataset": "chat.jsonl"}).validate() == []
    assert _sources({**text, "append_eos": False}, {**text, "append_eos": "<|endoftext|>"}).validate() == []
    with pytest.raises(ConfigError, match=r"data.sources\[1\].mode must be sft or pretrain, got 'cpt'"):
        _sources(text, {"dataset": "x", "mode": "cpt"}).validate()
    with pytest.raises(ConfigError, match=r"data.sources\[0\].append_eos must be true"):
        _sources({**text, "append_eos": 3}).validate()
    with pytest.raises(ConfigError, match=r"data.sources\[0\].append_eos applies to raw text"):
        _sources({"dataset": "chat.jsonl", "append_eos": True}).validate()
    with pytest.raises(ConfigError, match=r"data.sources\[0\].weight must be a finite number >= 0"):
        _sources({**text, "weight": -1}).validate()
    with pytest.raises(ConfigError, match=r"data.sources\[0\] must be a mapping"):
        _sources("corpus.jsonl").validate()
    cfg = Config()
    cfg.data.eval_sources = [{"name": "held", "dataset": "held.parquet", "mode": "raw"}]
    with pytest.raises(ConfigError, match=r"data.eval_sources\[0\].mode must be sft or pretrain"):
        cfg.validate()


def test_mix_weights_by_token_share_needs_weights():
    a, b = {"dataset": "a.jsonl", "mode": "pretrain", "weight": 0.8}, {"dataset": "b.jsonl", "mode": "pretrain"}
    assert Config().data.mix_weights == "sequences"
    assert _sources(a, {**b, "weight": 0.2}, mix_weights="tokens").validate() == []
    with pytest.raises(ConfigError, match="mix_weights: tokens needs a weight .* none on b.jsonl"):
        _sources(a, b, mix_weights="tokens").validate()
    with pytest.raises(ConfigError, match="mix_weights: tokens applies to data.sources"):
        _sources(mix_weights="tokens").validate()
    with pytest.raises(ConfigError, match="data.mix_weights must be sequences or tokens, got 'rows'"):
        _sources(a, mix_weights="rows").validate()


def test_random_initialisation_options():
    cfg = Config()
    assert cfg.model.init == "pretrained" and cfg.model.init_seed == -1
    cfg.model.init = "random"
    warnings = cfg.validate()
    assert any("model.init: random" in w and "learning_rate" in w for w in warnings)
    cfg.train.learning_rate = 3e-4
    assert cfg.validate() == []
    cfg.model.init = "scratch"
    with pytest.raises(ConfigError, match="model.init must be pretrained or random, got 'scratch'"):
        cfg.validate()
    cfg.model.init = "random"
    cfg.model.init_seed = -2
    with pytest.raises(ConfigError, match="model.init_seed"):
        cfg.validate()
    # dpo.enabled is refused too (tests/test_pretraining.py: validating a DPO config imports transformers)
    for section, option in (
        ("multimodal", "enabled"),
        ("train", "base_merge"),
        ("plugins", "pre_rl"),
        ("train", "freeze_non_attention"),
    ):
        bad = Config()
        bad.model.init, bad.train.learning_rate = "random", 3e-4
        setattr(getattr(bad, section), option, True)
        with pytest.raises(ConfigError, match=f"model.init: random does not support {section}.{option}"):
            bad.validate()


def test_raw_text_cleaning_options(tmp_path):
    cfg = Config()
    assert cfg.preprocess.near_dedupe == 0.0 and cfg.preprocess.decontaminate == []
    assert cfg.preprocess.decontaminate_ngram == 13
    path = tmp_path / "c.yaml"
    path.write_text(
        "preprocess:\n  near_dedupe: 0.8\n  decontaminate: [bench.jsonl]\n  decontaminate_ngram: 8\n  min_chars: 200\n"
    )
    loaded = Config.from_yaml(path)
    assert loaded.preprocess.near_dedupe == 0.8 and loaded.preprocess.decontaminate == ["bench.jsonl"]
    assert loaded.validate() == []
    for option, value in (("near_dedupe", 1.5), ("decontaminate_ngram", 0), ("min_chars", -1)):
        bad = Config()
        setattr(bad.preprocess, option, value)
        with pytest.raises(ConfigError, match=f"preprocess.{option}"):
            bad.validate()
    bad = Config()
    bad.preprocess.decontaminate = "bench.jsonl"
    with pytest.raises(ConfigError, match="preprocess.decontaminate must be a list"):
        bad.validate()
    fields = Config()
    fields.preprocess.decontaminate_fields = ["question", "answer"]
    assert fields.validate() == []
    fields.preprocess.decontaminate_fields = "question"
    with pytest.raises(ConfigError, match="preprocess.decontaminate_fields must be a list"):
        fields.validate()
    share = Config()
    assert share.preprocess.decontaminate_shared_share == -1
    for value in (0, 0.2, 1):
        share.preprocess.decontaminate_shared_share = value
        assert share.validate() == []
    share.preprocess.decontaminate_shared_share = 1.5
    with pytest.raises(ConfigError, match="preprocess.decontaminate_shared_share"):
        share.validate()


def test_data_stream_version_is_importable_without_heavy_dependencies():
    from palingenesis.config import DATA_STREAM_VERSION

    assert DATA_STREAM_VERSION == 2


# ══════════════════════════════════════════════════════════════════════════════
# TENSOR / PIPELINE PARALLELISM (parallel.tp_degree, parallel.pp_*)
# ══════════════════════════════════════════════════════════════════════════════


def _parallel(**options) -> Config:
    cfg = Config()
    for key, value in options.items():
        section, name = key.split("__")
        setattr(getattr(cfg, section), name, value)
    return cfg


def test_model_parallel_defaults_are_off():
    cfg = Config()
    assert (cfg.parallel.tp_degree, cfg.parallel.pp_degree, cfg.parallel.async_tp) == (1, 1, False)
    cfg.validate()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"parallel__tp_degree": 0}, "tp_degree must be an integer >= 1"),
        ({"parallel__pp_degree": True}, "pp_degree must be an integer >= 1"),
        ({"parallel__pp_microbatches": -1}, "pp_microbatches must be an integer >= 0"),
        ({"parallel__tp_degree": 2, "parallel__fsdp": False}, "need parallel.fsdp: true"),
        ({"parallel__tp_degree": 2, "memory__seco": True}, "does not support memory.seco"),
        ({"parallel__tp_degree": 2, "multimodal__enabled": True}, "does not support multimodal.enabled"),
        ({"parallel__tp_degree": 2, "parallel__cpu_offload": True}, "does not support parallel.cpu_offload"),
        ({"parallel__pp_degree": 2, "train__optimizer": "muon"}, "does not support train.optimizer=muon"),
        # activation offload and the fused optimizer are validated on one process only: refused, not run unchecked
        ({"parallel__tp_degree": 2, "memory__activation_offload": True}, "tp_degree > 1 does not support memory.act"),
        ({"parallel__pp_degree": 2, "memory__activation_offload": True}, "pp_degree > 1 does not support memory.act"),
        ({"parallel__tp_degree": 2, "train__fused_optimizer": True}, "tp_degree > 1 does not support train.fused_opt"),
        ({"parallel__pp_degree": 2, "train__fused_optimizer": True}, "pp_degree > 1 does not support train.fused_opt"),
        (
            {"parallel__tp_degree": 2, "parallel__pp_degree": 2, "train__fused_optimizer": True},
            "parallel.tp_degree / parallel.pp_degree > 1 does not support train.fused_optimizer",
        ),
        ({"parallel__async_tp": True}, "async_tp .* needs parallel.tp_degree > 1"),
        (
            {"parallel__async_tp": True, "parallel__tp_degree": 2, "model__compile": False},
            "async_tp needs model.compile",
        ),
        ({"parallel__pp_degree": 2, "parallel__pp_schedule": "pipedream"}, "pp_schedule must be one of"),
        ({"parallel__pp_degree": 2, "parallel__pp_schedule": "zbv", "parallel__pp_stages_per_rank": 3}, "zbv runs 2"),
        ({"parallel__pp_degree": 2, "parallel__pp_schedule": "1f1b", "parallel__pp_stages_per_rank": 2}, "1f1b runs 1"),
        ({"parallel__pp_degree": 2, "parallel__pp_microbatches": 3}, "must divide train.per_device_batch_size"),
        (
            {"parallel__pp_degree": 2, "parallel__context_parallel": True},
            "pp_degree > 1 does not support parallel.context",
        ),
        ({"parallel__pp_degree": 2, "train__ema": True}, "does not support train.ema"),
        ({"parallel__tp_degree": 2, "parallel__context_parallel": True}, "context_parallel is not validated"),
    ],
)
def test_model_parallel_errors(options, message):
    with pytest.raises(ConfigError, match=message):
        _parallel(**options).validate()


def test_model_parallel_valid_compositions():
    _parallel(parallel__tp_degree=4, data__packing=True, train__gradient_checkpointing="full").validate()
    _parallel(parallel__tp_degree=2, parallel__async_tp=True, model__compile=True).validate()
    for schedule in ("gpipe", "1f1b", "interleaved_1f1b", "looped_bfs", "interleaved_zero_bubble", "zbv", "dualpipev"):
        _parallel(parallel__pp_degree=4, parallel__pp_schedule=schedule, train__per_device_batch_size=8).validate()
    _parallel(
        parallel__pp_degree=2, parallel__tp_degree=2, parallel__pp_microbatches=2, train__per_device_batch_size=4
    ).validate()


def test_model_parallel_options_from_the_command_line():
    cfg = Config.from_cli(["--parallel.tp_degree", "2", "--parallel.pp_schedule", "zbv", "--parallel.pp_degree", "2"])
    assert (cfg.parallel.tp_degree, cfg.parallel.pp_degree, cfg.parallel.pp_schedule) == (2, 2, "zbv")


def test_parallel_dims_and_stage_split():
    from palingenesis.parallel_dims import ParallelDims, head_errors, split_layers, stage_indices

    dims = ParallelDims(16, tp=2, pp=2, cp=2)
    assert dims.dp == 2 and not dims.errors() and dims.describe() == "pp=2 x dp=2 x cp=2 x tp=2"
    assert "cannot split" in ParallelDims(6, tp=4).errors()[0]
    assert head_errors(4, {"num_key_value_heads": 2})[0].startswith("parallel.tp_degree=4 must divide")
    for layers, stages in ((28, 4), (36, 8), (4, 4), (80, 16), (5, 2)):
        split = split_layers(layers, stages)
        assert sum(split, []) == list(range(layers)) and all(split)
    # the embeddings and the head count as layers: the end stages get fewer
    assert [len(s) for s in split_layers(28, 4)] == [6, 8, 8, 6]
    with pytest.raises(ValueError, match="at least one layer"):
        split_layers(2, 4)
    assert stage_indices(1, 4, 2, v_shape=False) == [1, 5]
    assert stage_indices(1, 4, 2, v_shape=True) == [1, 6]
