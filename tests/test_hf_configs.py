"""An in-process vLLM engine registers its own config classes with AutoConfig (exist_ok=True) for model types
transformers already has (qwen3_5, ...): every later Hugging Face load of such a checkpoint in the process then
failed ("'Qwen3_5Config' object has no attribute 'vocab_size'"). restore_native_configs puts transformers' classes
back; the rollout engines call it after each engine start."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
qwen3_5 = pytest.importorskip("transformers.models.qwen3_5")

from palingenesis.hf_configs import restore_native_configs  # noqa: E402

foreign = type("module", (), {})  # classes defined outside transformers, as vLLM's are


@pytest.fixture()
def wrapper_checkpoint(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
            max_position_embeddings=64,
        ),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=32),
    )
    Qwen3_5ForConditionalGeneration(config).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.fixture()
def engine_registration():
    """What vLLM's get_config does for a qwen3_5 checkpoint: AutoConfig.register(model_type, its own class)."""
    from transformers import AutoConfig
    from transformers.models.auto import configuration_auto

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

    text = type("Qwen3_5TextConfig", (Qwen3_5TextConfig,), {"__module__": "vllm.transformers_utils.configs.qwen3_5"})
    wrapper = type(
        "Qwen3_5Config",
        (Qwen3_5Config,),
        {"__module__": "vllm.transformers_utils.configs.qwen3_5", "sub_configs": {**Qwen3_5Config.sub_configs, "text_config": text}},
    )
    AutoConfig.register("qwen3_5", wrapper, exist_ok=True)
    yield
    configuration_auto.CONFIG_MAPPING._extra_content.pop("qwen3_5", None)


def test_an_engine_registration_breaks_the_next_load(wrapper_checkpoint, engine_registration):
    from transformers import AutoModelForCausalLM

    with pytest.raises(AttributeError, match="vocab_size"):
        AutoModelForCausalLM.from_pretrained(wrapper_checkpoint, dtype=torch.float32)


def test_restored_configs_load_again(wrapper_checkpoint, engine_registration):
    from transformers import AutoConfig, AutoModelForCausalLM

    assert restore_native_configs() == ["qwen3_5"]
    assert type(AutoConfig.from_pretrained(wrapper_checkpoint)).__module__.startswith("transformers.")
    model = AutoModelForCausalLM.from_pretrained(wrapper_checkpoint, dtype=torch.float32)
    assert model.config.vocab_size == 64


def test_registrations_of_new_model_types_are_kept():
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto import configuration_auto

    custom = type("CustomConfig", (PretrainedConfig,), {"model_type": "pgs_custom_test", "__module__": "somewhere"})
    AutoConfig.register("pgs_custom_test", custom)
    try:
        assert restore_native_configs() == []
        assert configuration_auto.CONFIG_MAPPING["pgs_custom_test"] is custom
    finally:
        configuration_auto.CONFIG_MAPPING._extra_content.pop("pgs_custom_test", None)
