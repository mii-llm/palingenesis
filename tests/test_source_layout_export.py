"""Models loaded as a causal LM from a multimodal checkpoint (Qwen3.5) export in that
checkpoint's layout: its config, its weight names, its vision tower. vLLM serves Qwen3.5
only through the multimodal architecture."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from safetensors import safe_open  # noqa: E402

from palingenesis.checkpoint import save_hf_model  # noqa: E402


def _keys(path: Path) -> set[str]:
    keys = set()
    for file in path.glob("*.safetensors"):
        with safe_open(str(file), framework="pt") as f:
            keys |= set(f.keys())
    return keys


@pytest.fixture
def multimodal(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, linear_num_value_heads=2, linear_num_key_heads=1,
                         linear_key_head_dim=16, linear_value_head_dim=16, tie_word_embeddings=True),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32,
                           patch_size=4, spatial_merge_size=2, temporal_patch_size=2),
        tie_word_embeddings=True)
    path = tmp_path / "source"
    Qwen3_5ForConditionalGeneration(config).save_pretrained(path)
    (path / "preprocessor_config.json").write_text('{"processor_class": "Qwen3VLProcessor"}')
    return path


def test_export_keeps_the_source_layout(tmp_path, multimodal):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(multimodal)
    assert model.config.model_type != json.loads((multimodal / "config.json").read_text())["model_type"]
    with torch.no_grad():                               # "training"
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    out = tmp_path / "export"
    save_hf_model(model, None, out, source_layout=True)

    assert json.loads((out / "config.json").read_text())["model_type"] == "qwen3_5"   # the multimodal config
    assert _keys(out) == _keys(multimodal)                                            # the same weights, names
    assert (out / "preprocessor_config.json").exists()                               # what the architecture needs
    reloaded = AutoModelForCausalLM.from_pretrained(out)
    for (name, p), (_, q) in zip(model.named_parameters(), reloaded.named_parameters()):
        torch.testing.assert_close(p, q, msg=name)                                    # the trained ones
    with safe_open(str(next(multimodal.glob("*.safetensors"))), framework="pt") as a, \
            safe_open(str(next(out.glob("*.safetensors"))), framework="pt") as b:
        vision = [k for k in a.keys() if "visual" in k]
        assert vision
        for k in vision:                                                              # the copied ones
            torch.testing.assert_close(a.get_tensor(k), b.get_tensor(k))


def test_a_trained_weight_without_a_source_name_is_an_error(tmp_path, multimodal, monkeypatch):
    import transformers.core_model_loading as cml
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(multimodal)
    monkeypatch.setattr(cml, "revert_weight_conversion", lambda m, state: state)     # names left as the model's
    with pytest.raises(RuntimeError, match="without a counterpart"):
        save_hf_model(model, None, tmp_path / "export", source_layout=True)


def test_a_plain_causal_lm_saves_as_before(tmp_path):
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    model.save_pretrained(tmp_path / "src")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "src")
    save_hf_model(model, None, tmp_path / "out", source_layout=True)
    assert json.loads((tmp_path / "out" / "config.json").read_text())["model_type"] == "qwen3"
