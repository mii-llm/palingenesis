"""Saved models must load with plain `from_pretrained`: parameter names are the
architecture's, whatever wrappers training added (activation checkpointing,
torch.compile). A name mismatch makes from_pretrained initialise those weights
randomly, which reads as a model that "trained fine" and then generates garbage.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.checkpoint import load_checkpoint, save_checkpoint, save_final, save_hf_model  # noqa: E402
from palingenesis.kernels import apply_activation_checkpointing  # noqa: E402
from palingenesis.train import _compile_layers  # noqa: E402


def _tiny(tie=True):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=tie))


def _assert_same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        torch.testing.assert_close(sa[k], sb[k], rtol=0, atol=0, msg=k)


@pytest.mark.parametrize("tie", [True, False])
def test_trained_model_reloads_with_from_pretrained(tmp_path, tie):
    from transformers import LlamaForCausalLM

    model = _tiny(tie)
    apply_activation_checkpointing(model, mode="selective")
    _compile_layers(model)            # in place: no "_orig_mod." in names
    assert not any("_orig_mod" in k for k in model.state_dict())
    save_final(model, None, str(tmp_path))
    _assert_same_weights(_tiny(tie), LlamaForCausalLM.from_pretrained(tmp_path / "final"))


def test_wrapper_prefixes_are_stripped_on_save(tmp_path):
    """Models wrapped the old way (layer = torch.compile(layer)) save clean names too."""
    from transformers import LlamaForCausalLM

    model = _tiny()
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = torch.compile(layer)
    assert any("_orig_mod" in k for k in model.state_dict())
    save_hf_model(model, None, tmp_path / "m")
    _assert_same_weights(_tiny(), LlamaForCausalLM.from_pretrained(tmp_path / "m"))


def test_resume_refuses_a_checkpoint_missing_weights(tmp_path):
    from safetensors.torch import load_file, save_file

    model = _tiny(tie=False)
    opt = torch.optim.AdamW(model.parameters())
    save_checkpoint(model, None, opt, None, step=3, output_dir=str(tmp_path))
    weights = next((tmp_path / "step-3" / "model").glob("*.safetensors"))
    state = load_file(str(weights))
    state.pop("lm_head.weight")
    save_file(state, str(weights), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="lacks weights"):
        load_checkpoint(_tiny(tie=False), torch.optim.AdamW(_tiny(tie=False).parameters()), None, str(tmp_path / "step-3"))


def test_resume_restores_weights(tmp_path):
    model = _tiny()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    opt = torch.optim.AdamW(model.parameters())
    save_checkpoint(model, None, opt, None, step=5, output_dir=str(tmp_path))
    fresh = _tiny()
    apply_activation_checkpointing(fresh, mode="full")
    meta = load_checkpoint(fresh, torch.optim.AdamW(fresh.parameters()), None, str(tmp_path / "step-5"))
    assert meta["step"] == 5
    _assert_same_weights(model, fresh)
