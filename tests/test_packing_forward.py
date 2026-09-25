"""Packed forward: documents in one packed row must not see each other.

transformers keeps packed documents apart only when the forward gets the right
arguments (see palingenesis.packing); these tests pin PackedBatch's arguments and,
on a tiny Llama, that a document's logits are the same packed or alone.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.config import ConfigError  # noqa: E402
from palingenesis.loss import shift_labels  # noqa: E402
from palingenesis.packing import PackedBatch, check_packing_support  # noqa: E402


def _batch():
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 2, 3, 0, 1, 0, 1]])
    ids = torch.arange(16).view(2, 8) + 1
    return ids, shift_labels(ids.clone(), pos), pos


def test_rows_kept_without_flattening():
    ids, labels, pos = _batch()
    b = PackedBatch.build(ids, labels, pos, flatten=False, attn_implementation="sdpa")
    assert b.forward_kwargs["attention_mask"] is None  # lets transformers build the packed mask
    assert b.input_ids.shape == (2, 8) and "cu_seq_lens_q" not in b.forward_kwargs


def test_flattened_boundaries():
    ids, labels, pos = _batch()
    b = PackedBatch.build(ids, labels, pos, flatten=True, attn_implementation="flash_attention_2")
    kw = b.forward_kwargs
    assert b.input_ids.shape == (1, 16) and b.labels.shape == (1, 16) and kw["position_ids"].shape == (1, 16)
    assert kw["cu_seq_lens_q"].tolist() == [0, 3, 5, 8, 12, 14, 16]
    assert kw["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5]]
    assert kw["max_length_q"] == 4
    # the last token of each document never predicts the next document's first token
    assert b.labels[0, 2] == -100 and b.labels[0, 7] == -100


def test_documents_isolated_in_a_real_forward():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_implementation="sdpa",
        )
    ).eval()
    model.config.use_cache = False
    a, b = torch.randint(0, 64, (1, 5)), torch.randint(0, 64, (1, 7))
    ids = torch.cat([a, b], 1)
    pos = torch.cat([torch.arange(5), torch.arange(7)])[None]
    batch = PackedBatch.build(ids, shift_labels(ids.clone(), pos), pos, flatten=False, attn_implementation="sdpa")
    with torch.no_grad():
        alone = model(input_ids=b).logits
        packed = model(input_ids=batch.input_ids, **batch.forward_kwargs).logits[:, 5:]
        leaky = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos).logits[:, 5:]
    torch.testing.assert_close(packed, alone, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(leaky, alone, atol=1e-3), "a 2D mask of ones must be what leaks (sanity check)"


def test_packing_refused_for_recurrent_layers():
    class Cfg:
        layer_types = ["full_attention", "mamba"]

        def get_text_config(self):
            return self

    class M(torch.nn.Module):
        config = Cfg()

    with pytest.raises(ConfigError, match="mamba"):
        check_packing_support(M(), "sdpa")
