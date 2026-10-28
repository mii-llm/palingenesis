"""pgs learn adapters: the anchored MLP regex, exact stacking of a scope chain, the frozen ancestor stack,
gradients only into the scope's adapter, and a readable error without PEFT."""

import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from palingenesis.learn import adapter as adapters  # noqa: E402
from palingenesis.learn.config import MLP_TARGET_REGEX  # noqa: E402


def tiny_dense(seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return transformers.Qwen3ForCausalLM(config).eval()


def test_regex_matches_dense_mlp_projections_only():
    names = adapters.target_modules(tiny_dense(), MLP_TARGET_REGEX)
    assert len(names) == 6 and all(".mlp." in n and n.startswith("model.layers.") for n in names)


def test_regex_skips_vision_and_mtp_on_a_qwen35_style_model():
    config = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            tie_word_embeddings=True,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=32,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3_5ForConditionalGeneration(config)
    # An MTP-like block and a vision MLP with the same projection names must never match.
    model.mtp = torch.nn.Module()
    model.mtp.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.mtp.layers[0].mlp = torch.nn.Module()
    model.mtp.layers[0].mlp.gate_proj = torch.nn.Linear(4, 4)
    model.model.visual.blocks[0].mlp.gate_proj = torch.nn.Linear(4, 4)
    names = adapters.target_modules(model, MLP_TARGET_REGEX)
    assert len(names) == 6
    assert all(n.startswith("model.language_model.layers.") and ".mlp." in n for n in names)
    with pytest.raises(ValueError, match="no linear module"):
        adapters.target_modules(model, r"^nothing$")


def random_adapter(path, rank, alpha, seed, regex=MLP_TARGET_REGEX):
    """A saved PEFT LoRA with random A and B (a trained adapter's stand-in)."""
    peft = pytest.importorskip("peft")
    model = tiny_dense()
    torch.manual_seed(seed)
    config = peft.LoraConfig(r=rank, lora_alpha=alpha, target_modules=regex, init_lora_weights=False)
    wrapped = peft.get_peft_model(model, config)
    with torch.no_grad():
        for name, p in wrapped.named_parameters():
            if "lora_" in name:
                p.copy_(torch.randn_like(p) * 0.1)
    wrapped.save_pretrained(path)
    return path


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_stacking_is_exact(tmp_path):
    """Exact up to fp32 rounding (the export is fp32, as serving engines load it)."""
    peft = pytest.importorskip("peft")
    paths = [
        random_adapter(tmp_path / "org", 4, 8, 1),
        random_adapter(tmp_path / "team", 2, 4, 2, regex=r"^model\.layers\.\d+\.mlp\.(up|down)_proj$"),
        random_adapter(tmp_path / "user", 2, 2, 3),
    ]
    ids = torch.randint(0, 64, (2, 9))
    # Reference: the adapters applied one after the other (each merged into the weights).
    model = tiny_dense()
    for path in paths:
        model = peft.PeftModel.from_pretrained(model, path).merge_and_unload()
    reference = logits(model, ids)
    for pad_to in (None, 16):
        config = adapters.stack_dirs(paths, tmp_path / f"stacked{pad_to}", "tiny", pad_to=pad_to)
        assert config["r"] == (pad_to or 8) and config["lora_alpha"] == config["r"]
        stacked = peft.PeftModel.from_pretrained(tiny_dense(), tmp_path / f"stacked{pad_to}")
        assert torch.allclose(logits(stacked, ids), reference, atol=1e-5, rtol=0)
    assert not torch.allclose(logits(tiny_dense(), ids), reference, atol=1e-4)
    # The summed delta equals the sum of the deltas, module by module.
    parts = [adapters.read_adapter(p)[0] for p in paths]
    total = adapters.stack(parts)
    module = "model.layers.1.mlp.up_proj"
    expected = sum(b @ a for a, b in (p[module] for p in parts))
    assert torch.allclose(total[module][1] @ total[module][0], expected, atol=1e-6)


def test_frozen_delta_equals_the_merged_ancestors(tmp_path):
    peft = pytest.importorskip("peft")
    paths = [random_adapter(tmp_path / "org", 4, 8, 1), random_adapter(tmp_path / "team", 2, 4, 2)]
    ids = torch.randint(0, 64, (2, 7))
    merged = tiny_dense()
    for path in paths:
        merged = peft.PeftModel.from_pretrained(merged, path).merge_and_unload()
    model = tiny_dense()
    hooks = adapters.FrozenDelta(model, adapters.stack([adapters.read_adapter(p)[0] for p in paths]))
    assert torch.allclose(logits(model, ids), logits(merged, ids), atol=1e-6)
    hooks.remove()
    assert torch.allclose(logits(model, ids), logits(tiny_dense(), ids), atol=0)


def test_frozen_delta_swaps_to_another_chain_and_back(tmp_path):
    """Role previous after a parent update: the old ancestors' delta (or none) for a while, the live one after,
    on the base linears even once PEFT has wrapped them."""
    pytest.importorskip("peft")
    old, new = (adapters.read_adapter(random_adapter(tmp_path / n, 4, 8, s))[0] for n, s in (("old", 1), ("new", 2)))
    ids = torch.randint(0, 64, (2, 7))
    model = tiny_dense()
    frozen = adapters.FrozenDelta(model, new)
    adapters.attach_lora(model, 2, 4, MLP_TARGET_REGEX)  # B starts at zero: the LoRA adds nothing yet
    expected = {}
    for name, tensors in (("old", old), ("new", new), ("none", {})):
        reference = tiny_dense()
        adapters.FrozenDelta(reference, tensors)
        expected[name] = logits(reference, ids)
    assert torch.allclose(logits(model, ids), expected["new"], atol=1e-6)
    with frozen.swapped(frozen.place(old)):
        assert torch.allclose(logits(model, ids), expected["old"], atol=1e-6)
    with frozen.swapped({}):
        assert torch.allclose(logits(model, ids), expected["none"], atol=1e-6)
    assert torch.allclose(logits(model, ids), expected["new"], atol=1e-6)


def test_gradients_reach_only_the_scope_adapter(tmp_path):
    pytest.importorskip("peft")
    model = tiny_dense().train()
    ancestors = adapters.read_adapter(random_adapter(tmp_path / "org", 4, 8, 1))[0]
    frozen = adapters.FrozenDelta(model, ancestors)
    before_base = {n: p.detach().clone() for n, p in model.named_parameters()}
    before_anc = {k: (a.clone(), b.clone()) for k, (a, b) in frozen.tensors.items()}
    wrapped = adapters.attach_lora(model, 2, 4, MLP_TARGET_REGEX)
    previous = adapters.lora_state(wrapped)
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-2)
    ids = torch.randint(0, 64, (2, 6))
    for _ in range(3):
        loss = wrapped(input_ids=ids, labels=ids).loss
        optimizer.zero_grad()
        loss.backward()
        for name, p in wrapped.named_parameters():
            if "lora_" not in name:
                assert p.grad is None, name
        optimizer.step()
    for name, p in wrapped.get_base_model().named_parameters():
        if "lora_" not in name:
            key = name.replace(".base_layer", "")
            assert torch.equal(p, before_base[key]), name
    for k, (a, b) in frozen.tensors.items():
        assert torch.equal(a, before_anc[k][0]) and torch.equal(b, before_anc[k][1])
    # The "previous" snapshot can be swapped in and out without touching requires_grad or the trained values.
    trained = adapters.lora_state(wrapped)
    with adapters.swapped_lora(wrapped, previous):
        assert all(torch.equal(p, previous[n]) for n, p in wrapped.named_parameters() if n in previous)
    assert all(torch.equal(p, trained[n]) for n, p in wrapped.named_parameters() if n in trained)
    assert [n for n, p in wrapped.named_parameters() if p.requires_grad] == trainable


def test_readable_error_without_peft(monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(ImportError, match=r"palingenesis\[learn\]"):
        adapters.require_peft()
