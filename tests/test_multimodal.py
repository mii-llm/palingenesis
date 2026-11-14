"""Images in SFT: placeholder expansion and masking, M-RoPE positions (alone and under packing),
vision freeze options, FSDP layer discovery, evaluation, the export layout, and what is refused.
Tiny randomly initialised Qwen3.5 models (tests/multimodal_helpers.py); CPU only."""

import functools
import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from distill_helpers import cpu_config, run_cpu
from multimodal_helpers import save_tiny, tiny_image, tiny_image_processor, tiny_qwen3_5, tiny_tokenizer

from palingenesis import multimodal
from palingenesis.config import Config, ConfigError
from palingenesis.data import ChatDataset, PackedDataset, _collate_fn
from palingenesis.multimodal import (
    POSITION_FIELD,
    MultimodalError,
    VisionProcessor,
    forward_kwargs,
    split_images,
)

IGNORE = -100


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


@pytest.fixture(scope="module")
def vision(tok):
    return VisionProcessor(tiny_image_processor(), tok, "qwen3_5")


def _row(seed: int = 0, sizes=((24, 40), (32, 16))) -> dict:
    """A tool-calling conversation with an image in the user turn and one in the tool result."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Look at picture {seed} "},
                    {"type": "image", "image": tiny_image(*sizes[0], seed)},
                ],
            },
            {
                "role": "assistant",
                "content": "Taking a screenshot.",
                "tool_calls": [{"type": "function", "function": {"name": "screenshot", "arguments": {}}}],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "image", "image": tiny_image(*sizes[1], seed + 1)},
                ],
            },
            {"role": "assistant", "content": f"Done with picture {seed}."},
        ]
    }


def _text_row(i: int = 0) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"What is {i} plus one?"},
            {"role": "assistant", "content": f"It is {i + 1}."},
        ]
    }


# ----------------------------------------------------------------- positions


@pytest.mark.parametrize(
    "module, cls",
    [
        ("qwen3_5", "Qwen3_5Model"),
        ("qwen3_vl", "Qwen3VLModel"),
        ("qwen2_5_vl", "Qwen2_5_VLModel"),
        ("qwen2_vl", "Qwen2VLModel"),
    ],
)
def test_positions_match_transformers(tok, vision, module, cls):
    """Our per-document M-RoPE positions are the ones each Qwen-VL family computes in its forward."""
    import importlib

    model_cls = getattr(importlib.import_module(f"transformers.models.{module}.modeling_{module}"), cls)
    stub = types.SimpleNamespace(
        config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2, tokens_per_second=2))
    )
    if hasattr(model_cls, "get_vision_position_ids"):
        stub.get_vision_position_ids = functools.partial(model_cls.get_vision_position_ids, stub)
    for sizes in (((24, 40), (32, 16)), ((8, 8), (64, 8)), ((40, 24), (16, 16))):
        messages, refs = split_images(_row(0, sizes)["messages"])
        images = [vision.encode(r) for r in refs]
        text = vision.expand_text(tok.apply_chat_template(messages, tokenize=False), [im.tokens for im in images])
        ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
        ours = vision.position_ids(ids, [im.grid for im in images])
        grids = torch.tensor([im.grid for im in images])
        theirs, _ = model_cls.get_rope_index(
            stub, ids[None], mm_token_type_ids=(ids == vision.image_token_id).long()[None], image_grid_thw=grids
        )
        assert torch.equal(theirs[:, 0].T, ours)
        if sizes[0] == (24, 40):  # 3 x 5 tokens: the text after resumes at start + 5, not start + 15
            assert int(ours.max()) < ids.numel() - 1


def test_expansion_and_token_counts(tok, vision):
    image = tiny_image(24, 40)
    encoded = vision.encode(image)
    assert encoded.grid == (1, 6, 10) and encoded.tokens == 15
    assert vision.num_tokens(24, 40) == 15  # without touching the pixels
    assert encoded.pixel_values.shape[0] == 60  # patches before the 2 x 2 merge
    ids = [5, vision.image_token_id, 6, vision.image_token_id, 7]
    expanded, index = vision.expand_ids(ids, [3, 2])
    assert expanded == [5] + [vision.image_token_id] * 3 + [6] + [vision.image_token_id] * 2 + [7]
    assert index == [0, 3, 4, 6, 7]  # a placeholder maps to its last copy
    with pytest.raises(MultimodalError):
        vision.expand_ids(ids, [3])
    with pytest.raises(MultimodalError, match="does not render images"):
        vision.expand_text("no placeholder here", [3])


def test_pixel_bounds_set_the_token_count(tok):
    small = VisionProcessor(tiny_image_processor(), tok, "qwen3_5", max_pixels=256)
    assert small.encode(tiny_image(64, 64)).tokens == 4  # 64 x 64 resized to 16 x 16: 4 x 4 patches, 2 x 2 tokens
    with pytest.raises(MultimodalError):
        VisionProcessor(tiny_image_processor(), tok, "qwen3_5", min_pixels=512, max_pixels=256)


def test_unsupported_inputs_are_refused(tok, vision):
    with pytest.raises(MultimodalError, match="implemented for"):
        VisionProcessor(tiny_image_processor(), tok, "gemma3")
    with pytest.raises(MultimodalError, match="video"):
        split_images([{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}]}])
    with pytest.raises(MultimodalError, match="not downloaded"):
        multimodal.load_image("https://example.com/cat.png")


def test_arrow_filled_parts_render_as_text(tok, vision):
    """Arrow gives every part every key (None where absent): a text part must not render as an image."""
    messages, refs = split_images(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "image": None},
                    {"type": "image", "text": None, "image": tiny_image(8, 8)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "hello", "image": None}]},
        ]
    )
    assert len(refs) == 1
    assert messages[1]["content"] == "hello"  # text-only parts become the string maskers expect
    assert tok.apply_chat_template(messages, tokenize=False).count("<|image_pad|>") == 1


# ----------------------------------------------------------------- dataset


def test_dataset_expands_masks_and_positions(tok, vision):
    ds = ChatDataset([_row()], tok, 512, vision=vision)
    (sample,) = list(ds)
    ids, labels = sample["input_ids"], sample["labels"]
    is_image = ids == vision.image_token_id
    assert int(is_image.sum()) == 15 + 8
    assert (labels[is_image] == IGNORE).all()
    trained = tok.decode(ids[labels != IGNORE])
    assert "Taking a screenshot." in trained and "Done with picture 0." in trained
    assert "ok" not in trained and "Look at" not in trained
    assert sample["pixel_values"].shape[0] == 60 + 32
    assert sample["image_grid_thw"].tolist() == [[1, 6, 10], [1, 8, 4]]
    assert sample[POSITION_FIELD].shape == (ids.numel(), 3)
    # same tokens as the Hugging Face path: template, then per-image expansion
    messages, refs = split_images(_row()["messages"])
    text = tok.apply_chat_template(messages, tokenize=False)
    assert ids.tolist() == tok(vision.expand_text(text, [15, 8]), add_special_tokens=False)["input_ids"]


def test_observation_training_never_trains_image_tokens(tok, vision):
    ds = ChatDataset([_row()], tok, 512, vision=vision, include_observations=True)
    (sample,) = list(ds)
    assert (sample["labels"][sample["input_ids"] == vision.image_token_id] == IGNORE).all()
    assert "ok" in tok.decode(sample["input_ids"][sample["labels"] != IGNORE])  # the tool text is


def test_truncation_keeps_whole_turns_and_their_images(tok, vision):
    full = list(ChatDataset([_row()], tok, 512, vision=vision))[0]
    first_turn_end = 60  # room for the first assistant turn (15 image tokens), not for the tool image
    ds = ChatDataset([_row()], tok, first_turn_end, vision=vision)
    (sample,) = list(ds)
    assert sample["input_ids"].numel() < full["input_ids"].numel()
    assert int((sample["input_ids"] == vision.image_token_id).sum()) == 15
    assert sample["image_grid_thw"].tolist() == [[1, 6, 10]]
    assert sample["pixel_values"].shape[0] == 60
    assert ds.stats["truncated"] == 1
    # nothing fits: dropped, never cut inside an image
    ds = ChatDataset([_row()], tok, 12, vision=vision)
    assert list(ds) == []


def test_text_rows_and_image_rows_mix(tok, vision):
    samples = list(ChatDataset([_row(), _text_row()], tok, 512, vision=vision))
    assert "pixel_values" not in samples[1]
    assert torch.equal(samples[1][POSITION_FIELD][:, 0], torch.arange(samples[1]["input_ids"].numel()))
    batch = _collate_fn(samples, tok.pad_token_id)
    assert batch[POSITION_FIELD].shape == (2, batch["input_ids"].shape[1], 3)
    assert batch["image_grid_thw"].shape == (2, 3)


def test_images_without_multimodal_are_ignored_with_a_warning(tok, caplog):
    ds = ChatDataset([_row()], tok, 512)
    with caplog.at_level("WARNING"):
        (sample,) = list(ds)
    assert "pixel_values" not in sample and POSITION_FIELD not in sample
    assert "images are ignored" in caplog.text


def test_vision_processor_travels_with_the_tokenizer(tok, vision):
    """DataLoader workers get a pickled copy of the dataset: its tokenizer still carries the processor."""
    import copy
    import pickle

    carrier = copy.deepcopy(tok)
    multimodal.attach(carrier, vision)
    for clone in (pickle.loads(pickle.dumps(carrier)), copy.deepcopy(carrier)):
        assert multimodal.vision_of(clone).fingerprint == vision.fingerprint
        (sample,) = list(ChatDataset([_row()], clone, 512))  # no vision argument: the tokenizer's
        assert sample["image_grid_thw"].shape == (2, 3)
    assert multimodal.vision_of(tok) is None


def test_teacher_views_are_refused(tok, vision):
    with pytest.raises(MultimodalError, match="teacher views"):
        ChatDataset([], tok, 512, vision=vision, teacher_views=[(object(), object())])


# ----------------------------------------------------------------- forward


def _forward(model, batch, kwargs=None, flatten=False):
    kwargs = forward_kwargs(batch, kwargs or {}, flatten)
    return model(input_ids=batch["input_ids"], **kwargs).logits


def test_batch_forward_matches_transformers(tok, vision):
    """Our explicit positions and pixel order reproduce the model's own multimodal forward."""
    model = tiny_qwen3_5(tok).eval()
    samples = list(ChatDataset([_row(0), _row(3, ((16, 16), (8, 24)))], tok, 512, vision=vision))
    for sample in samples:
        ids = sample["input_ids"][None]
        reference = model(
            input_ids=ids,
            pixel_values=sample["pixel_values"],
            image_grid_thw=sample["image_grid_thw"],
            mm_token_type_ids=(ids == vision.image_token_id).long(),
        ).logits
        batch = _collate_fn([sample], tok.pad_token_id)
        assert torch.allclose(_forward(model, batch), reference, atol=1e-5)
    # padded batch of both rows: each row's logits are its own
    batch = _collate_fn(samples, tok.pad_token_id)
    logits = _forward(model, batch, {"attention_mask": None})
    for row, sample in enumerate(samples):
        alone = _forward(model, _collate_fn([sample], tok.pad_token_id))
        n = sample["input_ids"].numel()
        assert torch.allclose(logits[row, :n], alone[0], atol=1e-4)


def test_packed_documents_keep_their_own_positions_and_images(tok, vision):
    """Several documents and images in one packed row: every document's logits are those it has
    alone (positions restart per document on all axes, images follow in document order)."""
    from palingenesis.packing import PackedBatch

    model = tiny_qwen3_5(tok, layer_types=("full_attention", "full_attention")).eval()
    rows = [_row(0), _text_row(1), _row(3, ((16, 16), (8, 24))), _text_row(2)]
    samples = list(ChatDataset(rows, tok, 512, vision=vision))
    blocks = list(PackedDataset(samples, 4096, sort_buffer=0))
    assert len(blocks) == 1
    block = blocks[0]
    assert block["image_grid_thw"].shape[0] == 4
    starts = (block["position_ids"] == 0).nonzero().flatten().tolist()
    assert len(starts) == 4
    for start, sample in zip(starts, samples):  # each document's M-RoPE positions restart at 0
        n = sample["input_ids"].numel()
        assert torch.equal(block[POSITION_FIELD][start : start + n], sample[POSITION_FIELD])
    batch = _collate_fn([block], tok.pad_token_id)
    packed = PackedBatch.build(batch["input_ids"], batch["labels"], batch["position_ids"], False, "sdpa")
    kwargs = forward_kwargs(batch, packed.forward_kwargs)
    assert kwargs["position_ids"].shape[0] == 4  # text positions (the block-diagonal mask) + 3 M-RoPE axes
    logits = model(input_ids=packed.input_ids, **kwargs).logits[0]
    for start, sample in zip(starts, samples):
        alone = _forward(model, _collate_fn([sample], tok.pad_token_id))[0]
        n = sample["input_ids"].numel()
        assert torch.allclose(logits[start : start + n], alone, atol=1e-4)


def test_packed_flattened_layout(tok, vision):
    """The flattened layout (flash / linear-attention kernels): positions and pixels follow the rows."""
    from palingenesis.packing import PackedBatch

    samples = list(ChatDataset([_row(0), _row(1), _row(2), _text_row(0)], tok, 512, vision=vision))
    blocks = list(PackedDataset(samples, 140, sort_buffer=0))
    assert len(blocks) > 1
    batch = _collate_fn(blocks, tok.pad_token_id)
    packed = PackedBatch.build(batch["input_ids"], batch["labels"], batch["position_ids"], True, "flash_attention_2")
    kwargs = forward_kwargs(batch, packed.forward_kwargs, flatten=True)
    b, s = batch["input_ids"].shape
    assert kwargs["position_ids"].shape == (4, 1, b * s)
    assert torch.equal(kwargs["position_ids"][1:, 0].T, batch[POSITION_FIELD].reshape(b * s, 3))
    # pixels are in the order of the image runs of the flattened row
    runs = int(((packed.input_ids[0] == vision.image_token_id).long().diff() == 1).sum())
    assert runs == batch["image_grid_thw"].shape[0]
    tokens = (batch["image_grid_thw"].prod(-1) // 4).sum()
    assert int((packed.input_ids == vision.image_token_id).sum()) == int(tokens)


def test_packing_refuses_to_cut_a_document_with_images(tok, vision):
    (sample,) = list(ChatDataset([_row()], tok, 512, vision=vision))
    with pytest.raises(MultimodalError, match="cannot be cut"):
        list(PackedDataset([sample], 30, sort_buffer=0))


# ----------------------------------------------------------------- model


def test_vision_freeze_options(tok):
    model = tiny_qwen3_5(tok)
    tower, projector = multimodal.vision_parameters(model)
    assert tower and projector and not set(tower) & set(projector)
    assert all(".merger." in n for n in projector)
    for train_tower, train_projector in ((False, False), (False, True), (True, False), (True, True)):
        model.requires_grad_(True)
        multimodal.set_vision_trainable(model, train_tower, train_projector)
        assert all(p.requires_grad == train_tower for p in tower.values())
        assert all(p.requires_grad == train_projector for p in projector.values())
        assert model.lm_head.weight.requires_grad
    # never unfreezes what something else froze
    model.requires_grad_(False)
    multimodal.set_vision_trainable(model, True, True)
    assert not any(p.requires_grad for p in tower.values())


def test_layer_discovery_for_fsdp_and_checkpointing(tok):
    from palingenesis.distributed import _find_transformer_layers
    from palingenesis.kernels import _find_layers

    model = tiny_qwen3_5(tok)
    assert _find_transformer_layers(model) == list(model.model.language_model.layers)
    assert [m for _, m in _find_layers(model)] == list(model.model.language_model.layers)
    assert multimodal.vision_layers(model) == list(model.model.visual.blocks)


def test_check_model(tok, vision):
    model = tiny_qwen3_5(tok)
    multimodal.check_model(model, vision)
    other = VisionProcessor(tiny_image_processor(), tok, "qwen3_5")
    other.patch_size = 14
    with pytest.raises(MultimodalError, match="reads images as"):
        multimodal.check_model(model, other)
    with pytest.raises(MultimodalError, match="no vision tower"):
        multimodal.check_model(model.model.language_model, vision)


# ----------------------------------------------------------------- config


@pytest.mark.parametrize(
    "section, option",
    [("memory", "seco"), ("parallel", "context_parallel"), ("data", "pretokenize")],
)
def test_unsupported_combinations_are_refused(section, option):
    config = Config()
    config.multimodal.enabled = True
    if option == "context_parallel":
        config.parallel.fsdp = True
    setattr(getattr(config, section), option, True)
    with pytest.raises(ConfigError, match=f"multimodal.enabled with {section}.{option}"):
        config.validate()


def test_vision_training_needs_multimodal():
    config = Config()
    config.multimodal.train_projector = True
    with pytest.raises(ConfigError, match="need multimodal.enabled"):
        config.validate()


# ----------------------------------------------------------------- trainer


def _write_image_rows(tmp_path, n: int = 12) -> str:
    """JSONL rows whose content parts reference image files (what a dataset on disk looks like)."""
    path = tmp_path / "rows.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            row = _row(i, ((16 + 8 * (i % 3), 24), (16, 16 + 8 * (i % 2))))
            for message in row["messages"]:
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part["type"] == "image":
                            file = tmp_path / f"img_{i}_{id(part)}.png"
                            part["image"].save(file)
                            part["image"] = str(file)
                else:  # one schema for every message: parts everywhere
                    message["content"] = [{"type": "text", "text": message["content"]}]
            f.write(json.dumps(row) + "\n")
    return str(path)


def _mm_config(tmp_path, steps: int = 3):
    model_dir = save_tiny(tmp_path / "model")
    config = cpu_config(tmp_path, model_dir, _write_image_rows(tmp_path), steps=steps)
    config.multimodal.enabled = True
    config.data.max_seq_length = 256
    config.memory.chunked_loss = True
    return config


def _state(path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(Path(path) / "model.safetensors"))


@pytest.mark.parametrize("train_tower, train_projector", [(False, False), (False, True), (True, True)])
def test_sft_trains_and_exports_the_multimodal_layout(tmp_path, train_tower, train_projector):
    config = _mm_config(tmp_path)
    config.multimodal.train_vision_tower = train_tower
    config.multimodal.train_projector = train_projector
    config.data.eval_dataset = config.data.dataset
    config.data.eval_split = "train"
    config.data.eval_samples = 4
    config.data.eval_every = 2
    result = run_cpu(config)
    assert result.global_step == 3
    assert result.best_eval_loss is not None and result.best_eval_loss > 0
    before, after = _state(tmp_path / "model"), _state(Path(result.output_dir) / "final")
    assert set(before) == set(after)  # the source's layout: language model, vision tower, projector
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    tower = {k for k in before if ".visual." in k and ".merger." not in k}
    projector = {k for k in before if ".visual.merger." in k}
    assert tower and projector
    assert bool(changed & tower) == train_tower
    assert bool(changed & projector) == train_projector
    assert any(".language_model." in k for k in changed)
    # loadable as the multimodal architecture, with the processor files a server needs
    final = Path(result.output_dir) / "final"
    assert (final / "preprocessor_config.json").exists()
    from transformers import AutoModelForImageTextToText

    reloaded = AutoModelForImageTextToText.from_pretrained(final)
    assert type(reloaded).__name__ == "Qwen3_5ForConditionalGeneration"
    reloaded_vision = VisionProcessor.from_pretrained(str(final), tiny_tokenizer())
    assert (
        reloaded_vision.fingerprint
        == VisionProcessor.from_pretrained(config.model.name_or_path, tiny_tokenizer()).fingerprint
    )


def test_sft_with_packing(tmp_path):
    config = _mm_config(tmp_path)
    config.data.packing = True
    config.data.max_seq_length = 300  # two or three documents (four to six images) per packed row
    config.train.per_device_batch_size = 1
    config.train.gradient_accumulation_steps = 1
    # packing needs the linear-attention kernels on a hybrid model: the tiny student is attention only
    save_tiny(tmp_path / "model", layer_types=("full_attention", "full_attention"))
    result = run_cpu(config)
    assert result.global_step == 3
    assert result.metrics["train/loss"] > 0


def test_pixel_bounds_reach_the_export(tmp_path):
    config = _mm_config(tmp_path, steps=1)
    config.multimodal.max_pixels = 1024
    result = run_cpu(config)
    saved = json.loads((Path(result.output_dir) / "final" / "preprocessor_config.json").read_text())
    assert saved["size"]["longest_edge"] == 1024


def test_text_only_checkpoint_is_refused(tmp_path):
    from distill_helpers import save_model, write_rows

    tok = tiny_tokenizer()
    save_model(tmp_path / "text", tok)
    config = cpu_config(tmp_path, tmp_path / "text", write_rows(tmp_path / "rows.jsonl"))
    config.multimodal.enabled = True
    with pytest.raises(MultimodalError, match="no vision tower"):
        run_cpu(config)


def test_profile_counts_the_vision_tower(tmp_path):
    from palingenesis.agent_tooling.profile_memory import estimate_memory

    config = _mm_config(tmp_path)
    frozen = estimate_memory(config)
    assert frozen["vision_params_B"] > 0 and frozen["vision_memory_gb"] > 0
    assert 0 < frozen["trainable_ratio"] < 1  # the vision side is in the weights, not in the optimizer
    config.multimodal.train_vision_tower = config.multimodal.train_projector = True
    trained = estimate_memory(config)
    assert trained["trainable_ratio"] == 1.0
    assert trained["vision_memory_gb"] > frozen["vision_memory_gb"]  # activations kept for the backward
    assert trained["optimizer_memory_gb"] > frozen["optimizer_memory_gb"]
    config.multimodal.enabled = False
    config.multimodal.train_vision_tower = config.multimodal.train_projector = False
    assert estimate_memory(config)["vision_memory_gb"] == 0.0


def test_diagnostic_tools_read_images_like_the_trainer(tmp_path):
    from palingenesis.agent_tooling._pipeline import load_tokenizer

    config = _mm_config(tmp_path)
    assert multimodal.vision_of(load_tokenizer(config)) is not None
    config.multimodal.enabled = False
    assert multimodal.vision_of(load_tokenizer(config)) is None


def test_opd_refuses_images():
    from palingenesis.opd.formatting import encode_prompt

    messages = [{"role": "user", "content": [{"type": "image", "image": tiny_image(8, 8)}]}]
    with pytest.raises(MultimodalError, match="does not take images yet"):
        encode_prompt(tiny_tokenizer(), messages)
