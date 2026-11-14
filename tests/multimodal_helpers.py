"""Tiny randomly initialised vision-language models for the multimodal tests (no downloads beyond
the gpt2 tokenizer the other tests use): a Qwen-style tokenizer with the image tokens and the
Qwen3.5 chat template, a small-patch image processor, and Qwen-VL family models."""

from pathlib import Path

import numpy as np
import torch

TEMPLATE = Path(__file__).parent / "fixtures" / "qwen3_5_chat_template.jinja"
PATCH = 4  # pixels per patch; 2 x 2 patches merge into one token


def tiny_tokenizer():
    from transformers import AutoTokenizer

    try:  # the local cache first: no Hub request
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
            ]
        }
    )
    tok.add_tokens(["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"])
    tok.chat_template = TEMPLATE.read_text()
    tok.pad_token = "<|im_end|>"
    tok.eos_token = "<|im_end|>"
    return tok


def tiny_image_processor(min_pixels: int = 64, max_pixels: int = 4096):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    return Qwen2VLImageProcessorPil(
        patch_size=PATCH,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )


def tiny_image(height: int, width: int, seed: int = 0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def _ids(tok) -> dict:
    return dict(
        image_token_id=tok.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"),
    )


def _vision_config(hidden: int) -> dict:
    return dict(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=PATCH,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=hidden,
        num_position_embeddings=64,
    )


def tiny_qwen3_5(tok, hidden: int = 32, seed: int = 0, layer_types=("linear_attention", "full_attention")):
    """Qwen3_5ForConditionalGeneration: hybrid language model (Gated DeltaNet + attention), vision
    tower and patch merger."""
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=len(tok),
            hidden_size=hidden,
            intermediate_size=2 * hidden,
            num_hidden_layers=len(layer_types),
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=list(layer_types),
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        ),
        vision_config=_vision_config(hidden),
        **_ids(tok),
    )
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(config)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model


def save_tiny(path, tok=None, model=None, **model_kwargs) -> str:
    """A tiny multimodal checkpoint on disk: weights, tokenizer and image processor."""
    tok = tok or tiny_tokenizer()
    model = model or tiny_qwen3_5(tok, **model_kwargs)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    tiny_image_processor().save_pretrained(path)
    return str(path)
