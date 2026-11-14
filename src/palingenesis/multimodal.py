"""Images in the token stream: placeholders, pixels and positions for vision-language models.

A vision-language model reads an image as a run of placeholder tokens whose embeddings are
replaced by the vision tower's output. Three things must agree for that to be exact, in
training as at inference:

  * the number of placeholder tokens of each image (from the image processor's resize),
  * the pixels handed to the vision tower, in the order the placeholders appear,
  * the position of every token. Qwen-VL models use multimodal RoPE: a text token advances
    the temporal, height and width axes together; an image of h x w tokens keeps the temporal
    index, spreads over height and width, and the text after it resumes at start + max(h, w).

`VisionProcessor` owns all three for one model. The data pipeline renders the chat template
(which writes one placeholder per image), expands each placeholder to its image's token count
before tokenizing, and computes the positions per document, so packed rows (several documents
and images per row) carry exactly the positions each document has on its own. The rollout
engines receive the un-expanded tokens and the raw images (what vLLM expects) and the trainer
expands them with the same processor: `expand_ids`.

Supported: images with the Qwen-VL family (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5), whose
positions are checked against transformers' own `get_rope_index` in the tests. Video and other
families are refused with an error instead of being trained on unverified positions.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from palingenesis.config import ConfigError

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# model_type of the checkpoints whose image layout and positions are implemented (and tested
# against transformers).
MROPE_FAMILIES = frozenset({"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"})

IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})
VIDEO_PART_TYPES = frozenset({"video", "video_url", "input_video"})

# Batch fields that are per image, not per token: concatenated in row order by the collators.
PIXEL_FIELDS = ("pixel_values", "image_grid_thw")
POSITION_FIELD = "mm_position_ids"  # [tokens, 3]: temporal, height, width

# Parameters of the vision side. The projector maps vision features into the language model's
# embedding space (Qwen: the patch merger); everything else under the tower is the encoder.
_PROJECTOR = re.compile(r"(^|\.)(visual\.(merger|deepstack_merger_list)|multi_modal_projector|mm_projector)\.")
_TOWER = re.compile(r"(^|\.)(visual|vision_tower|vision_model)\.")


class MultimodalError(ConfigError):
    """A multimodal input or setting this library cannot train on exactly."""


# ------------------------------------------------------------------ images


def load_image(ref: Any):
    """A PIL RGB image from what datasets and tools hand over: a PIL image, raw bytes, a
    `datasets` Image dict ({"bytes": ..., "path": ...}), a file path, or a base64 data URL.
    Remote URLs are refused: a data worker must not download (store the images with the data)."""
    from PIL import Image

    if isinstance(ref, Image.Image):
        image = ref
    elif isinstance(ref, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(ref)))
    elif isinstance(ref, dict):
        if ref.get("bytes"):
            image = Image.open(io.BytesIO(ref["bytes"]))
        elif ref.get("path"):
            image = Image.open(ref["path"])
        elif ref.get("url"):
            return load_image(ref["url"])
        else:
            raise MultimodalError(f"image reference without bytes, path or url: keys {sorted(ref)}")
    elif isinstance(ref, str):
        if ref.startswith("data:"):
            image = Image.open(io.BytesIO(base64.b64decode(ref.split(",", 1)[1])))
        elif ref.startswith(("http://", "https://")):
            raise MultimodalError(
                f"remote image {ref[:80]!r}: images are not downloaded during training; store them in the "
                "dataset (bytes or local paths)"
            )
        else:
            image = Image.open(ref[7:] if ref.startswith("file://") else ref)
    else:
        raise MultimodalError(f"unsupported image reference of type {type(ref).__name__}")
    return image if image.mode == "RGB" else image.convert("RGB")


def _part_image(part: dict) -> Any:
    """The image reference of a content part (OpenAI image_url, Responses input_image, HF image)."""
    for key in ("image", "image_url", "input_image", "url", "path", "bytes"):
        value = part.get(key)
        if value is not None:
            return value
    raise MultimodalError(f"image part without an image: keys {sorted(part)}")


def has_image_parts(messages: list[dict]) -> bool:
    return any(
        isinstance(p, dict) and p.get("type") in IMAGE_PART_TYPES | VIDEO_PART_TYPES
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
    )


def split_images(messages: list[dict]) -> tuple[list[dict], list[Any]]:
    """(messages for the chat template, image references in rendering order). Image parts become
    bare {"type": "image"} parts, which templates render as one placeholder each; the pixels
    travel separately. Messages without image parts are returned as they are."""
    images: list[Any] = []
    out = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in VIDEO_PART_TYPES:
                raise MultimodalError("video inputs are not supported yet (images only)")
            if kind in IMAGE_PART_TYPES:
                images.append(_part_image(part))
                parts.append({"type": "image"})
            elif kind == "text":
                # only the text: Arrow fills the other parts' keys with None ("image": None), which
                # templates that test `'image' in part` would render as an image
                parts.append({"type": "text", "text": part.get("text") or ""})
            else:
                parts.append(part)
        if all(p.get("type") == "text" for p in parts if isinstance(p, dict)) and all(
            isinstance(p, dict) for p in parts
        ):  # text only: the string templates and maskers expect (parts concatenate without a separator)
            out.append({**message, "content": "".join(p["text"] for p in parts)})
        else:
            out.append({**message, "content": parts})
    return out, images


# ------------------------------------------------------------------ processor


@dataclass
class ImageInput:
    """One image as the model takes it."""

    pixel_values: torch.Tensor  # [patches, channels x temporal patch x patch x patch]
    grid: tuple[int, int, int]  # (t, h, w) in patches, before merging
    tokens: int  # placeholder tokens: t x h x w / merge^2


class VisionProcessor:
    """Image placeholders, pixels and positions of one vision-language model.

    image_processor: the checkpoint's Hugging Face image processor (its resize decides each
    image's token count; `min_pixels` / `max_pixels` override its bounds, and must be the ones
    the model is served with). tokenizer: the model's, with the image placeholder token."""

    def __init__(
        self,
        image_processor,
        tokenizer,
        model_type: str,
        image_token: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        if model_type not in MROPE_FAMILIES:
            raise MultimodalError(
                f"multimodal training is implemented for {sorted(MROPE_FAMILIES)}; this checkpoint is "
                f"{model_type!r}. Train it text-only (multimodal.enabled: false) or add its image layout "
                "to palingenesis.multimodal."
            )
        self.image_processor = image_processor
        self.model_type = model_type
        self.source = ""  # the checkpoint directory (from_pretrained)
        self.image_token = image_token or getattr(tokenizer, "image_token", None) or "<|image_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        if self.image_token_id is None or self.image_token_id == tokenizer.unk_token_id:
            raise MultimodalError(f"the tokenizer has no {self.image_token!r} token: not this model's tokenizer")
        self.merge_size = int(image_processor.merge_size)
        self.patch_size = int(image_processor.patch_size)
        size = dict(getattr(image_processor, "size", None) or {})
        self.min_pixels = int(min_pixels or size.get("shortest_edge") or 56 * 56)
        self.max_pixels = int(max_pixels or size.get("longest_edge") or 28 * 28 * 1280)
        if self.min_pixels > self.max_pixels:
            raise MultimodalError(f"multimodal.min_pixels {self.min_pixels} > max_pixels {self.max_pixels}")

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        tokenizer,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        trust_remote_code: bool = False,
    ) -> "VisionProcessor":
        from transformers import AutoConfig, AutoImageProcessor

        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        if not hasattr(config, "vision_config"):
            raise MultimodalError(f"{name_or_path} has no vision tower (no vision_config): it is a text-only model")
        try:
            image_processor = AutoImageProcessor.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        except ImportError as e:
            # transformers' image processors run on torchvision, which is what a server uses too.
            # Without it, the Qwen-VL processor's pillow backend: the same resize rule and token
            # counts, pixels that differ in the last bits of the interpolation.
            try:
                from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

                image_processor = Qwen2VLImageProcessorPil.from_pretrained(name_or_path)
            except Exception:  # noqa: BLE001
                raise MultimodalError(
                    "multimodal training needs torchvision and pillow for the image processor (both come "
                    "with the vllm extra; otherwise install torchvision for your torch build)"
                ) from e
            logger.warning(
                "torchvision is not installed: images are resized with pillow, whose pixels differ slightly "
                "from the torchvision processor a server uses. Install torchvision for training."
            )
        except Exception as e:  # noqa: BLE001 — a checkpoint saved without its processor files
            raise MultimodalError(f"{name_or_path} has no image processor (preprocessor_config.json): {e}") from e
        vision = cls(image_processor, tokenizer, config.model_type, min_pixels=min_pixels, max_pixels=max_pixels)
        try:  # where the checkpoint's other processor files are (exports copy them)
            from transformers.utils import cached_file

            vision.source = str(Path(cached_file(name_or_path, "config.json")).parent)
        except Exception:  # noqa: BLE001 — exports then carry the image processor only
            vision.source = ""
        return vision

    def save_pretrained(self, path) -> None:
        """The image processor's files, with the pixel bounds in use, next to a saved model; the
        source checkpoint's other processor files (video) are copied as they are."""
        import shutil

        path = Path(path)
        self.image_processor.size = {"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels}
        self.image_processor.save_pretrained(path)
        if self.source:
            for extra in Path(self.source).glob("*processor*.json"):
                if not (path / extra.name).exists():
                    shutil.copy(extra, path / extra.name)

    @property
    def fingerprint(self) -> dict:
        """What decides an image's tokens and pixels (two models can share images only when equal)."""
        return {
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "temporal_patch_size": int(getattr(self.image_processor, "temporal_patch_size", 1)),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "image_mean": [float(x) for x in (getattr(self.image_processor, "image_mean", None) or [])],
            "image_std": [float(x) for x in (getattr(self.image_processor, "image_std", None) or [])],
            "image_token_id": int(self.image_token_id),
        }

    def _bounds(self) -> dict:
        return {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}

    def num_tokens(self, height: int, width: int) -> int:
        """Placeholder tokens of an image of this size, without processing its pixels."""
        patches = self.image_processor.get_number_of_image_patches(height, width, self._bounds())
        return int(patches) // self.merge_size**2

    def encode(self, image: Any) -> ImageInput:
        image = load_image(image)
        out = self.image_processor(images=[image], return_tensors="pt", **self._bounds())
        grid = tuple(int(x) for x in out["image_grid_thw"][0])
        return ImageInput(out["pixel_values"], grid, grid[0] * grid[1] * grid[2] // self.merge_size**2)

    # -- placeholders

    def expand_text(self, text: str, counts: list[int]) -> str:
        """`text` with its i-th image placeholder repeated counts[i] times. The template must have
        written exactly one placeholder per image."""
        pieces = text.split(self.image_token)
        if len(pieces) - 1 != len(counts):
            raise MultimodalError(
                f"the chat template rendered {len(pieces) - 1} image placeholder(s) for {len(counts)} image(s): "
                "it does not render images in every role that carries one here (see the multimodal guide: "
                "image tool results)"
            )
        out = [pieces[0]]
        for count, piece in zip(counts, pieces[1:]):
            out.append(self.image_token * count)
            out.append(piece)
        return "".join(out)

    def expand_ids(self, ids: list[int], counts: list[int]) -> tuple[list[int], list[int]]:
        """(expanded ids, index): the i-th placeholder token of `ids` repeated counts[i] times;
        index[j] is where ids[j] lands (a placeholder: its last copy, the position whose output
        predicts the token after the image)."""
        expanded: list[int] = []
        index: list[int] = []
        k = 0
        for token in ids:
            if token == self.image_token_id:
                if k >= len(counts):
                    raise MultimodalError(f"more image placeholders in the tokens than images ({len(counts)})")
                expanded += [token] * counts[k]
                k += 1
            else:
                expanded.append(token)
            index.append(len(expanded) - 1)
        if k != len(counts):
            raise MultimodalError(f"{len(counts)} image(s) for {k} image placeholder(s) in the tokens")
        return expanded, index

    def count_images(self, ids) -> int:
        """Image placeholder tokens in un-expanded `ids` (one per image)."""
        if torch.is_tensor(ids):
            return int((ids == self.image_token_id).sum())
        return sum(1 for t in ids if t == self.image_token_id)

    # -- positions

    def position_ids(self, input_ids: torch.Tensor, grids: list[tuple[int, int, int]]) -> torch.Tensor:
        """Multimodal RoPE positions of one document: [tokens, 3] (temporal, height, width), from 0.
        `input_ids` holds the expanded placeholders; grids[i] is the i-th image's (t, h, w)."""
        n = input_ids.numel()
        positions = torch.empty(n, 3, dtype=torch.long)
        is_image = (input_ids == self.image_token_id).tolist()
        current = 0  # the next text position
        i = 0
        image = 0
        while i < n:
            if not is_image[i]:
                j = i
                while j < n and not is_image[j]:
                    j += 1
                positions[i:j] = (torch.arange(j - i) + current)[:, None]
                current += j - i
                i = j
                continue
            if image >= len(grids):
                raise MultimodalError(f"image tokens at {i} but only {len(grids)} image(s)")
            t, h, w = grids[image]
            h, w = h // self.merge_size, w // self.merge_size
            count = t * h * w
            if not all(is_image[i : i + count]) or (i + count < n and is_image[i + count] and t * h * w == 0):
                raise MultimodalError("image placeholder run shorter than its image's token count")
            if t != 1:
                raise MultimodalError("video grids (t > 1) are not supported")
            positions[i : i + count, 0] = current
            positions[i : i + count, 1] = torch.arange(h).repeat_interleave(w) + current
            positions[i : i + count, 2] = torch.arange(w).repeat(h) + current
            current += max(h, w)
            i += count
            image += 1
        if image != len(grids):
            raise MultimodalError(f"{len(grids)} image(s) but {image} image run(s) in the tokens")
        return positions

    def sequence(self, ids: list[int], images: list[Any]) -> "ImageSequence":
        """Model inputs of un-expanded `ids` and their images (rollout tokens -> training inputs)."""
        encoded = [im if isinstance(im, ImageInput) else self.encode(im) for im in images]
        expanded, index = self.expand_ids(ids, [e.tokens for e in encoded])
        input_ids = torch.tensor(expanded, dtype=torch.long)
        return ImageSequence(input_ids, index, self.position_ids(input_ids, [e.grid for e in encoded]), encoded)


@dataclass
class ImageSequence:
    input_ids: torch.Tensor  # expanded
    index: list[int]  # un-expanded token j -> its position in input_ids
    positions: torch.Tensor  # [tokens, 3]
    images: list[ImageInput]


# The tokenizer carries its model's vision processor, so every reader built from the tokenizer
# (training, evaluation, DPO pairs, the diagnostic tools) sees images the same way.
_ATTRIBUTE = "_palingenesis_vision"


def attach(tokenizer, vision: VisionProcessor | None) -> None:
    setattr(tokenizer, _ATTRIBUTE, vision)


def vision_of(tokenizer) -> VisionProcessor | None:
    return getattr(tokenizer, _ATTRIBUTE, None)


# ------------------------------------------------------------------ batches


def image_fields(input_ids: torch.Tensor, images: list[ImageInput], vision: VisionProcessor) -> dict[str, torch.Tensor]:
    """The multimodal fields of one tokenized document (expanded placeholders)."""
    fields = {POSITION_FIELD: vision.position_ids(input_ids, [im.grid for im in images])}
    if images:
        fields["pixel_values"] = torch.cat([im.pixel_values for im in images])
        fields["image_grid_thw"] = torch.tensor([im.grid for im in images], dtype=torch.long)
    return fields


def collate_images(items: list[dict], width: int) -> dict[str, torch.Tensor]:
    """The multimodal fields of a batch whose rows are padded to `width`: positions per token
    (padding continues each row's positions, as text would), pixels and grids concatenated in row
    order, which is the order the model fills the placeholders in. {} for a batch without positions."""
    if not any(POSITION_FIELD in x for x in items):
        return {}
    rows = []
    for x in items:
        n = x["input_ids"].size(0)
        pos = x.get(POSITION_FIELD)
        if pos is None:  # a text row among image rows
            pos = torch.arange(n)[:, None].expand(n, 3)
        start = int(pos.max()) + 1 if n else 0
        pad = (torch.arange(width - n) + start)[:, None].expand(width - n, 3)
        rows.append(torch.cat([pos, pad]))
    out = {POSITION_FIELD: torch.stack(rows)}
    for field in PIXEL_FIELDS:
        parts = [x[field] for x in items if field in x]
        if parts:
            out[field] = torch.cat(parts)
    return out


def forward_kwargs(batch: dict, kwargs: dict, flatten: bool = False) -> dict:
    """`kwargs` (a forward's keyword arguments) with the batch's images and positions.

    Without `position_ids` in kwargs (padded rows): the three M-RoPE axes, [3, B, S]. With them
    (packed rows, restarting at 0 per document): [4, B, S], the text positions first; transformers
    builds the block-diagonal mask from that row and rotates with the other three. `flatten`: the
    batch was flattened into one row (palingenesis.packing)."""
    positions = batch.get(POSITION_FIELD)
    if positions is None:
        return kwargs
    out = dict(kwargs)
    mrope = positions.permute(2, 0, 1)  # [3, B, S]
    if flatten:
        mrope = mrope.reshape(3, 1, -1)
    text = out.get("position_ids")
    out["position_ids"] = mrope if text is None else torch.cat([text[None].to(mrope.dtype), mrope])
    for field in PIXEL_FIELDS:
        if batch.get(field) is not None:
            out[field] = batch[field]
    return out


def mask_image_labels(labels: torch.Tensor, input_ids: torch.Tensor, vision: VisionProcessor) -> torch.Tensor:
    """Image placeholders are inputs, never targets, whatever the masking mode trains."""
    labels[input_ids == vision.image_token_id] = IGNORE_INDEX
    return labels


# ------------------------------------------------------------------ model


def load_model(name_or_path: str, **kwargs):
    """The checkpoint in its multimodal architecture (language model, vision tower, projector)."""
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(name_or_path, **kwargs)


def disable_cache(model: torch.nn.Module) -> None:
    """use_cache off in the model's config and its language model's (a multimodal checkpoint's text
    config has its own): a training forward must not build a K/V cache."""
    config = getattr(model, "config", None)
    for c in (config, getattr(config, "text_config", None)):
        if c is not None and hasattr(c, "use_cache"):
            c.use_cache = False


def vision_parameters(model: torch.nn.Module) -> tuple[dict[str, torch.nn.Parameter], dict[str, torch.nn.Parameter]]:
    """(vision tower, projector) parameters by name."""
    tower, projector = {}, {}
    for name, param in model.named_parameters():
        if _PROJECTOR.search(name):
            projector[name] = param
        elif _TOWER.search(name):
            tower[name] = param
    return tower, projector


def vision_tower(model: torch.nn.Module) -> torch.nn.Module | None:
    """The vision encoder module (with its projector), or None for a text-only model."""
    for owner in (getattr(model, "model", None), model):
        for name in ("visual", "vision_tower", "vision_model"):
            module = getattr(owner, name, None) if owner is not None else None
            if isinstance(module, torch.nn.Module):
                return module
    return None


def vision_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The vision tower's repeated blocks (what FSDP shards one by one)."""
    tower = vision_tower(model)
    if tower is None:
        return []
    largest: list[torch.nn.Module] = []
    for module in tower.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > len(largest):
            largest = list(module)
    return largest


def set_vision_trainable(model: torch.nn.Module, tower: bool, projector: bool) -> dict[str, int]:
    """Freeze or train the vision tower and the projector separately. It only freezes what is
    asked to be frozen and never unfreezes a parameter something else froze (a plugin's base
    weights). Returns the parameter counts."""
    tower_params, projector_params = vision_parameters(model)
    if not tower_params and not projector_params:
        raise MultimodalError("the model has no vision tower: load a multimodal checkpoint")
    for params, train in ((tower_params, tower), (projector_params, projector)):
        if not train:
            for p in params.values():
                p.requires_grad_(False)
    counts = {
        "vision_tower": sum(p.numel() for p in tower_params.values()),
        "projector": sum(p.numel() for p in projector_params.values()),
    }
    logger.info(
        "vision tower: %.1fM parameters (%s); projector: %.1fM (%s)",
        counts["vision_tower"] / 1e6,
        "trained" if tower else "frozen",
        counts["projector"] / 1e6,
        "trained" if projector else "frozen",
    )
    return counts


def check_model(model: torch.nn.Module, vision: VisionProcessor, label: str = "the model") -> None:
    """Raise when the model does not read images the way `vision` writes them."""
    config = model.config
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None or vision_tower(model) is None:
        raise MultimodalError(f"{label} has no vision tower: it cannot take the images in the data")
    found = {
        "image_token_id": getattr(config, "image_token_id", None),
        "patch_size": getattr(vision_config, "patch_size", None),
        "merge_size": getattr(vision_config, "spatial_merge_size", None),
    }
    expected = {k: vision.fingerprint[k] for k in found}
    if found != expected:
        raise MultimodalError(f"{label} reads images as {found}, the image processor writes {expected}")
