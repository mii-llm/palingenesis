"""Forward arguments that keep packed documents apart.

A packed row holds several documents; `position_ids` restart at 0 at each one. What
keeps a document from attending to the previous one depends on the layer:

  * attention (sdpa, eager, flex): transformers builds a block-diagonal causal mask
    from `position_ids`, but only when `attention_mask` is None. With a 2D mask of
    ones it builds a plain causal mask and every document sees the ones before it.
  * flash_attention_2: variable-length kernels, from `cu_seq_lens_q/k` and
    `max_length_q/k`, for one flattened row.
  * linear attention (Qwen3.5 / Qwen3-Next Gated DeltaNet): the recurrent state is
    reset at `cu_seq_lens_q` by flash-linear-attention's kernel and the short causal
    convolution at `seq_idx` by causal-conv1d. Both need one flattened row; the
    pure-torch fallbacks ignore document boundaries.

`PackedBatch` flattens the [B, S] packed batch into [1, B*S] when a kernel needs it:
the rows already end at document boundaries, so flattening changes nothing else.
"""

from dataclasses import dataclass

import torch

from palingenesis.config import ConfigError

# Layer types whose documents are separated by the attention mask.
_ATTENTION_LAYER_TYPES = {"full_attention", "sliding_attention", "chunked_attention", "attention"}


def _layer_types(model: torch.nn.Module) -> list[str]:
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return list(getattr(config, "layer_types", None) or [])


def has_linear_attention(model: torch.nn.Module) -> bool:
    return "linear_attention" in _layer_types(model)


def missing_linear_attention_kernels() -> list[str]:
    """Packages transformers needs for its fast linear-attention path, not installed."""
    from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

    return [
        name
        for name, ok in (
            ("flash-linear-attention", is_flash_linear_attention_available()),
            ("causal-conv1d", is_causal_conv1d_available()),
        )
        if not ok
    ]


def check_packing_support(model: torch.nn.Module, attn_implementation: str) -> None:
    """Raise ConfigError when packed documents cannot be kept apart in this model."""
    types = set(_layer_types(model))
    other = types - _ATTENTION_LAYER_TYPES - {"linear_attention"}
    if other:
        raise ConfigError(
            f"data.packing: this model has {sorted(other)} layers, whose state would carry from one packed "
            "document into the next. Set data.packing: false."
        )
    if "linear_attention" in types:
        missing = missing_linear_attention_kernels()
        if missing:
            raise ConfigError(
                "data.packing with linear-attention layers (Qwen3.5, Qwen3-Next) needs "
                f"{' and '.join(missing)}: without them transformers falls back to torch implementations that "
                "carry the recurrent state across packed documents. Install them (`uv sync --extra train "
                "--extra hybrid`, see the installation guide) or set data.packing: false."
            )
    if attn_implementation not in ("sdpa", "eager", "flash_attention_2", "flex_attention"):
        raise ConfigError(
            f"data.packing is not supported with attn_implementation={attn_implementation!r}: use sdpa, "
            "flash_attention_2 or flex_attention."
        )


@dataclass
class PackedBatch:
    """Inputs of one packed micro-batch, ready for the model's forward."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    forward_kwargs: dict
    loss_weights: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        flatten: bool,
        attn_implementation: str,
        loss_weights: torch.Tensor | None = None,
    ) -> "PackedBatch":
        """`labels` (and `loss_weights`) must already be shifted (per row); `flatten`
        joins the rows."""
        kwargs: dict = {"attention_mask": None}
        if flatten:
            input_ids, labels, position_ids = (t.reshape(1, -1) for t in (input_ids, labels, position_ids))
            if loss_weights is not None:
                loss_weights = loss_weights.reshape(1, -1)
            starts = position_ids[0] == 0
            starts[0] = True
            bounds = torch.nonzero(starts).flatten()
            cu_seqlens = torch.cat([bounds, bounds.new_tensor([position_ids.shape[1]])]).to(torch.int32)
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["seq_idx"] = (torch.cumsum(starts.to(torch.int32), 0) - 1).to(torch.int32)[None]
            if attn_implementation == "flash_attention_2":
                longest = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
                kwargs["max_length_q"] = kwargs["max_length_k"] = longest
        kwargs["position_ids"] = position_ids
        return cls(input_ids=input_ids, labels=labels, forward_kwargs=kwargs, loss_weights=loss_weights)
