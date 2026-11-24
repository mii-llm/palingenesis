"""Random initialisation: a model built from a config with fresh weights (model.init: random).

`pgs train` normally loads the weights at model.name_or_path. With `model.init: random` only the
architecture is read from there (config.json) and the weights are drawn by the architecture's own
initialiser (transformers' `_init_weights`), from `model.init_seed`. Every rank draws the same
weights: the seed does not depend on the rank, and the generators' states (CPU and CUDA) are restored
afterwards so the rest of the run (dropout, data order) is seeded as it would be otherwise.
"""

import logging

import torch

from palingenesis.config import ConfigError

logger = logging.getLogger(__name__)


def init_seed(config) -> int:
    """The seed of the random initialisation: model.init_seed, or train.seed when that is -1."""
    return config.train.seed if config.model.init_seed < 0 else config.model.init_seed


def _build(config, dtype: torch.dtype, kwargs: dict) -> torch.nn.Module:
    """The model of `config`, built under `dtype` as from_pretrained builds one: parameters are created
    in it, buffers the architecture computes in float32 (rotary frequencies) stay float32, and a rank
    never holds more than one copy of the model in `dtype`. An architecture whose initialiser cannot
    draw in a low-precision dtype is built in float32 and its parameters (only) are cast."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_config(config, dtype=dtype, **kwargs)
    except (RuntimeError, NotImplementedError) as e:
        if dtype == torch.float32:
            raise
        logger.warning(
            f"Random initialisation in {dtype} failed ({type(e).__name__}: {e}); initialising in float32 and "
            "casting the parameters (this needs the float32 model in host memory on every rank)."
        )
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32, **kwargs)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype)
    model.config.dtype = dtype
    return model


def random_init(
    name_or_path: str,
    seed: int,
    dtype: torch.dtype = torch.float32,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
    tokenizer=None,
) -> torch.nn.Module:
    """A causal LM with the architecture of `name_or_path`'s config and freshly initialised weights.

    The weights are drawn from `seed` directly in `dtype` (see _build), on the CPU, in every process
    that calls this: host memory per rank is one copy of the model in `dtype`, as when a checkpoint is
    loaded. The random generators of the CPU and of every CUDA device are forked around the
    initialisation, so the caller's seeding (per rank) is as it was. A multimodal checkpoint's config
    gives its language model alone. `tokenizer` (optional): refused when it has more tokens than the
    config's vocab_size, which would index past the embedding."""
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    vocab = getattr(
        model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config, "vocab_size", None
    )
    if tokenizer is not None and vocab is not None and len(tokenizer) > vocab:
        raise ConfigError(
            f"model.init: random: the tokenizer at {name_or_path} has {len(tokenizer)} tokens, but its config.json "
            f"gives vocab_size {vocab}. Set vocab_size to at least the tokenizer's size."
        )
    kwargs = {"trust_remote_code": trust_remote_code}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    # torch.manual_seed seeds the CUDA generators too: fork them all, or every rank would leave with the
    # same CUDA state whatever seed + rank it had set
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        try:
            model = _build(model_config, dtype, kwargs)
        except ValueError:
            # a multimodal checkpoint's config: its language model is the causal LM
            text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
            if text_config is model_config:
                raise
            torch.manual_seed(seed)
            model = _build(text_config, dtype, kwargs)
            # nothing to follow at export: the source holds another architecture (checkpoint._source_checkpoint)
            model.config._name_or_path = ""
    params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Random initialisation: {type(model).__name__} from {name_or_path}'s config, {params / 1e6:.1f}M "
        f"parameters in {str(dtype).removeprefix('torch.')}, seed {seed} (no pretrained weights are loaded)"
    )
    return model
