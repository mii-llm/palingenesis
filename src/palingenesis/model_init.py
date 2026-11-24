"""Random initialisation: a model built from a config with fresh weights (model.init: random).

`pgs train` normally loads the weights at model.name_or_path. With `model.init: random` only the
architecture is read from there (config.json) and the weights are drawn by the architecture's own
initialiser (transformers' `_init_weights`), from `model.init_seed`. Every rank draws the same
weights: the seed does not depend on the rank, and the generator's state is restored afterwards so
the rest of the run (dropout, data order) is seeded as it would be otherwise.
"""

import logging

import torch

from palingenesis.config import ConfigError

logger = logging.getLogger(__name__)


def init_seed(config) -> int:
    """The seed of the random initialisation: model.init_seed, or train.seed when that is -1."""
    return config.train.seed if config.model.init_seed < 0 else config.model.init_seed


def random_init(
    name_or_path: str,
    seed: int,
    dtype: torch.dtype = torch.float32,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
    tokenizer=None,
) -> torch.nn.Module:
    """A causal LM with the architecture of `name_or_path`'s config and freshly initialised weights.

    The weights are drawn in float32 from `seed` and then cast to `dtype`. A multimodal checkpoint's
    config gives its language model alone. `tokenizer` (optional): refused when it has more tokens
    than the config's vocab_size, which would index past the embedding."""
    from transformers import AutoConfig, AutoModelForCausalLM

    model_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    kwargs = {"trust_remote_code": trust_remote_code}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        try:
            model = AutoModelForCausalLM.from_config(model_config, **kwargs)
        except ValueError:
            # a multimodal checkpoint's config: its language model is the causal LM
            text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
            if text_config is model_config:
                raise
            torch.manual_seed(seed)
            model = AutoModelForCausalLM.from_config(text_config, **kwargs)
            # nothing to follow at export: the source holds another architecture (checkpoint._source_checkpoint)
            model.config._name_or_path = ""
    finally:
        torch.random.set_rng_state(state)
    model = model.to(dtype)
    vocab = getattr(model.config, "vocab_size", None)
    if tokenizer is not None and vocab is not None and len(tokenizer) > vocab:
        raise ConfigError(
            f"model.init: random: the tokenizer at {name_or_path} has {len(tokenizer)} tokens, but its config.json "
            f"gives vocab_size {vocab}. Set vocab_size to at least the tokenizer's size."
        )
    params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Random initialisation: {type(model).__name__} from {name_or_path}'s config, {params / 1e6:.1f}M "
        f"parameters, seed {seed} (no pretrained weights are loaded)"
    )
    return model
