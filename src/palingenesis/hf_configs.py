"""Keep transformers' own configuration classes after an in-process inference engine replaced them.

vLLM registers its own configuration classes with transformers' AutoConfig, with exist_ok=True, for model types
transformers already has (vllm/transformers_utils/config.py: qwen3_5, qwen3_5_moe, ...). It does this whenever it
loads such a model's config. From then on, every AutoConfig / AutoModelForCausalLM load in the process builds vLLM's
class. AutoModelForCausalLM hands a text-only class (Qwen3_5ForCausalLM) the wrapper's text_config only while the
wrapper's text sub-config class is the model class's own, so the next Hugging Face load of a Qwen3.5 checkpoint
fails ("'Qwen3_5Config' object has no attribute 'vocab_size'"). This hits a second trainer in a notebook, a
reference or teacher model loaded after the rollout engine, and evaluation in the same process.

vLLM reads the config only when an engine starts, and it registers its classes again then, so restoring
transformers' classes after each engine start is safe for vLLM.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def restore_native_configs() -> list[str]:
    """Drop AutoConfig registrations that shadow one of transformers' own model types with a class defined outside
    transformers. Returns the model types restored."""
    try:
        from transformers.models.auto import configuration_auto
    except ImportError:  # pragma: no cover (transformers is a dependency)
        return []
    mapping = configuration_auto.CONFIG_MAPPING
    extra = getattr(mapping, "_extra_content", None)
    if not isinstance(extra, dict):
        return []
    restored = []
    for model_type, cls in list(extra.items()):
        native = configuration_auto.CONFIG_MAPPING_NAMES.get(model_type)
        if native and not getattr(cls, "__module__", "").startswith("transformers."):
            del extra[model_type]
            restored.append(model_type)
    if restored:
        logger.debug("restored transformers' config classes for %s (shadowed by an inference engine)", restored)
    return restored
