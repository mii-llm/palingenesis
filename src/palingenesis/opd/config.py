"""OPD configuration — flat, typed, YAML + CLI, same shape as the SFT Config.

Loading and overrides mirror palingenesis.config.Config exactly:

    config = OPDConfig.from_yaml("configs/distill_opd.yaml")
    pgs distill --config configs/distill_opd.yaml --train.learning_rate 5e-6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class OPDModelConfig:
    student: str = "mii-llm/nesso-0.4B-agentic"
    teacher: str = "Coloss/nesso-3B"
    # Teacher placement; empty = same device as the student. On a multi-GPU
    # node "cuda:1" removes the largest memory consumer from the student's GPU.
    teacher_device: str = ""
    gradient_checkpointing: bool = False


@dataclass(slots=True)
class OPDBridgeConfig:
    # student token -> teacher token merged for end-of-turn supervision, e.g.
    # {"<|im_end|>": "<|eot_id|>"}. Empty = auto (student eos -> teacher eos,
    # only when the student's eos lies outside the shared vocab). Set it
    # explicitly when the teacher is a base model whose configured eos is not
    # its conversational terminator.
    eos_map: dict = field(default_factory=dict)
    # Additional student tokens that terminate a sampled completion.
    extra_stop_tokens: list = field(default_factory=list)
    # Extra texts checked for identical tokenization (appended to the built-in probes).
    probe_texts: list = field(default_factory=list)


@dataclass(slots=True)
class OPDDataConfig:
    # Prompt source kind (see palingenesis.opd.sources):
    #   "mcqa"     — multiple-choice pool (pool-row JSONL), letter-accuracy dev metric
    #   "messages" — generic chat prompts ({"messages": [...]} JSONL, last turn = user),
    #                held-out reverse-KL dev metric
    format: str = "mcqa"
    # Prompt file (pool-row JSONL for "mcqa", messages JSONL for "messages").
    prompts_path: str = "data/prompts.jsonl"
    # Held-out dev rows split off the pool (deterministic, hash-ranked, unique).
    dev_size: int = 500
    # System message for rendered prompts (mcqa; empty = the template default).
    system_message: str = ""
    # ---- mcqa-only fields ----
    # Prompt templates (empty = the library's neutral English defaults). To
    # train against a specific benchmark, put its VERBATIM templates here —
    # exact prompt bytes are policy, and they belong in the config (see
    # configs/distill_opd.yaml for ITALIC's). Placeholders: {question} and
    # {options} required; {topic} and {merged_letters} optional.
    fast_template: str = ""
    cot_template: str = ""
    # The benchmark's official few-shot file (empty = pool/zero-shot regimes only).
    shots_path: str = ""
    # Shot-regime mixture per sampled prompt: reference shots / k pool shots /
    # zero-shot with the remaining probability mass.
    p_reference_shots: float = 0.5
    p_pool_shots: float = 0.25
    pool_shots_max_k: int = 5


@dataclass(slots=True)
class OPDSamplingConfig:
    batch_prompts: int = 32       # prompts per optimizer step
    group_size: int = 1           # rollouts per prompt (>1 only useful for CoT)
    temperature: float = 1.0
    max_new_tokens: int = 16      # fast mode: the answer is a letter
    cot_fraction: float = 0.0     # fraction of prompts using the CoT template
    cot_max_new_tokens: int = 300
    gen_micro_seqs: int = 64      # sequences per generate() call


@dataclass(slots=True)
class OPDTrainConfig:
    output_dir: str = "./runs/opd"
    steps: int = 2000
    learning_rate: float = 1e-5
    warmup_steps: int = 50
    lr_scheduler: str = "cosine"  # "cosine" or "constant"
    max_grad_norm: float = 1.0
    loss_fn: str = "full_kl"      # "full_kl" or "sampled_rkl"
    seed: int = 0
    score_micro_seqs: int = 32    # sequences per scoring forward (student + teacher)
    eval_every: int = 200         # dev accuracy every N steps (0 = off)
    eval_dev_samples: int = 200
    save_steps: int = 500         # checkpoint every N steps (0 = final only)
    keep_checkpoints: int = 3     # newest step_* dirs kept on disk (0 = keep all)


@dataclass(slots=True)
class OPDLoggingConfig:
    log_every: int = 10
    use_wandb: bool = False
    project: str = "palingenesis-opd"
    run_name: str = ""


@dataclass(slots=True)
class OPDConfig:
    model: OPDModelConfig = field(default_factory=OPDModelConfig)
    bridge: OPDBridgeConfig = field(default_factory=OPDBridgeConfig)
    data: OPDDataConfig = field(default_factory=OPDDataConfig)
    sampling: OPDSamplingConfig = field(default_factory=OPDSamplingConfig)
    train: OPDTrainConfig = field(default_factory=OPDTrainConfig)
    logging: OPDLoggingConfig = field(default_factory=OPDLoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OPDConfig":
        with Path(path).open() as f:
            raw = yaml.safe_load(f) or {}
        config = cls()
        for section_name, section_data in raw.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for k, v in section_data.items():
                    if hasattr(section, k):
                        current = getattr(section, k)
                        if isinstance(current, bool) and isinstance(v, str):
                            v = v.lower() in ("true", "1", "yes")
                        elif isinstance(current, float) and isinstance(v, str):
                            v = float(v)
                        elif isinstance(current, int) and isinstance(v, str):
                            v = int(v)
                        setattr(section, k, v)
        return config

    @classmethod
    def from_cli(cls, args: list[str] | None = None) -> "OPDConfig":
        """Parse --config file.yaml and --section.field value overrides."""
        import sys

        args = args or sys.argv[1:]
        config = cls()

        # First pass: load YAML if specified
        i = 0
        while i < len(args):
            if args[i] == "--config" and i + 1 < len(args):
                config = cls.from_yaml(args[i + 1])
                i += 2
            else:
                i += 1

        # Second pass: apply overrides
        i = 0
        while i < len(args):
            if args[i] == "--config":
                i += 2
                continue
            if args[i].startswith("--") and i + 1 < len(args):
                key, value = args[i][2:], args[i + 1]
                parts = key.split(".")
                if len(parts) == 2:
                    section_name, field_name = parts
                    if hasattr(config, section_name):
                        section = getattr(config, section_name)
                        if hasattr(section, field_name):
                            current = getattr(section, field_name)
                            if isinstance(current, bool):
                                setattr(section, field_name, value.lower() in ("true", "1", "yes"))
                            elif isinstance(current, int):
                                setattr(section, field_name, int(value))
                            elif isinstance(current, float):
                                setattr(section, field_name, float(value))
                            else:
                                setattr(section, field_name, value)
                i += 2
            else:
                i += 1
        return config

    def validate(self) -> list[str]:
        """Validate config compatibility. Raises OPDConfigError on hard errors,
        returns a list of warnings for legal-but-suspicious combinations."""
        errors: list[str] = []
        warnings: list[str] = []

        if self.data.format not in ("mcqa", "messages"):
            errors.append(f"data.format must be 'mcqa' or 'messages', got {self.data.format!r}")
        for name in ("fast_template", "cot_template"):
            template = getattr(self.data, name)
            if template:
                errors.extend(_check_template(f"data.{name}", template))
        if self.train.loss_fn not in ("full_kl", "sampled_rkl"):
            errors.append(f"train.loss_fn must be 'full_kl' or 'sampled_rkl', got {self.train.loss_fn!r}")
        if self.train.lr_scheduler not in ("cosine", "constant"):
            errors.append(f"train.lr_scheduler must be 'cosine' or 'constant', got {self.train.lr_scheduler!r}")
        if not 0.0 <= self.sampling.cot_fraction <= 1.0:
            errors.append(f"sampling.cot_fraction must be in [0, 1], got {self.sampling.cot_fraction}")
        if self.data.p_reference_shots + self.data.p_pool_shots > 1.0:
            errors.append(
                "data.p_reference_shots + data.p_pool_shots must be <= 1.0 "
                f"(got {self.data.p_reference_shots} + {self.data.p_pool_shots}); "
                "the remainder is the zero-shot probability."
            )

        if self.data.format == "mcqa" and self.sampling.group_size > 1 and self.sampling.cot_fraction == 0.0:
            # mcqa-only: for messages-format long completions, groups add useful diversity
            warnings.append(
                "sampling.group_size > 1 with cot_fraction=0: fast-mode completions are "
                "~1 token, so extra rollouts per prompt add cost without signal."
            )
        if self.data.format == "mcqa" and self.data.p_reference_shots > 0 and not self.data.shots_path:
            warnings.append(
                "data.p_reference_shots > 0 but data.shots_path is empty — the reference-shot "
                "regime will silently fall back to zero-shot."
            )
        if self.data.format == "messages" and self.sampling.cot_fraction > 0:
            warnings.append(
                "sampling.cot_fraction is an mcqa-only knob (fast/CoT template mix) and is "
                "ignored by the messages source."
            )

        if errors:
            raise OPDConfigError(
                "OPD configuration has incompatible settings:\n" + "\n".join(f"  ✗ {e}" for e in errors)
            )
        return warnings


class OPDConfigError(Exception):
    """Raised when the OPD config has hard incompatibilities that prevent safe training."""


_TEMPLATE_FIELDS_REQUIRED = {"question", "options"}
_TEMPLATE_FIELDS_ALLOWED = _TEMPLATE_FIELDS_REQUIRED | {"topic", "merged_letters"}


def _check_template(name: str, template: str) -> list[str]:
    """Validate a prompt template's placeholders (str.format would KeyError at train time)."""
    import string as _string

    try:
        fields = {f for _, f, _, _ in _string.Formatter().parse(template) if f}
    except ValueError as e:
        return [f"{name} is not a valid format string: {e}"]
    errors = []
    if unknown := fields - _TEMPLATE_FIELDS_ALLOWED:
        errors.append(f"{name} has unknown placeholders {sorted(unknown)}; "
                      f"allowed: {sorted(_TEMPLATE_FIELDS_ALLOWED)}. Escape literal braces as '{{{{'.")
    if missing := _TEMPLATE_FIELDS_REQUIRED - fields:
        errors.append(f"{name} is missing required placeholders {sorted(missing)}.")
    return errors
