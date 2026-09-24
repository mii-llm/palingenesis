"""OPD configuration — typed sections, YAML + CLI, strict like the SFT Config.

    config = OPDConfig.from_yaml("configs/distill_math.yaml")
    pgs distill --config configs/distill_math.yaml --train.learning_rate 5e-6 \\
        --teachers.qwen.backend vllm --sources.gsm8k.max_new_tokens 384

Teachers and prompt sources are named mappings (``teachers: {name: {...}}``,
``sources: {name: {...}}``); each source routes its prompts to one teacher.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LOSSES = ("full_rkl", "topk_kl", "sampled_rkl", "xtok")
ROLLOUT_BACKENDS = ("hf", "vllm", "vllm_server")
TEACHER_BACKENDS = ("hf", "vllm")


@dataclass(slots=True)
class OPDModelConfig:
    # No default on purpose: the student/teacher pair is the experiment's
    # central decision. See configs/distill_*.yaml for worked examples.
    student: str = ""
    gradient_checkpointing: bool = False
    # Liger's fused kernels (RMSNorm, SwiGLU, RoPE, ...) in the student and hf teachers, on CUDA
    use_liger_kernel: bool = True
    # Student tokens that end a completion besides its eos/generation-config eos,
    # e.g. ["<|end_of_text|>"].
    stop_tokens: list = field(default_factory=list)
    # Extra apply_chat_template arguments for every model (student and teachers),
    # e.g. {enable_thinking: false} for Qwen3's non-thinking mode.
    chat_template_kwargs: dict = field(default_factory=dict)


@dataclass(slots=True)
class TeacherConfig:
    model: str = ""
    # Tokenizer to render and score with (empty = the model's).
    tokenizer: str = ""
    # "hf": in-process, frozen bf16, full-vocabulary logits (every loss).
    # "vllm": a vLLM server scoring prefill-only top-k log-probs (topk_kl,
    # sampled_rkl, xtok) — launched by the trainer, or reached at `url`.
    backend: str = "hf"
    # Loss for this teacher's samples; empty = full_rkl (hf) / topk_kl (vllm) for a
    # teacher sharing the student's vocabulary, xtok for a different tokenizer.
    loss: str = ""
    # hf: device (empty = the student's), and parking on CPU between scoring
    # calls (for several teachers that do not fit on the GPU together).
    device: str = ""
    offload: bool = False
    # vllm: an already running server (empty = launch one on the student's GPU).
    url: str = ""
    gpu_memory_utilization: float = 0.15
    # Shared vocabulary: student end-of-turn token -> teacher's, e.g.
    # {"<|im_end|>": "<|eot_id|>"} (empty = auto, see token_bridge).
    eos_map: dict = field(default_factory=dict)
    # Extra texts that must tokenize identically for a shared vocabulary.
    probe_texts: list = field(default_factory=list)


@dataclass(slots=True)
class SourceConfig:
    # "messages": chat JSONL of {"messages": [...]} ending with a user turn, optional
    #             "answer" (then greedy dev accuracy is reported too)
    # "mcqa":     multiple-choice pool (pool-row JSONL, see palingenesis.opd.pool)
    format: str = "messages"
    path: str = ""
    weight: float = 1.0
    # Teacher name (empty = the first teacher).
    teacher: str = ""
    max_new_tokens: int = 512
    # Held-out rows: split off `path` (deterministic, hash-ranked, unique), or all
    # of `dev_path` when set (e.g. a benchmark's test split, same format as path).
    dev_size: int = 200
    dev_path: str = ""
    # System message for rendered mcqa prompts (empty = the template default).
    system_message: str = ""
    # ---- mcqa-only ----
    # Prompt templates (empty = the library's neutral English defaults). To train
    # against a specific benchmark put its VERBATIM templates here. Placeholders:
    # {question} and {options} required; {topic} and {merged_letters} optional.
    fast_template: str = ""
    cot_template: str = ""
    # The benchmark's official few-shot file (empty = pool/zero-shot regimes only).
    shots_path: str = ""
    # Shot-regime mixture per prompt: reference shots / k pool shots / zero-shot.
    p_reference_shots: float = 0.5
    p_pool_shots: float = 0.25
    pool_shots_max_k: int = 5
    # Fraction of prompts rendered with the CoT template (max_new_tokens applies to
    # the fast template, cot_max_new_tokens to CoT).
    cot_fraction: float = 0.0
    cot_max_new_tokens: int = 300


@dataclass(slots=True)
class OPDRolloutConfig:
    # "hf": the trainer's own model.generate (always works, slow).
    # "vllm": an in-process vLLM engine on the student's GPU (sleeps while training).
    # "vllm_server": a separate vLLM server fed weights over CUDA IPC (experimental).
    backend: str = "hf"
    batch_prompts: int = 32           # prompts per optimizer step
    group_size: int = 1               # rollouts per prompt
    temperature: float = 1.0
    # Policy versions a batch may lag behind the weights it trains (0 = on-policy).
    # > 0 overlaps rollout with training (vllm backends only).
    max_staleness: int = 0
    micro_seqs: int = 64              # hf: sequences per generate() call
    gpu_memory_utilization: float = 0.3   # vllm: GPU fraction for weights + KV cache
    max_model_len: int = 4096         # vllm: prompt + completion tokens
    enforce_eager: bool = False       # vllm: no CUDA graphs (faster start, slower decode)
    # vllm, max_staleness 0: release the engine's memory while the trainer trains. Off keeps
    # it resident (no wake-up per step) when its gpu_memory_utilization fits beside training.
    sleep: bool = True
    url: str = ""                     # vllm_server: a running server (empty = launch one)


@dataclass(slots=True)
class OPDLossConfig:
    top_k: int = 8                    # teacher top-k for topk_kl and xtok's dense term
    beta: float = 1.0                 # topk_kl / dense: weight of reverse KL (1 - beta: forward)
    is_low: float = 0.5               # sampled_rkl / xtok: importance ratios outside
    is_high: float = 2.0              # [is_low, is_high] are zeroed (ICE-POP)
    length_norm: bool = False         # per-sequence mean instead of per-token mean
    xtok_spread: str = "chunk"        # "chunk" or "proportional" (see losses.xtok)
    xtok_dense_weight: float = 0.0    # xtok: top-k KL at one-to-one chunks
    mask_whitespace: bool = True      # xtok: no loss on whitespace-only chunks


@dataclass(slots=True)
class OPDTrainConfig:
    output_dir: str = "./runs/opd"
    steps: int = 1000
    learning_rate: float = 1e-6
    warmup_steps: int = 20
    lr_scheduler: str = "cosine"      # "cosine" or "constant"
    max_grad_norm: float = 1.0
    seed: int = 0
    score_micro_seqs: int = 16        # sequences per scoring forward (student and teacher)
    eval_every: int = 50              # dev metrics every N steps, and before the first (0 = off)
    eval_samples: int = 200           # dev prompts per source
    save_steps: int = 0               # checkpoint every N steps (0 = final only)
    keep_checkpoints: int = 3         # newest step_* dirs kept on disk (0 = keep all)
    resume_from: str = ""             # a step_* checkpoint dir, or "auto": the newest in output_dir


@dataclass(slots=True)
class OPDLoggingConfig:
    log_every: int = 1
    use_wandb: bool = False
    project: str = "palingenesis-opd"
    run_name: str = ""


# Sections and options of the first OPD config format, with where they went.
_MOVED = {
    "bridge": "moved: eos_map and probe_texts to teachers.<name>, extra_stop_tokens to model.stop_tokens.",
    "data": "replaced by `sources: {<name>: {format, path, ...}}` (prompts_path is now path).",
    "sampling": "replaced by `rollout:` (batch_prompts, group_size, temperature); max_new_tokens, "
                "cot_fraction and cot_max_new_tokens are per source.",
    "model.teacher": "moved to `teachers: {<name>: {model: ...}}`.",
    "model.teacher_device": "moved to teachers.<name>.device.",
    "train.loss_fn": "moved to teachers.<name>.loss (full_kl is now full_rkl, sampled_rkl is unchanged).",
    "train.eval_dev_samples": "renamed train.eval_samples.",
}


@dataclass(slots=True)
class OPDConfig:
    model: OPDModelConfig = field(default_factory=OPDModelConfig)
    teachers: dict = field(default_factory=dict)      # name -> TeacherConfig
    sources: dict = field(default_factory=dict)       # name -> SourceConfig
    rollout: OPDRolloutConfig = field(default_factory=OPDRolloutConfig)
    loss: OPDLossConfig = field(default_factory=OPDLossConfig)
    train: OPDTrainConfig = field(default_factory=OPDTrainConfig)
    logging: OPDLoggingConfig = field(default_factory=OPDLoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OPDConfig":
        """Load a YAML config. Unknown sections and options are errors (with a
        did-you-mean hint), as for training configs."""
        from palingenesis.config import ConfigError

        with Path(path).open() as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a mapping of sections (model:, teachers:, sources:, ...).")
        config = cls()
        for section, options in raw.items():
            if options is None:
                continue
            if not isinstance(options, dict):
                raise ConfigError(f"{path}: `{section}:` must be a mapping.")
            if section in ("teachers", "sources"):
                for name, entry in options.items():
                    if not isinstance(entry, dict):
                        raise ConfigError(f"{path}: `{section}.{name}:` must be a mapping of options.")
                    for key, value in entry.items():
                        config.set(f"{section}.{name}.{key}", value, where=str(path))
            else:
                for key, value in options.items():
                    config.set(f"{section}.{key}", value, where=str(path))
        return config

    @classmethod
    def from_cli(cls, args: list[str] | None = None) -> "OPDConfig":
        """Parse --config file.yaml and --section.option value overrides
        (--teachers.<name>.option / --sources.<name>.option for the mappings)."""
        import sys

        from palingenesis.config import ConfigError

        args = sys.argv[1:] if args is None else args
        config = cls()
        if "--config" in args:
            i = args.index("--config")
            if i + 1 >= len(args):
                raise ConfigError("--config needs a file.")
            config = cls.from_yaml(args[i + 1])
        i = 0
        while i < len(args):
            if args[i] == "--config":
                i += 2
            elif args[i].startswith("--") and i + 1 < len(args):
                config.set(args[i][2:], args[i + 1], where="command line")
                i += 2
            else:
                raise ConfigError(f"command line: unexpected argument {args[i]!r} (overrides are --section.option value).")
        return config

    def set(self, key: str, value, where: str) -> None:
        """Set one option by dotted key: section.option, or teachers/sources.<name>.option."""
        from palingenesis.config import ConfigError, _section, _set_option

        parts = key.split(".")
        if parts[0] in _MOVED or ".".join(parts[:2]) in _MOVED:
            moved = parts[0] if parts[0] in _MOVED else ".".join(parts[:2])
            raise ConfigError(f"{where}: `{moved}` {_MOVED[moved]}")
        if parts[0] in ("teachers", "sources"):
            if len(parts) != 3:
                raise ConfigError(f"{where}: {key}: use {parts[0]}.<name>.<option> (names cannot contain dots).")
            entries = getattr(self, parts[0])
            entry = entries.setdefault(parts[1], TeacherConfig() if parts[0] == "teachers" else SourceConfig())
            section, section_name, option = entry, f"{parts[0]}.{parts[1]}", parts[2]
        else:
            if len(parts) != 2:
                raise ConfigError(f"{where}: {key}: use <section>.<option>.")
            section, section_name, option = _section(self, parts[0], where=where), parts[0], parts[1]
        field_names = {f.name for f in dataclasses.fields(section)}
        if option in field_names and isinstance(getattr(section, option), (dict, list)) and isinstance(value, str):
            value = yaml.safe_load(value)       # mappings/lists given on the command line
        _set_option(section, section_name, option, value, where=where)

    def teacher_of(self, source: str) -> str:
        return self.sources[source].teacher or next(iter(self.teachers))

    def validate(self) -> list[str]:
        """Raise OPDConfigError on impossible settings; return warnings for suspicious ones.

        Checks that need the tokenizers (a loss that requires a shared vocabulary
        with a teacher that does not have one) run when the trainer loads them.
        """
        errors: list[str] = []
        warnings: list[str] = []
        rollout, loss = self.rollout, self.loss

        if not self.model.student:
            errors.append("model.student is required (see configs/distill_*.yaml for worked examples).")
        if not self.teachers:
            errors.append("at least one teacher is required: teachers: {<name>: {model: ...}}.")
        if not self.sources:
            errors.append("at least one prompt source is required: sources: {<name>: {path: ...}}.")
        for name, teacher in self.teachers.items():
            where = f"teachers.{name}"
            if not teacher.model:
                errors.append(f"{where}.model is required.")
            if teacher.backend not in TEACHER_BACKENDS:
                errors.append(f"{where}.backend must be one of {TEACHER_BACKENDS}, got {teacher.backend!r}.")
            if teacher.loss and teacher.loss not in LOSSES:
                errors.append(f"{where}.loss must be one of {LOSSES} (or empty for auto), got {teacher.loss!r}.")
            if teacher.loss == "full_rkl" and teacher.backend == "vllm":
                errors.append(f"{where}: full_rkl needs the teacher's full distribution; a vllm teacher only "
                              "returns its top-k. Use loss topk_kl or sampled_rkl, or backend hf.")
            if teacher.backend == "vllm" and (teacher.device or teacher.offload):
                errors.append(f"{where}: device and offload apply to hf teachers only.")
            if teacher.backend == "hf" and teacher.url:
                errors.append(f"{where}.url applies to vllm teachers only.")
        for name, source in self.sources.items():
            where = f"sources.{name}"
            if source.format not in ("messages", "mcqa"):
                errors.append(f"{where}.format must be 'messages' or 'mcqa', got {source.format!r}.")
            if not source.path:
                errors.append(f"{where}.path is required.")
            if source.teacher and source.teacher not in self.teachers:
                errors.append(f"{where}.teacher {source.teacher!r} is not one of the teachers {list(self.teachers)}.")
            if source.weight < 0:
                errors.append(f"{where}.weight must be >= 0.")
            for option in ("fast_template", "cot_template"):
                if getattr(source, option):
                    errors.extend(_check_template(f"{where}.{option}", getattr(source, option)))
            if not 0.0 <= source.cot_fraction <= 1.0:
                errors.append(f"{where}.cot_fraction must be in [0, 1], got {source.cot_fraction}.")
            if source.p_reference_shots + source.p_pool_shots > 1.0:
                errors.append(f"{where}: p_reference_shots + p_pool_shots must be <= 1.0 (the remainder is zero-shot).")
            if source.format == "messages" and source.cot_fraction > 0:
                warnings.append(f"{where}.cot_fraction is an mcqa-only option and is ignored.")
            if source.format == "mcqa" and source.p_reference_shots > 0 and not source.shots_path:
                warnings.append(f"{where}: p_reference_shots > 0 without shots_path: that regime falls back to zero-shot.")
        if self.sources and sum(s.weight for s in self.sources.values()) <= 0:
            errors.append("the source weights must not all be 0.")

        if rollout.backend not in ROLLOUT_BACKENDS:
            errors.append(f"rollout.backend must be one of {ROLLOUT_BACKENDS}, got {rollout.backend!r}.")
        if rollout.max_staleness < 0:
            errors.append("rollout.max_staleness must be >= 0.")
        if rollout.max_staleness > 0 and rollout.backend == "hf":
            errors.append("rollout.max_staleness > 0 overlaps generation with training, which the hf backend "
                          "cannot do (it generates with the model being trained). Use a vllm backend.")
        if rollout.temperature <= 0:
            errors.append("rollout.temperature must be > 0 (distillation samples from the student).")
        if rollout.url and rollout.backend != "vllm_server":
            errors.append("rollout.url applies to the vllm_server backend only.")
        if rollout.temperature != 1.0:
            warnings.append(f"rollout.temperature={rollout.temperature}: the losses compare the teacher with the "
                            "student's temperature-scaled distribution, not the student itself.")
        if not 0.0 <= loss.beta <= 1.0:
            errors.append(f"loss.beta must be in [0, 1], got {loss.beta}.")
        if loss.xtok_spread not in ("chunk", "proportional"):
            errors.append(f"loss.xtok_spread must be 'chunk' or 'proportional', got {loss.xtok_spread!r}.")
        if not 0 < loss.is_low <= 1.0 <= loss.is_high:
            errors.append("loss.is_low must be in (0, 1] and loss.is_high >= 1.")
        if loss.top_k < 1:
            errors.append(f"loss.top_k must be >= 1, got {loss.top_k}.")
        if self.train.lr_scheduler not in ("cosine", "constant"):
            errors.append(f"train.lr_scheduler must be 'cosine' or 'constant', got {self.train.lr_scheduler!r}.")

        if errors:
            raise OPDConfigError("OPD configuration has incompatible settings:\n" + "\n".join(f"  ✗ {e}" for e in errors))
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
