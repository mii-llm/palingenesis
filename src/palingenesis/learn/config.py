"""LearnConfig: typed sections, YAML + command-line overrides, strict like OPDConfig.

    config = LearnConfig.from_yaml("learn.yaml")
    pgs learn run stores/acme --scope org:acme --recipe.passes 2 --recipe.mode sft

Unknown sections and options are errors with a did-you-mean hint. A store keeps the
config it was created with in learn_state.yaml; run-time overrides are not saved.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LEVELS = ("org", "team", "user")
# vLLM's accepted --max-lora-rank values: a served (stacked) adapter is padded up to one of them.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
MLP_TARGET_REGEX = r"^(model\.language_model|model)\.layers\.\d+\.mlp\.(gate|up|down)_proj$"
DEFAULT_COMMAND_PATTERNS = ["remember", "ricorda", "ricordati", "recuerda", "souviens-toi", "merk dir"]


class LearnConfigError(Exception):
    """Raised when the learn config has incompatible settings."""


@dataclass(slots=True)
class LearnModelConfig:
    base: str = ""
    # Deployment system prompts, sampled per training row (student and teacher alike).
    system_prompts: list = field(default_factory=list)
    chat_template_kwargs: dict = field(default_factory=dict)
    # "auto" = cuda > mps > cpu.
    device: str = "auto"


@dataclass(slots=True)
class LearnScopesConfig:
    ranks: dict = field(default_factory=lambda: {"org": 32, "team": 16, "user": 16})


@dataclass(slots=True)
class LearnAdapterConfig:
    # Anchored on the decoder layers, so vision towers and MTP heads never match.
    target_regex: str = MLP_TARGET_REGEX
    alpha_over_rank: float = 2.0


@dataclass(slots=True)
class LearnNotesConfig:
    chunk_tokens: int = 400  # document chunk size (about 4 characters per token)
    teacher_neighbours: int = 4  # BM25 neighbours of a fact's note in the teacher context
    relevance_threshold: float = 0.0  # BM25 score a note needs to enter an anchor / near-miss teacher context
    contradiction_neighbours: int = 8  # active facts compared with each new fact


@dataclass(slots=True)
class LearnSignalsConfig:
    command_patterns: list = field(default_factory=lambda: list(DEFAULT_COMMAND_PATTERNS))
    max_note_chars: int = 1000


@dataclass(slots=True)
class LearnPIIConfig:
    # Personal identifiers (emails, phones, national ids): allow | redact | block (block = held for review).
    # Secrets (keys, passwords, cards, IBANs) are always rejected.
    user_scope_personal: str = "allow"
    shared_scope_personal: str = "redact"


@dataclass(slots=True)
class LearnGenerateConfig:
    # "self" = the base model with the note in context; "stub" = the deterministic rule-based generator
    # (tests, dry runs); anything else = a local model id or path.
    generator: str = "self"
    questions_per_fact: int = 4
    probes_per_fact: int = 2
    nearmiss_per_fact: int = 2
    max_new_tokens: int = 512
    probe_max_jaccard: float = 0.5  # token-set Jaccard a probe may share with any train question


@dataclass(slots=True)
class LearnRecipeConfig:
    mode: str = "distill"  # distill | sft
    divergence: str = "fkl"  # fkl | rkl | jsd
    jsd_beta: float = 0.5
    onpolicy_fraction: float = 0.5
    replay_teacher: str = "notes"  # notes | self
    anchor_teacher: str = "notes"  # notes | base | previous
    # Row shares per pass; replay rows come from replay.visits_per_fact instead of a share.
    mix: dict = field(default_factory=lambda: {"new": 0.45, "replay": 0.25, "nearmiss": 0.10, "anchor": 0.20})
    passes: int = 4
    rows_per_step: int = 16
    micro_rows: int = 8  # rows per forward pass (gradient accumulation inside a step)
    steps: int = 0  # 0 = auto from passes and rows
    max_steps_per_session: int = 4000
    learning_rate: float = 2.0e-4
    max_new_tokens: int = 256
    temperature: float = 1.0
    sampler: str = "hf"  # hf | vllm_colocate
    anchors_path: str = ""  # JSONL of {"messages": [...]} or {"prompt": ...}; empty = the built-in prompts
    max_anchor_rows: int = 256
    preflight: bool = True  # the teacher must answer the session's probes before training
    preflight_min: float = 0.8
    seed: int = 0


@dataclass(slots=True)
class LearnReplayConfig:
    visits_per_fact: int = 1
    max_prompts_per_fact: int = 4


@dataclass(slots=True)
class LearnConfig:
    model: LearnModelConfig = field(default_factory=LearnModelConfig)
    scopes: LearnScopesConfig = field(default_factory=LearnScopesConfig)
    adapter: LearnAdapterConfig = field(default_factory=LearnAdapterConfig)
    notes: LearnNotesConfig = field(default_factory=LearnNotesConfig)
    signals: LearnSignalsConfig = field(default_factory=LearnSignalsConfig)
    pii: LearnPIIConfig = field(default_factory=LearnPIIConfig)
    generate: LearnGenerateConfig = field(default_factory=LearnGenerateConfig)
    recipe: LearnRecipeConfig = field(default_factory=LearnRecipeConfig)
    replay: LearnReplayConfig = field(default_factory=LearnReplayConfig)

    @classmethod
    def from_dict(cls, raw: dict, where: str = "config") -> LearnConfig:
        if not isinstance(raw, dict):
            raise LearnConfigError(f"{where}: expected a mapping of sections (model:, recipe:, ...).")
        config = cls()
        for section, options in raw.items():
            if options is None:
                continue
            if not isinstance(options, dict):
                raise LearnConfigError(f"{where}: `{section}:` must be a mapping.")
            for key, value in options.items():
                config.set(f"{section}.{key}", value, where=where)
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> LearnConfig:
        with Path(path).open() as f:
            return cls.from_dict(yaml.safe_load(f) or {}, where=str(path))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def set(self, key: str, value, where: str = "command line") -> None:
        """Set one option by dotted key (section.option), coerced to the option's type."""
        from palingenesis.config import ConfigError, _section, _set_option

        parts = key.split(".")
        if len(parts) != 2:
            raise LearnConfigError(f"{where}: {key}: use <section>.<option>.")
        try:
            section = _section(self, parts[0], where=where)
            names = {f.name for f in dataclasses.fields(section)}
            if parts[1] in names and isinstance(getattr(section, parts[1]), (dict, list)) and isinstance(value, str):
                value = yaml.safe_load(value)
            _set_option(section, parts[0], parts[1], value, where=where)
        except ConfigError as e:
            raise LearnConfigError(str(e)) from None

    def apply_overrides(self, args: list[str]) -> None:
        """--section.option value pairs (the rest of a command line)."""
        i = 0
        while i < len(args):
            if args[i].startswith("--") and i + 1 < len(args):
                self.set(args[i][2:], args[i + 1])
                i += 2
            else:
                raise LearnConfigError(f"unexpected argument {args[i]!r} (overrides are --section.option value).")

    def validate(self) -> None:
        errors = []
        for level in LEVELS:
            rank = self.scopes.ranks.get(level)
            if not isinstance(rank, int) or rank < 1:
                errors.append(f"scopes.ranks.{level} must be a positive integer, got {rank!r}.")
        if extra := set(self.scopes.ranks) - set(LEVELS):
            errors.append(f"scopes.ranks: unknown levels {sorted(extra)} (levels: {LEVELS}).")
        if self.adapter.alpha_over_rank <= 0:
            errors.append("adapter.alpha_over_rank must be > 0.")
        for name in ("user_scope_personal", "shared_scope_personal"):
            if (value := getattr(self.pii, name)) not in ("allow", "redact", "block"):
                errors.append(f"pii.{name} must be allow, redact or block, got {value!r}.")
        recipe = self.recipe
        if recipe.mode not in ("distill", "sft"):
            errors.append(f"recipe.mode must be distill or sft, got {recipe.mode!r}.")
        if recipe.divergence not in ("fkl", "rkl", "jsd"):
            errors.append(f"recipe.divergence must be fkl, rkl or jsd, got {recipe.divergence!r}.")
        if not 0.0 < recipe.jsd_beta < 1.0:
            errors.append("recipe.jsd_beta must be in (0, 1).")
        if not 0.0 <= recipe.onpolicy_fraction <= 1.0:
            errors.append("recipe.onpolicy_fraction must be in [0, 1].")
        if recipe.replay_teacher not in ("notes", "self"):
            errors.append(f"recipe.replay_teacher must be notes or self, got {recipe.replay_teacher!r}.")
        if recipe.anchor_teacher not in ("notes", "base", "previous"):
            errors.append(f"recipe.anchor_teacher must be notes, base or previous, got {recipe.anchor_teacher!r}.")
        if set(recipe.mix) - {"new", "replay", "nearmiss", "anchor"} or recipe.mix.get("new", 0) <= 0:
            errors.append("recipe.mix takes new (> 0), replay, nearmiss and anchor shares.")
        if recipe.sampler not in ("hf", "vllm_colocate"):
            errors.append(f"recipe.sampler must be hf or vllm_colocate, got {recipe.sampler!r}.")
        if recipe.passes < 1 or recipe.rows_per_step < 1 or recipe.micro_rows < 1:
            errors.append("recipe.passes, recipe.rows_per_step and recipe.micro_rows must be >= 1.")
        if recipe.temperature <= 0:
            errors.append("recipe.temperature must be > 0 (on-policy rows sample from the student).")
        if self.generate.questions_per_fact < 1:
            errors.append("generate.questions_per_fact must be >= 1.")
        if self.signals.max_note_chars < 1:
            errors.append("signals.max_note_chars must be >= 1.")
        if errors:
            raise LearnConfigError(
                "learn configuration has incompatible settings:\n" + "\n".join(f"  - {e}" for e in errors)
            )


def served_rank(ranks: list[int]) -> int:
    """The rank a stacked adapter is served at: the sum, padded up to vLLM's next accepted rank."""
    total = sum(ranks)
    for allowed in VLLM_LORA_RANKS:
        if allowed >= total:
            return allowed
    raise LearnConfigError(
        f"stacked rank {total} (= {' + '.join(map(str, ranks))}) exceeds vLLM's largest LoRA rank "
        f"{VLLM_LORA_RANKS[-1]}; lower scopes.ranks."
    )
