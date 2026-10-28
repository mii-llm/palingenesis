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
    # Budget of the notes a served prompt carries between sessions (about 4 characters per token); notes
    # past it are left out (ancestors' and older notes first in, so the prefix stays stable).
    wake_max_tokens: int = 4000


@dataclass(slots=True)
class LearnSignalsConfig:
    command_patterns: list = field(default_factory=lambda: list(DEFAULT_COMMAND_PATTERNS))
    max_note_chars: int = 1000
    # Notes the proxy captures per API key and hour (commands, corrections and admin notes together).
    max_notes_per_hour: int = 20


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
    # Domain-adjacent questions the notes do not answer (an unstated attribute, an invented sibling entity);
    # the right answer declines. They measure confabulation.
    abstain_per_session: int = 20


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
class LearnEvalConfig:
    max_probes_per_session: int = 200  # acquisition and update probes of the session's new facts
    max_retention_probes: int = 200  # probes of the scope's earlier facts
    max_ancestor_probes: int = 100  # probes of the ancestors' facts, asked of the stacked chain
    general_qa_path: str = ""  # JSONL of {"q", "answers": [...]}; empty = the built-in multilingual set
    drift_path: str = ""  # JSONL of {"prompt"} or {"messages"}; empty = the built-in prompts
    max_drift_prompts: int = 32
    max_new_tokens: int = 64  # greedy answers of every closed-book and in-context probe
    drift_max_new_tokens: int = 32  # target completion length the drift KL is measured on
    seed: int = 0


@dataclass(slots=True)
class LearnGatesConfig:
    """Publishing gates. The defaults are uncalibrated: a GPU calibration on null sessions (a session rerun
    with another seed, notes already known, replay and anchors only) replaces them with thresholds set to a
    per-gate false-fail rate. Counts are paired against the live version on the same probes."""

    calibrated: bool = False
    acquisition_min: float = 0.80  # new facts answered closed-book, among the probes answered with notes in context
    update_min: float = 0.80  # superseded facts answered with the new value, same relative measure
    stale_max: float = 0.05  # share of answers about superseded facts that state an old value
    retention_max_net_loss: int = 3  # earlier facts: (live right, candidate wrong) minus the reverse
    ancestor_retention_max_net_loss: int = 3
    general_qa_max_net_loss: int = 5
    leak_max_net: int = 2  # changed, wrong answers to unrelated prompts that name a new entity of the scope
    confab_max_net: int = 2  # questions the notes do not answer: (live declined, candidate answered) minus the reverse
    meta_reference_max: float = 0.02  # closed-book answers that mention notes / context
    drift_increment_max: float = 0.03  # KL(target || candidate) - KL(target || live), nats per token
    drift_cap: float = 0.15  # KL(target || candidate), nats per token
    # The parity canary of a publish: the served adapter against HF + PEFT on the same tokens.
    parity_answer_max_net_loss: int = 1  # probes HF answers right and the server wrong, minus the reverse
    parity_effect_corr_min: float = 0.9  # correlation of the adapter's per-token logprob effect, HF vs server
    parity_k3_ratio_max: float = 1.5  # k3 mismatch with the adapter / the base model's own k3 on the same tokens
    parity_k3_floor: float = 1.0e-4  # a base k3 below this counts as this (an exact server would divide by 0)


@dataclass(slots=True)
class LearnServingConfig:
    """Publishing to inference servers (pgs learn publish / rollback / unload / proxy)."""

    engine: str = "vllm"  # vllm | sglang
    servers: list = field(default_factory=list)  # base URL of every replica, e.g. http://10.0.0.5:8000
    # api: each version is loaded through the server's LoRA load route; resolver: the vLLM resolver plugin
    # loads it from the store on its first request (the warm-up or canary request).
    transport: str = "api"
    base_name: str = ""  # the base model's name on the servers (default: the store's base)
    server_store: str = ""  # the store's path as the servers see it (default: the same path)
    drain_seconds: int = 60  # an old version stays loaded this long after a publish, for in-flight requests
    canary_probes: int = 16  # acquisition probes the parity canary asks (HF + PEFT reference vs server)
    canary_max_new_tokens: int = 48
    timeout: float = 120.0  # seconds per server request
    api_key_env: str = "PGS_LEARN_SERVER_KEY"  # environment variable holding the servers' --api-key


@dataclass(slots=True)
class LearnSleepConfig:
    """pgs learn sleep: one pass over the scopes with pending notes, under a compute budget."""

    budget_minutes: float = 0.0  # the whole pass; 0 = no limit
    scope_budget_minutes: float = 0.0  # one scope; 0 = no limit
    # gated: after a parent goes live, its published children are evaluated on the new parent and republished
    # when their gates pass; none: children keep serving the parent version they were published with.
    children_follow: str = "gated"


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
    eval: LearnEvalConfig = field(default_factory=LearnEvalConfig)
    gates: LearnGatesConfig = field(default_factory=LearnGatesConfig)
    serving: LearnServingConfig = field(default_factory=LearnServingConfig)
    sleep: LearnSleepConfig = field(default_factory=LearnSleepConfig)

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
        if self.generate.abstain_per_session < 0:
            errors.append("generate.abstain_per_session must be >= 0.")
        ev = self.eval
        if min(ev.max_probes_per_session, ev.max_retention_probes, ev.max_ancestor_probes, ev.max_drift_prompts) < 0:
            errors.append("eval.max_* counts must be >= 0.")
        if ev.max_new_tokens < 1 or ev.drift_max_new_tokens < 1:
            errors.append("eval.max_new_tokens and eval.drift_max_new_tokens must be >= 1.")
        g = self.gates
        for name in ("acquisition_min", "update_min", "stale_max", "meta_reference_max"):
            if not 0.0 <= getattr(g, name) <= 1.0:
                errors.append(f"gates.{name} is a rate in [0, 1].")
        for name in ("retention_max_net_loss", "ancestor_retention_max_net_loss", "general_qa_max_net_loss"):
            if getattr(g, name) < 0:
                errors.append(f"gates.{name} must be >= 0.")
        if g.drift_cap < 0:
            errors.append("gates.drift_cap must be >= 0.")
        if not -1.0 <= g.parity_effect_corr_min <= 1.0:
            errors.append("gates.parity_effect_corr_min is a correlation in [-1, 1].")
        if g.parity_k3_ratio_max <= 0 or g.parity_k3_floor <= 0:
            errors.append("gates.parity_k3_ratio_max and gates.parity_k3_floor must be > 0.")
        sv = self.serving
        if sv.engine not in ("vllm", "sglang"):
            errors.append(f"serving.engine must be vllm or sglang, got {sv.engine!r}.")
        if sv.transport not in ("api", "resolver"):
            errors.append(f"serving.transport must be api or resolver, got {sv.transport!r}.")
        elif sv.transport == "resolver" and sv.engine != "vllm":
            errors.append("serving.transport resolver is the vLLM resolver plugin: use it with serving.engine vllm.")
        if sv.drain_seconds < 0 or sv.canary_probes < 0 or sv.timeout <= 0:
            errors.append("serving.drain_seconds and serving.canary_probes must be >= 0, serving.timeout > 0.")
        if self.notes.wake_max_tokens < 0 or self.signals.max_notes_per_hour < 0:
            errors.append("notes.wake_max_tokens and signals.max_notes_per_hour must be >= 0.")
        if self.sleep.children_follow not in ("gated", "none"):
            errors.append(f"sleep.children_follow must be gated or none, got {self.sleep.children_follow!r}.")
        if self.sleep.budget_minutes < 0 or self.sleep.scope_budget_minutes < 0:
            errors.append("sleep.budget_minutes and sleep.scope_budget_minutes must be >= 0 (0 = no limit).")
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
