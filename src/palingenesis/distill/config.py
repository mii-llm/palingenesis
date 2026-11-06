"""Off-policy distillation settings: the `distill:` section of the training config.

    distill:
      enabled: true
      divergence: forward_kl          # forward_kl | reverse_kl | jsd | skew_forward_kl | skew_reverse_kl | tvd
      temperature: 1.0
      kd_weight: 1.0
      ce_weight: 0.0                  # cross-entropy on the gold tokens, mixed in
      teachers:
        big:   {model: Qwen/Qwen3-8B, placement: shard}     # online: the teacher runs every micro-batch
        cached: {cache: caches/qwen3_32b_top64}              # precomputed top-k (pgs distill-cache)

Teachers are validated into `TeacherSpec`s by `parse_teachers` (Config.validate calls it).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

DIVERGENCES = ("forward_kl", "reverse_kl", "jsd", "skew_forward_kl", "skew_reverse_kl", "tvd")
TOPK_MODES = ("tail", "renorm")
MISSING = ("error", "skip")
PLACEMENTS = ("auto", "replicate", "shard")
FEATURE_LOSSES = ("mse", "nmse", "cosine", "lens_kl")
PROJECTIONS = ("auto", "linear", "none")
CACHE_BACKENDS = ("hf", "vllm")


@dataclass(slots=True)
class DistillConfig:
    """Off-policy knowledge distillation inside `pgs train`: the student trains on the data
    pipeline's fixed sequences against its teachers' distributions. Off by default."""

    enabled: bool = False
    # name -> teacher settings (see TeacherSpec): `model` for an online teacher, `cache` for a
    # precomputed top-k cache. Several teachers mix per token by `weight`, restricted to their `sources`.
    teachers: dict = field(default_factory=dict)
    # Token-level divergence between teacher p and student q (temperature-scaled):
    #   forward_kl        KL(p || q), mode-covering; classic KD (Hinton et al., arXiv:1503.02531)
    #   reverse_kl        KL(q || p), mode-seeking (MiniLLM, arXiv:2306.08543)
    #   jsd               generalized JSD with mixture beta*p + (1-beta)*q (GKD, arXiv:2306.13649)
    #   skew_forward_kl   KL(p || beta*p + (1-beta)*q)   (DistiLLM, arXiv:2402.03898)
    #   skew_reverse_kl   KL(q || beta*q + (1-beta)*p)   (DistiLLM)
    #   tvd               total variation 0.5 * sum |p - q|
    divergence: str = "forward_kl"
    # jsd: weight of the teacher in the mixture (0 < beta < 1); skew_*: the skew coefficient alpha.
    beta: float = 0.5
    # Softmax temperature applied to both distributions. With temperature_scale the divergence is
    # multiplied by temperature^2, which keeps gradient magnitudes comparable across temperatures.
    temperature: float = 1.0
    temperature_scale: bool = True
    # The loss per trained token: kd_weight * divergence + ce_weight * cross-entropy on the gold token.
    kd_weight: float = 1.0
    ce_weight: float = 0.0
    # Sparse (cached top-k) teachers: "tail" keeps the teacher's top-k (plus the gold token) and one
    # bucket holding the rest of each side's mass, so the divergence stays a divergence between full
    # distributions coarse-grained on the same partition; "renorm" renormalises both sides over the
    # support (the teacher's tail mass is dropped).
    topk_mode: str = "tail"
    # A trained token no teacher covers (a cache miss, or a source routed to no teacher): "error"
    # stops (a stale cache is usually a mistake), "skip" trains it with the CE term only (if any).
    missing: str = "error"
    # Hidden-state / feature matching: [{student: 11, teacher: 23, weight: 1.0, loss: mse,
    # projection: auto}] (FitNets, arXiv:1412.6550; see hooks.FeatureSpec). Learnable projections
    # (when dimensions differ) train with the student at feature_lr (0 = train.learning_rate).
    features: list = field(default_factory=list)
    feature_lr: float = 0.0
    # Rows of [rows, vocab] logits projected at a time by the chunked loss (0 = automatic: about 512 MB
    # of fp32 logits per side).
    rows_per_slice: int = 0


@dataclass
class TeacherSpec:
    """One teacher of `distill.teachers`.

    Online (`model`): a frozen causal LM run on every micro-batch.
      dtype                bfloat16 | float16 | float32 (the teacher's weights)
      placement            auto (shard under FSDP, else the student's device) | replicate | shard
      device               single process: where the teacher lives (empty = the student's GPU, e.g. "cuda:1")
      attn_implementation  empty = the student's
      tokenizer            empty = the model's; must share the student's vocabulary (see token_bridge)
      eos_map, probe_texts the shared-vocabulary bridge (end-of-turn mapping, extra identity probes)
    Cached (`cache`): a directory written by `pgs distill-cache` (top-k log-probs per trained token).
      With both keys, `cache` is used for training and `model` (with the view) to build it.
    Both:
      weight               mixing weight per token among the teachers covering it
      sources              data source names this teacher supervises (empty = all)
      top_k                cache building: log-probs kept per position
    Teacher view (context distillation; the student sees the plain rendering):
      system_prompt        a system prompt only the teacher sees (replaces / creates the system turn)
      prefix_messages      messages placed before the conversation (after the system turn): few-shot
                           examples; a list, or a path to a JSON / JSONL file of messages
      privileged_field     row field with text only the teacher sees, placed by privileged_template
                           ({content}, {privileged}) into the last user turn or the system turn
                           (privileged_role); rows without the field render as they are
      messages_field       row field with the teacher's own full conversation (its trained turns
                           must match the student's)
      chat_template_kwargs the teacher's template options (e.g. {enable_thinking: true})
      view_fn              "module:function" (row, messages) -> teacher messages, for anything else
      max_seq_length       the teacher view's token limit (0 = 2 x data.max_seq_length)
    """

    name: str
    model: str = ""
    cache: str = ""
    tokenizer: str = ""
    dtype: str = "bfloat16"
    placement: str = "auto"
    device: str = ""
    attn_implementation: str = ""
    weight: float = 1.0
    sources: list = field(default_factory=list)
    top_k: int = 64
    eos_map: dict = field(default_factory=dict)
    probe_texts: list = field(default_factory=list)
    system_prompt: str = ""
    prefix_messages: list | str = field(default_factory=list)
    privileged_field: str = ""
    privileged_template: str = "{content}\n\n{privileged}"
    privileged_role: str = "user"
    messages_field: str = ""
    chat_template_kwargs: dict = field(default_factory=dict)
    view_fn: str = ""
    max_seq_length: int = 0

    @property
    def online(self) -> bool:
        return not self.cache

    @property
    def has_view(self) -> bool:
        return bool(
            self.system_prompt
            or self.prefix_messages
            or self.privileged_field
            or self.messages_field
            or self.chat_template_kwargs
            or self.view_fn
        )


def parse_teachers(raw: dict) -> list[TeacherSpec]:
    """`distill.teachers` as TeacherSpecs; raises ValueError naming the bad option."""
    if not isinstance(raw, dict):
        raise ValueError("distill.teachers must be a mapping {name: {model: ...} or {cache: ...}}")
    names = {f.name for f in dataclasses.fields(TeacherSpec)} - {"name"}
    specs = []
    for name, options in raw.items():
        where = f"distill.teachers.{name}"
        if isinstance(options, str):  # shorthand: {name: model_id}
            options = {"model": options}
        if not isinstance(options, dict):
            raise ValueError(f"{where} must be a mapping of options")
        unknown = set(options) - names
        if unknown:
            import difflib

            hints = [difflib.get_close_matches(u, names, n=1) for u in sorted(unknown)]
            hint = "; ".join(f"{u} (did you mean {h[0]}?)" if h else u for u, h in zip(sorted(unknown), hints))
            raise ValueError(f"{where}: unknown option(s) {hint}")
        spec = TeacherSpec(name=str(name), **options)
        if isinstance(spec.weight, int):
            spec.weight = float(spec.weight)
        specs.append(spec)
    return specs


def distill_errors(config) -> list[str]:
    """Hard configuration errors of the distill section (called from Config.validate)."""
    d: DistillConfig = config.distill
    errors: list[str] = []
    if not d.enabled:
        return errors
    if d.divergence not in DIVERGENCES:
        errors.append(f"distill.divergence={d.divergence!r} is not one of {', '.join(DIVERGENCES)}.")
    if d.divergence in ("jsd", "skew_forward_kl", "skew_reverse_kl") and not 0.0 < d.beta < 1.0:
        errors.append(f"distill.beta must be in (0, 1) for {d.divergence} (got {d.beta}).")
    if d.temperature <= 0:
        errors.append(f"distill.temperature must be > 0 (got {d.temperature}).")
    if d.kd_weight < 0 or d.ce_weight < 0:
        errors.append("distill.kd_weight and distill.ce_weight must be >= 0.")
    if d.kd_weight == 0 and d.ce_weight == 0 and not d.features:
        errors.append("distill: kd_weight and ce_weight are both 0 and there are no features: nothing to train on.")
    if d.topk_mode not in TOPK_MODES:
        errors.append(f"distill.topk_mode={d.topk_mode!r} is not one of {', '.join(TOPK_MODES)}.")
    if d.missing not in MISSING:
        errors.append(f"distill.missing={d.missing!r} is not one of {', '.join(MISSING)}.")
    if d.rows_per_slice < 0 or d.feature_lr < 0:
        errors.append("distill.rows_per_slice and distill.feature_lr must be >= 0.")
    try:
        teachers = parse_teachers(d.teachers)
    except (ValueError, TypeError) as e:
        return errors + [str(e)]
    if not teachers and d.kd_weight > 0:
        errors.append("distill.enabled needs at least one teacher in distill.teachers.")
    source_names = _source_names(config)
    for t in teachers:
        where = f"distill.teachers.{t.name}"
        if not t.model and not t.cache:
            errors.append(f"{where} needs `model` (an online teacher) or `cache` (a pgs distill-cache directory).")
        if t.placement not in PLACEMENTS:
            errors.append(f"{where}.placement={t.placement!r} is not one of {', '.join(PLACEMENTS)}.")
        if t.dtype not in ("bfloat16", "float16", "float32"):
            errors.append(f"{where}.dtype must be bfloat16, float16 or float32 (got {t.dtype!r}).")
        if t.weight <= 0:
            errors.append(f"{where}.weight must be > 0 (got {t.weight}).")
        if t.top_k < 1:
            errors.append(f"{where}.top_k must be >= 1 (got {t.top_k}).")
        if t.privileged_role not in ("user", "system"):
            errors.append(f"{where}.privileged_role must be 'user' or 'system' (got {t.privileged_role!r}).")
        if t.privileged_field:
            try:
                t.privileged_template.format(content="", privileged="")
                ok = all(f"{{{k}}}" in t.privileged_template for k in ("content", "privileged"))
            except (KeyError, IndexError, ValueError):
                ok = False
            if not ok:
                errors.append(f"{where}.privileged_template must contain {{content}} and {{privileged}} only.")
        unknown_sources = [s for s in t.sources if s not in source_names]
        if unknown_sources:
            errors.append(
                f"{where}.sources {unknown_sources} are not data source names ({sorted(source_names) or 'none'}; "
                "a data.sources entry is named by its `name`, else its `dataset`)."
            )
        if t.online and config.parallel.context_parallel:
            # each CP rank holds a slice of the sequence: an online teacher there would read a truncated context
            errors.append(
                f"{where}: an online teacher does not support parallel.context_parallel (it would see only this "
                "rank's slice of each sequence). Cache the teacher (pgs distill-cache): cached targets are computed "
                "on the whole sequence and sharded like the inputs."
            )
        if t.online and t.has_view:
            if config.data.packing:
                errors.append(
                    f"{where}: a teacher view (context distillation) with an online teacher needs data.packing: "
                    "false (each row has its own teacher rendering). Cache the teacher (pgs distill-cache) to pack."
                )
            if config.data.pretokenize:
                errors.append(f"{where}: a teacher view with an online teacher does not support data.pretokenize.")
        if t.has_view and (config.data.loop_recovery or config.data.reasoning_loop_recovery):
            errors.append(f"{where}: a teacher view cannot be combined with data.loop_recovery augmentation.")
        if t.online and t.sources and config.data.pretokenize:
            errors.append(
                f"{where}.sources (routing) needs the source of each row: not available with data.pretokenize."
            )
    if config.dpo.enabled:
        errors.append("distill.enabled cannot be combined with dpo.enabled.")
    gated = [n for n in ("dft", "cadft", "deft", "info_sft", "pre_rl") if getattr(config.plugins, n)]
    if gated:
        errors.append(f"distill.enabled replaces the SFT objective: disable plugins.{', plugins.'.join(gated)}.")
    if config.memory.seco and d.features:
        errors.append("distill.features (hidden-state matching) is not supported with memory.seco.")
    if config.memory.gradient_release:
        errors.append("distill.enabled does not support memory.gradient_release.")
    for i, f in enumerate(d.features):
        where = f"distill.features[{i}]"
        if not isinstance(f, dict) or "student" not in f or "teacher" not in f:
            errors.append(f"{where} must be a mapping with `student` and `teacher` (layer index or module path).")
            continue
        unknown = set(f) - {"student", "teacher", "weight", "loss", "projection", "name", "via"}
        if unknown:
            errors.append(f"{where}: unknown option(s) {sorted(unknown)}.")
        if f.get("loss", "mse") not in FEATURE_LOSSES:
            errors.append(f"{where}.loss must be one of {', '.join(FEATURE_LOSSES)}.")
        if f.get("projection", "auto") not in PROJECTIONS:
            errors.append(f"{where}.projection must be one of {', '.join(PROJECTIONS)}.")
        online = [t.name for t in teachers if t.online]
        via = f.get("via", online[0] if online else None)
        if via not in online:
            errors.append(f"{where}: feature matching needs an online teacher (`via`: one of {online or 'none'}).")
    return errors


def _source_names(config) -> set[str]:
    from palingenesis.data import source_name

    if config.data.sources:
        return {source_name(s, i) for i, s in enumerate(config.data.sources)}
    return set()
