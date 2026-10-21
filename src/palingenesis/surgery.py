"""Margin surgery: a minimal-norm weight edit that moves a trained model's decision margins back to target values.

The failure it repairs: a training stage (RL, distillation, SFT) shifts the overall level of one decision, e.g. the
log-odds of ending the turn right after a tool call, while the model's ability to tell when the decision is right
survives. Retraining to fix one propensity costs GPU-days; this edit costs minutes. It is the post-hoc counterpart of
common-mode rejection (palingenesis.cmr), which holds a margin during training.

The method (validated at 0.8B: a 2.2% edit restored parallel calling with every held-out measure equal or better):

- **Sides and groups.** Twins (palingenesis.margins format) give prompts at the decision position. Each family takes
  one side (pos or neg) of its twins, drops sides longer than `max_side_tokens`, and splits a seeded shuffle into
  `groups` groups of `group_size` sides. A group's margin m̄_i is the mean exact log-odds of the decision over its
  sides. Groups, not single prompts, are the constraints: one prompt's margin is noise, a group mean is a measurement.
- **Targets.** Per group: the reference model's margins on the same sides ("reference"; paired per side), the start
  model's ("hold", protects a decision you do not want to move), or a fixed value. `constraint: equal` always enters
  the system; `hinge_below` enters only while m̄_i is above its target (the margin must not exceed it) and
  `hinge_above` only while it is below.
- **Step.** For the active rows, with g_i = ∇θ m̄_i, Gram A = G Gᵀ and gap = target − m̄, the minimal-norm step that
  closes the linearised gaps is Δ = Gᵀ A⁺ gap. Group gradients of one decision are nearly collinear (cos .9-.99), so A
  is ill-conditioned and its small eigen-directions are group noise: eigenvalues below cutoff·λmax are dropped
  (truncated pseudo-inverse, c = V_k diag(1/λ_k) V_kᵀ gap). The step is scaled to a trust radius per iteration and a
  cumulative cap, both fractions of ‖θ0 − θ_ref‖ over the edited parameters (the size of the stage that caused the
  shift) or absolute norms.
- **Loop.** Re-measure, repeat until |mean gap| < max(min_gap, stop_se·SE of the mean gap) over the equality rows and
  no hinge is active, or `iterations` steps.
- **Doses.** θ(α) = θ0 + α(θ_final − θ0) for each α in `doses`. The twin target is a proxy for behaviour (our 0.8B case
  needed α = 3): choose α on a behavioural split, not on the twins.

Memory (the only real cost; compute is a few dozen group forward/backward passes):

- **The step never materialises G**: Δ = Σ c_i g_i is the gradient of Σ c_i m̄_i, one weighted backward over the
  active groups' sides (micro-batched), accumulated in fp32 by post-accumulate-grad hooks that free each bf16 .grad as
  soon as it is produced.
- **The Gram matrix** (`gram`): `device` keeps the K flattened gradients on the device in bf16 (2·K·P_edit bytes);
  `cpu` streams each to (pinned) host memory and computes the dots on the device chunk by chunk (host 2·K·P_edit);
  `recompute` keeps nothing: for each i it computes g_i once and recomputes every g_j, j > i, into a second buffer
  (K(K+1)/2 backward passes instead of K, two bf16 gradient buffers). Recompute is exact (bf16 backward is not bitwise
  deterministic on GPU, ~1e-3 relative, far below the 1e-2 truncation) and its cost is minutes even at 70B because the
  groups are tiny; a sketch would be cheaper but approximate. `auto` picks device if it fits in free memory with
  headroom, else cpu if host RAM allows, else recompute, and logs the choice and the estimated bytes. Dots are float32
  chunk dots (2^26 elements) summed in float64.
- **Parameter subset** (`params.include` / `exclude`, regexes on parameter names, every alias of a tied tensor
  checked): only matching parameters get requires_grad and are edited; frozen ones need no gradient memory and are
  saved bit-for-bit as they were.
- Totals: device ≈ 2·P_work + 4·P_edit (fp32 master, unless master: cpu) + 4·P_edit (fp32 gradient accumulator)
  + 2·K·P_edit + activations; cpu moves the 2·K·P_edit to host; recompute replaces it with 4·P_edit (two bf16
  buffers). Host also holds θ0 of the edited parameters in the source dtype (to measure the edit norm exactly).

Precision (the edit is often smaller than a bf16 step):

- The forward/backward runs on a working model in `work_dtype` (default bf16 on CUDA, fp32 on CPU). The edit
  accumulates in an fp32 **master copy of the edited parameters only** (`master: device | cpu`, cpu = pinned host).
- After every update the working weights are re-derived from the master by **the same cast that saving will use**:
  S = cast(master, save_dtype, rounding, seed) (palingenesis.precision.cast_state_dict, per-tensor seed from (seed,
  name), always on the run's device so the random stream is identical), then W = S.to(work_dtype) (what a loader
  does). `rounding` defaults to stochastic when the saved dtype is 16-bit: a bf16-native source sits on the grid and
  round-to-nearest would delete a sub-ulp edit deterministically, while unbiased rounding keeps it in expectation.
- Gaps, the stop rule and every reported margin are measured on W, never on the fp32 master, so the saved checkpoint is
  the measured one (a test asserts the saved checkpoint reproduces the last measured margins exactly).
  edit_survival is reported for every dose (and per simulated serving format): read the projection; the norm ratio
  is dominated by the whole-ulp jumps of elements that cross a grid boundary.

Reference model: loaded once (CPU, its stored dtype), ‖θ0 − θ_ref‖ computed tensor by tensor against θ0, then moved to
the device in the working dtype to measure the target margins, and freed before the start model goes to the device.

    pgs surgery run --config surgery.yaml [--set trust=0.2 --set families.cont.groups=8]
    pgs surgery margins --model runs/ref --model runs/cur --twins twins.jsonl --decision "<|im_end|>" --prefix-side both
    pgs surgery span --model runs/cur --groups twins.jsonl --targets bfcl_like.jsonl --decision "<|im_end|>" --k 1,4,16

Limits: two applications at 0.8B, one confirmed on held-out tests, one that restored the twin margins but not the
behaviour (its twins' negatives were easy, the benchmark's hard: twins must represent the target distribution's
positives AND hard negatives, and discrimination must be checked on a held-out behavioural set); decisions inside a
thinking block are unsupported; GPTQ/AWQ survival is unmeasured; run a constant logit-bias baseline before crediting
the weight edit. Single device.
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import json
import logging
import math
import os
import random
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from palingenesis.margins import MarginReader, decision_ids, load_twins, log_odds
from palingenesis.precision import SurvivalMeter, cast_tensor, simulate_cast

logger = logging.getLogger(__name__)

SIDES = ("pos", "neg")
CONSTRAINTS = ("equal", "hinge_below", "hinge_above")
GRAM_MODES = ("auto", "device", "cpu", "recompute")
DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}
CHUNK = 1 << 26


class SurgeryConfigError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------- config


@dataclass
class FamilyConfig:
    """One constraint family: `groups` groups of `group_size` sides from one side of a twin file."""

    path: str = ""
    side: str = "pos"  # pos | neg: which side of each twin is a prompt of this family
    decision: str | list = "<tool_call>"  # a token, or a token set (each element one token)
    target: str | float = "reference"  # reference | hold | <float>
    constraint: str = "equal"  # equal | hinge_below (margin <= target) | hinge_above (margin >= target)
    groups: int = 8
    group_size: int = 8


@dataclass
class EvalConfig:
    """Held-out twins, never in a group: margins reported for the reference, the start, the end and every dose."""

    path: str = ""
    side: str = "pos"  # pos | neg | both (pos, neg, common = m+ + m-, differential = m+ - m-)
    decision: str | list = "<tool_call>"
    limit: int | None = None


@dataclass
class ParamsConfig:
    include: list = field(default_factory=lambda: [".*"])  # regexes (re.search) on parameter names
    exclude: list = field(default_factory=list)


@dataclass
class SurgeryConfig:
    model: str = ""  # θ0, the model to edit
    reference: str | None = None  # needed for target: reference and trust_ref: reference_distance
    output_dir: str = ""
    families: dict = field(default_factory=dict)  # name -> FamilyConfig
    eval: dict = field(default_factory=dict)  # name -> EvalConfig
    iterations: int = 6  # maximum number of steps
    cutoff: float = 1e-2  # drop Gram eigenvalues below cutoff * lambda_max
    trust: float = 0.1  # per-step |Δ| <= trust * R
    cap: float = 0.25  # cumulative |θ - θ0| <= cap * R
    trust_ref: str = "reference_distance"  # R = |θ0 - θ_ref| over the edited parameters, or "absolute" (R = 1)
    min_gap: float = 0.5  # nats; stop when |mean gap| < max(min_gap, stop_se * SE)
    stop_se: float = 2.0
    doses: list = field(default_factory=lambda: [1.0])
    params: ParamsConfig = field(default_factory=ParamsConfig)
    gram: str = "auto"  # auto | device | cpu | recompute
    master: str = "device"  # device | cpu (pinned host memory)
    work_dtype: str | None = None  # default: bf16 on CUDA, fp32 on CPU
    save_dtype: str | None = None  # default: the source checkpoint's stored dtype
    rounding: str | None = None  # nearest | stochastic; default stochastic when save_dtype is 16-bit
    serving_formats: list = field(default_factory=list)  # e.g. [fp8_e4m3]: edit survival under simulated serving
    micro_batch: int = 4  # sides per forward / backward
    gradient_checkpointing: bool = False
    max_side_tokens: int | None = None  # sides longer than this are dropped (reported per family)
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    seed: int = 0
    device: str | None = None  # default: cuda if available
    overwrite: bool = False

    def __post_init__(self):
        """Plain mappings (Python API) become the option dataclasses."""
        if isinstance(self.params, dict):
            self.params = ParamsConfig(**self.params)
        self.families = {k: FamilyConfig(**v) if isinstance(v, dict) else v for k, v in self.families.items()}
        self.eval = {k: EvalConfig(**v) if isinstance(v, dict) else v for k, v in self.eval.items()}

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_dict(cls, raw: dict, where: str = "config") -> SurgeryConfig:
        if not isinstance(raw, dict):
            raise SurgeryConfigError(f"{where}: expected a mapping")
        if set(raw) == {"surgery"}:
            raw = raw["surgery"] or {}
        config = cls()
        for key, value in raw.items():
            if key in ("families", "eval"):
                if not isinstance(value, dict):
                    raise SurgeryConfigError(f"{where}: `{key}:` must be a mapping of name -> options")
                for name, entry in value.items():
                    config.set(f"{key}.{name}", entry, where)
            elif key == "params":
                if not isinstance(value, dict):
                    raise SurgeryConfigError(f"{where}: `params:` must be a mapping (include:, exclude:)")
                for k, v in value.items():
                    config.set(f"params.{k}", v, where)
            else:
                config.set(key, value, where)
        return config

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Sequence[str] = ()) -> SurgeryConfig:
        """A YAML mapping of the fields (optionally under `surgery:`), then KEY=VALUE overrides (dotted keys:
        trust=0.2, families.cont.groups=4, params.exclude=['embed']; values parsed as YAML)."""
        import yaml

        with Path(path).open() as f:
            config = cls.from_dict(yaml.safe_load(f) or {}, str(path))
        config.apply_overrides(overrides)
        return config

    def apply_overrides(self, overrides: Sequence[str]) -> None:
        import yaml

        for item in overrides:
            key, sep, value = item.partition("=")
            if not sep:
                raise SurgeryConfigError(f"--set {item!r}: expected KEY=VALUE")
            self.set(key.strip(), yaml.safe_load(value), "command line")

    def set(self, key: str, value, where: str = "config") -> None:
        parts = key.split(".")
        if parts[0] in ("families", "eval"):
            entries = getattr(self, parts[0])
            cls = FamilyConfig if parts[0] == "families" else EvalConfig
            if len(parts) == 2:
                if not isinstance(value, dict):
                    raise SurgeryConfigError(f"{where}: `{key}:` must be a mapping of options")
                entry = entries.setdefault(parts[1], cls())
                for k, v in value.items():
                    _set_field(entry, k, v, f"{where}: {key}")
            elif len(parts) == 3:
                _set_field(entries.setdefault(parts[1], cls()), parts[2], value, f"{where}: {parts[0]}.{parts[1]}")
            else:
                raise SurgeryConfigError(f"{where}: {key}: use {parts[0]}.<name>.<option>")
        elif parts[0] == "params" and len(parts) == 2:
            _set_field(self.params, parts[1], value, f"{where}: params")
        elif len(parts) == 1 and parts[0] not in ("params",):
            _set_field(self, parts[0], value, where)
        else:
            raise SurgeryConfigError(f"{where}: unknown key {key!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    # ---------------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Errors for impossible or unsupported settings (empty list = valid). Multi-token decisions and decisions
        inside a think block are refused when the tokenizer is loaded (they need it)."""
        e: list[str] = []
        if not self.model:
            e.append("model is required")
        if not self.output_dir:
            e.append("output_dir is required")
        if not self.families:
            e.append("families: at least one constraint family is required")
        for name, f in self.families.items():
            w = f"families.{name}"
            if not f.path:
                e.append(f"{w}.path is required")
            if f.side not in SIDES:
                e.append(f"{w}.side must be pos or neg, not {f.side!r}")
            if f.constraint not in CONSTRAINTS:
                e.append(f"{w}.constraint must be one of {CONSTRAINTS}, not {f.constraint!r}")
            if not (f.target in ("reference", "hold") or _is_number(f.target)):
                e.append(f"{w}.target must be reference, hold or a number, not {f.target!r}")
            if f.target == "reference" and not self.reference:
                kind = "a hinge" if f.constraint != "equal" else "an equality"
                e.append(f"{w}: {kind} on target: reference needs `reference:` (or set a numeric target)")
            if not _is_int(f.groups) or f.groups < 1 or not _is_int(f.group_size) or f.group_size < 1:
                e.append(f"{w}.groups and group_size must be integers >= 1")
            if not f.decision:
                e.append(f"{w}.decision is required")
        for name, ev in self.eval.items():
            if not ev.path:
                e.append(f"eval.{name}.path is required")
            if ev.side not in (*SIDES, "both"):
                e.append(f"eval.{name}.side must be pos, neg or both, not {ev.side!r}")
        if not _is_int(self.iterations) or self.iterations < 0:
            e.append("iterations must be an integer >= 0")
        if not (0 < self.cutoff < 1):
            e.append("cutoff must be in (0, 1)")
        if self.trust <= 0 or self.cap <= 0:
            e.append("trust and cap must be > 0")
        if self.trust_ref not in ("reference_distance", "absolute"):
            e.append(f"trust_ref must be reference_distance or absolute, not {self.trust_ref!r}")
        if self.trust_ref == "reference_distance" and not self.reference:
            e.append("trust_ref: reference_distance needs `reference:` (or trust_ref: absolute)")
        if self.min_gap < 0 or self.stop_se < 0:
            e.append("min_gap and stop_se must be >= 0")
        if not self.doses or any(not _is_number(a) or a <= 0 for a in self.doses):
            e.append(f"doses must be a non-empty list of numbers > 0, not {self.doses!r}")
        if self.gram not in GRAM_MODES:
            e.append(f"gram must be one of {GRAM_MODES}, not {self.gram!r}")
        if self.master not in ("device", "cpu"):
            e.append(f"master must be device or cpu, not {self.master!r}")
        for k in ("work_dtype", "save_dtype"):
            v = getattr(self, k)
            if v is not None and str(v) not in DTYPES:
                e.append(f"{k} must be one of {sorted(DTYPES)}, not {v!r}")
        if self.rounding not in (None, "nearest", "stochastic"):
            e.append(f"rounding must be nearest or stochastic, not {self.rounding!r}")
        from palingenesis.precision import FP8

        bad = [f for f in self.serving_formats if f not in ("bf16", "fp16", *FP8)]
        if bad:
            e.append(f"serving_formats: unknown {bad}; one of bf16, fp16, {', '.join(FP8)}")
        if not _is_int(self.micro_batch) or self.micro_batch < 1:
            e.append("micro_batch must be an integer >= 1")
        if self.max_side_tokens is not None and (not _is_int(self.max_side_tokens) or self.max_side_tokens < 1):
            e.append("max_side_tokens must be an integer >= 1 (or null)")
        if (self.chat_template_kwargs or {}).get("enable_thinking"):
            e.append("chat_template_kwargs.enable_thinking: thinking-mode decisions are not supported")
        for k in ("include", "exclude"):
            pats = getattr(self.params, k)
            if not isinstance(pats, list):
                e.append(f"params.{k} must be a list of regexes")
                continue
            for p in pats:
                try:
                    re.compile(p)
                except (re.error, TypeError) as err:
                    e.append(f"params.{k}: invalid regex {p!r} ({err})")
        return e


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _set_field(obj, name: str, value, where: str) -> None:
    names = [f.name for f in dataclasses.fields(obj)]
    if name not in names:
        hint = difflib.get_close_matches(name, names, n=1)
        raise SurgeryConfigError(
            f"{where}: unknown option {name!r}" + (f" (did you mean {hint[0]!r}?)" if hint else "")
        )
    current = getattr(obj, name)
    if isinstance(current, float) and _is_int(value):
        value = float(value)
    if name == "doses" and _is_number(value):
        value = [value]
    setattr(obj, name, value)


# ---------------------------------------------------------------------------------------------------------- groups


@dataclass
class Group:
    """A constraint row: sides (pre-tokenised prompts up to the decision) of one family, its decision set and target.
    `targets` holds the per-side target margins (float64) once known, so reference / hold gaps are paired per side."""

    family: str
    index: int
    ids: list
    token_ids: list
    constraint: str = "equal"
    target: str | float = "reference"
    targets: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return f"{self.family}/{self.index}"

    def __len__(self) -> int:
        return len(self.ids)


def _side_ids(reader: MarginReader, side: dict) -> list[int]:
    """Prompt ids up to the decision; refuses a decision inside an open think block (thinking-mode twins)."""
    ids = reader.prompt_ids(side)
    text = reader.tok.decode(ids)
    if text.rfind("<think>") > text.rfind("</think>"):
        raise SurgeryConfigError(
            "a side's decision position is inside an open <think> block: thinking-mode decisions are not supported "
            "(render with enable_thinking: false)"
        )
    return ids


def load_sides(
    tok, path: str, side: str, *, chat_template_kwargs: dict | None = None, max_side_tokens: int | None = None
) -> tuple[list, int]:
    """Token ids of one side of every twin in `path` ("pos", "neg", or "both" -> (pos_ids, neg_ids) pairs), with the
    number of twins dropped for exceeding max_side_tokens (padded long prompts are what exhausts memory)."""
    reader = MarginReader(None, tok, chat_template_kwargs, device="cpu")
    out, dropped = [], 0
    for row in load_twins(path):
        sides = SIDES if side == "both" else (side,)
        ids = [_side_ids(reader, row[s]) for s in sides]
        if max_side_tokens is not None and any(len(x) > max_side_tokens for x in ids):
            dropped += 1
            continue
        out.append(tuple(ids) if side == "both" else ids[0])
    return out, dropped


def make_groups(
    name: str,
    ids: list,
    token_ids: list,
    n_groups: int,
    group_size: int,
    *,
    seed: int = 0,
    constraint: str = "equal",
    target: str | float = "reference",
) -> list[Group]:
    """`n_groups` disjoint groups of `group_size` sides from a seeded shuffle (seed and family name) of `ids`."""
    need = n_groups * group_size
    if len(ids) < need:
        raise SurgeryConfigError(
            f"family {name}: {n_groups} groups x {group_size} sides need {need} sides, only {len(ids)} available "
            "(after the max_side_tokens filter)"
        )
    order = list(range(len(ids)))
    random.Random(f"{seed}:{name}").shuffle(order)
    return [
        Group(
            name, g, [ids[i] for i in order[g * group_size : (g + 1) * group_size]], list(token_ids), constraint, target
        )
        for g in range(n_groups)
    ]


def build_groups(tok, name: str, fam: FamilyConfig, config: SurgeryConfig) -> tuple[list[Group], int]:
    ids, dropped = load_sides(
        tok,
        fam.path,
        fam.side,
        chat_template_kwargs=config.chat_template_kwargs,
        max_side_tokens=config.max_side_tokens,
    )
    tids = decision_ids(tok, fam.decision)
    groups = make_groups(
        name, ids, tids, fam.groups, fam.group_size, seed=config.seed, constraint=fam.constraint, target=fam.target
    )
    return groups, dropped


# ---------------------------------------------------------------------------------------------------------- margins


@contextlib.contextmanager
def _mode(model, train: bool):
    was = model.training
    model.train(train)
    try:
        yield
    finally:
        model.train(was)


def _reader(model, tok, reader=None) -> MarginReader:
    return reader if reader is not None else MarginReader(model, tok)


def _by_length(ids: list) -> list[int]:
    """Indices sorted by length: micro-batches of similar lengths waste the least memory on padding."""
    return sorted(range(len(ids)), key=lambda i: len(ids[i]))


@torch.no_grad()
def measure(reader: MarginReader, ids: list, token_ids, micro_batch: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """(margins float64, greedy bool) on CPU for prompts sharing one decision set: the exact log-odds, and whether
    greedy decoding would pick a token of the set. Eval mode, no gradient, micro-batched by length."""
    n = len(ids)
    margins = torch.empty(n, dtype=torch.float64)
    greedy = torch.empty(n, dtype=torch.bool)
    tset = torch.tensor(list(token_ids) if not isinstance(token_ids, int) else [token_ids])
    order = _by_length(ids)
    with _mode(reader.model, False):
        for s in range(0, n, micro_batch):
            idx = order[s : s + micro_batch]
            logits = reader.logits_at_last([ids[i] for i in idx])
            m = log_odds(logits, tset.tolist())
            g = torch.isin(logits.argmax(-1).cpu(), tset)
            margins[idx] = m.double().cpu()
            greedy[idx] = g
            del logits
    return margins, greedy


def group_margins(model, tok, groups: list[Group], *, micro_batch: int = 4, reader=None) -> list[torch.Tensor]:
    """Per-group per-side margins (float64, CPU) at the model's current weights. Groups sharing a decision set are
    measured together, sorted by length."""
    reader = _reader(model, tok, reader)
    out: list[torch.Tensor | None] = [None] * len(groups)
    by_set: dict[tuple, list[int]] = {}
    for i, g in enumerate(groups):
        by_set.setdefault(tuple(g.token_ids), []).append(i)
    for tset, members in by_set.items():
        ids = [x for i in members for x in groups[i].ids]
        m, _ = measure(reader, ids, list(tset), micro_batch)
        o = 0
        for i in members:
            out[i] = m[o : o + len(groups[i])]
            o += len(groups[i])
    return out  # type: ignore[return-value]


def _stats(m: torch.Tensor, greedy: torch.Tensor | None = None) -> dict:
    n = m.numel()
    se = float(m.std(unbiased=True) / math.sqrt(n)) if n > 1 else 0.0
    out = {"n": n, "mean": float(m.mean()) if n else float("nan"), "se": se}
    if greedy is not None:
        out["greedy"] = float(greedy.double().mean()) if n else float("nan")
    return out


def eval_margins(reader: MarginReader, sets: dict, micro_batch: int = 4) -> dict:
    """{name: stats} for held-out sets {name: {"side", "ids", "token_ids"}} (side both: ids are (pos, neg) pairs)."""
    out = {}
    for name, s in sets.items():
        if s["side"] == "both":
            mp, gp = measure(reader, [p for p, _ in s["ids"]], s["token_ids"], micro_batch)
            mn, gn = measure(reader, [q for _, q in s["ids"]], s["token_ids"], micro_batch)
            out[name] = {
                "pos": _stats(mp, gp),
                "neg": _stats(mn, gn),
                "common": float((mp + mn).mean()),
                "differential": float((mp - mn).mean()),
            }
        else:
            out[name] = _stats(*measure(reader, s["ids"], s["token_ids"], micro_batch))
    return out


# ---------------------------------------------------------------------------------------------------------- gradients


class _GradSink:
    """fp32 per-parameter gradient accumulators. A post-accumulate-grad hook adds each parameter's .grad (bf16 under a
    bf16 working model) into an fp32 buffer and frees the .grad at once, so gradients never accumulate in bf16 across
    micro-batches and at most one parameter's bf16 gradient is alive at a time."""

    def __init__(self, params: list[torch.Tensor]):
        self.params = params
        self.acc: list[torch.Tensor | None] = [None] * len(params)
        self.handles = []

    def __enter__(self):
        for i, p in enumerate(self.params):
            p.grad = None
            self.handles.append(p.register_post_accumulate_grad_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        for p in self.params:
            p.grad = None
        self.acc = [None] * len(self.params)

    def _hook(self, i: int):
        def hook(p):
            g = p.grad.detach()
            if self.acc[i] is None:
                self.acc[i] = g if g.dtype == torch.float32 else g.float()
            else:
                self.acc[i].add_(g)
            p.grad = None

        return hook

    def take(self) -> list[torch.Tensor]:
        out = [
            a if a is not None else torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            for a, p in zip(self.acc, self.params)
        ]
        self.acc = [None] * len(self.params)
        return out


def _backward(reader: MarginReader, items: list[tuple], micro_batch: int) -> None:
    """Backward of Σ weight * margin over items (ids, token set, weight), micro-batched by length; rows of one
    micro-batch may have different decision sets."""
    order = _by_length([it[0] for it in items])
    for s in range(0, len(order), micro_batch):
        chunk = [items[i] for i in order[s : s + micro_batch]]
        logits = reader.logits_at_last([c[0] for c in chunk])
        loss = logits.new_zeros(())
        by_set: dict[tuple, list[int]] = {}
        for r, c in enumerate(chunk):
            by_set.setdefault(tuple(c[1]), []).append(r)
        for tset, rows in by_set.items():
            w = torch.tensor([chunk[r][2] for r in rows], dtype=logits.dtype, device=logits.device)
            loss = loss + (log_odds(logits[rows], list(tset)) * w).sum()
        loss.backward()
        del logits, loss


def _grads_of(reader, params, items, micro_batch, train: bool, sink: _GradSink | None = None) -> list[torch.Tensor]:
    with _mode(reader.model, train):
        if sink is not None:
            _backward(reader, items, micro_batch)
            return sink.take()
        with _GradSink(params) as own:
            _backward(reader, items, micro_batch)
            return own.take()


def group_gradient(
    model, tok, group: Group, params: list[torch.Tensor], *, micro_batch: int = 4, reader=None, train: bool = False
) -> list[torch.Tensor]:
    """∇θ m̄ (the group-mean margin) for `params`, fp32, one tensor per parameter."""
    reader = _reader(model, tok, reader)
    items = [(x, group.token_ids, 1.0 / len(group)) for x in group.ids]
    return _grads_of(reader, params, items, micro_batch, train)


def weighted_step(
    model,
    tok,
    groups: list[Group],
    coefs: Sequence[float],
    params: list[torch.Tensor],
    *,
    micro_batch: int = 4,
    reader=None,
    train: bool = False,
) -> list[torch.Tensor]:
    """Δ = Σ_i c_i ∇θ m̄_i as ONE weighted backward of Σ_i c_i m̄_i over all groups' sides (micro-batched): never
    materialises the K gradients. fp32, one tensor per parameter."""
    reader = _reader(model, tok, reader)
    items = [(x, g.token_ids, float(c) / len(g)) for g, c in zip(groups, coefs) if c != 0 for x in g.ids]
    if not items:
        return [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
    return _grads_of(reader, params, items, micro_batch, train)


# ---------------------------------------------------------------------------------------------------------- gram


def _flat_into(grads: list[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
    o = 0
    for g in grads:
        n = g.numel()
        out[o : o + n].copy_(g.reshape(-1))
        o += n
    return out


def chunked_dot(a: torch.Tensor, b: torch.Tensor, device=None, chunk: int = CHUNK) -> float:
    """<a, b> as float32 chunk dots summed in float64; chunks of a host-resident operand are moved to `device`."""
    device = device if device is not None else a.device
    total = 0.0
    for s in range(0, a.numel(), chunk):
        x = a[s : s + chunk].to(device, non_blocking=True).float()
        y = x if b is a else b[s : s + chunk].to(device, non_blocking=True).float()
        total += float(torch.dot(x, y))
    return total


def gram_matrix(
    model,
    tok,
    groups: list[Group],
    params: list[torch.Tensor],
    *,
    mode: str = "device",
    micro_batch: int = 4,
    reader=None,
    train: bool = False,
    chunk: int = CHUNK,
) -> torch.Tensor:
    """A_ij = <∇m̄_i, ∇m̄_j> (float64, CPU, K x K) over `params`, from bf16-stored gradients in every mode, so the modes
    agree to float rounding:
    - device: the K flat bf16 gradients stay on the device (2 K P bytes);
    - cpu: each is streamed to (pinned) host memory; dots move host chunks to the device (host 2 K P bytes);
    - recompute: two bf16 buffers; g_j (j > i) recomputed for every i: K(K+1)/2 backward passes."""
    if mode not in ("device", "cpu", "recompute"):
        raise ValueError(f"gram mode must be device, cpu or recompute here, not {mode!r}")
    reader = _reader(model, tok, reader)
    device = params[0].device
    n = sum(p.numel() for p in params)
    K = len(groups)
    A = torch.zeros(K, K, dtype=torch.float64)

    def grad_flat(i, out=None):
        items = [(x, groups[i].token_ids, 1.0 / len(groups[i])) for x in groups[i].ids]
        with _mode(reader.model, train), _GradSink(params) as sink:
            _backward(reader, items, micro_batch)
            grads = sink.take()
        out = out if out is not None else torch.empty(n, dtype=torch.bfloat16, device=device)
        _flat_into(grads, out)
        del grads
        return out

    if mode in ("device", "cpu"):
        stored: list[torch.Tensor] = []
        pin = device.type == "cuda"
        for i in range(K):
            flat = grad_flat(i)
            for j, s in enumerate(stored):
                A[i, j] = A[j, i] = chunked_dot(flat, s, device, chunk)
            A[i, i] = chunked_dot(flat, flat, device, chunk)
            if mode == "cpu" and device.type != "cpu":
                host = torch.empty(n, dtype=torch.bfloat16, pin_memory=pin)
                host.copy_(flat)
                stored.append(host)
                del flat
            else:
                stored.append(flat)
        del stored
    else:
        gi = torch.empty(n, dtype=torch.bfloat16, device=device)
        gj = torch.empty(n, dtype=torch.bfloat16, device=device)
        for i in range(K):
            grad_flat(i, gi)
            A[i, i] = chunked_dot(gi, gi, device, chunk)
            for j in range(i + 1, K):
                grad_flat(j, gj)
                A[i, j] = A[j, i] = chunked_dot(gi, gj, device, chunk)
        del gi, gj
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return A


def _host_available() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError, AttributeError):
            return 0


def choose_gram_mode(mode: str, k: int, p_edit: int, device) -> tuple[str, dict]:
    """Resolve gram: auto. device if 2·K·P_edit fits in free device memory after a headroom (max(4 GiB, 15% of the
    device) plus the fp32 accumulator and two bf16 flats, 8·P_edit); else cpu if it fits in 70% of available host RAM;
    else recompute. Returns (mode, info with the estimated bytes)."""
    device = torch.device(device)
    need = 2 * k * p_edit
    info = {
        "requested": mode,
        "K": k,
        "edited_params": p_edit,
        "stored_gradients_bytes": need,
        "recompute_buffers_bytes": 4 * p_edit,
        "host_available_bytes": _host_available(),
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        headroom = max(4 << 30, int(0.15 * total)) + 8 * p_edit
        info.update(device_free_bytes=int(free), device_headroom_bytes=headroom)
        fits_device = need + headroom <= free
    else:
        fits_device = need <= 0.7 * info["host_available_bytes"]
    fits_host = need <= 0.7 * info["host_available_bytes"]
    if mode == "auto":
        mode = "device" if fits_device else "cpu" if fits_host else "recompute"
    elif mode == "device" and not fits_device:
        logger.warning(
            "surgery: gram: device may not fit (%.2f GB of gradients); consider cpu or recompute", need / 1e9
        )
    elif mode == "cpu" and not fits_host:
        logger.warning("surgery: gram: cpu may not fit in host RAM (%.2f GB); consider recompute", need / 1e9)
    info["mode"] = mode
    logger.info(
        "surgery: gram mode %s (K=%d, P_edit=%d: stored gradients %.2f GB, recompute buffers %.2f GB)",
        mode,
        k,
        p_edit,
        need / 1e9,
        4 * p_edit / 1e9,
    )
    return mode, info


# ---------------------------------------------------------------------------------------------------------- solve


def solve_coefficients(A: torch.Tensor, gap: torch.Tensor, cutoff: float = 1e-2) -> tuple[torch.Tensor, dict]:
    """c = V_k diag(1/λ_k) V_kᵀ gap over the eigenvalues λ_k > cutoff·λmax of the Gram matrix A (truncated
    pseudo-inverse): Gᵀc is the minimal-norm step closing the linearised gaps inside the kept eigen-directions. Small
    eigen-directions of nearly collinear group gradients are group noise; inverting them chases noise with huge steps.
    Returns (c float64, info: relative spectrum (descending), kept count, kept relative eigenvalues, lambda_max)."""
    A = A.double().cpu()
    gap = torch.as_tensor(gap, dtype=torch.float64).cpu()
    ev, V = torch.linalg.eigh(0.5 * (A + A.T))
    lmax = float(ev.max()) if ev.numel() else 0.0
    if lmax <= 0:
        return torch.zeros_like(gap), {"spectrum": [], "kept": 0, "kept_eigenvalues": [], "lambda_max": lmax}
    keep = ev > cutoff * lmax
    Vk = V[:, keep]
    c = Vk @ ((Vk.T @ gap) / ev[keep])
    rel = (ev / lmax).flip(0)
    return c, {
        "spectrum": [float(x) for x in rel],
        "kept": int(keep.sum()),
        "kept_eigenvalues": [float(x) for x in rel[: int(keep.sum())]],
        "lambda_max": lmax,
    }


def constraint_rows(groups: list[Group], margins: list[torch.Tensor]) -> dict:
    """Gaps (target − margin, paired per side), the active rows and the stop statistics. An equality group is always a
    row; hinge_below is a row only while its mean margin is above the target (gap < 0), hinge_above only while below
    (gap > 0). mean / se are over the equality rows (SE of the mean gap from the per-side spread in each group)."""
    gaps, rows, active, ses = [], [], [], []
    for i, (g, m) in enumerate(zip(groups, margins)):
        d = g.targets.to(m.dtype) - m
        gi = float(d.mean())
        gaps.append(gi)
        if g.constraint == "equal":
            rows.append(i)
            ses.append(float(d.var(unbiased=True)) / len(d) if len(d) > 1 else 0.0)
        elif (g.constraint == "hinge_below" and gi < 0) or (g.constraint == "hinge_above" and gi > 0):
            active.append(i)
    eq = [gaps[i] for i in rows]
    mean = sum(eq) / len(eq) if eq else 0.0
    se = math.sqrt(sum(ses)) / len(eq) if eq else 0.0
    return {"gaps": gaps, "rows": rows + active, "equality": rows, "active": active, "mean": mean, "se": se}


# ---------------------------------------------------------------------------------------------------------- editing


def select_parameters(model, include: Sequence[str], exclude: Sequence[str]) -> list[tuple[str, torch.nn.Parameter]]:
    """Floating parameters matching an include regex and no exclude regex, checked against every alias of a tied
    tensor (an edit of lm_head is an edit of a tied embedding). Each tensor once, under its first name."""
    aliases: dict[int, list[str]] = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(n)
    out = []
    for name, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        names = aliases[id(p)]
        inc = any(re.search(r, n) for r in include for n in names)
        exc = any(re.search(r, n) for r in exclude for n in names)
        if inc and not exc:
            out.append((name, p))
    return out


def canonical_names(model) -> dict[int, str]:
    """id(parameter) -> the smallest of its names (tied tensors have several): the name precision.cast_state_dict seeds
    an aliased tensor with, so a cast here and the cast inside save_hf_model draw the same noise."""
    out: dict[int, str] = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        out[id(p)] = min(out.get(id(p), n), n)
    return out


class _Editor:
    """θ0 (host, source dtype), the fp32 master of the edited parameters, and the cast that derives the working
    weights from it: S = cast_tensor(x, save_dtype, rounding, seed, name) (precision; seeded per tensor, noise drawn on
    the CPU so it does not depend on the device), W = S.to(work_dtype)."""

    def __init__(self, cnames, params, theta0, *, device, master, work_dtype, save_dtype, rounding, seed):
        self.cnames, self.params, self.theta0 = cnames, params, theta0
        self.device, self.work_dtype, self.save_dtype = torch.device(device), work_dtype, save_dtype
        self.rounding, self.seed = rounding, seed
        # fp32 working weights saved in fp32: the parameters are their own master (no copy)
        self.alias = work_dtype == torch.float32 and save_dtype == torch.float32
        pin = self.device.type == "cuda"
        self._master: list[torch.Tensor] = []
        if not self.alias:
            mdev = self.device if master == "device" else torch.device("cpu")
            for t in theta0:
                m = torch.empty(t.shape, dtype=torch.float32, device=mdev, pin_memory=pin and mdev.type == "cpu")
                m.copy_(t)
                self._master.append(m)

    @property
    def master(self) -> list[torch.Tensor]:
        """fp32 master tensors; when aliased, the parameters' current data (valid after model.to(device))."""
        return [p.data for p in self.params] if self.alias else self._master

    def cast(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """x cast to the save dtype exactly as save_hf_model(dtype=save_dtype, rounding, seed) casts tensor `name`."""
        return cast_tensor(x.to(self.device), self.save_dtype, self.rounding, self.seed, name)

    def saved(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return self.cast(self.cnames[i], x)

    @torch.no_grad()
    def refresh(self) -> None:
        """Working weights from the master."""
        if self.alias:
            return
        for i, p in enumerate(self.params):
            p.data.copy_(self.saved(i, self.master[i]).to(self.work_dtype))

    @torch.no_grad()
    def edit_terms(self, step: list[torch.Tensor] | None = None) -> tuple[float, float]:
        """(|E|², <E, Δ>) with E = master − θ0, streamed tensor by tensor (θ0 lives on the host)."""
        ee = ed = 0.0
        for i, m in enumerate(self.master):
            e = m.to(self.device).float() - self.theta0[i].to(self.device, non_blocking=True).float()
            ee += float(torch.sum(e * e, dtype=torch.float64))
            if step is not None:
                ed += float(torch.sum(e * step[i], dtype=torch.float64))
            del e
        return ee, ed

    @torch.no_grad()
    def apply(self, step: list[torch.Tensor], scale: float) -> None:
        for i, m in enumerate(self.master):
            m.add_(step[i].to(m.device, non_blocking=True), alpha=scale)
        self.refresh()

    def freeze(self) -> None:
        """Before doses overwrite the working weights: an aliased master gets its own copy."""
        if self.alias:
            self._master = [m.clone() for m in self.master]
            self.alias = False

    @torch.no_grad()
    def dose(self, i: int, alpha: float) -> torch.Tensor:
        m = self.master[i].to(self.device).float()
        if alpha == 1.0:
            return m
        t0 = self.theta0[i].to(self.device).float()
        return t0 + alpha * (m - t0)


# ---------------------------------------------------------------------------------------------------------- run


def stored_dtype(path: str) -> torch.dtype:
    """The dtype most floating tensors of a safetensors checkpoint are stored in (read from the headers)."""
    from palingenesis.merge import Checkpoint

    ck = Checkpoint(path)
    counts: dict[torch.dtype, int] = {}
    for name in ck.keys():
        dt = _ST_DTYPES.get(ck._handle(ck.weight_map[name]).get_slice(name).get_dtype())
        if dt is not None:
            counts[dt] = counts.get(dt, 0) + 1
    if not counts:
        raise SurgeryConfigError(f"{path}: no floating tensors found")
    return max(counts, key=counts.get)


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _load(path, dtype, device="cpu"):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True)
    return m.to(device) if str(device) != "cpu" else m


def _empty_cache(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _prepare_out(path: str, overwrite: bool) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SurgeryConfigError(f"{out} exists and is not empty (overwrite: true to replace it)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _dose_dir(alpha: float) -> str:
    return f"dose_{alpha:g}"


def run_surgery(config: SurgeryConfig) -> dict:
    """Run margin surgery end to end (see the module docstring). Writes one checkpoint per dose
    (output_dir/dose_<α>, in the source checkpoint's layout) and output_dir/surgery_report.json; returns the report."""
    from transformers import AutoTokenizer

    errors = config.validate()
    if errors:
        raise SurgeryConfigError("; ".join(errors))
    t_start = time.time()
    timings: dict[str, float] = {}
    out = _prepare_out(config.output_dir, config.overwrite)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(config.model)

    # ---- precision plan
    source_dtype = stored_dtype(config.model)
    work_dtype = (
        DTYPES[config.work_dtype] if config.work_dtype else (torch.bfloat16 if device.type == "cuda" else torch.float32)
    )
    save_dtype = DTYPES[config.save_dtype] if config.save_dtype else source_dtype
    rounding = config.rounding or ("stochastic" if save_dtype in (torch.bfloat16, torch.float16) else "nearest")
    if save_dtype in (torch.bfloat16, torch.float16) and source_dtype == save_dtype and rounding == "nearest":
        logger.warning(
            "surgery: a %s-native source written by nearest rounding: a sub-ulp edit is deleted deterministically",
            _dtype_name(save_dtype),
        )
    if save_dtype == torch.float32 and config.rounding == "stochastic":
        logger.warning("surgery: rounding only applies to 16-bit saves; the fp32 save is exact")

    # ---- groups and held-out sets (tokenizer only)
    groups: list[Group] = []
    dropped: dict[str, int] = {}
    for name, fam in config.families.items():
        gs, d = build_groups(tok, name, fam, config)
        groups += gs
        dropped[name] = d
        if d:
            logger.info("surgery: family %s: %d twins dropped (> %s tokens)", name, d, config.max_side_tokens)
    eval_sets = {}
    for name, ev in config.eval.items():
        if ev.path in {f.path for f in config.families.values()}:
            logger.warning("surgery: eval.%s uses a constraint family's file: it is not held out", name)
        ids, d = load_sides(
            tok,
            ev.path,
            ev.side,
            chat_template_kwargs=config.chat_template_kwargs,
            max_side_tokens=config.max_side_tokens,
        )
        if ev.limit:
            ids = ids[: ev.limit]
        eval_sets[name] = {"side": ev.side, "ids": ids, "token_ids": decision_ids(tok, ev.decision)}
        dropped[f"eval.{name}"] = d
    timings["prepare"] = time.time() - t_start

    # ---- the working model (CPU for now) and the edited subset
    t0 = time.time()
    model = _load(config.model, work_dtype)
    selected = select_parameters(model, config.params.include, config.params.exclude)
    if not selected:
        raise SurgeryConfigError("params: no parameter matches include / exclude")
    names = [n for n, _ in selected]
    params = [p for _, p in selected]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    p_edit = sum(p.numel() for p in params)
    p_total = sum(p.numel() for p in model.parameters())
    logger.info("surgery: editing %d tensors, %d of %d parameters", len(params), p_edit, p_total)

    # exact θ0 of the edited parameters (and the source values of the frozen ones, if the save cast changes them)
    pin = device.type == "cuda"
    if source_dtype == work_dtype:
        src_params = dict(model.named_parameters())
        theta0 = [p.detach().to("cpu", copy=True) for p in params]
    else:
        src = _load(config.model, source_dtype)
        src_params = dict(src.named_parameters())
        theta0 = [src_params[n].detach() for n in names]
    if pin:
        theta0 = [t.pin_memory() for t in theta0]
    theta0_norm = math.sqrt(sum(float(torch.sum(t.float() ** 2, dtype=torch.float64)) for t in theta0))
    cn = canonical_names(model)
    editor = _Editor(
        [cn[id(p)] for p in params],
        params,
        theta0,
        device=device,
        master=config.master,
        work_dtype=work_dtype,
        save_dtype=save_dtype,
        rounding=rounding,
        seed=config.seed,
    )
    with torch.no_grad():  # frozen tensors: the save cast of the source (a no-op when the save dtype is the source's)
        if save_dtype != source_dtype:
            edited = {id(p) for p in params}
            for n, p in model.named_parameters():
                if id(p) not in edited:
                    p.data = editor.cast(cn[id(p)], src_params[n].detach()).to(work_dtype).cpu()
    del src_params
    if source_dtype != work_dtype:
        del src
    timings["load"] = time.time() - t0

    # ---- reference: distance, target margins, held-out margins; freed before θ0 goes to the device
    reference_distance = None
    ref_group, ref_eval = None, {}
    if config.reference:
        t0 = time.time()
        _check_reference_tokenizer(tok, config.reference)
        ref = _load(config.reference, stored_dtype(config.reference))
        ref_params = dict(ref.named_parameters())
        d2 = 0.0
        for n, t in zip(names, theta0):
            if n not in ref_params or tuple(ref_params[n].shape) != tuple(t.shape):
                raise SurgeryConfigError(f"reference: parameter {n} missing or of another shape")
            d2 += float(torch.sum((ref_params[n].detach().float() - t.float()) ** 2, dtype=torch.float64))
        reference_distance = math.sqrt(d2)
        del ref_params
        ref = ref.to(device=device, dtype=work_dtype)
        reader_ref = MarginReader(ref, tok, device=device)
        ref_group = group_margins(ref, tok, groups, micro_batch=config.micro_batch, reader=reader_ref)
        ref_eval = eval_margins(reader_ref, eval_sets, config.micro_batch)
        del ref, reader_ref
        _empty_cache(device)
        timings["reference"] = time.time() - t0
        if reference_distance == 0 and config.trust_ref == "reference_distance":
            raise SurgeryConfigError("θ0 equals the reference on the edited parameters: nothing to restore")

    # ---- the working model on the device
    model.to(device)
    editor.params = params  # .to() keeps the Parameter objects
    editor.refresh()
    _empty_cache(device)
    train_mode = False
    if config.gradient_checkpointing:
        if getattr(model, "supports_gradient_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            train_mode = True  # transformers checkpoints only in training mode
        else:
            logger.warning("surgery: gradient_checkpointing requested but %s does not support it", type(model).__name__)
    reader = MarginReader(model, tok, device=device)

    t0 = time.time()
    start_group = group_margins(model, tok, groups, micro_batch=config.micro_batch, reader=reader)
    start_eval = eval_margins(reader, eval_sets, config.micro_batch)
    for i, g in enumerate(groups):
        if g.target == "reference":
            g.targets = ref_group[i].clone()
        elif g.target == "hold":
            g.targets = start_group[i].clone()
        else:
            g.targets = torch.full((len(g),), float(g.target), dtype=torch.float64)
    timings["start_margins"] = time.time() - t0

    R = reference_distance if config.trust_ref == "reference_distance" else 1.0
    gram_mode, memory = choose_gram_mode(config.gram, len(groups), p_edit, device)
    memory.update(
        master=config.master if not editor.alias else "aliased (fp32 working weights)",
        work_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        master_bytes=0 if editor.alias else 4 * p_edit,
        accumulator_bytes=4 * p_edit,
        theta0_host_bytes=sum(t.numel() * t.element_size() for t in theta0),
    )

    # ---- iterations
    iters = []
    current = start_group
    stop_reason = "max_iterations"
    edit_sq = 0.0
    t_loop = time.time()
    for it in range(config.iterations + 1):
        t_it = time.time()
        st = constraint_rows(groups, current)
        threshold = max(config.min_gap, config.stop_se * st["se"])
        rec = {
            "it": it,
            "gap_mean": st["mean"],
            "se": st["se"],
            "threshold": threshold,
            "gaps": [round(x, 4) for x in st["gaps"]],
            "active_hinges": [groups[i].name for i in st["active"]],
            "rows": len(st["rows"]),
            "edit_rel": math.sqrt(edit_sq) / R if R else None,
        }
        converged = (not st["equality"] or abs(st["mean"]) < threshold) and not st["active"]
        if converged:
            stop_reason = "converged" if it else ("nothing_to_do" if not st["equality"] else "converged")
        elif it == config.iterations:
            stop_reason = "max_iterations"
        elif math.sqrt(edit_sq) >= 0.999 * config.cap * R:
            stop_reason = "cap"
        if converged or it == config.iterations or stop_reason == "cap":
            iters.append(rec)
            logger.info("surgery: it %d: gap %.3f (threshold %.3f), stop: %s", it, st["mean"], threshold, stop_reason)
            break
        rows = [groups[i] for i in st["rows"]]
        gap = torch.tensor([st["gaps"][i] for i in st["rows"]], dtype=torch.float64)
        A = gram_matrix(
            model, tok, rows, params, mode=gram_mode, micro_batch=config.micro_batch, reader=reader, train=train_mode
        )
        coef, sinfo = solve_coefficients(A, gap, config.cutoff)
        if sinfo["kept"] == 0:
            rec.update(sinfo)
            iters.append(rec)
            stop_reason = "no_gradient"
            break
        step = weighted_step(
            model, tok, rows, coef.tolist(), params, micro_batch=config.micro_batch, reader=reader, train=train_mode
        )
        sn = math.sqrt(sum(float(torch.sum(s * s, dtype=torch.float64)) for s in step))
        ee, ed = editor.edit_terms(step)
        cap_abs = config.cap * R
        if sn > 0:
            disc = ed * ed - sn * sn * (ee - cap_abs * cap_abs)
            s_cap = max(0.0, (-ed + math.sqrt(max(disc, 0.0))) / (sn * sn))
            scale = min(1.0, config.trust * R / sn, s_cap)
        else:
            scale = 0.0
        editor.apply(step, scale)
        del step
        _empty_cache(device)
        edit_sq = max(ee + 2 * scale * ed + scale * scale * sn * sn, 0.0)
        nrm = torch.sqrt(torch.clamp(A.diag(), min=1e-300))
        C = A / nrm[:, None] / nrm[None, :]
        k = len(rows)
        rec.update(
            spectrum=sinfo["spectrum"][:8],
            kept=sinfo["kept"],
            kept_eigenvalues=sinfo["kept_eigenvalues"],
            coef_norm=float(coef.norm()),
            step_norm=sn,
            predicted_step_norm=math.sqrt(max(float(coef @ A @ coef), 0.0)),
            scale=scale,
            edit_rel_after=math.sqrt(edit_sq) / R if R else None,
            group_cos_mean=float((C.sum() - k) / (k * (k - 1))) if k > 1 else 1.0,
            seconds=round(time.time() - t_it, 2),
        )
        iters.append(rec)
        logger.info(
            "surgery: it %d: gap %.3f, rows %d, kept %d, |step| %.4g, scale %.3g, edit_rel %.4g",
            it,
            st["mean"],
            k,
            sinfo["kept"],
            sn,
            scale,
            rec["edit_rel_after"] or 0.0,
        )
        current = group_margins(model, tok, groups, micro_batch=config.micro_batch, reader=reader)
    timings["iterations"] = time.time() - t_loop
    end_group = current
    end_eval = eval_margins(reader, eval_sets, config.micro_batch)
    ee, _ = editor.edit_terms()
    edit_norm = math.sqrt(ee)

    # ---- doses: working weights = the saved cast; measured, then saved
    t0 = time.time()
    if config.gradient_checkpointing and train_mode:
        model.gradient_checkpointing_disable()
    editor.freeze()
    doses = []
    save_model = None
    for alpha in config.doses:
        alpha = float(alpha)
        meter = SurvivalMeter()  # the saved edit vs the intended one (precision.edit_survival, streamed)
        served = {fmt: SurvivalMeter() for fmt in config.serving_formats}
        saved_edit = []
        with torch.no_grad():
            for i, p in enumerate(params):
                x = editor.dose(i, alpha)
                s = editor.saved(i, x)
                before = editor.saved(i, theta0[i])  # θ0 as saved (itself when stored in the save dtype)
                meter.update(before, x, s)
                for fmt, m in served.items():  # q = served θ0; does the edit survive serving in fmt?
                    q = simulate_cast(before, fmt)
                    m.update(q, q + (x - before.float()), simulate_cast(s, fmt))
                p.data.copy_(s.to(work_dtype))
                if save_dtype != work_dtype:
                    saved_edit.append(s.cpu())
                del x, s, before
        survival = meter.result()
        survival_served = {fmt: m.result() for fmt, m in served.items()}
        dg = group_margins(model, tok, groups, micro_batch=config.micro_batch, reader=reader)
        dst = constraint_rows(groups, dg)
        de = eval_margins(reader, eval_sets, config.micro_batch)
        path = out / _dose_dir(alpha)
        if save_dtype == work_dtype:  # the working weights ARE the saved cast
            _save(model, tok, path, config.model)
        else:  # e.g. fp32 save of a bf16 working model: θ0's other tensors from the source, cast by save_hf_model
            if save_model is None:
                save_model = _load(config.model, source_dtype)
            with torch.no_grad():
                sp = dict(save_model.named_parameters())
                for n, s in zip(names, saved_edit):
                    sp[n].data = s
            _save(save_model, tok, path, config.model, dtype=save_dtype, rounding=rounding, seed=config.seed)
        (path / "surgery_config.json").write_text(
            json.dumps(
                {"method": "margin_surgery", "alpha": alpha, "model": config.model, "reference": config.reference},
                indent=2,
            )
        )
        doses.append(
            {
                "alpha": alpha,
                "path": str(path),
                "edit_rel": alpha * edit_norm / R if R else None,
                "edit_rel_weights": alpha * edit_norm / theta0_norm if theta0_norm else None,
                "edit_survival": survival,  # read the projection; the norm ratio is inflated by whole-ulp jumps
                "edit_survival_served": survival_served,
                "gap_mean": dst["mean"],
                "gap_se": dst["se"],
                "active_hinges": [groups[i].name for i in dst["active"]],
                "group_means": _family_means(groups, dg),
                "eval": de,
            }
        )
        logger.info(
            "surgery: dose %g -> %s (gap %.3f, survival projection %.3f)",
            alpha,
            path,
            dst["mean"],
            survival["projection"],
        )
    timings["doses"] = time.time() - t0
    timings["total"] = time.time() - t_start
    if torch.device(device).type == "cuda":  # measured peak next to choose_gram_mode's estimate
        memory = dict(
            memory,
            peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved(device) / 2**30, 2),
        )

    report = {
        "config": config.to_dict(),
        "source_dtype": _dtype_name(source_dtype),
        "work_dtype": _dtype_name(work_dtype),
        "save_dtype": _dtype_name(save_dtype),
        "rounding": rounding,
        "device": str(device),
        "edited": {"tensors": len(params), "params": p_edit, "total_params": p_total, "names": names[:50]},
        "dropped_sides": dropped,
        "memory": memory,
        "reference_distance": reference_distance,
        "theta0_norm": theta0_norm,
        "groups": [
            {"name": g.name, "constraint": g.constraint, "target_mean": float(g.targets.mean())} for g in groups
        ],
        "families": {
            "reference": _family_means(groups, ref_group) if ref_group is not None else None,
            "start": _family_means(groups, start_group),
            "end": _family_means(groups, end_group),
        },
        "iterations": iters,
        "stop_reason": stop_reason,
        "edit": {
            "norm": edit_norm,
            "rel_reference": edit_norm / reference_distance if reference_distance else None,
            "rel_weights": edit_norm / theta0_norm if theta0_norm else None,
        },
        "eval": {
            name: {
                "reference": ref_eval.get(name),
                "start": start_eval[name],
                "end": end_eval[name],
                **{_dose_dir(d["alpha"]): d["eval"][name] for d in doses},
            }
            for name in eval_sets
        },
        "doses": doses,
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    (out / "surgery_report.json").write_text(json.dumps(report, indent=2))
    return report


def _family_means(groups: list[Group], margins: list[torch.Tensor]) -> dict:
    out: dict[str, list[float]] = {}
    for g, m in zip(groups, margins):
        out.setdefault(g.family, []).append(round(float(m.mean()), 4))
    return out


def _check_reference_tokenizer(tok, reference: str) -> None:
    """Margins at a position mean nothing if the reference renders the prompts differently."""
    from transformers import AutoTokenizer

    try:
        rt = AutoTokenizer.from_pretrained(reference)
    except Exception as e:  # noqa: BLE001
        logger.warning("surgery: reference tokenizer not readable (%s): assuming it matches the model's", e)
        return
    if len(rt) != len(tok) or (rt.chat_template or "") != (tok.chat_template or ""):
        raise SurgeryConfigError("reference: tokenizer vocabulary or chat template differs from the model's")


def _save(model, tok, path: Path, source: str, **cast) -> None:
    from transformers import AutoConfig

    from palingenesis.checkpoint import save_hf_model

    save_hf_model(model, tok, path, source_layout=True, **cast)
    saved = json.loads((path / "config.json").read_text()).get("model_type")
    want = AutoConfig.from_pretrained(source).model_type
    if saved != want:
        raise RuntimeError(f"saved model_type {saved!r} differs from the source's {want!r}")


# ---------------------------------------------------------------------------------------------------------- diagnostics


def span_coverage(
    model,
    tok,
    groups: list[Group],
    target: Group,
    ks: Sequence[int],
    params: list[torch.Tensor] | None = None,
    *,
    mode: str = "device",
    micro_batch: int = 4,
) -> dict:
    """Fraction of the target set's margin gradient inside the span of the first K group gradients, for each K:
    |P_K t| / |t| = sqrt(bᵀ A_K⁻¹ b) / |t| with A the groups' Gram matrix and b_i = <g_i, t> (tiny ridge for
    stability). High coverage with a low value for an unrelated decision = the groups represent this decision."""
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    A = gram_matrix(model, tok, list(groups) + [target], params, mode=mode, micro_batch=micro_batch)
    K = len(groups)
    tt = float(A[K, K])
    b = A[:K, K]
    out = {}
    for k in ks:
        k = min(int(k), K)
        Ak = A[:k, :k] + 1e-9 * torch.eye(k, dtype=torch.float64) * A.diag()[:k].mean()
        out[k] = math.sqrt(max(float(b[:k] @ torch.linalg.solve(Ak, b[:k])), 0.0) / tt) if tt > 0 else 0.0
    nrm = torch.sqrt(torch.clamp(A.diag()[:K], min=1e-300))
    C = A[:K, :K] / nrm[:, None] / nrm[None, :]
    return {"coverage": out, "group_cos_mean": float((C.sum() - K) / (K * (K - 1))) if K > 1 else 1.0}


# ---------------------------------------------------------------------------------------------------------- CLI


def _cli_device(arg):
    return torch.device(arg or ("cuda" if torch.cuda.is_available() else "cpu"))


def _cli_dtype(arg, device):
    return DTYPES[arg] if arg else (torch.bfloat16 if device.type == "cuda" else torch.float32)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="pgs surgery", description="margin surgery (palingenesis.surgery)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run margin surgery from a YAML config")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key (repeatable)")
    p.add_argument("--overwrite", action="store_true", help="replace a non-empty output_dir")

    def model_opts(p):
        p.add_argument("--decision", required=True, help="decision token (a single token)")
        p.add_argument("--micro-batch", type=int, default=4)
        p.add_argument("--max-side-tokens", type=int, default=None)
        p.add_argument("--dtype", choices=sorted(DTYPES), default=None, help="default: bf16 on CUDA, fp32 on CPU")
        p.add_argument("--device", default=None)
        p.add_argument("--limit", type=int, default=None, help="use the first N twins")

    p = sub.add_parser("margins", help="mean / SE / greedy share of a decision's margin per model (a diagnostic)")
    p.add_argument("--model", action="append", required=True, help="repeatable")
    p.add_argument("--twins", required=True)
    p.add_argument("--prefix-side", "--side", dest="side", choices=("pos", "neg", "both"), default="both")
    p.add_argument("--json", action="store_true")
    model_opts(p)
    p = sub.add_parser("span", help="fraction of a target set's margin gradient inside the span of K group gradients")
    p.add_argument("--model", required=True)
    p.add_argument("--groups", required=True, help="twins giving the groups")
    p.add_argument("--targets", required=True, help="twins giving the target set")
    p.add_argument("--k", default="1,4,16", help="comma-separated K values (nested: the first K groups)")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--side", choices=SIDES, default="pos")
    p.add_argument("--target-side", choices=SIDES, default="pos")
    p.add_argument("--target-decision", default=None, help="default: --decision")
    p.add_argument("--gram", choices=("device", "cpu", "recompute"), default="device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    model_opts(p)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if a.cmd == "run":
            cfg = SurgeryConfig.from_yaml(a.config, a.set)
            if a.overwrite:
                cfg.overwrite = True
            report = run_surgery(cfg)
            first, last = report["iterations"][0], report["iterations"][-1]
            print(
                f"surgery: {len(report['iterations']) - 1} step(s), gap {first['gap_mean']:.3f} -> "
                f"{last['gap_mean']:.3f} ({report['stop_reason']}), edit {report['edit']['norm']:.4g}"
                + (f" = {report['edit']['rel_reference']:.4f} of |θ0 - ref|" if report["edit"]["rel_reference"] else "")
                + f" -> {cfg.output_dir}"
            )
            return 0
        from transformers import AutoTokenizer

        device = _cli_device(a.device)
        dtype = _cli_dtype(a.dtype, device)
        if a.cmd == "margins":
            rows = []
            for path in a.model:
                tok = AutoTokenizer.from_pretrained(path)
                ids, dropped = load_sides(tok, a.twins, a.side, max_side_tokens=a.max_side_tokens)
                ids = ids[: a.limit] if a.limit else ids
                model = _load(path, dtype, device)
                stats = eval_margins(
                    MarginReader(model, tok, device=device),
                    {"m": {"side": a.side, "ids": ids, "token_ids": decision_ids(tok, a.decision)}},
                    a.micro_batch,
                )["m"]
                rows.append({"model": path, "dropped": dropped, **stats})
                del model
                _empty_cache(device)
            if a.json:
                print(json.dumps(rows, indent=2))
            else:
                _print_margins(rows, a.side)
            return 0
        # span
        tok = AutoTokenizer.from_pretrained(a.model)
        ks = [int(k) for k in a.k.split(",") if k.strip()]
        ids, _ = load_sides(tok, a.groups, a.side, max_side_tokens=a.max_side_tokens)
        ids = ids[: a.limit] if a.limit else ids
        n_groups = min(max(ks), len(ids) // a.group_size)
        if n_groups < 1:
            raise SurgeryConfigError(f"span: fewer than --group-size {a.group_size} sides in {a.groups}")
        groups = make_groups("groups", ids, decision_ids(tok, a.decision), n_groups, a.group_size, seed=a.seed)
        tids, _ = load_sides(tok, a.targets, a.target_side, max_side_tokens=a.max_side_tokens)
        tids = tids[: a.limit] if a.limit else tids
        target = Group("target", 0, tids, decision_ids(tok, a.target_decision or a.decision))
        model = _load(a.model, dtype, device)
        res = span_coverage(model, tok, groups, target, ks, mode=a.gram, micro_batch=a.micro_batch)
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"group cosine (mean) {res['group_cos_mean']:.3f}")
            for k, v in res["coverage"].items():
                print(f"K={k:<4d} coverage {v:.3f}")
        return 0
    except (SurgeryConfigError, ValueError) as e:
        print(f"pgs surgery: {e}", file=sys.stderr)
        return 1


def _print_margins(rows: list[dict], side: str) -> None:
    if side == "both":
        print(
            f"{'model':40s} {'n':>5s} {'pos':>16s} {'neg':>16s} {'common':>8s} {'diff':>8s} {'greedy+':>7s} {'greedy-':>7s}"
        )
        for r in rows:
            p, n = r["pos"], r["neg"]
            print(
                f"{r['model'][-40:]:40s} {p['n']:5d} {p['mean']:8.3f} ±{p['se']:6.3f} {n['mean']:8.3f} ±{n['se']:6.3f} "
                f"{r['common']:8.3f} {r['differential']:8.3f} {p['greedy']:7.3f} {n['greedy']:7.3f}"
            )
    else:
        print(f"{'model':40s} {'n':>5s} {'mean':>8s} {'se':>7s} {'greedy':>7s}")
        for r in rows:
            print(f"{r['model'][-40:]:40s} {r['n']:5d} {r['mean']:8.3f} {r['se']:7.3f} {r['greedy']:7.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
