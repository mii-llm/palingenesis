"""The off-policy distillation step: teacher targets for a micro-batch, and the loss against them.

Per micro-batch the trainer calls

    targets = distiller.targets(batch, input_ids, labels, forward_kwargs, ...)  # teachers' side
    with distiller.capturing():                                                # student features
        hidden = student backbone(...)
    loss = distiller.loss(hidden, targets, head, denom)                        # chunked over rows

The loss per trained token t (labels shifted: position t predicts token t+1) is

    kd_weight * sum_i w_i D(p_i(.|t) || q(.|t)) / sum_i w_i   +   ce_weight * -log q(gold_t)

over the teachers i covering t (routing by data source, cache hits), then summed with the data's
per-token weights and divided by the trainer's global token count, like the SFT loss. Feature terms
(hidden-state matching) and plugin terms are added on top.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field, replace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from palingenesis.distill import losses as L
from palingenesis.distill.cache import CacheLookup, TopKCache, batch_field
from palingenesis.distill.chunked import chunked_rows_loss, rows_per_slice
from palingenesis.distill.config import DistillConfig, TeacherSpec, parse_teachers
from palingenesis.distill.features import ModuleCapture, model_hidden_size, probe_output_dim
from palingenesis.distill.teachers import OnlineTeacher
from palingenesis.distill.views import TeacherView, input_field, positions_field
from palingenesis.hooks import FeatureSpec

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
AUX_MODULE = "distill_aux"  # student submodule holding learnable projections (not exported)


class MissingTeacherError(RuntimeError):
    """Trained tokens without any teacher while distill.missing is "error"."""


# ─────────────────────────────────────────────────────────────── targets


@dataclass
class TeacherRows:
    """One teacher's targets at the micro-batch's trained rows (aligned with TeacherTargets.rows).

    Online teachers have `hidden` [N, H_t] (its final hidden states; logits are projected per slice
    inside the loss); cached teachers have `ids` / `lp` [N, k] and `gold_lp` [N]. `cover` [N] marks
    the rows this teacher supervises; `features` holds its captured feature rows {name: [N, D]}."""

    name: str
    weight: float
    cover: Tensor
    hidden: Tensor | None = None
    ids: Tensor | None = None
    lp: Tensor | None = None
    gold_lp: Tensor | None = None
    features: dict = field(default_factory=dict)

    def subset(self, index: Tensor) -> TeacherRows:
        def pick(t):
            return None if t is None else t.index_select(0, index.to(t.device))

        return replace(
            self,
            cover=pick(self.cover),
            hidden=pick(self.hidden),
            ids=pick(self.ids),
            lp=pick(self.lp),
            gold_lp=pick(self.gold_lp),
            features={k: pick(v) for k, v in self.features.items()},
        )


@dataclass
class TeacherTargets:
    """The teachers' targets for one micro-batch.

    rows [N]: flat indices of the trained positions in the student's hidden states (B*T);
    gold [N]: the tokens they predict; token_weights [N]: per-token loss weights (the data's, which a
    plugin's transform_targets may change); teachers: name -> TeacherRows."""

    rows: Tensor
    gold: Tensor
    token_weights: Tensor
    teachers: dict[str, TeacherRows]
    seq_len: int = 0

    def covered(self) -> Tensor:
        cover = torch.zeros_like(self.gold, dtype=torch.bool)
        for t in self.teachers.values():
            cover |= t.cover.to(cover.device)
        return cover

    def subset(self, index: Tensor, rows: Tensor) -> TeacherTargets:
        """The targets at `index` (positions into rows), with new flat row indices `rows`."""
        return TeacherTargets(
            rows=rows,
            gold=self.gold.index_select(0, index),
            token_weights=self.token_weights.index_select(0, index),
            teachers={k: v.subset(index) for k, v in self.teachers.items()},
            seq_len=self.seq_len,
        )


# ─────────────────────────────────────────────────────────────── the distiller


class Distiller:
    """Teachers, feature captures and the loss of off-policy distillation for one training run."""

    def __init__(
        self,
        config: DistillConfig,
        specs: list[TeacherSpec],
        online: dict[str, OnlineTeacher],
        caches: dict[str, TopKCache],
        source_index: dict[str, int] | None = None,
        features: list[FeatureSpec] | None = None,
        student_vocab: int = 0,
        max_seq_length: int = 8192,
    ):
        self.config = config
        self.max_seq_length = max_seq_length  # the student's; a teacher view's default limit is twice it
        self.specs = {s.name: s for s in specs}
        self.online = online
        self.caches = caches
        self.source_index = source_index or {}
        self.features = list(features or [])
        self.student_vocab = student_vocab
        self.temperature = config.temperature
        self.t2 = config.temperature**2 if config.temperature_scale else 1.0
        self._student_capture: ModuleCapture | None = None
        self._teacher_capture: dict[str, ModuleCapture] = {}
        self._feature_via: dict[str, str] = {}
        self._lenses: dict[str, tuple] = {}
        self.projections: nn.ModuleDict | None = None
        self.metrics: dict[str, float] = {}
        self._counts: dict[str, float] = {}
        for name, cache in caches.items():
            if config.topk_mode == "tail" and abs(cache.meta.temperature - config.temperature) > 1e-6:
                raise ValueError(
                    f"distill.teachers.{name}: the cache holds log-probs at temperature {cache.meta.temperature}, "
                    f"distill.temperature is {config.temperature}. With topk_mode tail the teacher's tail mass "
                    "is only known at the cache's temperature: rebuild the cache at this temperature, or use "
                    "topk_mode: renorm."
                )

    # -- data pipeline

    def routes(self) -> dict[str, set[int]]:
        """teacher -> data source indices it supervises (teachers without `sources` are absent)."""
        out = {}
        for name, spec in self.specs.items():
            if spec.sources:
                out[name] = {self.source_index[s] for s in spec.sources}
        return out

    @property
    def needs_source_tags(self) -> bool:
        return any(s.sources for s in self.specs.values())

    def batch_transform(self) -> CacheLookup | None:
        """The collate-time lookup of the cached teachers' targets (None without cached teachers)."""
        if not self.caches:
            return None
        return CacheLookup(self.caches, {k: v for k, v in self.routes().items() if k in self.caches})

    def teacher_views(self) -> list:
        """[(TeacherView, tokenizer, max length)] of the online teachers that have a view."""
        out = []
        for name, teacher in self.online.items():
            spec = teacher.spec
            if spec.has_view:
                limit = spec.max_seq_length or 2 * self.max_seq_length
                out.append((TeacherView(spec, teacher.bridge), teacher.tokenizer, limit))
        return out

    # -- features

    def attach_features(self, student: nn.Module, device) -> nn.ModuleDict | None:
        """Hooks on the student and teacher modules of every FeatureSpec, and learnable projections
        (added to the student as `distill_aux`, before FSDP wraps it: they train and checkpoint with it
        and are dropped from exported models)."""
        if not self.features:
            return None
        online = [n for n, t in self.online.items()]
        student_targets, projections = {}, nn.ModuleDict()
        teacher_targets: dict[str, dict] = {}
        for i, spec in enumerate(self.features):
            spec.name = spec.name or f"f{i}"
            via = getattr(spec, "via", None) or (online[0] if online else None)
            if via not in self.online:
                raise ValueError(f"feature {spec.name}: needs an online teacher (have {online or 'none'})")
            self._feature_via[spec.name] = via
            student_targets[spec.name] = spec.student
            teacher_targets.setdefault(via, {})[spec.name] = spec.teacher
            if spec.loss == "lens_kl":  # each side is read through its own final norm and head: no projection
                teacher = self.online[via]
                self._lenses[spec.name] = (Lens(student), Lens(teacher.model, teacher.device))
                continue
            s_dim = _module_width(student, spec.student, device)
            t_dim = _module_width(self.online[via].model, spec.teacher, self.online[via].device)
            if spec.projection == "linear" or (spec.projection == "auto" and s_dim != t_dim):
                proj = nn.Linear(s_dim, t_dim, bias=False)
                # the same initial projection on every rank and in every run (not the rank-seeded RNG)
                with torch.no_grad():
                    gen = torch.Generator().manual_seed(1_000_003 + i)
                    proj.weight.copy_(torch.randn(proj.weight.shape, generator=gen) * s_dim**-0.5)
                projections[_key(spec.name)] = proj
            elif s_dim != t_dim:
                raise ValueError(f"feature {spec.name}: widths differ ({s_dim} vs {t_dim}) and projection is none")
        self._student_capture = ModuleCapture(student, student_targets)
        for via, targets in teacher_targets.items():
            self._teacher_capture[via] = ModuleCapture(self.online[via].model, targets)
        if len(projections):
            dtype = next(student.parameters()).dtype
            projections.to(device=device, dtype=dtype)
            setattr(student, AUX_MODULE, projections)
            self.projections = projections
        return self.projections

    @contextlib.contextmanager
    def capturing(self):
        """Record the student's feature modules during the forward inside this block."""
        if self._student_capture is None:
            yield
            return
        with self._student_capture.capturing():
            yield

    # -- targets

    @torch.no_grad()
    def targets(
        self,
        batch: dict,
        input_ids: Tensor,
        labels: Tensor,
        forward_kwargs: dict,
        loss_weights: Tensor | None = None,
        teacher_inputs: dict | None = None,
        reshape=None,
        compute_dtype=torch.bfloat16,
        autocast: bool = True,
    ) -> TeacherTargets:
        """The teachers' targets at the trained positions of a micro-batch.

        input_ids / labels (shifted) / forward_kwargs: the student's inputs as the trainer feeds them
        (after packing / context-parallel sharding). `reshape(t)` maps a per-token batch tensor [B, T, ...]
        to the student's layout (the trainer's flattening / sharding); identity by default.
        teacher_inputs: per online teacher, its own (input_ids, forward kwargs) when they differ from the
        student's (a packed batch flattened for one model but not the other)."""
        reshape = reshape or (lambda t: t)
        flat_labels = labels.reshape(-1)
        rows = torch.nonzero(flat_labels != IGNORE_INDEX).flatten()
        gold = flat_labels.index_select(0, rows)
        if loss_weights is not None:
            weights = loss_weights.reshape(-1).index_select(0, rows).float()
        else:
            weights = torch.ones(rows.numel(), device=rows.device, dtype=torch.float32)
        seq_len = labels.shape[-1]
        sources = reshape(batch["source_ids"]).reshape(-1).index_select(0, rows) if "source_ids" in batch else None
        routes = self.routes()
        teachers: dict[str, TeacherRows] = {}
        for name, spec in self.specs.items():
            route = torch.ones_like(rows, dtype=torch.bool)
            if name in routes and sources is not None:
                allowed = torch.tensor(sorted(routes[name]), device=rows.device)
                route = torch.isin(sources, allowed)
            if name in self.caches:
                ids = reshape(batch[batch_field(name, "ids")])
                k = ids.shape[-1]
                ids = ids.reshape(-1, k).index_select(0, rows)
                lp = reshape(batch[batch_field(name, "lp")]).reshape(-1, k).index_select(0, rows)
                gold_lp = reshape(batch[batch_field(name, "gold")]).reshape(-1).index_select(0, rows)
                hit = reshape(batch[batch_field(name, "hit")]).reshape(-1).index_select(0, rows)
                teachers[name] = TeacherRows(
                    name, spec.weight, route & hit.bool(), ids=ids, lp=lp.float(), gold_lp=gold_lp.float()
                )
            else:
                teachers[name] = self._online_rows(
                    name, batch, input_ids, rows, route, forward_kwargs, teacher_inputs, compute_dtype, autocast
                )
        targets = TeacherTargets(rows=rows, gold=gold, token_weights=weights, teachers=teachers, seq_len=seq_len)
        if self.config.missing == "error" and self.config.kd_weight > 0:
            uncovered = (~targets.covered()).sum().reshape(1).to(torch.long)

            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                # every rank checks at every micro-batch and all raise together: a rank raising alone
                # would leave the others waiting in the next FSDP collective
                dist.all_reduce(uncovered, op=dist.ReduceOp.MAX)
            uncovered = int(uncovered.item())
            if uncovered:
                raise MissingTeacherError(
                    f"{uncovered} of {rows.numel()} trained tokens of a micro-batch (on the worst rank) have no teacher target (a cache "
                    "built from other data or masking, or a data source routed to no teacher). Rebuild the cache "
                    "with the training config, or set distill.missing: skip (those tokens then get only the CE term)."
                )
        return targets

    def _online_rows(
        self, name, batch, input_ids, rows, route, forward_kwargs, teacher_inputs, compute_dtype, autocast
    ) -> TeacherRows:
        teacher = self.online[name]
        spec = teacher.spec
        view_ids = batch.get(input_field(name))
        capture = self._teacher_capture.get(name)
        if view_ids is not None:
            # context distillation: the teacher reads its own rendering (never packed: validated)
            t_ids = view_ids
            kwargs = {"attention_mask": None}
            positions = batch[positions_field(name)].reshape(-1).index_select(0, rows)
            t_rows = (rows // input_ids.shape[-1]) * t_ids.shape[-1] + positions.clamp(min=0)
            cover = route & (positions >= 0)
        else:
            t_ids, kwargs = (teacher_inputs or {}).get(name, (input_ids, forward_kwargs))
            t_rows = rows
            cover = route
        ctx = capture.capturing() if capture is not None else contextlib.nullcontext()
        with ctx:
            hidden = teacher.hidden_states(t_ids, kwargs, compute_dtype, autocast)
        t_rows = t_rows.to(hidden.device)
        h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, t_rows)
        feats = {}
        if capture is not None:
            for key, out in capture.take().items():
                feats[key] = out.reshape(-1, out.shape[-1]).index_select(0, t_rows.to(out.device))
        del hidden
        return TeacherRows(name, spec.weight, cover.to(rows.device), hidden=h, features=feats)

    def release(self) -> None:
        for teacher in self.online.values():
            teacher.release()

    # -- loss

    def _slice_fn(self, targets: TeacherTargets, stats: dict):
        cfg = self.config
        temperature = self.temperature
        names = list(targets.teachers)

        def fn(logits: Tensor, a: int, b: int):
            gold = targets.gold[a:b]
            kd_sum = logits.new_zeros(b - a)
            w_sum = logits.new_zeros(b - a)
            out_stats: dict[str, Tensor | float] = {}
            if cfg.kd_weight > 0:
                for name in names:
                    t = targets.teachers[name]
                    cover = t.cover[a:b]
                    n = int(cover.sum())
                    if n == 0:
                        continue
                    sel = torch.nonzero(cover).flatten()
                    z = logits.index_select(0, sel)
                    if t.hidden is not None:
                        teacher = self.online[name]
                        tl = teacher.logits(t.hidden[a:b].index_select(0, sel.to(t.hidden.device)))
                        lp, lq, mask = self._dense(z, tl, teacher)
                    else:
                        cache = self.caches[name]
                        lp, lq, mask = L.sparse_support(
                            z,
                            t.ids[a:b].index_select(0, sel),
                            t.lp[a:b].index_select(0, sel),
                            gold.index_select(0, sel),
                            t.gold_lp[a:b].index_select(0, sel),
                            temperature,
                            cfg.topk_mode,
                            cache.meta.temperature,
                        )
                        if cfg.topk_mode == "tail":  # the teacher's mass on its stored support
                            out_stats[f"topk_mass/{name}"] = (lp[:, :-1].exp() * mask[:, :-1]).sum().detach()
                    d = L.divergence(lp, lq, mask, cfg.divergence, cfg.beta) * self.t2
                    kd_sum = kd_sum.index_add(0, sel, t.weight * d)
                    w_sum = w_sum.index_add(0, sel, torch.full_like(d, t.weight))
                    out_stats[f"kd_sum/{name}"] = d.detach().sum()
                    out_stats[f"kd_tokens/{name}"] = n
            kd = torch.where(w_sum > 0, kd_sum / w_sum.clamp(min=1e-12), torch.zeros_like(kd_sum))
            per_row = cfg.kd_weight * kd
            ce = L.gold_cross_entropy(logits, gold)  # always reported (the logits are here anyway)
            out_stats["ce_sum"] = ce.detach().sum()
            if cfg.ce_weight > 0:
                per_row = per_row + cfg.ce_weight * ce
            out_stats["kd_sum"] = (kd.detach() * (w_sum > 0)).sum()
            out_stats["kd_tokens"] = int((w_sum > 0).sum())
            out_stats["tokens"] = b - a
            return (targets.token_weights[a:b] * per_row).sum(), out_stats

        return fn

    def _dense(self, z: Tensor, teacher_logits: Tensor, teacher: OnlineTeacher):
        lq_full = F.log_softmax(z.float() / self.temperature, -1)
        size = teacher.bridge.shared_vocab_size
        swap = teacher.bridge.swap
        lq = lq_full[:, :size]
        if swap:
            src = torch.tensor(list(swap), device=z.device)
            dst = torch.tensor(list(swap.values()), device=z.device)
            lq = lq.index_copy(1, dst, torch.logaddexp(lq[:, dst], lq_full[:, src]))
        lp = F.log_softmax(teacher_logits.float() / self.temperature, -1)[:, :size]
        if lp.shape[-1] < size:  # a teacher head narrower than its tokenizer: pad with no mass
            lp = torch.nn.functional.pad(lp, (0, size - lp.shape[-1]), value=-1e30)
        return lp, lq, None

    def loss(
        self,
        hidden: Tensor,
        targets: TeacherTargets,
        head: nn.Module,
        denom: float,
        student_features: dict | None = None,
        progress: float = 0.0,
    ) -> tuple[Tensor, dict]:
        """The distillation loss of a micro-batch (differentiable; the head's gradients are already
        accumulated, the backbone's flow at the caller's backward)."""
        stats: dict[str, float] = {}
        slice_rows = rows_per_slice(int(head.weight.shape[0]), self.config.rows_per_slice)
        loss = chunked_rows_loss(hidden, targets.rows, head, self._slice_fn(targets, stats), denom, slice_rows, stats)
        feature_loss, feature_stats = self.feature_loss(targets, student_features, denom, progress)
        if feature_loss is not None:
            loss = loss + feature_loss
        stats.update(feature_stats)
        self._accumulate(stats)
        return loss, stats

    def student_feature_rows(self, targets: TeacherTargets) -> dict[str, Tensor]:
        """The student's captured feature outputs at the trained rows (with gradient), projected."""
        if self._student_capture is None:
            return {}
        out = {}
        for key, value in self._student_capture.take().items():
            rows = value.reshape(-1, value.shape[-1]).index_select(0, targets.rows.to(value.device))
            proj = (
                self.projections[_key(key)] if self.projections is not None and _key(key) in self.projections else None
            )
            out[key] = proj(rows) if proj is not None else rows
        return out

    def feature_pairs(self, targets: TeacherTargets, student_rows: dict[str, Tensor]) -> dict:
        """{name: (student [n, D], teacher [n, D])} at the rows the feature's teacher covers."""
        pairs = {}
        for spec in self.features:
            via = self._feature_via.get(spec.name)
            s = student_rows.get(spec.name)
            t_rows = targets.teachers.get(via)
            if s is None or t_rows is None or spec.name not in t_rows.features:
                continue
            cover = torch.nonzero(t_rows.cover).flatten()
            pairs[spec.name] = (
                s.index_select(0, cover.to(s.device)),
                t_rows.features[spec.name].index_select(0, cover.to(t_rows.features[spec.name].device)).to(s.device),
            )
        return pairs

    def feature_loss(self, targets, student_features, denom, progress) -> tuple[Tensor | None, dict]:
        if not self.features or student_features is None:
            return None, {}
        total, stats = None, {}
        pairs = self.feature_pairs(targets, student_features)
        for spec in self.features:
            if spec.name not in pairs:
                continue
            s, t = pairs[spec.name]
            via = self._feature_via[spec.name]
            cover = torch.nonzero(targets.teachers[via].cover).flatten()
            w = targets.token_weights.index_select(0, cover).to(s.device)
            weight = spec.weight_at(progress)
            if spec.loss == "lens_kl":
                term, per_row = self._lens_loss(spec, s, t, w, denom)
                term = weight * term
            else:
                per_row = L.feature_loss(s, t, spec.loss)
                term = weight * (w * per_row).sum() / denom
            total = term if total is None else total + term
            stats[f"feature_sum/{spec.name}"] = float(per_row.detach().sum())
            stats[f"feature_tokens/{spec.name}"] = float(per_row.numel())
        return total, stats

    def _lens_loss(self, spec, s: Tensor, t: Tensor, w: Tensor, denom: float) -> tuple[Tensor, Tensor]:
        """Logit-lens target: forward KL between the teacher's and the student's intermediate states read
        through each model's own final norm and output head (frozen for this term), a slice at a time."""
        student_lens, teacher_lens = self._lenses[spec.name]
        vocab = int(student_lens.head.weight.shape[0])
        per_row = torch.zeros(s.shape[0], device=s.device)
        temperature = self.temperature

        def fn(logits, a, b):
            with torch.no_grad():
                t_logits = teacher_lens(t[a:b].to(teacher_lens.device)).float().to(logits.device)
            size = min(logits.shape[-1], t_logits.shape[-1])
            lq = F.log_softmax(logits / temperature, -1)[:, :size]
            lp = F.log_softmax(t_logits / temperature, -1)[:, :size]
            d = L.divergence(lp, lq, None, "forward_kl") * self.t2
            per_row[a:b] = d.detach()
            return (w[a:b] * d).sum(), {}

        rows = torch.arange(s.shape[0], device=s.device)
        term = chunked_rows_loss(s, rows, student_lens, fn, denom, rows_per_slice(vocab, self.config.rows_per_slice))
        return term, per_row

    # -- SeCO (chunk-wise forward/backward over long sequences)

    def seco_loss_fn(self, targets: TeacherTargets, head: nn.Module):
        """loss_fn(hidden [B, L, H], lo, hi, denom) for seco.chunkwise_forward_backward: the loss of the
        trained rows inside the window [lo, hi) of every row."""
        seq = targets.seq_len
        r_b = targets.rows // seq
        r_t = targets.rows % seq

        def fn(hidden: Tensor, lo: int, hi: int, denom: float) -> Tensor:
            inside = torch.nonzero((r_t >= lo) & (r_t < hi)).flatten()
            local = r_b.index_select(0, inside) * (hi - lo) + (r_t.index_select(0, inside) - lo)
            sub = targets.subset(inside, local)
            stats: dict[str, float] = {}
            slice_rows = rows_per_slice(int(head.weight.shape[0]), self.config.rows_per_slice)
            out = chunked_rows_loss(hidden, sub.rows, head, self._slice_fn(sub, stats), denom, slice_rows, stats)
            if torch.is_grad_enabled():  # stage 2 sees each chunk once: count it there
                self._accumulate(stats)
            return out

        return fn

    # -- metrics

    def _accumulate(self, stats: dict) -> None:
        for k, v in stats.items():
            self._counts[k] = self._counts.get(k, 0.0) + float(v)

    def pop_metrics(self) -> dict[str, float]:
        """Means since the last call: train/kd_loss (per covered token), train/kd/<teacher>,
        train/gold_ce (CE on the gold tokens), train/feature/<name>, train/kd_coverage, train/topk_mass/<teacher>."""
        c, self._counts = self._counts, {}
        out = {}
        if c.get("kd_tokens"):
            out["train/kd_loss"] = c["kd_sum"] / c["kd_tokens"]
        if c.get("tokens"):
            out["train/gold_ce"] = c.get("ce_sum", 0.0) / c["tokens"]
            out["train/kd_coverage"] = c.get("kd_tokens", 0.0) / c["tokens"]
        for k, v in c.items():
            if k.startswith("kd_sum/"):
                name = k.split("/", 1)[1]
                if c.get(f"kd_tokens/{name}"):
                    out[f"train/kd/{name}"] = v / c[f"kd_tokens/{name}"]
                    if f"topk_mass/{name}" in c:
                        out[f"train/topk_mass/{name}"] = c[f"topk_mass/{name}"] / c[f"kd_tokens/{name}"]
            if k.startswith("feature_sum/"):
                name = k.split("/", 1)[1]
                if c.get(f"feature_tokens/{name}"):
                    out[f"train/feature/{name}"] = v / c[f"feature_tokens/{name}"]
        return out


class Lens(nn.Module):
    """A model's final norm and output head applied to intermediate hidden states, with the parameters
    detached (a logit-lens target trains the layers that produce the states, not the head)."""

    _NORMS = ("norm", "final_layernorm", "ln_f", "final_layer_norm")

    def __init__(self, model: nn.Module, device=None):
        from palingenesis.logits import backbone_of, output_head

        super().__init__()
        backbone = backbone_of(model)
        norm = next(
            (getattr(backbone, n) for n in self._NORMS if isinstance(getattr(backbone, n, None), nn.Module)), None
        )
        if norm is None:
            raise ValueError(
                f"lens_kl: cannot find the final norm of {type(model).__name__} ({', '.join(self._NORMS)})"
            )
        # plain attributes, not submodules: the lens must not re-register the model's parameters
        self.__dict__["norm"] = norm
        self.__dict__["head"] = output_head(model)
        self.device = device

    def forward(self, hidden: Tensor) -> Tensor:
        from torch.func import functional_call

        norm_params = {k: v.detach() for k, v in self.norm.named_parameters()}
        head_params = {k: v.detach() for k, v in self.head.named_parameters()}
        return functional_call(self.head, head_params, (functional_call(self.norm, norm_params, (hidden,)),))


def _key(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _module_width(model: nn.Module, where, device) -> int:
    from palingenesis.distill.features import decoder_layers

    if isinstance(where, int) or (isinstance(where, str) and where.lstrip("-").isdigit()):
        if decoder_layers(model):
            return model_hidden_size(model)
    return probe_output_dim(model, where, device)


# ─────────────────────────────────────────────────────────────── building


def build_distiller(
    config,
    tokenizer,
    device,
    mesh=None,
    extra_features: list[FeatureSpec] | None = None,
    teacher_models: dict | None = None,
) -> Distiller:
    """Teachers (online models loaded, caches opened and verified against the student's tokenizer)
    and feature specs for `config` (a palingenesis Config with distill.enabled).

    teacher_models: name -> (model, tokenizer) to use instead of loading (tests, notebooks)."""
    from palingenesis.data import source_name

    d = config.distill
    specs = parse_teachers(d.teachers)
    teacher_models = teacher_models or {}
    online, caches = {}, {}
    for spec in specs:
        if spec.cache:
            cache = TopKCache(spec.cache)
            cache.verify_tokenizer(tokenizer)
            caches[spec.name] = cache
            logger.info(
                "Teacher %s: cache %s (top-%d, %d documents, built from %s)",
                spec.name,
                spec.cache,
                cache.k,
                len(cache),
                cache.meta.teacher,
            )
        else:
            model, tok = teacher_models.get(spec.name, (None, None))
            online[spec.name] = OnlineTeacher(
                spec,
                tokenizer,
                device,
                attn_implementation=config.model.attn_implementation,
                trust_remote_code=config.model.trust_remote_code,
                mesh=mesh,
                parallel_config=config.parallel,
                bf16=config.train.bf16,
                model=model,
                tokenizer=tok,
            )
    sources = {source_name(s, i): i for i, s in enumerate(config.data.sources)}
    features = []
    for f in d.features:
        spec = FeatureSpec(
            student=f["student"],
            teacher=f["teacher"],
            weight=f.get("weight", 1.0),
            loss=f.get("loss", "mse"),
            projection=f.get("projection", "auto"),
            name=f.get("name", ""),
            via=f.get("via", ""),
        )
        features.append(spec)
    features.extend(extra_features or [])
    return Distiller(
        d,
        specs,
        online,
        caches,
        sources,
        features,
        student_vocab=len(tokenizer),
        max_seq_length=config.data.max_seq_length,
    )
