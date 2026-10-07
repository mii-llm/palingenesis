"""Muon (momentum orthogonalized by Newton-Schulz) for hidden weight matrices, AdamW for everything else.

One optimizer, two kinds of parameter groups (``use_muon`` True / False), so LR schedulers, checkpoints and grad
clipping see a single torch optimizer. Design follows three sources:

- Muon (K. Jordan, 2024): the update of a 2-D weight is the orthogonalized (Nesterov) momentum, i.e. the nearest
  semi-orthogonal matrix U V^T of the momentum's SVD, computed by an odd matrix polynomial iteration.
- Moonlight / "Muon is Scalable for LLM Training" (Liu et al., 2025): scale each Muon update by
  0.2 * sqrt(max(rows, cols)) so its RMS matches AdamW's (~0.2), which lets Muon matrices share the AdamW learning
  rate and decoupled weight decay (``scale="moonlight"``, the default). ``scale="spectral"`` is Jordan's
  sqrt(max(1, rows / cols)), which needs a larger LR.
- Gram Newton-Schulz (T. Dao et al., 2026): for a rectangular n x m matrix (n <= m) iterate on the n x n Gram matrix
  X X^T instead of X, with one restart (re-forming the Gram matrix from Q X) after iteration 2 to stop half-precision
  negative eigenvalues from diverging; only two n x m products remain (about half the FLOPs of standard
  Newton-Schulz at aspect ratio 4). Coefficients: Polar Express (Amsel et al., arXiv 2505.16932) with a 1.05 safety
  factor, computed in float16 after Frobenius normalisation. Square matrices use standard Newton-Schulz.

Same-shaped matrices are orthogonalized together as one batch (one batched matmul per iteration instead of one per
layer). FSDP2 (DTensor) parameters are gathered whole for the orthogonalization (Newton-Schulz needs the full matrix),
and each rank keeps its own shard of the result.

Parameter routing (``muon_param_groups``): Muon takes 2-D hidden weights whose smaller side is at least ``min_dim``;
embeddings, the output head (tied or not), 1-D parameters (norms, biases, gates' log-scales), convolution kernels and
very thin matrices (e.g. rank-16 gate projections, which orthogonalization would force to all-ones singular values)
go to AdamW. Note (Moonlight): fine-tuning a model pre-trained with AdamW using Muon is an optimizer mismatch; measure
it against AdamW before adopting it for SFT/RL.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_POLAR_EXPRESS = [  # Amsel et al. 2025, arXiv 2505.16932 (5 iterations)
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
_SAFETY = 1.05  # roundoff margin for half precision (Gram Newton-Schulz uses 1.05 instead of 1.02)
POLAR_EXPRESS_COEFFICIENTS = [(a / _SAFETY, b / _SAFETY**3, c / _SAFETY**5) for a, b, c in _POLAR_EXPRESS]
JORDAN_COEFFICIENTS = [(3.4445, -4.7750, 2.0315)] * 5  # the original Muon quintic


class _TorchOps:
    """Dense torch GEMMs (cuBLAS / CPU): A100 and any GPU or CPU."""

    name = "torch"

    @staticmethod
    def sym_mm(A, B):  # A @ B where the result is symmetric
        return A @ B

    @staticmethod
    def sym_baddbmm(C, A, B, alpha=1.0, beta=1.0):  # beta * C + alpha * A @ B, symmetric result
        return torch.baddbmm(C, A, B, beta=beta, alpha=alpha)

    @staticmethod
    def mm(A, B):
        return A @ B

    @staticmethod
    def mm_add(C, A, B, beta=1.0):  # beta * C + A @ B
        return torch.baddbmm(C, A, B, beta=beta)


class _QuackOps:
    """quack (Dao-AILab, Apache-2.0) CuTeDSL GEMMs for Hopper / Blackwell (sm90, sm100; RTX 50): the symmetric ones
    compute one triangle of tiles and mirror it, so the n x n products of Newton-Schulz cost about half."""

    name = "quack"

    def __init__(self):
        from quack.gemm_interface import gemm, gemm_add, gemm_symmetric

        self._gemm, self._add, self._sym = gemm, gemm_add, gemm_symmetric

    def sym_mm(self, A, B):
        return self._sym(A, B)

    def sym_baddbmm(self, C, A, B, alpha=1.0, beta=1.0):
        return self._sym(A, B, C=C, alpha=alpha, beta=beta)

    def mm(self, A, B):
        return self._gemm(A, B, tuned=False)

    def mm_add(self, C, A, B, beta=1.0):
        return self._add(A, B, C=C, beta=beta, tuned=False)


QUACK_MIN_SIDE = 256  # quack's symmetric kernels tile at 256: smaller matrices stay on torch


def _quack_ops():
    """The quack backend when installed and the GPU is sm90 or newer, else None."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return None
    try:
        return _QuackOps()
    except Exception:  # noqa: BLE001 — not installed / unsupported toolkit: torch backend
        return None


def newton_schulz(X: Tensor, coefficients=POLAR_EXPRESS_COEFFICIENTS, eps: float = 1e-7, dtype=None, ops=_TorchOps) -> Tensor:
    """Standard Newton-Schulz on a batch (b, n, m): X <- a X + (b A + c A^2) X with A = X X^T (wide orientation)."""
    dtype = dtype or (torch.float16 if X.is_cuda else torch.float32)
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X.float()
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + eps)).to(dtype).contiguous()
    for a, b, c in coefficients:
        A = ops.sym_mm(X, X.mT)
        B = ops.sym_baddbmm(A, A, A, alpha=c, beta=b)
        X = ops.mm_add(X, B, X, beta=a)  # a X + B X
    return X.mT if tall else X


def gram_newton_schulz(
    X: Tensor, coefficients=POLAR_EXPRESS_COEFFICIENTS, restarts=(2,), eps: float = 1e-7, dtype=None, ops=_TorchOps
) -> Tensor:
    """Gram Newton-Schulz on a batch (b, n, m): the same polynomial as newton_schulz, applied through the n x n Gram
    matrix R = X X^T (n = the smaller side): Z_t = b R + c R^2, Q <- Q (Z_t + a I), R <- R (Z_t + a I)^2, output Q X.
    ``restarts``: iterations at which X <- Q X and R, Q are re-formed (stability in half precision). Every n x n
    product is symmetric (all are polynomials in R), so a symmetric-GEMM backend computes half of each."""
    dtype = dtype or (torch.float16 if X.is_cuda else torch.float32)
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X.float()
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + eps)).to(dtype).contiguous()
    restarts = set(restarts)
    n = X.size(-2)
    eye = torch.eye(n, device=X.device, dtype=dtype).expand(X.size(0), n, n)
    R = ops.sym_mm(X, X.mT)
    Q = None
    last = len(coefficients) - 1
    for i, (a, b, c) in enumerate(coefficients):
        if i in restarts and i != 0:
            X = ops.mm(Q, X)
            R = ops.sym_mm(X, X.mT)
            Q = None
        Z = ops.sym_baddbmm(R, R, R, alpha=c, beta=b)  # b R + c R^2
        # Q (Z + a I) = a Q + Q Z, written without adding a*I to Z (keeps precision in fp16)
        Q = Z + a * eye if Q is None else ops.sym_baddbmm(Q, Q, Z, beta=a)
        if i < last and (i + 1) not in restarts:
            RZ = ops.sym_baddbmm(R, R, Z, beta=a)  # R (Z + a I)
            R = ops.sym_baddbmm(RZ, Z, RZ, beta=a)  # (Z + a I) R (Z + a I)
    X = ops.mm(Q, X)
    return X.mT if tall else X


_COMPILED: dict = {}
_COMPILE_FAILED: set = set()


def _kernel(method: str, ops, compile_: bool, shape: tuple):
    """The Newton-Schulz function for (method, backend, batch shape). Compiled functions are kept one per shape (each
    a separate torch.compile object with its own graph): a model's handful of matrix shapes compile once each and are
    reused every step, and no single function runs into torch's recompile limit (one shared function with
    fullgraph=True failed hard at the 9th shape). torch.compile fuses the normalisation, casts and scalar epilogues
    around the GEMMs."""
    fn = gram_newton_schulz if method == "gram" else newton_schulz

    def run(X, fn=fn, ops=ops):
        return fn(X, POLAR_EXPRESS_COEFFICIENTS, ops=ops)

    if not compile_ or (method, ops.name) in _COMPILE_FAILED:
        return run
    key = (method, ops.name, shape)
    if key not in _COMPILED:
        _COMPILED[key] = torch.compile(run, fullgraph=True, dynamic=False)
    compiled = _COMPILED[key]

    def safe(X):
        try:
            return compiled(X)
        except Exception as e:  # noqa: BLE001 — compilation must never stop training: fall back to eager
            logger.warning("Muon: compiling %s Newton-Schulz (%s) failed, running eager from now on: %s", method,
                           ops.name, str(e)[:200])
            _COMPILE_FAILED.add((method, ops.name))
            return run(X)

    return safe


def orthogonalize(G: Tensor, method: str = "gram", coefficients=None, dtype=None, backend: str = "auto",
                  compile: bool = False) -> Tensor:
    """Approximate U V^T of a batch of matrices (b, n, m) or one matrix (n, m); returns G's dtype and shape.

    backend: "torch" (dense GEMMs, any device), "quack" (symmetric CuTeDSL kernels, sm90+; needs quack-kernels),
    "auto" (quack on sm90+ when installed and both sides are >= QUACK_MIN_SIDE, else torch). Square matrices use
    standard Newton-Schulz (no Gram saving at aspect ratio 1)."""
    if method not in ("gram", "standard"):
        raise ValueError(f"unknown Newton-Schulz method {method!r} (gram or standard)")
    one = G.ndim == 2
    X = G.unsqueeze(0) if one else G
    ops = _TorchOps
    if backend in ("auto", "quack") and X.is_cuda and min(X.shape[-2:]) >= QUACK_MIN_SIDE:
        ops = _quack_ops() or _TorchOps
        if backend == "quack" and ops is _TorchOps:
            raise RuntimeError("Muon backend quack: needs quack-kernels and an sm90+ GPU (H100, B200/B300, RTX 50)")
    use = "gram" if method == "gram" and X.size(-2) != X.size(-1) else "standard"
    if coefficients is not None or dtype is not None:  # explicit settings: the uncompiled reference path
        fn = gram_newton_schulz if use == "gram" else newton_schulz
        out = fn(X, coefficients or POLAR_EXPRESS_COEFFICIENTS, dtype=dtype, ops=ops)
    else:
        out = _kernel(use, ops, compile and X.is_cuda, tuple(X.shape))(X)
    out = out.to(G.dtype)
    return out.squeeze(0) if one else out


def _update_scale(shape, scale: str) -> float:
    rows, cols = shape[-2], shape[-1]
    if scale == "moonlight":
        return 0.2 * math.sqrt(max(rows, cols))
    if scale == "spectral":
        return math.sqrt(max(1.0, rows / cols))
    raise ValueError(f"unknown Muon update scale {scale!r} (moonlight or spectral)")


def _full(t: Tensor) -> Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - old torch
        return t
    return t.full_tensor() if isinstance(t, DTensor) else t


def _like(full: Tensor, ref: Tensor) -> Tensor:
    """The local part of `full` laid out as `ref` (a DTensor shard, or a plain tensor)."""
    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
    except ImportError:  # pragma: no cover
        return full
    if isinstance(ref, DTensor):
        return distribute_tensor(full, ref.device_mesh, ref.placements, src_data_rank=None)
    return full


class MuonAdamW(torch.optim.Optimizer):
    """Muon on parameter groups with ``use_muon=True``, AdamW on the others (decoupled weight decay on both)."""

    def __init__(
        self,
        param_groups,
        lr: float = 2e-5,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_method: str = "gram",
        scale: str = "moonlight",
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        backend: str = "auto",
        compile: bool = False,
        distribute: bool = True,
    ):
        """backend: Newton-Schulz GEMMs ("auto": quack's symmetric kernels on sm90+ when installed, torch otherwise);
        compile: torch.compile the Newton-Schulz function (CUDA only); distribute: under torch.distributed, each rank
        orthogonalizes 1/world of the matrices and the results are all-gathered (instead of every rank computing all)."""
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_method=ns_method,
                        scale=scale, betas=betas, eps=eps, use_muon=False)
        super().__init__(param_groups, defaults)
        self.backend, self.compile, self.distribute = backend, compile, distribute

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_group(group)
            else:
                self._adamw_group(group)
        return loss

    def _muon_group(self, group):
        lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
        batches = defaultdict(list)
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(mom).add_(g)
            u = g.add(buf, alpha=mom) if group["nesterov"] else buf
            full = _full(u)
            batches[(tuple(full.shape), full.dtype, full.device)].append((p, full))
        for (shape, _, _), items in batches.items():
            ortho = self._orthogonalize_batch([u for _, u in items], group["ns_method"])
            s = _update_scale(shape, group["scale"])
            for (p, _), o in zip(items, ortho.unbind(0)):
                o = _like(o.to(p.dtype), p.grad)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(o, alpha=-lr * s)

    def _orthogonalize_batch(self, updates: list[Tensor], method: str) -> Tensor:
        """Orthogonalize same-shaped full matrices; with several ranks, each takes a contiguous slice of the batch and an
        all-gather returns the whole (every rank holds every full update, so ownership needs no communication)."""
        import torch.distributed as dist

        kw = dict(method=method, backend=self.backend, compile=self.compile)
        world = dist.get_world_size() if self.distribute and dist.is_available() and dist.is_initialized() else 1
        if world == 1 or len(updates) == 1:
            return orthogonalize(torch.stack(updates), **kw)
        per = -(-len(updates) // world)
        rank = dist.get_rank()
        mine = updates[rank * per : (rank + 1) * per]
        local = torch.zeros((per, *updates[0].shape), dtype=updates[0].dtype, device=updates[0].device)
        if mine:
            local[: len(mine)] = orthogonalize(torch.stack(mine), **kw)
        out = torch.empty((world * per, *updates[0].shape), dtype=local.dtype, device=local.device)
        try:
            dist.all_gather_into_tensor(out, local)  # NCCL: one collective into a contiguous buffer
        except (RuntimeError, NotImplementedError):  # backends without it (e.g. some gloo builds)
            dist.all_gather(list(out.chunk(world)), local)
        return out[: len(updates)]

    def _adamw_group(self, group):
        """AdamW with multi-tensor (foreach) kernels: one launch per op for the whole group, not one per tensor."""
        lr, wd, (b1, b2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
        by_step = defaultdict(lambda: ([], [], [], []))  # tensors sharing a step count share bias corrections
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["step"] += 1
            ps, gs, ms, vs = by_step[state["step"]]
            ps.append(p)
            gs.append(p.grad)
            ms.append(state["exp_avg"])
            vs.append(state["exp_avg_sq"])
        for t, (ps, gs, ms, vs) in by_step.items():
            torch._foreach_lerp_(ms, gs, 1 - b1)
            torch._foreach_mul_(vs, b2)
            torch._foreach_addcmul_(vs, gs, gs, value=1 - b2)
            denom = torch._foreach_div(vs, 1 - b2**t)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, eps)
            if wd:
                torch._foreach_mul_(ps, 1 - lr * wd)
            torch._foreach_addcdiv_(ps, ms, denom, value=-lr / (1 - b1**t))


_ADAMW_NAMES = ("embed", "lm_head", "wte", "wpe", "output.weight")


def muon_param_groups(model, weight_decay: float, min_dim: int = 32):
    """(groups, counts): Muon for 2-D hidden weights with min side >= min_dim; AdamW (with / without weight decay) for
    the rest. `model`: a module, or (name, parameter) pairs (e.g. fp32 master copies named after the module's
    parameters). Tied embeddings are seen once (named_parameters de-duplicates them) and stay on AdamW."""
    named = model.named_parameters() if isinstance(model, torch.nn.Module) else model
    muon, adam_decay, adam_no_decay = [], [], []
    counts = {"muon": 0, "adamw": 0}
    for name, p in named:
        if not p.requires_grad:
            continue
        lname = name.lower()
        if p.ndim == 2 and min(p.shape) >= min_dim and not any(k in lname for k in _ADAMW_NAMES):
            muon.append(p)
            counts["muon"] += p.numel()
        else:
            (adam_no_decay if p.ndim <= 1 or "norm" in lname or "bias" in lname else adam_decay).append(p)
            counts["adamw"] += p.numel()
    groups = []
    if muon:
        groups.append({"params": muon, "use_muon": True, "weight_decay": weight_decay})
    if adam_decay:
        groups.append({"params": adam_decay, "use_muon": False, "weight_decay": weight_decay})
    if adam_no_decay:
        groups.append({"params": adam_no_decay, "use_muon": False, "weight_decay": 0.0})
    return groups, counts


def build_muon(
    model,
    lr: float,
    weight_decay: float,
    momentum: float = 0.95,
    ns_method: str = "gram",
    scale: str = "moonlight",
    min_dim: int = 32,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    backend: str = "auto",
    compile: bool = False,
    distribute: bool = True,
) -> MuonAdamW:
    groups, counts = muon_param_groups(model, weight_decay, min_dim)
    logger.info(
        "Muon: %.1fM params orthogonalized (%s Newton-Schulz, %s scale), %.1fM on AdamW; lr %.2e",
        counts["muon"] / 1e6, ns_method, scale, counts["adamw"] / 1e6, lr,
    )
    return MuonAdamW(groups, lr=lr, weight_decay=weight_decay, momentum=momentum, ns_method=ns_method, scale=scale,
                     betas=betas, eps=eps, backend=backend, compile=compile, distribute=distribute)
