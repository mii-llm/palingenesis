"""AdamW on bfloat16 weights and moments, with stochastic rounding: fp32-master quality at bf16 memory.

Why: a bf16 weight has 8 significant bits, so one ulp is 2^-8 to 2^-7 of |w|. A fine-tuning update (AdamW at
lr 2e-5 on weights of RMS 0.02 moves them by about 1e-3 of their size) is below half an ulp almost everywhere, and
round-to-nearest returns the old weight: plain AdamW on bf16 parameters (torch's AdamW keeps its moments in the
parameter's dtype) silently stops training most weights. The usual fix, fp32 master weights, costs 4 more bytes per
parameter plus fp32 moments (16 B/param with gradients against 8 here).

This optimizer keeps the weight and both moments in bf16 (6 B/param, plus the bf16 gradient) and does every
element's update in fp32, rounding only when it stores the results:

    weight          stochastic rounding (SR): rounds up with probability equal to the dropped fraction of an ulp,
                    so E[stored] = the fp32 result and small updates accumulate over steps instead of vanishing
    second moment   SR: a non-negative running average near its own rounding floor, where nearest rounding biases
                    it (and with it sqrt(v), the effective step size) by tens of percent
    first moment    nearest: a signed, roughly zero-mean average, whose nearest rounding is already unbiased

rounding="kahan" replaces the weight's stochastic rounding by compensated (Kahan) summation: a bf16 buffer keeps what
nearest rounding dropped (8 B/param of weight and state), so the weight carries ~16 significant bits and no rounding
noise, which matters at very small learning rates (stochastic rounding is unbiased, but its error is a random walk).

The update reads the fp32 moments before they are rounded. Parameters that are not bf16 (fp32 adapters, norms kept
in fp32) take the plain fp32 AdamW update, identical to torch.optim.AdamW.

Noise: each element's two 16-bit draws come from a counter-based integer hash of (its offset in the tensor, a
per-step seed), not from a random generator. The seed is a hash of (optimizer seed, parameter name, the parameter's
step count), so the result is bit-for-bit reproducible, a resumed run rounds exactly as the uninterrupted one
(nothing but the step count, which the optimizer state holds, is needed), data-parallel replicas holding the same
parameter and gradient round identically, and the same inputs give the same bits on CPU and CUDA.

Speed: on CUDA the per-parameter update is compiled once (torch.compile, dynamic shapes) into a single fused
kernel: one read of weight, gradient and moments and one write of weight and moments, 14 bytes per element moved
for bf16 gradients. Without compilation (CPU, or `compile=False`) the same function runs eagerly in chunks.
`multi_tensor=True` steps every parameter in one Triton launch per dtype bucket instead (palingenesis.fused_optim:
the same operations without FMA contraction, so the same bits as the eager update), and can fold gradient clipping
into that launch (`clip_grad_norm_`).

After: stochastic rounding for low-precision training (Gupta et al. 2015, arXiv:1502.02551; Zamirai et al. 2020,
arXiv:2010.06192, which finds bf16 weights with SR or Kahan summation match fp32 masters; Kahan-compensated bf16
optimizers as in the optimi library, MIT); SR of the second moment and nearest
rounding of the first follows the design notes of the Halo framework's AdamWBF16 (Apache-2.0; re-implemented here,
no code copied). The integer hash is a 32-bit avalanche mixer in the style of C. Wellons' "lowbias32" (public
domain), with multipliers below 2^31 so that every product fits in int64.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)

_MASK32 = 0xFFFFFFFF
_MIX1 = 0x7FEB352D  # odd multipliers < 2^31: (x < 2^32) * c < 2^63, no signed int64 overflow
_MIX2 = 0x5BD1E995
EAGER_CHUNK = 1 << 22  # elements per chunk on the eager path (int64 temporaries: 32 MB)


# ----------------------------------------------------------------------------------------------------------------
# Counter-based noise and stochastic rounding (pure tensor functions: run eagerly or under torch.compile)
# ----------------------------------------------------------------------------------------------------------------


def hash32(index: torch.Tensor, seed: torch.Tensor | int) -> torch.Tensor:
    """A 32-bit hash (as int64 in [0, 2^32)) of each element of `index` (int64) under `seed`: uniform, and
    uncorrelated between neighbouring indices or seeds. Pure integer ops, so identical on every device."""
    x = (index + seed) & _MASK32
    x = x ^ (x >> 16)
    x = (x * _MIX1) & _MASK32
    x = x ^ (x >> 15)
    x = (x * _MIX2) & _MASK32
    return x ^ (x >> 16)


def round_bf16(x: torch.Tensor, noise16: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16, rounding stochastically with `noise16` (int64 in [0, 2^16), one draw per element).

    bf16 is the top 16 bits of fp32: adding the noise to the low 16 bits of the bit pattern and truncating rounds the
    magnitude up with probability (dropped bits) / 2^16, so E[result] = x. IEEE floats are sign-magnitude, so the
    integer addition acts on the magnitude for negative values too. A carry from the largest finite bf16 into the
    infinity exponent is truncated instead (saturates); inf and nan take the plain cast."""
    bits = x.view(torch.int32).to(torch.int64)
    finite = torch.isfinite(x)
    rounded = (bits + torch.where(finite, noise16, torch.zeros_like(noise16))) & ~0xFFFF
    overflow = (rounded & 0x7FFFFFFF) == 0x7F800000
    rounded = torch.where(overflow & finite, bits & ~0xFFFF, rounded)
    out = (rounded >> 16).to(torch.int16).view(torch.bfloat16)  # low 16 bits: the bf16 pattern (sign included)
    return torch.where(finite, out, x.to(torch.bfloat16))


def round_bf16_nearest(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 -> bf16 with round-to-nearest-even, done on the bit pattern, and the same value as fp32. A cast followed
    by an upcast (x.to(bf16).float()) is not used for the residual x - round(x): torch.compile may drop such
    intermediate casts in a fused kernel, which would make the residual 0."""
    bits = x.view(torch.int32).to(torch.int64)
    finite = torch.isfinite(x)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & ~0xFFFF
    as_f32 = torch.where(finite, rounded.to(torch.int32).view(torch.float32), x)
    out = (rounded >> 16).to(torch.int16).view(torch.bfloat16)
    return torch.where(finite, out, x.to(torch.bfloat16)), as_f32


def _bf16_adamw_update(
    p: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    hp: torch.Tensor,
    seed: torch.Tensor,
    offset: int = 0,
) -> None:
    """One AdamW step of flat bf16 tensors in place. hp (fp32 on p's device): [decay, beta1, beta2, step_size,
    1 / sqrt(bias_correction2), eps], with decay = 1 - lr * weight_decay and step_size = lr / bias_correction1.
    `offset` is the position of p[0] in the whole tensor (eager chunks), so each element draws the same noise
    whatever the chunking."""
    decay, beta1, beta2, step_size, inv_bc2_sqrt, eps = hp[0], hp[1], hp[2], hp[3], hp[4], hp[5]
    gf = g.float()
    mf = m.float() * beta1 + gf * (1.0 - beta1)
    vf = v.float() * beta2 + gf * gf * (1.0 - beta2)
    pf = p.float() * decay - step_size * mf / (vf.sqrt() * inv_bc2_sqrt + eps)
    index = torch.arange(offset, offset + p.numel(), device=p.device, dtype=torch.int64)
    r = hash32(index, seed)
    p.copy_(round_bf16(pf, r & 0xFFFF))
    v.copy_(round_bf16(vf, r >> 16))
    m.copy_(mf.to(torch.bfloat16))


def _bf16_adamw_kahan_update(
    p: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    hp: torch.Tensor,
    seed: torch.Tensor,
    offset: int = 0,
) -> None:
    """The same step with compensated (Kahan) summation of the weight instead of stochastic rounding: c holds what
    the last nearest rounding of the weight dropped (in bf16), so the stored pair p + c carries about 16 significant
    bits and every update is applied exactly up to that precision, with no rounding noise. The second moment is still
    rounded stochastically, the first to nearest."""
    decay, beta1, beta2, step_size, inv_bc2_sqrt, eps = hp[0], hp[1], hp[2], hp[3], hp[4], hp[5]
    gf = g.float()
    mf = m.float() * beta1 + gf * (1.0 - beta1)
    vf = v.float() * beta2 + gf * gf * (1.0 - beta2)
    pf = (p.float() + c.float()) * decay - step_size * mf / (vf.sqrt() * inv_bc2_sqrt + eps)
    stored, stored_f32 = round_bf16_nearest(pf)
    index = torch.arange(offset, offset + p.numel(), device=p.device, dtype=torch.int64)
    r = hash32(index, seed)
    p.copy_(stored)
    c.copy_((pf - stored_f32).to(torch.bfloat16))
    v.copy_(round_bf16(vf, r >> 16))
    m.copy_(mf.to(torch.bfloat16))


ROUNDINGS = ("stochastic", "kahan")


# ----------------------------------------------------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------------------------------------------------


def _local(t: torch.Tensor) -> torch.Tensor:
    """The rank's local shard of an FSDP2 (DTensor) tensor, the tensor itself otherwise."""
    return t.to_local() if hasattr(t, "to_local") else t


def _seed(key: int, step: int, base: int) -> int:
    """A 32-bit seed for (parameter key, step): crc32 chained over the three integers."""
    return zlib.crc32(step.to_bytes(8, "little"), zlib.crc32(key.to_bytes(8, "little"), base & _MASK32))


class BF16AdamW(torch.optim.Optimizer):
    """AdamW (decoupled weight decay) for bf16 parameters with bf16 moments and stochastic rounding (module
    docstring). A drop-in torch optimizer: parameter groups, LR schedulers, state_dict / load_state_dict, gradient
    clipping and FSDP2 (DTensor) parameters work as with torch.optim.AdamW.

    params:  parameters or parameter groups (lr, betas, eps, weight_decay per group).
    named:   (name, parameter) pairs, e.g. model.named_parameters(). The name keys each parameter's rounding noise,
             so replicas and resumed runs round identically even if their parameter lists are ordered differently.
             Parameters without a name are keyed by their position in the groups.
    seed:    the optimizer's noise seed (the run's seed); changes every rounding decision, nothing else.
    rounding: how the weight keeps updates below its ulp. "stochastic" (6 B/param of weight and state): unbiased,
             but each element's rounding error is a random walk, about ulp * sqrt(steps * update / ulp) after many
             steps of the same small update. "kahan" (8 B/param): a bf16 compensation buffer holds what nearest
             rounding dropped, so the weight carries ~16 significant bits with no noise; preferable at very small
             learning rates (RL at 1e-6, where an update is ~1/100 of an ulp).
    compile: "auto" (compile the fused update on CUDA), True, or False (eager everywhere).
    multi_tensor: step all parameters with palingenesis.fused_optim's Triton kernels (one launch per dtype bucket)
             on CUDA; parameters off CUDA, or every parameter when Triton is missing, take the per-parameter path.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        named: Iterable[tuple[str, torch.Tensor]] | None = None,
        seed: int = 0,
        rounding: str = "stochastic",
        compile: bool | str = "auto",
        multi_tensor: bool = False,
    ):
        if rounding not in ROUNDINGS:
            raise ValueError(f"rounding must be one of {', '.join(ROUNDINGS)}, got {rounding!r}")
        if lr < 0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if compile not in (True, False, "auto"):
            raise ValueError(f"compile must be True, False or 'auto', got {compile!r}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))
        self.seed = int(seed)
        self.rounding = rounding
        self.compile = compile
        names = {id(p): n for n, p in named} if named is not None else {}
        self._keys: dict[int, int] = {}
        position = 0
        for group in self.param_groups:
            for p in group["params"]:
                name = names.get(id(p))
                self._keys[id(p)] = zlib.crc32(name.encode()) if name is not None else (1 << 32) + position
                position += 1
        self._compiled = None
        self._key_crc: dict[int, int] = {}  # parameter -> crc32 of (its key) under the seed: the step's seed chains on
        self.multi_tensor = bool(multi_tensor)
        self.triton = False
        if self.multi_tensor:
            from palingenesis.fused_optim import triton_available

            self.triton = triton_available()[0]
        self.grad_scale: torch.Tensor | None = None  # the folded clip factor of the next step (multi_tensor)
        self.last_launches = 0

    # -- internals ---------------------------------------------------------------------------------------------

    def _update_fn(self, device: torch.device):
        use = self.compile is True or (self.compile == "auto" and device.type == "cuda")
        if not use:
            return None
        if self._compiled is None:
            self._compiled = torch.compile(self._update, dynamic=True, fullgraph=True)
        return self._compiled

    @property
    def _update(self):
        return _bf16_adamw_kahan_update if self.rounding == "kahan" else _bf16_adamw_update

    def _init_state(self, p: torch.Tensor) -> dict:
        state = self.state[p]
        if not state:
            moment = torch.bfloat16 if p.dtype == torch.bfloat16 else p.dtype
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, dtype=moment, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=moment, memory_format=torch.preserve_format)
            if self.rounding == "kahan" and p.dtype == torch.bfloat16:
                state["compensation"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state

    def _tensors(self, p: torch.Tensor, state: dict) -> list[torch.Tensor]:
        """The flat local tensors the update function takes, in order."""
        names = ("exp_avg", "exp_avg_sq", "compensation") if self.rounding == "kahan" else ("exp_avg", "exp_avg_sq")
        return [_local(t).view(-1) for t in (p, p.grad, *(state[n] for n in names))]

    def _key(self, p: torch.Tensor) -> int:
        key = self._keys.get(id(p))
        if key is None:  # a parameter added with add_param_group after construction
            key = self._keys[id(p)] = (1 << 32) + len(self._keys)
        return key

    def _step_seed(self, p: torch.Tensor, step: int) -> int:
        """_seed(key, step, seed), with the key's part of the chain cached."""
        crc = self._key_crc.get(id(p))
        if crc is None:
            crc = self._key_crc[id(p)] = zlib.crc32(self._key(p).to_bytes(8, "little"), self.seed & _MASK32)
        return zlib.crc32(step.to_bytes(8, "little"), crc)

    @staticmethod
    def _scalars(group: dict, step: int) -> list[float]:
        lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        return [1.0 - lr * wd, beta1, beta2, lr / bc1, bc2**-0.5, eps]

    @torch.no_grad()
    def _step_fp32(self, p: torch.Tensor, group: dict, state: dict) -> None:
        """torch.optim.AdamW's update, for parameters that are not bf16."""
        decay, beta1, beta2, step_size, inv_bc2_sqrt, eps = self._scalars(group, state["step"])
        g, m, v = p.grad, state["exp_avg"], state["exp_avg_sq"]
        p.mul_(decay)
        m.lerp_(g, 1.0 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        p.addcdiv_(m, (v.sqrt() * inv_bc2_sqrt).add_(eps), value=-step_size)

    def _run_bf16(self, items: list) -> None:
        """Step bf16 parameters: items of (param, group, state). All per-step scalars and seeds go to the device in
        one pinned, non-blocking copy, so the loop never waits for the GPU."""
        by_device: dict[torch.device, list] = {}
        for item in items:
            by_device.setdefault(_local(item[0]).device, []).append(item)
        for device, its in by_device.items():
            hp = torch.tensor([self._scalars(g, s["step"]) for _, g, s in its], dtype=torch.float32)
            seeds = torch.tensor([self._step_seed(p, s["step"]) for p, _, s in its], dtype=torch.int64)
            if device.type == "cuda":
                hp, seeds = (
                    hp.pin_memory().to(device, non_blocking=True),
                    seeds.pin_memory().to(device, non_blocking=True),
                )
            elif device.type != "cpu":
                hp, seeds = hp.to(device), seeds.to(device)
            fn = self._update_fn(device)
            update = self._update
            for i, (p, _, state) in enumerate(its):
                flat = self._tensors(p, state)
                if fn is not None:
                    fn(*flat, hp[i], seeds[i])
                else:
                    n = flat[0].numel()
                    for start in range(0, n, EAGER_CHUNK):
                        part = [t[start : start + EAGER_CHUNK] for t in flat]
                        update(*part, hp[i], seeds[i], start)

    def _check(self, p: torch.Tensor) -> None:
        if p.grad.is_sparse:
            raise RuntimeError("BF16AdamW does not support sparse gradients")
        if p.dtype == torch.bfloat16 and not _local(p).is_contiguous():
            raise RuntimeError("BF16AdamW needs contiguous bf16 parameters (it steps their flat storage)")

    # -- public API --------------------------------------------------------------------------------------------

    def _params(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    yield p, group

    def _fusable(self, p: torch.Tensor, state: dict) -> bool:
        """Whether the multi-tensor kernels step `p` (CUDA, contiguous local tensors of a supported dtype)."""
        from palingenesis.fused_optim import local

        if not self.triton:
            return False
        lp, lg = local(p), local(p.grad)
        if not (lp.is_cuda and lp.is_contiguous() and lg.is_contiguous()):
            return False
        return lp.dtype == torch.bfloat16 or (lp.dtype == torch.float32 and lg.dtype in (torch.float32, torch.bfloat16))

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """With multi_tensor: the global L2 norm of the gradients (fp32), and the clip factor
        min(1, max_norm / (norm + 1e-6)) kept for the next step(), which multiplies it into each fp32 gradient in the
        kernel instead of a pass that rescales every gradient. The .grad tensors keep their unclipped values. One
        process only (the norm is not reduced across ranks). max_norm <= 0: the norm only."""
        from palingenesis.fused_optim import _clip

        return _clip(self, max_norm)

    def zero_grad(self, set_to_none: bool = True):
        self.grad_scale = None
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        scale, self.grad_scale = self.grad_scale, None
        bf16, fused = [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._check(p)
                state = self._init_state(p)
                state["step"] += 1
                if self.multi_tensor and self._fusable(p, state):
                    fused.append((p, group, state))
                    continue
                if scale is not None:  # a folded clip factor, on a parameter the kernels do not step
                    p.grad.mul_(scale.to(p.grad.device).to(p.grad.dtype))
                if p.dtype == torch.bfloat16:
                    bf16.append((p, group, state))
                else:
                    self._step_fp32(p, group, state)
        if bf16:
            self._run_bf16(bf16)
        self.last_launches = self._run_fused(fused, scale) if fused else 0
        return loss

    def _run_fused(self, items: list, scale: torch.Tensor | None) -> int:
        """Step parameters with the multi-tensor kernels (palingenesis.fused_optim): bf16 ones with this optimizer's
        update and rounding, fp32 ones with AdamW's. Returns the number of launches."""
        from palingenesis.fused_optim import Bf16Row, Fp32Row, adamw, adamw_bf16, local

        kahan = self.rounding == "kahan"
        bf16_rows, fp32_rows, hp_cache = [], [], {}
        for p, group, state in items:
            key = (id(group), state["step"])
            hp = hp_cache.get(key)
            if hp is None:
                hp = hp_cache[key] = self._scalars(group, state["step"])
            flat = [local(t).view(-1) for t in (p, p.grad, state["exp_avg"], state["exp_avg_sq"])]
            if p.dtype == torch.bfloat16:
                c = local(state["compensation"]).view(-1) if kahan else None
                bf16_rows.append(Bf16Row(*flat, c, hp, self._step_seed(p, state["step"])))
            else:
                fp32_rows.append(Fp32Row(*flat, hp))
        launches = adamw_bf16(bf16_rows, scale) if bf16_rows else 0
        return launches + (adamw(fp32_rows, scale) if fp32_rows else 0)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict) -> None:
        """Step one parameter (its .grad must be set): what memory.gradient_release calls from the backward hook.
        Gives the same result as step() over that parameter."""
        if p.grad is None:
            return
        self._check(p)
        state = self._init_state(p)
        state["step"] += 1
        if p.dtype == torch.bfloat16:
            self._run_bf16([(p, group, state)])
        else:
            self._step_fp32(p, group, state)

    def state_dict(self):
        out = super().state_dict()
        out["bf16_adamw_seed"] = self.seed
        return out

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.seed = int(state_dict.pop("bf16_adamw_seed", self.seed))
        self._key_crc.clear()
        super().load_state_dict(state_dict)
        for state in self.state.values():  # the step count is a plain int (it seeds the noise on the host)
            if torch.is_tensor(state.get("step")):
                state["step"] = int(state["step"].item())


def state_bytes_per_param(dtype: torch.dtype, rounding: str = "stochastic") -> int:
    """Optimizer state bytes per parameter (both moments, and the compensation buffer with "kahan") for a
    parameter of `dtype`."""
    if dtype != torch.bfloat16:
        return 2 * torch.empty((), dtype=dtype).element_size()
    return 6 if rounding == "kahan" else 4
