"""Attention of a SeCO chunk against a stored prefix, without copying the prefix.

A chunk's queries attend to (a) every token of the prefix and (b) the chunk's own
tokens causally. The two parts, and the prefix itself block by block, are computed
separately and merged through their log-sum-exp, the same exact combination that
flash attention and ring attention use internally:

    out = sum_b exp(lse_b - lse) * out_b,        lse = logsumexp_b(lse_b)

The backward runs block by block from the MERGED output and log-sum-exp (as the
flash backward kernel expects), so gradients are exact. Prefix-K/V gradients are
accumulated straight into the store's gradient buffer, which SeCO relays to the
chunk that produced them.

So the prefix is never concatenated with the chunk (the copy that
`DynamicLayer.update` makes), and it can live in pinned CPU memory, streamed to
the GPU one block at a time (`memory.seco_kv_offload`).

Kernels: flash (bf16/fp16 on CUDA; grouped K/V heads natively) or an exact math
path (any dtype/device) with the same interface, used for fp32/fp64 and on CPU.
"""

from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)


def _host_bytes(name: str) -> int | None:
    try:
        return os.sysconf(name) * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


_HOST_USED = 0        # bytes of host memory this call's offloaded stores hold


def reset_host_budget() -> None:
    global _HOST_USED
    _HOST_USED = 0


# Share of the machine's RAM the offloaded K/V may hold. Pinned pages cannot be
# swapped, and the host still needs page cache and the training process's own
# memory: measured on a 117 GiB host, 28 GiB of pinned K/V ran fine while 56 GiB
# (48%) left the machine unusable and the GPU waiting on page faults.
HOST_BUDGET_FRACTION = 0.4


def check_host_budget(need: int) -> None:
    """Raise unless `need` bytes of K/V can live in host memory on this machine."""
    total, available = _host_bytes("SC_PHYS_PAGES"), _host_bytes("SC_AVPHYS_PAGES")
    limit = min(HOST_BUDGET_FRACTION * total if total else float("inf"), available or float("inf"))
    if need > limit:
        raise RuntimeError(
            f"memory.seco_kv_offload would hold {need / 2**30:.1f} GiB of K/V in host memory, over the "
            f"{limit / 2**30:.1f} GiB this machine can safely use ({(total or 0) / 2**30:.0f} GiB RAM, "
            f"{(available or 0) / 2**30:.0f} GiB free). Use a shorter sequence, a smaller batch, or turn "
            "offloading off (the K/V then stay on the GPU)."
        )


def _host_tensor(shape, dtype: torch.dtype) -> torch.Tensor:
    """Host memory for an offloaded store: pinned when possible (async copies).

    Offloading holds the whole sequence's K/V for EVERY attention layer, so the
    total is checked against the machine's RAM; going past it makes the host swap
    and everything crawls (the GPU then waits on page faults)."""
    global _HOST_USED
    need = math.prod(shape) * torch.finfo(dtype).bits // 8
    total, available = _host_bytes("SC_PHYS_PAGES"), _host_bytes("SC_AVPHYS_PAGES")
    limit = min(HOST_BUDGET_FRACTION * total if total else float("inf"), (available or float("inf")) + _HOST_USED)
    if _HOST_USED + need > limit:
        raise RuntimeError(
            f"memory.seco_kv_offload would hold {(_HOST_USED + need) / 2**30:.1f} GiB of K/V in host memory, over "
            f"the {limit / 2**30:.1f} GiB it can safely use on this machine "
            f"({(total or 0) / 2**30:.0f} GiB RAM, {(available or 0) / 2**30:.0f} GiB free). Use a shorter "
            "sequence, a smaller batch, or turn offloading off."
        )
    _HOST_USED += need
    try:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    except RuntimeError as exc:      # pinning can fail (locked-memory limits): plain host memory still works
        logger.warning("SeCO: could not pin host memory for the K/V store (%s); using pageable memory", exc)
        return torch.empty(shape, dtype=dtype, device="cpu")


class KVStore:
    """One attention layer's K/V for the whole sequence, [B, Hkv, S, D].

    Stage 1 writes each chunk's K/V; stage 2 reads prefixes block by block.
    With `offload`, the store lives in pinned CPU memory. The gradient buffer
    (stage 2) always lives on the compute device: every chunk adds to the whole
    prefix's gradient, so it cannot be streamed."""

    def __init__(self, like: torch.Tensor, seq_len: int, offload: bool):
        shape = (like.shape[0], like.shape[1], seq_len, like.shape[3])
        self.device, self.dtype = like.device, like.dtype
        if not offload:
            self.k = torch.empty(shape, dtype=like.dtype, device=like.device)
            self.v = torch.empty(shape, dtype=like.dtype, device=like.device)
        else:
            self.k, self.v = _host_tensor(shape, like.dtype), _host_tensor(shape, like.dtype)
        self.grad_k = self.grad_v = None

    def write(self, lo: int, k: torch.Tensor, v: torch.Tensor) -> None:
        n = k.shape[-2]
        self.k[..., lo:lo + n, :].copy_(k.detach(), non_blocking=True)
        self.v[..., lo:lo + n, :].copy_(v.detach(), non_blocking=True)

    def load(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.k[..., start:end, :].to(self.device, non_blocking=True),
                self.v[..., start:end, :].to(self.device, non_blocking=True))

    def start_gradients(self) -> None:
        self.grad_k = torch.zeros(self.k.shape, dtype=self.dtype, device=self.device)
        self.grad_v = torch.zeros(self.v.shape, dtype=self.dtype, device=self.device)

    def nbytes_on_device(self) -> int:
        grads = 0 if self.grad_k is None else 2 * self.grad_k.numel() * self.grad_k.element_size()
        store = 0 if self.k.device.type == "cpu" else 2 * self.k.numel() * self.k.element_size()
        return grads + store


# ── block kernels: (out, lse, aux) forward and (dq, dk, dv) backward ────────────


def _flash_ok(q: torch.Tensor, k: torch.Tensor) -> bool:
    return (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and q.shape[-1] % 8 == 0
            and q.shape[-1] <= 256 and k.shape[-1] == q.shape[-1])


def _block_forward(q, k, v, causal: bool, scale: float):
    if _flash_ok(q, k):
        out, lse, cq, ck, mq, mk, seed, offset, _ = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, causal, False, scale=scale)
        return out, lse, ("flash", cq, ck, mq, mk, seed, offset)
    groups = q.shape[1] // k.shape[1]
    kr, vr = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    dtype = torch.promote_types(q.dtype, torch.float32)
    s = (q.to(dtype) @ kr.to(dtype).transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    out = (torch.exp(s - lse.unsqueeze(-1)) @ vr.to(dtype)).to(q.dtype)
    return out, lse, ("math",)


def _block_backward(dout, q, k, v, out, lse, causal: bool, scale: float, aux):
    if aux[0] == "flash":
        _, cq, ck, mq, mk, seed, offset = aux
        return torch.ops.aten._scaled_dot_product_flash_attention_backward(
            dout, q, k, v, out, lse, cq, ck, mq, mk, 0.0, causal, seed, offset, scale=scale)
    groups = q.shape[1] // k.shape[1]
    dtype = torch.promote_types(q.dtype, torch.float32)
    qf, dof, of = q.to(dtype), dout.to(dtype), out.to(dtype)
    kr, vr = k.repeat_interleave(groups, 1).to(dtype), v.repeat_interleave(groups, 1).to(dtype)
    s = (qf @ kr.transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), float("-inf"))
    p = torch.exp(s - lse.to(dtype).unsqueeze(-1))           # probabilities under the MERGED normaliser
    dv = p.transpose(-1, -2) @ dof
    dp = dof @ vr.transpose(-1, -2)
    ds = p * (dp - (dof * of).sum(-1, keepdim=True))
    dq = (ds @ kr) * scale
    dk = (ds.transpose(-1, -2) @ qf) * scale
    if groups > 1:
        dk = dk.view(k.shape[0], k.shape[1], groups, *dk.shape[2:]).sum(2)
        dv = dv.view(v.shape[0], v.shape[1], groups, *dv.shape[2:]).sum(2)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def _merge(out_acc, lse_acc, out, lse):
    """Online log-sum-exp combination of partial attention outputs (at least fp32)."""
    out = out.to(torch.promote_types(out.dtype, torch.float32))
    if out_acc is None:
        return out, lse
    new = torch.logaddexp(lse_acc, lse)
    return out_acc * torch.exp(lse_acc - new).unsqueeze(-1).to(out.dtype) + out * torch.exp(lse - new).unsqueeze(-1).to(out.dtype), new


class ChunkAttention(torch.autograd.Function):
    """softmax attention of chunk queries over [stored prefix; chunk] with causal
    masking inside the chunk. q: [B, H, L, D]; k/v: the chunk's own [B, Hkv, L, D].

    `prefix` is the stored prefix length, the same for every row, or one per row
    (branches of a tree continuing the stored trunk from different positions; the
    store then holds the trunk once, batch 1, and serves every row). Prefix blocks
    that every row covers are one call; a block that only some rows reach is
    computed row by row over each row's own keys."""

    @staticmethod
    def forward(ctx, q, k, v, store: KVStore, prefix, scale: float, block: int):
        rows = q.shape[0]
        prefixes = [prefix] * rows if isinstance(prefix, int) else list(prefix)
        shared = store.k.shape[0] != rows              # one stored row serving every query row
        out_acc = lse_acc = None
        parts = []
        for start in range(0, max(prefixes, default=0), block):
            end = min(start + block, max(prefixes))
            kb, vb = store.load(start, end)
            valid = [min(max(p - start, 0), end - start) for p in prefixes]
            if all(n == end - start for n in valid):
                kx, vx = (_broadcast(kb, rows), _broadcast(vb, rows)) if shared else (kb, vb)
                out, lse, aux = _block_forward(q, kx, vx, False, scale)
                parts.append((start, end, None, aux))
            else:
                out = q.new_zeros(q.shape, dtype=torch.promote_types(q.dtype, torch.float32))
                lse = torch.full(q.shape[:3], float("-inf"), dtype=torch.float32, device=q.device)
                per_row = []
                for b, n in enumerate(valid):
                    if n == 0:
                        continue
                    src = slice(0, 1) if shared else slice(b, b + 1)
                    o, l_, a = _block_forward(q[b:b + 1], kb[src][..., :n, :], vb[src][..., :n, :], False, scale)
                    out[b:b + 1], lse[b:b + 1] = o.to(out.dtype), l_.to(lse.dtype)
                    per_row.append((b, n, a))
                parts.append((start, end, per_row, None))
            out_acc, lse_acc = _merge(out_acc, lse_acc, out, lse)
        out, lse, aux_local = _block_forward(q, k, v, True, scale)
        out_acc, lse_acc = _merge(out_acc, lse_acc, out, lse)
        out = out_acc.to(q.dtype)
        ctx.save_for_backward(q, k, v, out, lse_acc)
        ctx.store, ctx.scale, ctx.parts, ctx.aux_local, ctx.shared = store, scale, parts, aux_local, shared
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dout = dout.to(q.dtype).contiguous()
        store, rows = ctx.store, q.shape[0]
        dq = torch.zeros(q.shape, dtype=torch.promote_types(q.dtype, torch.float32), device=q.device)
        for start, end, per_row, aux in ctx.parts:
            kb, vb = store.load(start, end)
            if per_row is None:
                kx, vx = (_broadcast(kb, rows), _broadcast(vb, rows)) if ctx.shared else (kb, vb)
                dqb, dkb, dvb = _block_backward(dout, q, kx, vx, out, lse, False, ctx.scale, aux)
                dq += dqb
                if store.grad_k is not None:
                    if ctx.shared:
                        dkb, dvb = dkb.sum(0, keepdim=True), dvb.sum(0, keepdim=True)
                    store.grad_k[..., start:end, :].add_(dkb)
                    store.grad_v[..., start:end, :].add_(dvb)
                continue
            for b, n, a in per_row:
                src = slice(0, 1) if ctx.shared else slice(b, b + 1)
                r = slice(b, b + 1)
                dqb, dkb, dvb = _block_backward(dout[r], q[r], kb[src][..., :n, :], vb[src][..., :n, :], out[r],
                                                lse[r], False, ctx.scale, a)
                dq[r] += dqb
                if store.grad_k is not None:
                    store.grad_k[src][..., start:start + n, :].add_(dkb)
                    store.grad_v[src][..., start:start + n, :].add_(dvb)
        dql, dkl, dvl = _block_backward(dout, q, k, v, out, lse, True, ctx.scale, ctx.aux_local)
        dq += dql
        return dq.to(q.dtype), dkl, dvl, None, None, None, None


def _broadcast(t: torch.Tensor, rows: int) -> torch.Tensor:
    """A stored [1, ...] block for `rows` query rows (the kernels need matching batch sizes)."""
    return t.expand(rows, *t.shape[1:]).contiguous()


def chunk_attention(q, k, v, store: KVStore, prefix: int, scale: float | None, block: int) -> torch.Tensor:
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    return ChunkAttention.apply(q.contiguous(), k.contiguous(), v.contiguous(), store, prefix, scale, block)
