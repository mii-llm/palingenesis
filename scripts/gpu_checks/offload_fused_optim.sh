#!/usr/bin/env bash
# GPU checks for memory.activation_offload (palingenesis.activation_offload) and train.fused_optimizer
# (palingenesis.fused_optim): exactness, memory, speed and launch counts, measured on one NVIDIA GPU.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/offload_fused_optim.sh
#
# Written for one A100 80GB PCIe (torch 2.11 cu129, triton). Models from the Hugging Face cache, offline:
# Qwen/Qwen3.5-2B, Qwen/Qwen3.5-0.8B, Qwen/Qwen3-0.6B. Every line on stdout is one JSON object:
#   {"check": ..., "status": "PASS" | "FAIL" | "INFO" | "SKIP", ...numbers}
# Timings are medians of 3 timed windows after warm-up. Logs of the Python processes go to stderr only when a check
# fails. Exit code 1 when any check FAILs.
#
# Checks (each its own process):
#   env               versions, GPU, kernels compile
#   optim_exact       multi-tensor adamw_bf16 (stochastic, kahan) bit-exact vs the eager per-parameter update, with
#                     LLRD-style groups, odd sizes, misaligned views, extreme magnitudes, fp32 parameters and a
#                     folded clip factor; FusedAdamW vs torch AdamW (fp32); compiled per-parameter path agreement
#   optim_resume      6 steps straight == 3 steps, state_dict round trip, 3 steps (bits), both roundings + FusedAdamW
#   optim_bench       step time, host time and GPU launches per step on 0.8B and 2B parameter shapes: per-parameter
#                     adamw_bf16 vs multi-tensor, torch fused AdamW vs FusedAdamW (fp32 masters), clipping folded
#   offload_grads     2B, 4k tokens: gradients with offload vs without (none / selective / full checkpointing)
#   offload_bench     2B, 4k and 16k tokens: peak memory and forward+backward time, 6 settings
#   offload_compile   Qwen3-0.6B, 4k, compiled layers: gradients, step time and graph breaks with offload
#   offload_trainer   the real trainer (pgs fit's measurement) on 2B at 16k: full checkpointing with and without
#                     offload, adamw_bf16 on the multi-tensor kernels
#   sft               a 30-step SFT run (Qwen3.5-0.8B): fp32-master adamw vs multi-tensor adamw_bf16 vs per-parameter
#                     adamw_bf16, the loss curves compared
# PGS_CHECK_SMOKE=1 runs the code paths on the CPU with tiny random models (measures nothing; for development).
set -uo pipefail

PGS_PY="${PGS_PY:-python}"
PGS_SRC="${PGS_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src" && pwd)}"
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/pgs_gpu_check.XXXXXX")"
export PGS_CHECK_WORK="$WORK"
cleanup() {  # file by file, then the empty directories
  find "$WORK" -type f -delete 2>/dev/null
  find "$WORK" -depth -type d -empty -delete 2>/dev/null
}
trap cleanup EXIT

read -r -d '' PROG <<'PY'
import contextlib, json, math, os, statistics, sys, time

import torch
import torch.nn as nn

SMOKE = os.environ.get("PGS_CHECK_SMOKE") == "1"
DEV = torch.device("cpu" if SMOKE else "cuda")
WORK = os.environ["PGS_CHECK_WORK"]
FAILS = 0
M_2B, M_08B, M_06B = "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"


def _round(v):
    if isinstance(v, float):
        return float(f"{v:.5g}") if math.isfinite(v) else str(v)
    if isinstance(v, (list, tuple)):
        return [_round(x) for x in v]
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    return v


def emit(check, status, **kw):
    global FAILS
    FAILS += status == "FAIL"
    print(json.dumps({"check": check, "status": status, **{k: _round(v) for k, v in kw.items()}}), flush=True)


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def free():
    import gc

    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()


def timed(fn, iters=1, windows=3, warmup=2):
    """Median seconds per call over `windows` windows of `iters` calls, after `warmup` calls."""
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(windows):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        out.append((time.perf_counter() - t) / iters)
    return statistics.median(out)


def host_time(fn, calls=7):
    """Median host seconds a call takes to return (the GPU drained before each)."""
    out = []
    for _ in range(calls):
        sync()
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    sync()
    return statistics.median(out)


def launches(fn):
    """(GPU kernels, memcpy / memset) one call issues, from the profiler's device activities."""
    if DEV.type != "cuda":
        return None, None
    from torch.profiler import ProfilerActivity, profile

    sync()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        sync()
    kernels = copies = 0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            if e.name.startswith(("Memcpy", "Memset")) or "memcpy" in e.name.lower():
                copies += 1
            else:
                kernels += 1
    return kernels, copies


def tiny_config(layers=4):
    from transformers import AutoConfig

    return AutoConfig.for_model(
        "qwen3", vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=32768,
        tie_word_embeddings=False,
    )


def load(model_id, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    if SMOKE:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(tiny_config()).to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    return model.to(DEV).train()


def vocab_of(model):
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return cfg.vocab_size


def bits_equal(a, b):
    a, b = a.detach(), b.detach()
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    nan = torch.isnan(a.float())
    if not torch.equal(nan, torch.isnan(b.float())):
        return False
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return torch.equal(a[~nan].view(view), b[~nan].view(view))


# ── optimizer: exactness ────────────────────────────────────────────────────────────────────────────────────────


def _param_set(dtype=torch.bfloat16, misaligned=True, extremes=True, fp32_extra=True):
    g = torch.Generator().manual_seed(0)
    shapes = [(4096, 1024), (1000,), (3,), (8193,), (2, 3, 5), (257, 129), (1, 1)]
    if SMOKE:
        shapes = [(64, 32), (1000,), (3,), (8193,), (2, 3, 5)]
    tensors = [torch.randn(s, generator=g) * 0.02 for s in shapes]
    if extremes:
        tensors.append(torch.randn(10000, generator=g) * torch.logspace(-30, 30, 10000))
    params = [nn.Parameter(t.to(dtype).to(DEV)) for t in tensors]
    if misaligned:  # views at odd element offsets of one buffer: the kernels' unaligned variant
        buf = (torch.randn(20011, generator=g) * 0.02).to(dtype).to(DEV)
        params += [nn.Parameter(buf[1:5001]), nn.Parameter(buf[5003:20010])]
    if fp32_extra:  # fp32 parameters inside BF16AdamW take AdamW's update (the fp32 kernel)
        params.append(nn.Parameter((torch.randn(64, 64, generator=g) * 0.02).to(DEV)))
    return params


def _grads_for(params, step):
    g = torch.Generator().manual_seed(1000 + step)
    out = []
    for p in params:
        gr = torch.randn(p.shape, generator=g) * 1e-2
        if step % 3 == 2:  # a few large values, as after a spike
            gr[..., :1] *= 1e4
        out.append(gr.to(p.dtype).to(DEV))
    return out


def _groups(params):
    half = len(params) // 2
    return [
        {"params": params[:half], "lr": 1e-3, "weight_decay": 0.1},
        {"params": params[half:], "lr": 3e-4, "weight_decay": 0.0},
    ]


def _clone(params):
    return [nn.Parameter(p.detach().clone()) for p in params]


def optim_exact():
    from palingenesis import bf16_adamw as B
    from palingenesis.bf16_adamw import BF16AdamW
    from palingenesis.fused_optim import FusedAdamW, triton_available

    ok, why = triton_available()
    if not ok and not SMOKE:  # the smoke run exercises the code on the fallback path
        emit("optim_exact", "SKIP", reason=why)
        return
    for rounding in ("stochastic", "kahan"):
        fused_p = _param_set()
        ref_p, comp_p = _clone(fused_p), _clone(fused_p)
        names = lambda ps: [(f"p{i}", p) for i, p in enumerate(ps)]  # noqa: E731
        fused = BF16AdamW(_groups(fused_p), named=names(fused_p), seed=7, rounding=rounding, multi_tensor=True)
        ref = BF16AdamW(_groups(ref_p), named=names(ref_p), seed=7, rounding=rounding, compile=False)
        comp = BF16AdamW(_groups(comp_p), named=names(comp_p), seed=7, rounding=rounding, compile=not SMOKE)
        exact, fp32_rel, mismatch_compiled = True, 0.0, []
        for step in range(4):
            grads = _grads_for(fused_p, step)
            for ps in (fused_p, ref_p, comp_p):
                for p, gr in zip(ps, grads):
                    p.grad = gr.clone()
            fused.step()
            ref.step()
            comp.step()
            for p, q, c in zip(fused_p, ref_p, comp_p):
                if p.dtype == torch.bfloat16:
                    keys = ("exp_avg", "exp_avg_sq") + (("compensation",) if rounding == "kahan" else ())
                    same = bits_equal(p, q) and all(bits_equal(fused.state[p][k], ref.state[q][k]) for k in keys)
                    exact = exact and same
                    mismatch_compiled.append((p.detach() != c.detach()).float().mean().item())
                else:
                    rel = ((p - q).abs().max() / q.abs().max().clamp_min(1e-30)).item()
                    fp32_rel = max(fp32_rel, rel)
        # the folded clip factor: the fp32 gradient times the factor, against the eager update fed that gradient
        scale = torch.tensor([0.37], device=DEV)
        grads = _grads_for(fused_p, 7)
        for p, gr in zip(fused_p, grads):
            p.grad = gr.clone()
        fused.grad_scale = scale
        fused.step()
        update = B._bf16_adamw_kahan_update if rounding == "kahan" else B._bf16_adamw_update
        scaled_exact = True
        for i, q in enumerate(ref_p):
            group = next(gp for gp in ref.param_groups if any(x is q for x in gp["params"]))
            st = ref._init_state(q)
            st["step"] += 1
            if q.dtype != torch.bfloat16:
                q.grad = grads[i].float() * scale
                ref._step_fp32(q, group, st)
                continue
            hp = torch.tensor(ref._scalars(group, st["step"]), dtype=torch.float32, device=DEV)
            seed = torch.tensor(ref._step_seed(q, st["step"]), device=DEV)
            keys = ("exp_avg", "exp_avg_sq") + (("compensation",) if rounding == "kahan" else ())
            with torch.no_grad():
                update(q.data.view(-1), grads[i].float().view(-1) * scale, *(st[k].view(-1) for k in keys), hp, seed)
        for p, q in zip(fused_p, ref_p):
            if p.dtype == torch.bfloat16:
                scaled_exact = scaled_exact and bits_equal(p, q)
        # off CUDA the fallback multiplies the bf16 gradient by the factor (torch's clipping), not the fp32 one
        scaled_exact = scaled_exact if fused.triton else None
        emit(
            f"optim_exact_bf16_{rounding}",
            "PASS" if exact and scaled_exact is not False and fp32_rel < 1e-5 else "FAIL",
            bit_exact_vs_eager=exact,
            bit_exact_with_clip_factor=scaled_exact,
            fp32_params_max_rel_diff=fp32_rel,
            launches_per_step=fused.last_launches,
            compiled_per_param_mismatch_share=max(mismatch_compiled) if mismatch_compiled else 0.0,
        )
        del fused, ref, comp, fused_p, ref_p, comp_p
        free()

    # denormal intermediates (|g| < 1e-19: g^2 below fp32's normal range): Triton may flush them to zero where the
    # eager update keeps them. Reported, not judged: such gradients do not move a weight either way.
    ps = _param_set(misaligned=False, extremes=False, fp32_extra=False)
    qs = _clone(ps)
    a = BF16AdamW(ps, named=[(str(i), p) for i, p in enumerate(ps)], multi_tensor=True)
    b = BF16AdamW(qs, named=[(str(i), p) for i, p in enumerate(qs)], compile=False)
    g = torch.Generator().manual_seed(5)
    for p, q in zip(ps, qs):
        gr = (torch.randn(p.shape, generator=g) * 1e-21).to(torch.bfloat16).to(DEV)
        p.grad, q.grad = gr.clone(), gr.clone()
    a.step()
    b.step()
    share = lambda x, y: sum((x_ != y_).sum().item() for x_, y_ in zip(x, y)) / sum(t.numel() for t in x)  # noqa: E731
    emit(
        "optim_exact_denormals",
        "INFO",
        weight_mismatch_share=share(ps, qs),
        second_moment_mismatch_share=share([a.state[p]["exp_avg_sq"] for p in ps], [b.state[q]["exp_avg_sq"] for q in qs]),
    )
    del a, b, ps, qs
    free()

    # FusedAdamW (fp32 weights) vs torch AdamW (fused CUDA kernel and foreach)
    ps = _param_set(torch.float32, extremes=False, fp32_extra=False)
    qs, rs = _clone(ps), _clone(ps)
    lr = 1e-3
    fused = FusedAdamW(_groups(ps), lr=lr, betas=(0.9, 0.95))
    tfused = torch.optim.AdamW(_groups(qs), lr=lr, betas=(0.9, 0.95), fused=not SMOKE)
    tforeach = torch.optim.AdamW(_groups(rs), lr=lr, betas=(0.9, 0.95), foreach=True)
    for step in range(5):
        grads = _grads_for(ps, step)
        for group in (ps, qs, rs):
            for p, gr in zip(group, grads):
                p.grad = gr.clone()
        fused.step()
        tfused.step()
        tforeach.step()
    d_fused = max(((p - q).abs().max().item() for p, q in zip(ps, qs)))
    d_foreach = max(((p - q).abs().max().item() for p, q in zip(ps, rs)))
    d_torch = max(((q - r).abs().max().item() for q, r in zip(qs, rs)))
    emit(
        "optim_exact_fp32_adamw",
        "PASS" if d_fused / lr < 1e-2 else "FAIL",
        max_abs_diff_over_lr_vs_torch_fused=d_fused / lr,
        max_abs_diff_over_lr_vs_torch_foreach=d_foreach / lr,
        torch_fused_vs_foreach_over_lr=d_torch / lr,
        launches_per_step=fused.last_launches,
    )


def optim_resume():
    import io

    from palingenesis.bf16_adamw import BF16AdamW
    from palingenesis.fused_optim import FusedAdamW, triton_available

    ok, why = triton_available()
    if not ok and not SMOKE:  # the smoke run exercises the code on the fallback path
        emit("optim_resume", "SKIP", reason=why)
        return

    def make(kind, ps):
        if kind == "fp32":
            return FusedAdamW(_groups(ps), lr=1e-3)
        return BF16AdamW(_groups(ps), named=[(f"p{i}", p) for i, p in enumerate(ps)], seed=3, rounding=kind,
                         multi_tensor=True)

    for kind in ("stochastic", "kahan", "fp32"):
        dtype = torch.float32 if kind == "fp32" else torch.bfloat16
        straight = _param_set(dtype, fp32_extra=kind != "fp32")
        resumed = _clone(straight)
        a = make(kind, straight)
        for step in range(6):
            for p, gr in zip(straight, _grads_for(straight, step)):
                p.grad = gr
            a.step()
        b = make(kind, resumed)
        for step in range(3):
            for p, gr in zip(resumed, _grads_for(resumed, step)):
                p.grad = gr
            b.step()
        buf = io.BytesIO()
        torch.save({"opt": b.state_dict(), "params": [p.detach().cpu() for p in resumed]}, buf)
        buf.seek(0)
        saved = torch.load(buf, weights_only=False)
        fresh = [nn.Parameter(t.to(DEV)) for t in saved["params"]]
        c = make(kind, fresh)
        c.load_state_dict(saved["opt"])
        for step in range(3, 6):
            for p, gr in zip(fresh, _grads_for(fresh, step)):
                p.grad = gr
            c.step()
        same = all(bits_equal(p, q) for p, q in zip(straight, fresh))
        emit(f"optim_resume_{kind}", "PASS" if same else "FAIL", bit_exact=same)
        del a, b, c, straight, resumed, fresh
        free()


# ── optimizer: speed ────────────────────────────────────────────────────────────────────────────────────────────


def optim_bench(model_id):
    from palingenesis.bf16_adamw import BF16AdamW
    from palingenesis.fused_optim import FusedAdamW, triton_available

    ok, why = triton_available()
    if not ok and not SMOKE:
        emit(f"optim_bench[{model_id}]", "SKIP", reason=why)
        return
    model = load(model_id)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    count = sum(p.numel() for p in params)
    g = torch.Generator(device=DEV).manual_seed(0)
    grads = [torch.randn(p.shape, device=DEV, dtype=torch.bfloat16, generator=g) * 1e-3 for p in params]
    for p, gr in zip(params, grads):
        p.grad = gr

    def bf16(rounding, multi):
        return BF16AdamW(params, lr=1e-5, named=named, rounding=rounding, multi_tensor=multi)

    results = {}

    def measure(name, opt, clip=None):
        if clip == "torch":
            def fn():
                torch.nn.utils.clip_grad_norm_(opt_params(opt), 1.0)
                opt.step()
        elif clip == "folded":
            def fn():
                opt.clip_grad_norm_(1.0)
                opt.step()
        else:
            fn = opt.step
        step_s = timed(fn, iters=5, warmup=3)
        host_s = host_time(fn)
        kernels, copies = launches(fn)
        results[name] = step_s
        emit(
            f"optim_bench[{model_id}]",
            "INFO",
            regime=name,
            params_M=count / 1e6,
            tensors=len(params),
            step_ms=step_s * 1e3,
            host_ms=host_s * 1e3,
            gpu_kernels=kernels,
            memcpy=copies,
            bytes_moved_GBps=(count * (14 if "kahan" not in name else 18)) / step_s / 1e9 if "fp32" not in name else None,
        )

    def opt_params(opt):
        return [p for gp in opt.param_groups for p in gp["params"]]

    for rounding in ("stochastic", "kahan"):
        for multi in (False, True):
            opt = bf16(rounding, multi)
            measure(f"adamw_bf16_{rounding}_{'multi_tensor' if multi else 'per_param'}", opt)
            if rounding == "stochastic":
                measure(f"adamw_bf16_{rounding}_{'multi_tensor' if multi else 'per_param'}+clip", opt,
                        "folded" if multi else "torch")
            del opt
            free()
    # fp32 master weights (model.torch_dtype float32): torch's fused AdamW against FusedAdamW
    masters = [nn.Parameter(p.detach().float()) for p in params]
    for m, gr in zip(masters, grads):
        m.grad = gr.float()
    for name, make in (
        ("adamw_fp32_torch_fused", lambda: torch.optim.AdamW(masters, lr=1e-5, fused=True)),
        ("adamw_fp32_torch_foreach", lambda: torch.optim.AdamW(masters, lr=1e-5, foreach=True)),
        ("adamw_fp32_multi_tensor", lambda: FusedAdamW(masters, lr=1e-5)),
    ):
        opt = make()
        measure(name, opt)
        measure(name + "+clip", opt, "folded" if isinstance(opt, FusedAdamW) else "torch")
        del opt
        free()
    speedup = results["adamw_bf16_stochastic_per_param"] / results["adamw_bf16_stochastic_multi_tensor"]
    speedup_clip = results["adamw_bf16_stochastic_per_param+clip"] / results["adamw_bf16_stochastic_multi_tensor+clip"]
    vs_torch = results["adamw_fp32_torch_fused"] / results["adamw_fp32_multi_tensor"]
    emit(
        f"optim_bench[{model_id}]",
        "INFO" if SMOKE else ("PASS" if speedup >= 1.0 and speedup_clip >= 1.0 else "FAIL"),
        multi_tensor_speedup_vs_per_param=speedup,
        with_clipping_speedup=speedup_clip,
        kahan_speedup=results["adamw_bf16_kahan_per_param"] / results["adamw_bf16_kahan_multi_tensor"],
        fp32_multi_tensor_vs_torch_fused=vs_torch,
        fp32_clip_folded_vs_torch=results["adamw_fp32_torch_fused+clip"] / results["adamw_fp32_multi_tensor+clip"],
    )


# ── activation offload ──────────────────────────────────────────────────────────────────────────────────────────


def _prepare(model_id, mode):
    from palingenesis.kernels import apply_activation_checkpointing
    from palingenesis.seco import use_chunk_attention

    model = load(model_id)
    apply_activation_checkpointing(model, mode=mode, respect_saved_tensors_hooks=True if mode == "selective" else None)
    use_chunk_attention(model)
    return model


def _offloader(model, **opts):
    from palingenesis.activation_offload import ActivationOffloader, OffloadOptions
    from palingenesis.distributed import _find_transformer_layers

    if SMOKE:
        opts.setdefault("min_bytes", 0)  # the tiny models' tensors are all below the default
    return ActivationOffloader(_find_transformer_layers(model), DEV, OffloadOptions(**opts), params=model.parameters())


def _step_fn(model, ids, offloader):
    from palingenesis.loss import chunked_cross_entropy_loss, shift_labels
    from palingenesis.train import _dynamic_num_chunks, _get_hidden_states, _get_lm_head

    head, vocab = _get_lm_head(model), vocab_of(model)
    labels = shift_labels(ids.clone())

    def step():
        with offloader.saving() if offloader is not None else contextlib.nullcontext():
            hidden = _get_hidden_states(model, input_ids=ids)
            loss = chunked_cross_entropy_loss(
                hidden, labels, head, num_chunks=_dynamic_num_chunks(ids.numel(), vocab),
                global_valid_tokens=float(ids.numel()),
            )
        loss.backward()
        return loss

    return step


def _ids(model, seq, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, min(vocab_of(model), 100000), (1, seq), generator=g).to(DEV)


def _grad_copy(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _rel_l2(a, b):
    num = sum(((a[k].double() - b[k].double()) ** 2).sum().item() for k in a)
    den = sum((b[k].double() ** 2).sum().item() for k in b)
    return math.sqrt(num / max(den, 1e-300))


def _grads_of(step, model):
    model.zero_grad(set_to_none=True)
    step()
    out = _grad_copy(model)
    model.zero_grad(set_to_none=True)
    return out


def offload_grads(model_id, seq):
    for mode in ("none", "selective", "full"):
        model = _prepare(model_id, mode)
        ids = _ids(model, seq)
        base = _grads_of(_step_fn(model, ids, None), model)
        repeat = _grads_of(_step_fn(model, ids, None), model)
        off = _offloader(model)
        got = _grads_of(_step_fn(model, ids, off), model)
        stats = off.stats()
        off.close()
        rel, noise = _rel_l2(got, base), _rel_l2(repeat, base)
        identical = all(bits_equal(got[k], base[k]) for k in base)
        emit(
            f"offload_grads[{mode}]",
            "PASS" if identical or rel <= max(4 * noise, 2e-3) else "FAIL",
            model=model_id, seq=seq, bit_identical=identical, rel_l2_vs_no_offload=rel,
            rel_l2_no_offload_repeat=noise, offloaded_tensors=stats["tensors"], offloaded_gb=stats["gb"],
        )
        del model, base, repeat, got
        free()


def offload_bench(model_id, seq):
    rows = {}
    for mode in ("none", "selective", "full"):
        try:
            model = _prepare(model_id, mode)
        except torch.OutOfMemoryError:
            free()
            continue
        ids = _ids(model, seq)
        for offload in (False, True):
            name = f"{mode}{'+offload' if offload else ''}"
            off = _offloader(model) if offload else None
            step = _step_fn(model, ids, off)

            def run():
                step()
                model.zero_grad(set_to_none=True)

            row = {}
            try:
                run()
                run()
                sync()
                if DEV.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                row["fwd_bwd_s"] = timed(run, warmup=0)
                if DEV.type == "cuda":
                    row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                    row["reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
                row["tokens_per_s"] = seq / row["fwd_bwd_s"]
                if off is not None:
                    row.update({f"offload_{k}": v for k, v in off.stats().items()})
            except torch.OutOfMemoryError:
                row["oom"] = True
            except RuntimeError as e:
                if "activation_offload" not in str(e):
                    raise
                row["host_limit"] = str(e)[:160]
            model.zero_grad(set_to_none=True)
            if off is not None:
                off.close()
            rows[name] = row
            emit(f"offload_bench[{seq}]", "INFO", model=model_id, setting=name, timed="forward+backward", **row)
            free()
        del model
        free()
    verdicts, ok = {}, True
    for mode in ("none", "selective", "full"):
        base, off = rows.get(mode, {}), rows.get(mode + "+offload", {})
        if "peak_gb" in off and "peak_gb" in base:
            verdicts[mode] = {
                "peak_saved_gb": base["peak_gb"] - off["peak_gb"],
                "time_ratio": off["fwd_bwd_s"] / base["fwd_bwd_s"],
            }
            ok = ok and off["peak_gb"] < base["peak_gb"]
        elif "peak_gb" in off or "fwd_bwd_s" in off:
            verdicts[mode] = {"baseline": "oom" if base.get("oom") else "not measured"}
        elif off.get("host_limit"):  # refused before pinning more than its share of RAM: correct behaviour
            verdicts[mode] = {"offload": "over the host memory limit"}
        elif off.get("oom"):
            verdicts[mode] = {"offload": "oom", "baseline": "oom" if base.get("oom") else "fits"}
            ok = ok and bool(base.get("oom"))  # offload must not fail where the baseline fits
    sel, full_off = rows.get("selective", {}), rows.get("full+offload", {})
    emit(
        f"offload_bench[{seq}]",
        "PASS" if verdicts and ok else "FAIL",
        model=model_id,
        vs_same_checkpointing=verdicts,
        full_offload_vs_selective_peak=(full_off.get("peak_gb", 0) / sel["peak_gb"]) if sel.get("peak_gb") else None,
        full_offload_vs_selective_time=(full_off.get("fwd_bwd_s", 0) / sel["fwd_bwd_s"]) if sel.get("fwd_bwd_s") else None,
    )


def offload_compile(model_id, seq):
    from torch._dynamo.utils import counters

    from palingenesis.train import _compile_layers

    model = _prepare(model_id, "selective")
    if not SMOKE:
        _compile_layers(model)
    else:
        from palingenesis.distributed import _find_transformer_layers

        for layer in _find_transformer_layers(model):
            layer.compile(backend="aot_eager")
    ids = _ids(model, seq)
    plain = _step_fn(model, ids, None)
    base = _grads_of(plain, model)
    repeat = _grads_of(plain, model)
    t_plain = timed(lambda: (plain(), model.zero_grad(set_to_none=True)), warmup=1)
    breaks_before = sum(counters["graph_break"].values())
    off = _offloader(model)
    offl = _step_fn(model, ids, off)
    got = _grads_of(offl, model)
    t_off = timed(lambda: (offl(), model.zero_grad(set_to_none=True)), warmup=1)
    breaks = sum(counters["graph_break"].values()) - breaks_before
    stats = off.stats()
    off.close()
    rel, noise = _rel_l2(got, base), _rel_l2(repeat, base)
    emit(
        "offload_compile",
        "PASS" if rel <= max(4 * noise, 2e-3) else "FAIL",
        model=model_id, seq=seq, rel_l2_vs_no_offload=rel, rel_l2_repeat=noise,
        fwd_bwd_s_compiled=t_plain, fwd_bwd_s_compiled_offload=t_off, new_graph_breaks=breaks,
        offloaded_tensors=stats["tensors"],
    )


def offload_trainer(setting):
    """pgs fit's measurement (the real trainer on worst-case rows) of one setting; written for the comparison."""
    from palingenesis.agent_tooling.profile_memory import measure_memory
    from palingenesis.config import Config

    c = Config()
    c.model.name_or_path = M_2B
    c.model.compile = False
    c.data.max_seq_length = 16384
    c.train.per_device_batch_size = 1
    c.train.gradient_accumulation_steps = 1
    c.train.gradient_checkpointing = "full"
    c.train.optimizer = "adamw_bf16"
    c.train.fused_optimizer = True
    c.memory.activation_offload = setting == "offload"
    try:
        result = measure_memory(c, steps=3)
    except torch.OutOfMemoryError:
        result = {"oom": True}
    with open(os.path.join(WORK, f"trainer_{setting}.json"), "w") as f:
        json.dump(result, f)
    emit("offload_trainer", "INFO", setting=f"full checkpointing{' + offload' if setting == 'offload' else ''}", **result)


def offload_trainer_compare():
    rows = {}
    for s in ("plain", "offload"):
        path = os.path.join(WORK, f"trainer_{s}.json")
        rows[s] = json.load(open(path)) if os.path.exists(path) else {}
    a, b = rows["plain"], rows["offload"]
    ok = "measured_peak_gb" in b and (a.get("oom") or b["measured_peak_gb"] < a.get("measured_peak_gb", 0))
    emit(
        "offload_trainer",
        "PASS" if ok else "FAIL",
        peak_gb={"full": a.get("measured_peak_gb"), "full+offload": b.get("measured_peak_gb")},
        tokens_per_s={"full": a.get("tokens_per_sec"), "full+offload": b.get("tokens_per_sec")},
    )


# ── a short SFT run: loss curves ────────────────────────────────────────────────────────────────────────────────


def _sft_rows(n=600):
    import random

    rng = random.Random(0)
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
             "quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
    rows = []
    for _ in range(n):
        seq = [rng.choice(words) for _ in range(rng.randint(6, 14))]
        rows.append({"messages": [
            {"role": "user", "content": "Write these words in reverse order, separated by commas: " + " ".join(seq)},
            {"role": "assistant", "content": ", ".join(reversed(seq))},
        ]})
    return rows


def sft(regime):
    import palingenesis.logging as L
    from palingenesis.config import Config
    from palingenesis.train import train

    data = os.path.join(WORK, "sft_rows.jsonl")
    if not os.path.exists(data):
        with open(data, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in _sft_rows())
    losses = []
    original = L.Tracker.log

    def log(self, metrics, step=None):
        if "train/loss" in metrics:
            losses.append(float(metrics["train/loss"]))
        return original(self, metrics, step)

    L.Tracker.log = log
    c = Config()
    c.model.name_or_path = M_08B
    c.model.compile = False
    c.model.torch_dtype = "float32" if regime == "fp32_master" else "bfloat16"
    c.train.optimizer = "adamw" if regime == "fp32_master" else "adamw_bf16"
    c.train.fused_optimizer = regime == "adamw_bf16_multi_tensor"
    c.data.dataset, c.data.dataset_split, c.data.streaming, c.data.num_workers = data, "train", False, 0
    c.data.max_seq_length = 512
    c.train.max_steps = 30
    c.train.per_device_batch_size = 8
    c.train.gradient_accumulation_steps = 1
    c.train.learning_rate, c.train.min_learning_rate, c.train.warmup_ratio = 1e-5, 1e-6, 0.1
    c.train.save_steps, c.train.save_final = 0, False
    c.train.output_dir = os.path.join(WORK, f"sft_{regime}")
    c.train.spike_detection = False
    c.logging.use_wandb = c.logging.use_trackio = False
    t = time.perf_counter()
    train(c)
    with open(os.path.join(WORK, f"sft_{regime}.json"), "w") as f:
        json.dump({"losses": losses, "seconds": time.perf_counter() - t}, f)
    emit("sft", "INFO", regime=regime, model=M_08B, steps=len(losses), first=losses[0], last=losses[-1],
         seconds=time.perf_counter() - t)


def sft_compare():
    runs = {}
    for r in ("fp32_master", "adamw_bf16_multi_tensor", "adamw_bf16_per_param"):
        path = os.path.join(WORK, f"sft_{r}.json")
        if os.path.exists(path):
            runs[r] = json.load(open(path))["losses"]
    if len(runs) < 3:
        emit("sft", "FAIL", reason=f"runs missing: {sorted(set(('fp32_master', 'adamw_bf16_multi_tensor', 'adamw_bf16_per_param')) - set(runs))}")
        return
    tail = {k: statistics.mean(v[-10:]) for k, v in runs.items()}
    ref = tail["fp32_master"]
    gap = (tail["adamw_bf16_multi_tensor"] - ref) / ref
    gap_pp = (tail["adamw_bf16_multi_tensor"] - tail["adamw_bf16_per_param"]) / tail["adamw_bf16_per_param"]
    learned = runs["fp32_master"][-1] < runs["fp32_master"][0]
    emit(
        "sft",
        "PASS" if learned and abs(gap) <= 0.03 and abs(gap_pp) <= 0.03 else "FAIL",
        last10_mean=tail,
        multi_tensor_vs_fp32_master=gap,
        multi_tensor_vs_per_param=gap_pp,
        curves={k: [round(x, 4) for x in v] for k, v in runs.items()},
    )


# ── env ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def env():
    info = {"torch": torch.__version__, "smoke": SMOKE}
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    if DEV.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_gb=props.total_memory / 2**30, cuda=torch.version.cuda)
    from palingenesis.kernels import sac_routes_saved_tensors_hooks

    info["sac_routes_saved_tensors_hooks"] = sac_routes_saved_tensors_hooks()
    from palingenesis.activation_offload import host_limit_bytes

    info["offload_host_limit_gb"] = (host_limit_bytes() or 0) / 2**30
    from palingenesis.fused_optim import _Launcher, triton_available

    if triton_available()[0]:  # compile every kernel once on aligned tensors: did the vectorized variant compile?
        from palingenesis.bf16_adamw import BF16AdamW
        from palingenesis.fused_optim import FusedAdamW

        try:
            for make, dtype in (
                (lambda ps: BF16AdamW(ps, multi_tensor=True), torch.bfloat16),
                (lambda ps: BF16AdamW(ps, multi_tensor=True, rounding="kahan"), torch.bfloat16),
                (lambda ps: FusedAdamW(ps), torch.float32),
            ):
                ps = [nn.Parameter(torch.zeros(4096, dtype=dtype, device=DEV))]
                ps[0].grad = torch.ones_like(ps[0])
                opt = make(ps)
                opt.clip_grad_norm_(1.0)
                opt.step()
            sync()
            info["kernels_compile"] = True
            info["aligned_variant"] = not _Launcher.unaligned_only
        except Exception as e:  # noqa: BLE001 - reported
            info["kernels_compile"] = f"{type(e).__name__}: {str(e)[:300]}"
    emit("env", "INFO" if info.get("kernels_compile", True) is True else "FAIL", **info)


CHECKS = {
    "env": env,
    "optim_exact": optim_exact,
    "optim_resume": optim_resume,
    "optim_bench": lambda m: optim_bench(m),
    "offload_grads": lambda m, s: offload_grads(m, int(s)),
    "offload_bench": lambda m, s: offload_bench(m, int(s)),
    "offload_compile": lambda m, s: offload_compile(m, int(s)),
    "offload_trainer": offload_trainer,
    "offload_trainer_compare": offload_trainer_compare,
    "sft": sft,
    "sft_compare": sft_compare,
}

if __name__ == "__main__":
    name, *args = sys.argv[1:]
    if not SMOKE and not torch.cuda.is_available():
        emit(name, "FAIL", reason="no CUDA device")
        sys.exit(1)
    try:
        CHECKS[name](*args)
    except Exception as e:  # noqa: BLE001 - reported as the check's failure; the other checks still run
        import traceback

        traceback.print_exc()
        emit(name, "FAIL", args=args, error=f"{type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if FAILS else 0)
PY

FAILED=0
START=$(date +%s)
run() {
  local log="$WORK/log_$1_$RANDOM.txt"
  if ! "$PGS_PY" -c "$PROG" "$@" 2>"$log"; then
    FAILED=1
    echo "---- $* failed; last lines of its log:" >&2
    tail -n 40 "$log" >&2
  fi
}

M2B="Qwen/Qwen3.5-2B"
if [ "${PGS_CHECK_SMOKE:-0}" = "1" ]; then SEQ_SHORT=256; SEQ_LONG=1024; else SEQ_SHORT=4096; SEQ_LONG=16384; fi

run env
run optim_exact
run optim_resume
run optim_bench "Qwen/Qwen3.5-0.8B"
run optim_bench "Qwen/Qwen3.5-2B"
run offload_grads "$M2B" "$SEQ_SHORT"
run offload_bench "$M2B" "$SEQ_SHORT"
run offload_bench "$M2B" "$SEQ_LONG"
run offload_compile "Qwen/Qwen3-0.6B" "$SEQ_SHORT"
if [ "${PGS_CHECK_SMOKE:-0}" != "1" ]; then
  run offload_trainer plain
  run offload_trainer offload
  run offload_trainer_compare
  run sft fp32_master
  run sft adamw_bf16_multi_tensor
  run sft adamw_bf16_per_param
  run sft_compare
fi
echo "{\"check\": \"summary\", \"status\": \"$([ $FAILED = 0 ] && echo PASS || echo FAIL)\", \"seconds\": $(( $(date +%s) - START ))}"
exit $FAILED
