#!/usr/bin/env bash
# Tensor / pipeline parallelism on real CUDA kernels, on ONE GPU (palingenesis.tensor_parallel,
# palingenesis.pipeline_parallel). One JSON line per check: {"check", "status": PASS|FAIL|SKIP, numbers...}.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/pgs/src bash scripts/gpu_checks/tp_pp.sh
#
# Needs: torch (2.11 tested target), transformers, Qwen/Qwen3-0.6B and Qwen/Qwen3.5-0.8B in the HF cache
# (HF_HUB_OFFLINE=1 is set). Optional: PGS_SEQ (tokens per row, default 512), PGS_DEVICE=cpu (a CPU smoke run),
# PGS_DTYPES (default "fp32,bf16" on CUDA, "fp32" on the CPU), PGS_QWEN3 / PGS_QWEN35 (other model ids or
# local folders of the same architectures).
#
# What one GPU can check (NCCL refuses two ranks on one device):
#   tp1_*        the tensor-parallel plan on a one-rank mesh: every DTensor layout, hook, local view and
#                rank-by-rank (Gated DeltaNet) parameter on CUDA kernels, against the plain model (loss, every
#                gradient); and the DTensor overhead (tokens/s of both; eager DTensor dispatch is Python per
#                operation, compile_tp shows what model.compile makes of it);
#   pp_*         a two-stage pipeline on one rank (zbv / dualpipev keep both stages of the V on the same rank),
#                FSDP2 inside: loss and gradients against the plain model;
#   pp_tp_*      both at once;
#   compile_tp_* torch.compile of the tensor-parallel layers against eager (Qwen3, and Qwen3.5's Gated DeltaNet);
#   async_tp     whether this torch enables micro-pipelined TP (symmetric memory);
#   tp2_cpu_*    two gloo processes on the CPU (float32): a real tp=2 split, the reductions over the group
#                included, against the plain model. Always runs;
#   tp2_gloo_cuda  the same two processes sharing the GPU (CUDA tensors through gloo). Gloo is a CPU transport:
#                a probe of the collectives the split needs runs first in processes of its own, and the check
#                is SKIPped, with the collective that failed or crashed, when this build cannot run them.
#
# Every gradient check runs once per dtype and names the parameters that differ most ("worst"):
#   fp32   float32 weights, no autocast: the correctness check. Tight tolerances (loss 1e-4, gradients 2e-3);
#          a wrong reduction or layout is an error of order 1 here.
#   bf16   bfloat16 weights under bf16 autocast, as deployed. A bf16 backward is only accurate to a few percent
#          per parameter, and much less for the norm weights (their gradient is a sum of ~10^4 cancelling
#          bf16 terms, accumulated in bf16): any change in the order of operations (micro-batches, a compiled
#          graph, another reduction) moves them by that much. So each parameter is judged against the float32
#          gradient of the plain model, next to the plain bf16 model's own distance from it ("noise"):
#          FAIL when diff > 3 x noise + 0.02.
# Exit code: 0 when no check FAILed.
set -u
PY="${PGS_PY:-python}"
SRC="${PGS_SRC:-$(cd "$(dirname "$0")/../../src" && pwd)}"
export HF_HUB_OFFLINE=1 PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
TMP="$(mktemp -d)"
CHECKS="$TMP/tp_pp_checks.py"
cleanup() { rm -f "$CHECKS" "$TMP"/*.json; rmdir "$TMP" 2>/dev/null; }
trap cleanup EXIT

cat > "$CHECKS" <<'PYEOF'
import copy
import json
import os
import socket
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist

DEVICE = os.environ.get("PGS_DEVICE", "cuda")
SEQ = int(os.environ.get("PGS_SEQ", "512"))
DTYPES = [d for d in os.environ.get("PGS_DTYPES", "fp32,bf16" if DEVICE == "cuda" else "fp32").split(",") if d]
QWEN3 = os.environ.get("PGS_QWEN3", "Qwen/Qwen3-0.6B")
QWEN35 = os.environ.get("PGS_QWEN35", "Qwen/Qwen3.5-0.8B")
ROWS = 2
FAILED = []
# relative tolerances (loss, gradient) of the float32 pass; bf16: the loss, and the noise rule for gradients
FP32_TOL = (1e-4, 2e-3)
BF16_LOSS_TOL = 2e-3
BF16_NOISE_FACTOR, BF16_NOISE_FLOOR = 3.0, 0.02


def emit(check, status, **numbers):
    line = {"check": check, "status": status}
    line.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in numbers.items()})
    print(json.dumps(line), flush=True)
    if status == "FAIL":
        FAILED.append(check)


def run_check(name, fn, *args):
    try:
        fn(name, *args)
    except Exception as e:  # noqa: BLE001 - every check reports, none stops the others
        emit(name, "FAIL", error=f"{type(e).__name__}: {e}"[:600], trace=traceback.format_exc()[-1500:])


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def one_rank_group():
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    for attempt in range(5):  # another process can take the port between the probe and the listen
        os.environ["MASTER_PORT"] = str(free_port())
        try:
            dist.init_process_group("nccl" if DEVICE == "cuda" else "gloo", rank=0, world_size=1)
            return
        except RuntimeError:
            if attempt == 4:
                raise


def pipeline_on_this_backend():
    """None when the one-rank V-pipeline can run here, else why not. torch 2.11's schedules warm the stages'
    connections up with a send from the rank to itself, which NCCL accepts and gloo (the CPU smoke run) crashes
    on; later versions pass same-rank stages' data in memory."""
    from torch.distributed.pipelining.schedules import PipelineScheduleMulti

    if DEVICE == "cuda" or hasattr(PipelineScheduleMulti, "_warmup_p2p"):
        return None
    return f"torch {torch.__version__} pipeline schedules send from a rank to itself, which gloo cannot do (CPU run)"


def load(name, dtype, untie=False, device=None):
    from transformers import AutoModelForCausalLM

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    if untie:  # pipeline stages keep the embeddings and the head apart
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    return model.to(device or DEVICE)


def batch(vocab, seed=0, device=None):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (ROWS, SEQ), generator=g)
    labels = ids.clone()
    labels[:, : SEQ // 4] = -100  # a prompt without loss
    from palingenesis.loss import shift_labels

    return ids.to(device or DEVICE), shift_labels(labels).to(device or DEVICE)


def autocast(dtype):
    return torch.amp.autocast(DEVICE, dtype=torch.bfloat16, enabled=dtype == "bf16")


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def plain_step(model, ids, labels, dtype):
    from palingenesis.logits import final_hidden_states, output_head
    from palingenesis.loss import chunked_cross_entropy_loss

    model.zero_grad(set_to_none=True)
    n = int((labels != -100).sum())
    with autocast(dtype):
        hidden = final_hidden_states(model, ids)
        loss = chunked_cross_entropy_loss(hidden, labels, output_head(model), num_chunks=4, global_valid_tokens=n)
    loss.backward()
    return float(loss.detach())


def tp_step(model, tp, ids, labels, dtype):
    from palingenesis.logits import final_hidden_states, output_head
    from palingenesis.loss import chunked_cross_entropy_loss

    model.zero_grad(set_to_none=True)
    n = int((labels != -100).sum())
    with autocast(dtype):
        hidden = final_hidden_states(model, tp.pad(ids))
        loss = chunked_cross_entropy_loss(
            hidden, tp.shard(labels, -100), output_head(model), num_chunks=4, global_valid_tokens=n
        )
    loss.backward()
    total = loss.detach().float().clone()
    if dist.is_initialized() and tp.size > 1:
        dist.all_reduce(total, group=tp.group)
    return float(total)


def grads(model):
    from torch.distributed.tensor import DTensor

    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        out[name] = g.detach().float().clone()
    return out


def distances(ref, got):
    """Per parameter: ||got - ref|| / ||ref||; and the parameters of `ref` without a gradient in `got`."""
    out = {}
    for name, g in ref.items():
        if name in got:
            out[name] = float((got[name].to(g.device) - g).norm() / g.norm().clamp_min(1e-12))
    return out, sorted(set(ref) - set(got))


_FP32_REFERENCE = {}


def fp32_reference(model_id, untie, seed):
    """Loss and gradients of the plain float32 model (kept for the bf16 pass of the same check)."""
    key = (model_id, untie, seed)
    if key not in _FP32_REFERENCE:
        _FP32_REFERENCE.clear()  # one model's gradients at a time
        model = load(model_id, "fp32", untie)
        ids, labels = batch(model.config.vocab_size, seed)
        loss = plain_step(model, ids, labels, "fp32")
        _FP32_REFERENCE[key] = (loss, grads(model))
        del model
    return _FP32_REFERENCE[key]


def judge(dtype, loss_ref, loss, g_ref, g, exact=None):
    """The verdict of one gradient comparison and the numbers that explain it.

    fp32: `g` against `g_ref` (the plain model, same dtype), tight tolerances.
    bf16: `g` and `g_ref` (the plain bf16 model) both against `exact` (the plain float32 gradients): a parameter
    fails when it is further from the float32 gradient than BF16_NOISE_FACTOR x the plain bf16 model is, plus
    BF16_NOISE_FLOOR."""
    direct, missing = distances(g_ref, g)
    loss_rel = abs(loss - loss_ref) / abs(loss_ref)
    worst_direct = max(direct, key=direct.get) if direct else ""
    numbers = {
        "dtype": dtype,
        "loss_ref": loss_ref,
        "loss": loss,
        "loss_rel_diff": loss_rel,
        "grad_max_rel_diff": direct.get(worst_direct, 0.0),
        "grad_worst_param": worst_direct,
        "missing_grads": missing[:5],
    }
    if dtype == "fp32" or exact is None:
        loss_tol, grad_tol = FP32_TOL
        over = {n: d for n, d in direct.items() if d >= grad_tol}
        top = sorted(direct, key=direct.get, reverse=True)[:5]
        numbers.update(
            loss_tol=loss_tol,
            grad_tol=grad_tol,
            params_over_tol=len(over),
            worst=[{"param": n, "diff": round(direct[n], 6)} for n in top],
        )
    else:
        loss_tol = BF16_LOSS_TOL
        noise, _ = distances(exact, g_ref)
        diff, _ = distances(exact, g)
        excess = {n: diff[n] - (BF16_NOISE_FACTOR * noise.get(n, 0.0) + BF16_NOISE_FLOOR) for n in diff}
        over = {n: e for n, e in excess.items() if e > 0}
        top = sorted(excess, key=excess.get, reverse=True)[:5]
        numbers.update(
            loss_tol=loss_tol,
            rule=f"diff_vs_fp32 <= {BF16_NOISE_FACTOR} x bf16_noise + {BF16_NOISE_FLOOR}",
            params_over_tol=len(over),
            median_bf16_noise=statistics.median(noise.values()) if noise else 0.0,
            max_bf16_noise=max(noise.values()) if noise else 0.0,
            worst=[
                {"param": n, "diff_vs_fp32": round(diff[n], 6), "bf16_noise": round(noise.get(n, 0.0), 6)} for n in top
            ],
        )
    ok = loss_rel < loss_tol and not over and not missing
    return ok, numbers


def timed(fn, repeats=3):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def per_dtype(check):
    """Run `check(name, dtype, ...)` once per dtype, as <name>_<dtype>; a failure in one does not stop the other."""

    def runner(name, *args):
        for dtype in DTYPES:
            run_check(f"{name}_{dtype}", check, dtype, *args)

    return runner


def check_tp1(name, dtype, model_id):
    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.tensor_parallel import apply_tensor_parallel

    one_rank_group()
    exact = fp32_reference(model_id, False, 0)[1] if dtype == "bf16" else None
    ref = load(model_id, dtype)
    model = copy.deepcopy(ref)
    tp = apply_tensor_parallel(model, init_device_mesh(DEVICE, (1,), mesh_dim_names=("tp",)))
    ids, labels = batch(ref.config.vocab_size)
    loss_ref = plain_step(ref, ids, labels, dtype)
    g_ref = grads(ref)
    loss = tp_step(model, tp, ids, labels, dtype)
    ok, numbers = judge(dtype, loss_ref, loss, g_ref, grads(model), exact)
    del g_ref
    t_ref = timed(lambda: plain_step(ref, ids, labels, dtype))
    t_tp = timed(lambda: tp_step(model, tp, ids, labels, dtype))
    emit(
        name,
        "PASS" if ok else "FAIL",
        **numbers,
        tok_s_ref=ROWS * SEQ / t_ref,
        tok_s_tp=ROWS * SEQ / t_tp,
        dtensor_overhead_pct=100 * (t_tp / t_ref - 1),  # eager dispatch; informative, never a failure
    )


def check_pp(name, dtype, model_id, schedule, tp_on=False):
    from types import SimpleNamespace

    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.distributed import apply_fsdp
    from palingenesis.pipeline_parallel import PipelineParallel
    from palingenesis.tensor_parallel import apply_tensor_parallel

    problem = pipeline_on_this_backend()
    if problem:
        emit(name, "SKIP", reason=problem)
        return
    one_rank_group()
    exact = fp32_reference(model_id, True, 0)[1] if dtype == "bf16" else None
    ref = load(model_id, dtype, untie=True)
    model = copy.deepcopy(ref)
    names = ("pp", "dp", "tp") if tp_on else ("pp", "dp")
    mesh = init_device_mesh(DEVICE, (1,) * len(names), mesh_dim_names=names)
    parallel = SimpleNamespace(
        pp_schedule=schedule,
        pp_stages_per_rank=0,
        pp_microbatches=ROWS,
        pp_first_stage_less_layers=1,
        pp_last_stage_less_layers=1,
        reshard_after_forward=True,
        cpu_offload=False,
    )
    pipe = PipelineParallel(model, mesh, parallel, seq_len=SEQ, flatten_packed=False)
    if tp_on:
        pipe.tp = apply_tensor_parallel(pipe.model, mesh["tp"], pipeline=True)
    pipe.loss_kind, pipe.num_chunks = "chunked", (lambda n: 4)
    pipe.build(torch.device(DEVICE), lambda m: apply_fsdp(m, mesh, parallel, bf16=dtype == "bf16", reshard_layers=False))
    pipe.train(True)
    ids, labels = batch(ref.config.vocab_size)
    loss_ref = plain_step(ref, ids, labels, dtype)
    g_ref = grads(ref)
    n = int((labels != -100).sum())

    def step():
        pipe.model.zero_grad(set_to_none=True)
        with autocast(dtype):
            return pipe.step(ids, labels, None, None, n)

    loss = step()
    ok, numbers = judge(dtype, loss_ref, loss, g_ref, grads(pipe.model), exact)
    del g_ref
    t_ref = timed(lambda: plain_step(ref, ids, labels, dtype))
    t_pp = timed(step)
    emit(
        name,
        "PASS" if ok else "FAIL",
        stages=pipe.num_stages,
        **numbers,
        tok_s_ref=ROWS * SEQ / t_ref,
        tok_s_pp=ROWS * SEQ / t_pp,
    )


def check_compile(name, dtype, model_id):
    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.tensor_parallel import apply_tensor_parallel
    from palingenesis.train import _compile_layers

    if DEVICE != "cuda":
        emit(name, "SKIP", reason="inductor check on CUDA only")
        return
    one_rank_group()
    exact = fp32_reference(model_id, False, 1)[1] if dtype == "bf16" else None
    ref = load(model_id, dtype)
    eager = copy.deepcopy(ref)
    tp_e = apply_tensor_parallel(eager, init_device_mesh(DEVICE, (1,), mesh_dim_names=("tp",)))
    compiled = copy.deepcopy(ref)
    tp_c = apply_tensor_parallel(compiled, init_device_mesh(DEVICE, (1,), mesh_dim_names=("tp",)))
    del ref
    _compile_layers(compiled)
    ids, labels = batch(compiled.config.vocab_size, seed=1)
    t = time.perf_counter()
    loss_c = tp_step(compiled, tp_c, ids, labels, dtype)
    compile_s = time.perf_counter() - t
    g_c = grads(compiled)
    loss_e = tp_step(eager, tp_e, ids, labels, dtype)
    ok, numbers = judge(dtype, loss_e, loss_c, grads(eager), g_c, exact)  # eager is the reference
    del g_c
    t_e = timed(lambda: tp_step(eager, tp_e, ids, labels, dtype))
    t_c = timed(lambda: tp_step(compiled, tp_c, ids, labels, dtype))
    emit(
        name,
        "PASS" if ok else "FAIL",
        **numbers,
        first_step_s=compile_s,
        tok_s_eager=ROWS * SEQ / t_e,
        tok_s_compiled=ROWS * SEQ / t_c,
    )


def check_async_tp(name):
    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.tensor_parallel import TensorParallel
    from palingenesis.train import _enable_async_tp

    one_rank_group()
    tp = TensorParallel.from_mesh(init_device_mesh(DEVICE, (1,), mesh_dim_names=("tp",)))
    _enable_async_tp(tp)
    import torch._inductor.config as inductor_config

    emit(name, "PASS" if inductor_config._micro_pipeline_tp else "FAIL", torch=torch.__version__)


# ── two processes over gloo ─────────────────────────────────────────────────────


def pair_worker(rank, port, model_id, device, out):
    """A real tp=2 split in float32 against the plain model. Rank 0 holds the reference and writes the result."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from palingenesis.tensor_parallel import apply_tensor_parallel

        loss_ref, g_ref = 0.0, {}
        if rank == 0:
            ref = load(model_id, "fp32", device=device)
            ids, labels = batch(ref.config.vocab_size, device=device)
            loss_ref = plain_step(ref, ids, labels, "fp32")
            g_ref = grads(ref)
            del ref
        model = load(model_id, "fp32", device=device)
        ids, labels = batch(model.config.vocab_size, device=device)
        tp = apply_tensor_parallel(model, init_device_mesh(device, (2,), mesh_dim_names=("tp",)))
        loss = tp_step(model, tp, ids, labels, "fp32")
        # the embeddings / head are whole on each rank and see only its tokens: sum them over the group (FSDP does)
        g = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if hasattr(p.grad, "full_tensor"):
                full = p.grad.full_tensor().float()
            else:
                full = p.grad.detach().float().clone()
                dist.all_reduce(full)
            g[name] = tp.sections[name].canonical(full, 2) if name in tp.sections else full
            p.grad = None
        if rank == 0:
            ok, numbers = judge("fp32", loss_ref, loss, g_ref, g)
            with open(out, "w") as f:
                json.dump({"ok": ok, "numbers": numbers}, f)
    finally:
        dist.destroy_process_group()


def probe_worker(rank, port, device, progress):
    """The collectives a tensor-parallel step runs, on small tensors of `device`: gloo is a CPU transport, and a
    build that cannot run one of them on CUDA memory raises or crashes here, not in the middle of a model."""
    import torch.distributed._functional_collectives as funcol

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)

    def mark(step):
        if rank == 0:
            with open(progress, "w") as f:
                f.write(step)

    try:
        group = dist.group.WORLD
        x = torch.full((2, 4, 6), float(rank + 1), device=device)
        mark("all_reduce")
        t = x.clone()
        dist.all_reduce(t)
        assert float(t[0, 0, 0]) == 3.0
        mark("all_gather_tensor")
        gathered = funcol.all_gather_tensor(x, 1, group)
        gathered = gathered.wait() if hasattr(gathered, "wait") else gathered
        assert tuple(gathered.shape) == (2, 8, 6) and float(gathered.sum()) == 3.0 * x.numel()
        mark("reduce_scatter_tensor")
        scattered = funcol.reduce_scatter_tensor(x, "sum", 1, group)
        scattered = scattered.wait() if hasattr(scattered, "wait") else scattered
        assert tuple(scattered.shape) == (2, 2, 6) and float(scattered[0, 0, 0]) == 3.0
        mark("done")
    finally:
        dist.destroy_process_group()


def gloo_cuda_problem(tmp):
    """None when gloo runs the split's collectives on CUDA tensors in this build, else why it does not."""
    import torch.multiprocessing as mp

    progress = os.path.join(tmp, "probe.json")
    try:
        mp.spawn(probe_worker, args=(free_port(), "cuda", progress), nprocs=2, join=True)
    except Exception as e:  # noqa: BLE001 - a raise or a crash (ProcessExitedException) of the probe
        step = open(progress).read() if os.path.exists(progress) else "init_process_group"
        return f"gloo cannot run {step} on CUDA tensors in this build ({type(e).__name__}: {str(e).strip()[:200]})"
    step = open(progress).read() if os.path.exists(progress) else ""
    return None if step == "done" else f"the gloo probe stopped at {step or 'start'}"


def check_tp2(name, model_id, device, tmp):
    import torch.multiprocessing as mp

    if device == "cuda":
        problem = gloo_cuda_problem(tmp)
        if problem:
            emit(name, "SKIP", reason=problem + "; tp=2 is checked on CPU ranks (tp2_cpu_*)")
            return
    out = os.path.join(tmp, f"{name}.json")
    mp.spawn(pair_worker, args=(free_port(), model_id, device, out), nprocs=2, join=True)
    with open(out) as f:
        r = json.load(f)
    emit(name, "PASS" if r["ok"] else "FAIL", device=device, **r["numbers"])


def main():
    tmp = sys.argv[1]
    emit(
        "environment",
        "PASS",
        torch=torch.__version__,
        device=DEVICE,
        cuda=torch.cuda.is_available(),
        seq=SEQ,
        dtypes=DTYPES,
    )
    # two processes first: the one-rank checks below initialise this process's own group
    run_check("tp2_cpu_qwen3", check_tp2, QWEN3, "cpu", tmp)
    run_check("tp2_cpu_qwen35", check_tp2, QWEN35, "cpu", tmp)
    if DEVICE == "cuda":
        run_check("tp2_gloo_cuda_qwen3", check_tp2, QWEN3, "cuda", tmp)
    per_dtype(check_tp1)("tp1_qwen3", QWEN3)
    per_dtype(check_tp1)("tp1_qwen35", QWEN35)
    per_dtype(check_pp)("pp_zbv_qwen3", QWEN3, "zbv")
    per_dtype(check_pp)("pp_dualpipev_qwen3", QWEN3, "dualpipev")
    per_dtype(check_pp)("pp_zbv_qwen35", QWEN35, "zbv")
    per_dtype(check_pp)("pp_tp_zbv_qwen35", QWEN35, "zbv", True)
    per_dtype(check_compile)("compile_tp_qwen3", QWEN3)
    per_dtype(check_compile)("compile_tp_qwen35", QWEN35)  # the Gated DeltaNet local views inside a compiled layer
    run_check("async_tp", check_async_tp)
    print(json.dumps({"check": "summary", "status": "FAIL" if FAILED else "PASS", "failed": FAILED}), flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
PYEOF

"$PY" "$CHECKS" "$TMP"
