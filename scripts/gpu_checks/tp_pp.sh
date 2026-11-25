#!/usr/bin/env bash
# Tensor / pipeline parallelism on real CUDA kernels, on ONE GPU (palingenesis.tensor_parallel,
# palingenesis.pipeline_parallel). One JSON line per check: {"check", "status": PASS|FAIL|SKIP, numbers...}.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/pgs/src bash scripts/gpu_checks/tp_pp.sh
#
# Needs: torch (2.11 tested target), transformers, Qwen/Qwen3-0.6B and Qwen/Qwen3.5-0.8B in the HF cache
# (HF_HUB_OFFLINE=1 is set). Optional: PGS_SEQ (tokens per row, default 512), PGS_DEVICE=cpu (a CPU smoke run).
#
# What one GPU can check (NCCL refuses two ranks on one device):
#   tp1_*        the tensor-parallel plan on a one-rank mesh: every DTensor layout, hook, local view and
#                rank-by-rank (Gated DeltaNet) parameter on CUDA kernels, against the plain model (loss, every
#                gradient); and the DTensor overhead (tokens/s of both);
#   pp_*         a two-stage pipeline on one rank (zbv / dualpipev keep both stages of the V on the same rank),
#                FSDP2 inside: loss and gradients against the plain model;
#   pp_tp_*      both at once;
#   compile_tp   torch.compile of the tensor-parallel layers against eager;
#   async_tp     whether this torch enables micro-pipelined TP (symmetric memory);
#   tp2_gloo     two processes sharing the GPU over gloo (CUDA tensors through gloo): a real tp=2 split
#                against the plain model; SKIP when this gloo build lacks a CUDA collective it needs.
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
ROWS = 2
FAILED = []


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
    os.environ.setdefault("MASTER_PORT", str(free_port()))
    dist.init_process_group("nccl" if DEVICE == "cuda" else "gloo", rank=0, world_size=1)


def load(name, untie=False):
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    if untie:  # pipeline stages keep the embeddings and the head apart
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    return model.to(DEVICE)


def batch(vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (ROWS, SEQ), generator=g)
    labels = ids.clone()
    labels[:, : SEQ // 4] = -100  # a prompt without loss
    from palingenesis.loss import shift_labels

    return ids.to(DEVICE), shift_labels(labels).to(DEVICE)


def autocast():
    return torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda")


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def plain_step(model, ids, labels):
    from palingenesis.logits import final_hidden_states, output_head
    from palingenesis.loss import chunked_cross_entropy_loss

    model.zero_grad(set_to_none=True)
    n = int((labels != -100).sum())
    with autocast():
        hidden = final_hidden_states(model, ids)
        loss = chunked_cross_entropy_loss(hidden, labels, output_head(model), num_chunks=4, global_valid_tokens=n)
    loss.backward()
    return float(loss)


def tp_step(model, tp, ids, labels):
    from palingenesis.logits import final_hidden_states, output_head
    from palingenesis.loss import chunked_cross_entropy_loss

    model.zero_grad(set_to_none=True)
    n = int((labels != -100).sum())
    with autocast():
        hidden = final_hidden_states(model, tp.pad(ids))
        loss = chunked_cross_entropy_loss(
            hidden, tp.shard(labels, -100), output_head(model), num_chunks=4, global_valid_tokens=n
        )
    loss.backward()
    total = loss.detach().float().clone()
    if dist.is_initialized() and tp.size > 1:
        dist.all_reduce(total, group=tp.group)
    return float(total)


def grads(model, canonical=None):
    from torch.distributed.tensor import DTensor

    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        g = g.detach().float()
        out[name] = canonical(name, g) if canonical else g
    return out


def compare(ref, got):
    worst, where = 0.0, ""
    missing = sorted(set(ref) - set(got))
    for name, g in ref.items():
        if name not in got:
            continue
        rel = float((got[name].to(g.device) - g).norm() / g.norm().clamp_min(1e-12))
        if rel > worst:
            worst, where = rel, name
    return worst, where, missing


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


def tolerance():
    return (2e-3, 3e-2) if DEVICE == "cuda" else (1e-5, 1e-3)  # (loss, gradient) relative


def check_tp1(name, model_id):
    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.tensor_parallel import apply_tensor_parallel

    one_rank_group()
    ref = load(model_id)
    model = copy.deepcopy(ref)
    tp = apply_tensor_parallel(model, init_device_mesh(DEVICE, (1,), mesh_dim_names=("tp",)))
    ids, labels = batch(ref.config.vocab_size)
    loss_ref = plain_step(ref, ids, labels)
    g_ref = grads(ref)
    loss = tp_step(model, tp, ids, labels)
    g = grads(model)
    worst, where, missing = compare(g_ref, g)
    t_ref = timed(lambda: plain_step(ref, ids, labels))
    t_tp = timed(lambda: tp_step(model, tp, ids, labels))
    lt, gt = tolerance()
    rel = abs(loss - loss_ref) / abs(loss_ref)
    ok = rel < lt and worst < gt and not missing
    emit(
        name,
        "PASS" if ok else "FAIL",
        loss_ref=loss_ref,
        loss=loss,
        loss_rel_diff=rel,
        grad_max_rel_diff=worst,
        grad_worst_param=where,
        missing_grads=missing[:5],
        tok_s_ref=ROWS * SEQ / t_ref,
        tok_s_tp=ROWS * SEQ / t_tp,
        dtensor_overhead_pct=100 * (t_tp / t_ref - 1),
    )


def check_pp(name, model_id, schedule, tp_on=False):
    from types import SimpleNamespace

    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.distributed import apply_fsdp
    from palingenesis.pipeline_parallel import PipelineParallel
    from palingenesis.tensor_parallel import apply_tensor_parallel

    one_rank_group()
    ref = load(model_id, untie=True)
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
    pipe.build(torch.device(DEVICE), lambda m: apply_fsdp(m, mesh, parallel, bf16=DEVICE == "cuda", reshard_layers=False))
    pipe.train(True)
    ids, labels = batch(ref.config.vocab_size)
    loss_ref = plain_step(ref, ids, labels)
    g_ref = grads(ref)
    n = int((labels != -100).sum())

    def step():
        pipe.model.zero_grad(set_to_none=True)
        with autocast():
            return pipe.step(ids, labels, None, None, n)

    loss = step()
    g = grads(pipe.model)
    worst, where, missing = compare(g_ref, g)
    t_ref = timed(lambda: plain_step(ref, ids, labels))
    t_pp = timed(step)
    lt, gt = tolerance()
    rel = abs(loss - loss_ref) / abs(loss_ref)
    ok = rel < lt and worst < gt and not missing
    emit(
        name,
        "PASS" if ok else "FAIL",
        stages=pipe.num_stages,
        loss_ref=loss_ref,
        loss=loss,
        loss_rel_diff=rel,
        grad_max_rel_diff=worst,
        grad_worst_param=where,
        missing_grads=missing[:5],
        tok_s_ref=ROWS * SEQ / t_ref,
        tok_s_pp=ROWS * SEQ / t_pp,
    )


def check_compile(name, model_id):
    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.tensor_parallel import apply_tensor_parallel
    from palingenesis.train import _compile_layers

    if DEVICE != "cuda":
        emit(name, "SKIP", reason="inductor check on CUDA only")
        return
    one_rank_group()
    ref = load(model_id)
    eager = copy.deepcopy(ref)
    tp_e = apply_tensor_parallel(eager, init_device_mesh(DEVICE, (1,), mesh_dim_names=("tp",)))
    compiled = copy.deepcopy(ref)
    tp_c = apply_tensor_parallel(compiled, init_device_mesh(DEVICE, (1,), mesh_dim_names=("tp",)))
    del ref
    _compile_layers(compiled)
    ids, labels = batch(compiled.config.vocab_size, seed=1)
    t = time.perf_counter()
    loss_c = tp_step(compiled, tp_c, ids, labels)
    compile_s = time.perf_counter() - t
    loss_e = tp_step(eager, tp_e, ids, labels)
    worst, where, _ = compare(grads(eager), grads(compiled))
    t_e = timed(lambda: tp_step(eager, tp_e, ids, labels))
    t_c = timed(lambda: tp_step(compiled, tp_c, ids, labels))
    rel = abs(loss_c - loss_e) / abs(loss_e)
    ok = rel < 5e-3 and worst < 5e-2
    emit(
        name,
        "PASS" if ok else "FAIL",
        loss_eager=loss_e,
        loss_compiled=loss_c,
        loss_rel_diff=rel,
        grad_max_rel_diff=worst,
        grad_worst_param=where,
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


def pair_worker(rank, port, model_id, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from palingenesis.tensor_parallel import apply_tensor_parallel

        ref = load(model_id)
        model = copy.deepcopy(ref)
        tp = apply_tensor_parallel(model, init_device_mesh(DEVICE, (2,), mesh_dim_names=("tp",)))
        ids, labels = batch(ref.config.vocab_size)
        loss_ref = plain_step(ref, ids, labels)
        g_ref = grads(ref)
        loss = tp_step(model, tp, ids, labels)
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
            key = name
            g[key] = tp.sections[key].canonical(full, 2) if key in tp.sections else full
        worst, where, missing = compare(g_ref, g)
        if rank == 0:
            with open(out, "w") as f:
                json.dump({"loss_ref": loss_ref, "loss": loss, "worst": worst, "where": where, "missing": missing}, f)
    finally:
        dist.destroy_process_group()


def check_tp2_gloo(name, model_id, tmp):
    import torch.multiprocessing as mp

    out = os.path.join(tmp, f"{name}.json")
    try:
        mp.spawn(pair_worker, args=(free_port(), model_id, out), nprocs=2, join=True)
    except Exception as e:  # noqa: BLE001
        text = str(e)
        if "gloo" in text.lower() and ("not supported" in text.lower() or "unsupported" in text.lower()):
            emit(name, "SKIP", reason=text.strip().splitlines()[-1][:300])
            return
        raise
    r = json.load(open(out))
    lt, gt = tolerance()
    rel = abs(r["loss"] - r["loss_ref"]) / abs(r["loss_ref"])
    ok = rel < lt and r["worst"] < gt and not r["missing"]
    emit(
        name,
        "PASS" if ok else "FAIL",
        loss_ref=r["loss_ref"],
        loss=r["loss"],
        loss_rel_diff=rel,
        grad_max_rel_diff=r["worst"],
        grad_worst_param=r["where"],
        missing_grads=r["missing"][:5],
    )


def main():
    tmp = sys.argv[1]
    q3, q35 = "Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"
    emit("environment", "PASS", torch=torch.__version__, device=DEVICE, cuda=torch.cuda.is_available(), seq=SEQ)
    # two processes first: the one-rank checks below initialise this process's own group
    run_check("tp2_gloo_qwen3", check_tp2_gloo, q3, tmp)
    run_check("tp1_qwen3", check_tp1, q3)
    run_check("tp1_qwen35", check_tp1, q35)
    run_check("pp_zbv_qwen3", check_pp, q3, "zbv")
    run_check("pp_dualpipev_qwen3", check_pp, q3, "dualpipev")
    run_check("pp_zbv_qwen35", check_pp, q35, "zbv")
    run_check("pp_tp_zbv_qwen35", check_pp, q35, "zbv", True)
    run_check("compile_tp_qwen3", check_compile, q3)
    run_check("async_tp", check_async_tp)
    print(json.dumps({"check": "summary", "status": "FAIL" if FAILED else "PASS", "failed": FAILED}), flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
PYEOF

"$PY" "$CHECKS" "$TMP"
