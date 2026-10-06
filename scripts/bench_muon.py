"""Benchmark Muon's orthogonalization on a model's real matrix shapes, per Newton-Schulz variant and GEMM backend.

Run it on each machine (A100, H100, B200/B300, ...) to pick the fastest setting and to keep docs/performance.md
measured, not estimated:

    python scripts/bench_muon.py --model Qwen/Qwen3.5-0.8B [--iters 20] [--json out.jsonl]

For every combination of {standard, gram} x {torch, quack (sm90+, if installed)} x {eager, compiled} it reports the
time of one full Muon orthogonalization step (all of the model's Muon matrices, batched by shape as the optimizer
does), the speedup over eager standard Newton-Schulz, and the accuracy (cosine to the exact polar factor U V^T and
the singular-value range) on one matrix of each shape.
"""

import argparse
import json
import time
from collections import Counter

import torch

from palingenesis.muon import _quack_ops, muon_param_groups, orthogonalize


def model_shapes(model_id: str, min_dim: int) -> Counter:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    groups, _ = muon_param_groups(model, 0.0, min_dim)
    return Counter(tuple(p.shape) for g in groups if g["use_muon"] for p in g["params"])


def bench(shapes: Counter, method: str, backend: str, compile_: bool, iters: int) -> dict:
    batches = {s: torch.randn(n, *s, device="cuda") for s, n in shapes.items()}

    def step():
        for X in batches.values():
            orthogonalize(X, method, backend=backend, compile=compile_)

    for _ in range(3):  # warm-up (and compilation)
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    acc = {}
    for s in shapes:
        G = torch.randn(*s, device="cuda")
        orth = orthogonalize(G, method, backend=backend, compile=compile_).double()
        U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
        sv = torch.linalg.svdvals(orth)
        acc["x".join(map(str, s))] = {
            "cos": float(torch.nn.functional.cosine_similarity(orth.flatten(), (U @ Vh).flatten(), dim=0)),
            "sv": [round(float(sv.min()), 3), round(float(sv.max()), 3)],
        }
    return {"method": method, "backend": backend, "compiled": compile_, "ms": round(ms, 3), "accuracy": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--min-dim", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    shapes = model_shapes(a.model, a.min_dim)
    print(f"{dev} (sm{cap[0]}{cap[1]}), {a.model}: {sum(shapes.values())} Muon matrices in {len(shapes)} shapes "
          f"{dict(shapes)}")
    backends = ["torch"] + (["quack"] if _quack_ops() is not None else [])
    rows = []
    for backend in backends:
        for method in ("standard", "gram"):
            for compile_ in (False, True):
                try:
                    rows.append(bench(shapes, method, backend, compile_, a.iters))
                except Exception as e:  # noqa: BLE001 — report and keep going
                    rows.append({"method": method, "backend": backend, "compiled": compile_, "error": str(e)[:200]})
    base = next((r["ms"] for r in rows if r.get("ms") and r["method"] == "standard" and r["backend"] == "torch"
                 and not r["compiled"]), None)
    for r in rows:
        r.update(device=dev, capability=f"sm{cap[0]}{cap[1]}", model=a.model, torch=torch.__version__)
        if "ms" in r:
            worst = min(v["cos"] for v in r["accuracy"].values())
            print(f"  {r['method']:9s} {r['backend']:6s} {'compiled' if r['compiled'] else 'eager':8s} "
                  f"{r['ms']:8.2f} ms  x{base / r['ms']:.2f}  min cos {worst:.4f}")
        else:
            print(f"  {r['method']:9s} {r['backend']:6s} {'compiled' if r['compiled'] else 'eager':8s} ERROR {r['error']}")
    if a.json:
        with open(a.json, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
