"""`pgs internals`: read a model's internals from the command line.

pgs internals run --config internals.yaml [--set layers=12-20]
pgs internals capture --model M --data rows.jsonl --out acts/m_train [--position last_prompt] [--layers all]
pgs internals probe --acts acts/m_train --out probes/m [--test it=acts/m_it] [--kind logistic]
pgs internals compare --a acts/base_train --b acts/sft_train
pgs internals lens --model M --data chats.jsonl [--position assistant_all] [--decision "<tool_call>"] [--tuned lens.st]
pgs internals tuned-lens --model M --data chats.jsonl --out lens.safetensors [--steps 250]
pgs internals causal --model M --probe probes/m/best.safetensors --data rows.jsonl --decision "I" [--alphas -2,-1,0,1,2]
pgs internals patch --model M --twins twins.jsonl --decision "<tool_call>" [--offsets -1,-2,-3] [--mode denoise]
pgs internals steer --model M --probe P --prompt "..." [--alpha 1,2] [--positions all]
pgs internals score --probe P --data rows.jsonl --out scored.jsonl [--column known] [--gate fpr_0.05]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch


def _kwargs(text: str | None) -> dict:
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--chat-template-kwargs must be a JSON object")
    return value


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _fmt(v, width: int = 6) -> str:
    return f"{v:{width}.3f}" if isinstance(v, (int, float)) and v == v else " " * (width - 3) + "nan"


def print_sweep(report: dict) -> None:
    first = next(iter(report["layers"].values()))
    tests = [k for k, v in first.items() if isinstance(v, dict) and k != "test"]  # the extra test sets
    head = f"{'r':>3s} {'mixer':17s} {'val':>6s} {'test [95% CI]':>22s} {'dmean':>6s} {'ctrl':>6s} {'sel':>6s}"
    print(head + "".join(f" {t[:12]:>12s}" for t in tests))
    for r, e in report["layers"].items():
        t = e["test"]
        ci = f"{_fmt(t['auroc'])} [{_fmt(t['lo'], 5)},{_fmt(t['hi'], 5)}]"
        line = f"{int(r):3d} {str(e.get('mixer') or ''):17s} {_fmt(e['val_auroc'])} {ci:>22s} {_fmt(e.get('diffmean_test_auroc'))} "
        line += f"{_fmt(e.get('control_test_auroc'))} {_fmt(e.get('selectivity'))}"
        line += "".join(f" {_fmt(e[name]['auroc'], 12)}" for name in tests)
        print(line)
    length = report.get("length_auroc", {})
    print(
        f"best layer (validation): {report['best_layer']}   length-only AUROC: "
        + ", ".join(f"{k} {_fmt(v).strip()}" for k, v in length.items())
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pgs internals", description="read a model's internals (palingenesis.internals)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_opts(p, model=True):
        if model:
            p.add_argument("--model", required=True)
        p.add_argument(
            "--dtype",
            choices=("bfloat16", "float16", "float32"),
            default=None,
            help="default: bf16 on CUDA, fp32 on CPU",
        )
        p.add_argument("--device", default=None)
        p.add_argument("--chat-template-kwargs", default=None, help="JSON, e.g. '{\"enable_thinking\": false}'")
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--max-tokens", type=int, default=None, help="drop rows longer than this")
        p.add_argument("--limit", type=int, default=None, help="use the first N rows")

    p = sub.add_parser("run", help="capture + probe sweeps + comparisons from a YAML config")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--overwrite", action="store_true", help="recapture existing stores")

    p = sub.add_parser("capture", help="residuals at a position for every row -> a sharded store")
    model_opts(p)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--position", default="last_prompt")
    p.add_argument("--layers", default="all", help='"all", "17-19", "0,6,12", "every:4", -1 = last')
    p.add_argument("--store-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("probe", help="per-layer probe sweep on a store (grouped splits, controls, transfer)")
    p.add_argument("--acts", required=True)
    p.add_argument("--out", default=None, help="save the probes (layer_<r>, best) and sweep.json here")
    p.add_argument("--test", action="append", default=[], metavar="NAME=STORE", help="extra test store (repeatable)")
    p.add_argument("--kind", choices=("logistic", "diffmean", "lda"), default="logistic")
    p.add_argument("--layers", default=None)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--label-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("compare", help="two checkpoints' stores on the same rows: CKA, cosine, probe transfer")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--kind", choices=("logistic", "diffmean", "lda"), default="logistic")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("lens", help="logit / tuned lens per layer: KL to final, top-1, next-token CE, decision margin")
    model_opts(p)
    p.add_argument("--data", required=True)
    p.add_argument("--position", default="assistant_all")
    p.add_argument("--layers", default="all")
    p.add_argument("--decision", action="append", default=[], help="decision token (repeatable: a token set)")
    p.add_argument("--tuned", default=None, help="a tuned lens saved by `tuned-lens`")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("tuned-lens", help="train a tuned lens (one affine translator per layer)")
    model_opts(p)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--position", default="assistant_all")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)

    p = sub.add_parser("causal", help="is the probe's direction used? steering dose-response + ablation vs random")
    model_opts(p)
    p.add_argument("--probe", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--decision", action="append", required=True, help="decision token (repeatable: a token set)")
    p.add_argument("--alphas", default="-2,-1,0,1,2", help="multiples of |mu+ - mu-|")
    p.add_argument("--n-random", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("patch", help="activation patching between twin sides, per layer and position")
    model_opts(p)
    p.add_argument("--twins", required=True, help="palingenesis.margins twins: {pos: side, neg: side}")
    p.add_argument("--decision", action="append", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--offsets", default="-1", help="positions from the end of each prompt, e.g. -1,-2,-3")
    p.add_argument("--mode", choices=("denoise", "noise"), default="denoise")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("steer", help="generate with the probe's diff-of-means direction added at its layer")
    model_opts(p)
    p.add_argument("--probe", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--alpha", default="0,1", help="multiples of |mu+ - mu-| (0 = baseline)")
    p.add_argument("--positions", choices=("all", "last_and_new"), default="all")
    p.add_argument("--max-new-tokens", type=int, default=128)

    p = sub.add_parser("score", help="add a probe's calibrated probability to every row of a dataset")
    p.add_argument("--probe", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=None, help="default: the model recorded in the probe")
    p.add_argument("--column", default="probe")
    p.add_argument("--gate", default=None, help="also write <column>_flag at this threshold, e.g. fpr_0.05")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)

    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return _dispatch(a)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as e:
        print(f"pgs internals: {e}", file=sys.stderr)
        return 1


def _dispatch(a) -> int:
    from palingenesis.internals.capture import ActivationStore, capture_dataset, load_rows
    from palingenesis.internals.run import load_model

    if a.cmd == "run":
        from palingenesis.internals.config import InternalsConfig
        from palingenesis.internals.run import run_internals

        cfg = InternalsConfig.from_yaml(a.config, a.set)
        if a.overwrite:
            cfg.overwrite = True
        report = run_internals(cfg)
        for name, res in report["sweeps"].items():
            print(f"\n== {name}")
            print_sweep(res)
        for pair, res in report["compare"].items():
            print(f"\n== compare {pair}")
            _print_compare(res)
        print(f"\nreport: {cfg.output_dir}/report.json")
        return 0

    if a.cmd == "probe":
        from palingenesis.internals.probes import sweep
        from palingenesis.internals.sites import parse_layers

        store = ActivationStore(a.acts)
        tests = {}
        for item in a.test:
            name, sep, path = item.partition("=")
            if not sep:
                raise ValueError(f"--test {item!r}: expected NAME=STORE")
            tests[name] = ActivationStore(path)
        layers = None
        if a.layers:
            layers = parse_layers(a.layers, max(store.layers) + 1)
        report = sweep(
            store,
            layers=layers,
            kind=a.kind,
            tests=tests,
            val=a.val,
            test=a.test_fraction,
            seed=a.seed,
            resamples=a.bootstrap,
            label_threshold=a.label_threshold,
            save_dir=a.out,
        )
        report.pop("probes")
        if a.json:
            print(json.dumps(report, indent=2))
        else:
            print_sweep(report)
        return 0

    if a.cmd == "compare":
        from palingenesis.internals.similarity import compare_stores

        res = compare_stores(
            ActivationStore(a.a), ActivationStore(a.b), probe_kind=a.kind, resamples=a.bootstrap, seed=a.seed
        )
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            _print_compare(res)
        return 0

    if a.cmd == "score":
        from palingenesis.internals.score import ProbeScorer, score_rows

        scorer = ProbeScorer(a.probe, a.model, device=a.device, batch_size=a.batch_size)
        rows = score_rows(scorer, load_rows(a.data), a.column, gate_at=a.gate)
        with open(a.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        scored = [r[a.column] for r in rows if r[a.column] is not None]
        print(f"scored {len(scored)}/{len(rows)} rows -> {a.out} (mean {sum(scored) / max(len(scored), 1):.3f})")
        return 0

    model, tok = load_model(a.model, a.dtype, a.device)
    kwargs = _kwargs(a.chat_template_kwargs)

    if a.cmd == "capture":
        store = capture_dataset(
            model,
            tok,
            load_rows(a.data, a.limit),
            a.out,
            layers=a.layers,
            position=a.position,
            chat_template_kwargs=kwargs,
            batch_size=a.batch_size,
            max_tokens=a.max_tokens,
            dtype=a.store_dtype,
            overwrite=a.overwrite,
            extra_meta={"source": a.data},
        )
        print(f"{len(store)} rows x {len(store.layers)} residuals -> {a.out} (dropped {store.meta['dropped']})")
        return 0

    if a.cmd in ("lens", "tuned-lens"):
        from palingenesis.internals.lens import TunedLens, fit_tuned_lens, lens_report
        from palingenesis.margins import decision_ids

        rows = load_rows(a.data, a.limit)
        if a.cmd == "tuned-lens":
            lens = fit_tuned_lens(
                model,
                tok,
                rows,
                position=a.position,
                steps=a.steps,
                lr=a.lr,
                batch_size=a.batch_size,
                chat_template_kwargs=kwargs,
                max_tokens=a.max_tokens,
            )
            lens.save(a.out)
            print(f"tuned lens ({lens.n_layers} translators) -> {a.out}")
            return 0
        tuned = TunedLens.load(a.tuned).to(next(model.parameters()).device) if a.tuned else None
        res = lens_report(
            model,
            tok,
            rows,
            position=a.position,
            layers=a.layers,
            decision=decision_ids(tok, a.decision) if a.decision else None,
            tuned=tuned,
            chat_template_kwargs=kwargs,
            batch_size=a.batch_size,
            max_tokens=a.max_tokens,
        )
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            print(
                f"{'r':>3s} {'mixer':17s} {'KL':>8s} {'top1':>6s} {'CE':>7s}"
                + (f" {'margin':>7s}" if a.decision else "")
            )
            for r, e in res["layers"].items():
                line = f"{int(r):3d} {e['mixer']:17s} {_fmt(e['kl'], 8)} {_fmt(e['top1'])} {_fmt(e['ce'], 7)}"
                print(line + (f" {_fmt(e['margin'], 7)}" if a.decision else ""))
            print(f"{res['positions']} positions ({'tuned' if res['tuned'] else 'logit'} lens)")
        return 0

    from palingenesis.internals.probes import Probe

    if a.cmd == "causal":
        from palingenesis.internals.causal import direction_effect
        from palingenesis.margins import decision_ids

        res = direction_effect(
            model,
            tok,
            load_rows(a.data, a.limit),
            Probe.load(a.probe),
            decision_ids(tok, a.decision),
            alphas=_floats(a.alphas),
            n_random=a.n_random,
            chat_template_kwargs=kwargs or None,
            batch_size=a.batch_size,
            seed=a.seed,
        )
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"layer {res['layer']}, {res['n']} prompts, base margin {res['base_margin']:.3f}")
            for al, m in zip(res["alphas"], res["margins"]):
                print(f"  alpha {al:+.2f}: margin {m:.3f}")
            rs = torch.tensor(res["random_slopes"])
            print(f"slope {res['slope']:.3f} vs random {float(rs.mean()):.3f} ± {float(rs.std()):.3f} (z {res['z']})")
            if "ablation" in res:
                ra = torch.tensor(res["ablation"]["random_deltas"])
                print(
                    f"ablation: margin change {res['ablation']['delta']:.3f} vs random {float(ra.mean()):.3f} ± {float(ra.std()):.3f}"
                )
        return 0

    if a.cmd == "patch":
        from palingenesis.internals.causal import patching_sweep
        from palingenesis.margins import decision_ids, load_twins

        twins = load_twins(a.twins)[: a.limit] if a.limit else load_twins(a.twins)
        res = patching_sweep(
            model,
            tok,
            twins,
            decision_ids(tok, a.decision),
            layers=a.layers,
            offsets=_ints(a.offsets),
            mode=a.mode,
            chat_template_kwargs=kwargs,
            batch_size=a.batch_size,
        )
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            print(
                f"{a.mode}: source margin {res['source_margin']:.3f}, base {res['base_margin']:.3f}, {res['n']} twins"
            )
            print(f"{'r':>3s} {'mixer':17s}" + "".join(f" {o:>7d}" for o in res["offsets"]))
            for r, vals in res["layers"].items():
                print(f"{int(r):3d} {res['mixers'][r]:17s}" + "".join(f" {_fmt(v, 7)}" for v in vals))
        return 0

    if a.cmd == "steer":
        from palingenesis.internals.causal import generate

        probe = Probe.load(a.probe)
        messages = [{"role": "user", "content": a.prompt}]
        kw = kwargs or probe.meta.get("chat_template_kwargs", {})
        for alpha in _floats(a.alpha):
            vectors = {probe.layer: alpha * probe.mean_diff} if alpha else None
            text = generate(
                model,
                tok,
                messages,
                vectors=vectors,
                positions=a.positions,
                chat_template_kwargs=kw,
                max_new_tokens=a.max_new_tokens,
            )
            print(f"--- alpha {alpha:+g} (layer {probe.layer})\n{text}\n")
        return 0
    raise ValueError(f"unknown command {a.cmd}")


def _print_compare(res: dict) -> None:
    from palingenesis.internals.similarity import summarise_transfer

    print(f"{res['a']}  vs  {res['b']}  ({res['n']} rows, {res['position']})")
    for r, e in res["layers"].items():
        print(f"{int(r):3d}  {summarise_transfer(e)}")


if __name__ == "__main__":
    raise SystemExit(main())
