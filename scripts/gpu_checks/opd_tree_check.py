"""GPU checks of train.packing (palingenesis.opd.packing) in on-policy distillation: padded vs varlen vs tree.

Called by scripts/gpu_checks/opd_tree.sh; every result is one JSON line on stdout
({"check": ..., "status": "PASS" | "FAIL" | "INFO" | "SKIP", ...numbers}).

  env                 versions, kernels, GPU memory free at the start
  equality            on real models (bf16 autocast, as the trainer runs), one sampled batch: the teacher's log-probs
                      and top-k, the loss statistics, dev_kl and the student's gradients, varlen / tree against
                      padded, for full_rkl (the fused head kernel) and topk_kl; the noise between two padded layouts
                      (other micro-batch sizes) is the floor
  speed               a step's time per packing mode at group sizes 4 / 8 / 16: the rollout (once, the same batch for
                      every mode), the teacher's scoring, the student's forward / backward / optimizer step; tokens/s,
                      the teacher's share, peak memory, median of --repeats after a warm-up; and the pre-registered
                      target at group 8

Prompts are English text of realistic lengths (a shared system prompt, then a task of 150-700 tokens); completions
are what the student samples at temperature 1 (up to --max-new-tokens).
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

TARGET_GROUP, TARGET, KILL, DEV_KL_TOLERANCE = 8, 0.25, 0.10, 0.01

WORDS = (
    "the of and to in is that for it as with was on be by at this have from or one had not but what all were when "
    "we there can an your which their said if do will each about how up out them then she many some so these would "
    "other into has more her two like him see time could no make than first been its who now people my made over "
    "did down only way find use may water long little very after words called just where most know get through "
    "back much before go good new write our used me man too any day same right look think also around another "
    "came come work three word must because does part even place well such here take why things help put years "
    "different away again off went old number great tell men say small every found still between name should home "
    "big give air line set own under read last never us left end along while might next sound below saw something "
    "thought both few those always looked show large often together asked house world going want school important "
    "until form food keep children feet land side without boy once animals life enough took sometimes four head "
    "above kind began almost live page got earth need far hand high year mother light parts country father let "
    "night following picture being study second eyes soon times story boys since white days ever paper hard near "
    "sentence better best across during today others however sure means knew try told young miles sun ways thing "
    "whole hear example heard several change answer room sea against top turned learn point city play toward five"
).split()


def emit(check: str, status: str, **values) -> None:
    print(json.dumps({"check": check, "status": status, **values}), flush=True)


def sync(empty: bool = False) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if empty:
            torch.cuda.empty_cache()


def free_gib() -> float | None:
    import torch

    return torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else None


# ── the trainer and its batches ────────────────────────────────────────────────


def text(rng: random.Random, words: int) -> str:
    out, sentence = [], []
    for _ in range(words):
        sentence.append(rng.choice(WORDS))
        if len(sentence) >= rng.randint(6, 18):
            out.append(" ".join(sentence).capitalize() + ".")
            sentence = []
    return " ".join(out + ([" ".join(sentence).capitalize() + "."] if sentence else []))


def conversations(count: int, seed: int, prompt_words: tuple[int, int]) -> list[list[dict]]:
    rng = random.Random(seed)
    system = "You are a careful assistant. Read the passage and answer the question that follows it. " + text(rng, 150)
    return [
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Passage {i}: {text(rng, rng.randint(*prompt_words))}\n\nQuestion: which sentence of "
                "the passage matters most, and why? Answer in a short paragraph.",
            },
        ]
        for i in range(count)
    ]


def build_trainer(args, loss: str):
    """An OPDTrainer on the real models with train.packing tree (the models are checked and routed for packed
    rows); the modes are switched on it afterwards (set_mode)."""
    from palingenesis.opd.config import OPDConfig
    from palingenesis.opd.trainer import OPDTrainer

    os.makedirs(args.work, exist_ok=True)
    prompts = os.path.join(args.work, "prompts.jsonl")
    with open(prompts, "w") as f:
        for messages in conversations(64, 0, args.prompt_words):
            f.write(json.dumps({"messages": messages}) + "\n")
    config = OPDConfig()
    settings = {
        "model.student": args.student,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "teachers.teacher.model": args.teacher,
        "teachers.teacher.loss": loss,
        "sources.bench.path": prompts,
        "sources.bench.teacher": "teacher",
        "sources.bench.max_new_tokens": args.max_new_tokens,
        "sources.bench.dev_size": 8,
        "rollout.backend": args.rollout,
        "rollout.batch_prompts": args.prompts,
        "rollout.group_size": TARGET_GROUP,
        "rollout.max_model_len": args.max_model_len,
        "loss.top_k": 16,
        "train.output_dir": os.path.join(args.work, f"run_{loss}"),
        "train.steps": 1,
        "train.learning_rate": 0.0,  # timed optimizer steps that leave the weights where they are
        "train.eval_every": 0,
        "train.score_micro_seqs": args.micro_seqs,
        "train.packing": "tree",
        "logging.trackers": [],
    }
    for key, value in settings.items():
        try:
            config.set(key, value, "check")
        except Exception:  # noqa: BLE001 (an option this version does not have, e.g. logging.trackers)
            if key != "logging.trackers":
                raise
    return OPDTrainer(config)


def set_mode(trainer, mode: str, micro_seqs: int | None = None) -> None:
    trainer.config.train.packing = mode
    if micro_seqs is not None:
        trainer.config.train.score_micro_seqs = micro_seqs
    for route in trainer.routes.values():
        if hasattr(route.teacher, "packing"):
            route.teacher.packing = trainer._packing()


def requests_for(trainer, prompts: int, group: int, args, seed: int):
    out = []
    for messages in conversations(prompts, seed, args.prompt_words):
        request = trainer.pipeline.request(messages, args.max_new_tokens, {}, "bench", "teacher")
        if trainer._fit(request):
            out += [request] * group
    return out


def score_teacher(trainer, batch) -> float:
    """The teacher's scoring of the batch, as Pipeline.run does it; seconds."""
    sync()
    start = time.perf_counter()
    for name, route in trainer.routes.items():
        group = [s for s in batch.samples if s.teacher == name]
        scores = route.teacher.score(
            [s.view for s in group],
            top_k=route.top_k,
            keep_hidden=route.keep_hidden,
            micro_seqs=trainer.config.train.score_micro_seqs,
        )
        for s, sc in zip(group, scores):
            s.scores = sc
    sync()
    return time.perf_counter() - start


def train_student(trainer, batch, optimizer: bool = True) -> tuple[float, dict]:
    """The student's forward / backward over the batch, clipping and the optimizer step (learning rate 0);
    (seconds, the loss statistics). The gradients stay in .grad when `optimizer` is off."""
    import torch

    trainer.student.train()
    trainer.opt.zero_grad(set_to_none=True)
    sync()
    start = time.perf_counter()
    stats = trainer._scores(batch, train=True)
    if optimizer:
        torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), trainer.config.train.max_grad_norm)
        trainer.opt.step()
        trainer.opt.zero_grad(set_to_none=True)
    sync()
    return time.perf_counter() - start, dict(stats)


def forwarded_tokens(trainer, batch, mode: str) -> dict:
    """Tokens each model forwards for the batch in `mode` (padded: rows x their width)."""
    out = {}
    packing = trainer._packing()
    for side, sequences in (
        ("student", [s.request.prompt_ids + s.completion[:-1] for s in batch.samples]),
        ("teacher", [s.view.input_ids[:-1] for s in batch.samples]),
    ):
        if packing is None:
            micro = trainer.config.train.score_micro_seqs
            order = sorted(sequences, key=len)
            chunks = [order[i : i + micro] for i in range(0, len(order), micro)]
            out[f"{side}_forwarded_tokens"] = sum(len(chunk[-1]) * len(chunk) for chunk in chunks)
        else:
            parts = packing.micro_batches(sequences, [1] * len(sequences), trainer.config.train.score_micro_seqs)
            rows = [packing.row([sequences[i] for i in part]) for part in parts]
            out[f"{side}_forwarded_tokens"] = sum(row.num_tokens for row in rows)
            out[f"{side}_rows"] = len(rows)
            out[f"{side}_levels"] = max(len(row.levels) for row in rows)
    return out


def dev_kl(trainer, batch) -> float:
    import torch

    with torch.no_grad():
        stats = trainer._scores(batch, train=False)
    return stats["k1/teacher"] / stats["tokens/teacher"]


# ── environment ────────────────────────────────────────────────────────────────


def env_report(args) -> None:
    import torch
    import transformers

    def has(module: str) -> bool:
        try:
            __import__(module)
            return True
        except Exception:  # noqa: BLE001
            return False

    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_free_gib": free_gib(),
        "gpu_total_gib": torch.cuda.mem_get_info()[1] / 2**30 if torch.cuda.is_available() else None,
        "flash_linear_attention": has("fla"),
        "causal_conv1d": has("causal_conv1d"),
        "vllm": has("vllm"),
        "student": args.student,
        "teacher": args.teacher,
        "rollout": args.rollout,
    }
    emit("env", "INFO", **info)
    if torch.cuda.is_available():
        from palingenesis.tree_packing import _VERIFIED, verify_flash_tree

        try:
            verify_flash_tree("cuda:0", torch.bfloat16)
            emit("tree_attention_selfcheck", "PASS", max_errors=next(iter(_VERIFIED.values())))
        except Exception as error:  # noqa: BLE001
            emit("tree_attention_selfcheck", "FAIL", error=str(error)[:500])


# ── equality ───────────────────────────────────────────────────────────────────


def grads_of(trainer) -> dict:
    return {
        n: p.grad.detach().to("cpu", copy=True) for n, p in trainer.student.named_parameters() if p.grad is not None
    }


def compare_grads(ref: dict, got: dict) -> dict:
    num = den = dot = norm_got = 0.0
    for n, g in ref.items():
        g, h = g.float(), got[n].float()
        num += float((h - g).pow(2).sum())
        den += float(g.pow(2).sum())
        dot += float((h * g).sum())
        norm_got += float(h.pow(2).sum())
    return {
        "grad_rel_l2": math.sqrt(num / max(den, 1e-30)),
        "grad_cosine": dot / max(math.sqrt(den) * math.sqrt(norm_got), 1e-30),
    }


def teacher_diffs(batch, reference: list) -> dict:
    """Max / mean |difference| of the teacher's outputs against `reference` (per sample TeacherScores)."""
    import torch

    out = {}
    for field in ("token_lp", "topk_lp", "hidden"):
        pairs = [(getattr(s.scores, field), getattr(r, field)) for s, r in zip(batch.samples, reference)]
        pairs = [(a.float().cpu(), b.float().cpu()) for a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        diff = torch.cat([(a - b).abs().flatten() for a, b in pairs])
        diff = diff[diff.isfinite()]
        scale = torch.cat([b.abs().flatten() for _, b in pairs])
        out[f"teacher_{field}_max_abs_diff"] = float(diff.max()) if diff.numel() else 0.0
        out[f"teacher_{field}_mean_abs_diff"] = float(diff.mean()) if diff.numel() else 0.0
        out[f"teacher_{field}_mean_abs"] = float(scale[scale.isfinite()].mean())
    ids = [(s.scores.topk_ids, r.topk_ids) for s, r in zip(batch.samples, reference) if r.topk_ids is not None]
    if ids:  # the top-1 token is the same wherever the top two are not a bf16 tie
        same = sum(int((a[:, 0] == b[:, 0]).sum()) for a, b in ids)
        out["teacher_top1_agreement"] = same / max(1, sum(a.shape[0] for a, _ in ids))
    return out


def equality(args) -> None:
    """One loss per process: vLLM's sleep-mode allocator serves one engine per process, so a second trainer (another
    loss, or fewer prompts after an out-of-memory error) cannot start its engine here. Every loss but the first, and
    a retry with half the prompts, re-executes this script."""
    import torch

    loss, rest = args.losses[0], args.losses[1:]
    prompts = min(args.prompts, 4)
    try:
        equality_of(args, loss, prompts)
    except torch.OutOfMemoryError as error:
        if prompts > 1:
            reexec(args, [loss, *rest], prompts // 2)
        emit("equality", "SKIP", loss=loss, reason=f"out of memory at 1 prompt x {TARGET_GROUP} ({str(error)[:300]})")
    if rest:
        reexec(args, rest, args.prompts)


def reexec(args, losses: list[str], prompts: int) -> None:
    """Replace this process by the equality part for `losses` at `prompts` prompts (a fresh CUDA context and engine)."""
    sys.stdout.flush()
    argv = list(sys.argv[1:])
    for flag in ("--losses", "--prompts"):
        if flag in argv:
            i = argv.index(flag)
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                j += 1
            del argv[i:j]
    argv += ["--losses", *losses, "--prompts", str(prompts)]
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *argv])


def equality_of(args, loss: str, prompts: int) -> None:
    trainer = build_trainer(args, loss)
    set_mode(trainer, "padded", args.micro_seqs)
    batch = trainer.pipeline.run(requests_for(trainer, prompts, TARGET_GROUP, args, seed=1), 1.0)
    tokens = sum(len(s.completion) for s in batch.samples)
    results = {}
    for label, mode, micro in (
        ("padded", "padded", args.micro_seqs),
        ("padded_alt", "padded", max(2, args.micro_seqs // 2 + 1)),  # other rows per forward: the noise floor
        ("varlen", "varlen", args.micro_seqs),
        ("tree", "tree", args.micro_seqs),
    ):
        set_mode(trainer, mode, micro)
        reference = [s.scores for s in batch.samples] if label == "padded" else results["padded"]["scores"]
        score_teacher(trainer, batch)
        teacher = {} if label == "padded" else teacher_diffs(batch, reference)
        scores = [s.scores for s in batch.samples]
        # the student against the SAME teacher targets in every mode: its own differences only
        for s, r in zip(batch.samples, reference):
            s.scores = r
        _, stats = train_student(trainer, batch, optimizer=False)
        grads = grads_of(trainer)
        trainer.opt.zero_grad(set_to_none=True)
        results[label] = {
            "scores": scores if label == "padded" else None,
            "teacher": teacher,
            "kl": stats["kl/teacher"] / tokens,
            "k1": stats["k1/teacher"] / tokens,
            "loss": stats["loss/teacher"],
            "dev_kl": dev_kl(trainer, batch),
            "grads": compare_grads(results["padded"]["raw_grads"], grads) if label != "padded" else {},
            "raw_grads": grads if label == "padded" else None,
        }
        del grads
    base, floor = results["padded"], results["padded_alt"]
    common = {"student": args.student, "teacher": args.teacher, "loss": loss, "completion_tokens": tokens}
    emit("equality_noise_floor", "INFO", **common, **floor["teacher"], **floor["grads"],
         loss_rel_diff=abs(floor["loss"] - base["loss"]) / max(abs(base["loss"]), 1e-30))  # fmt: skip
    for label in ("varlen", "tree"):
        got = results[label]
        values = {
            **common,
            "mode": label,
            **got["teacher"],
            **got["grads"],
            "kl_per_token": got["kl"],
            "kl_per_token_padded": base["kl"],
            "loss_rel_diff": abs(got["loss"] - base["loss"]) / max(abs(base["loss"]), 1e-30),
            "dev_kl": got["dev_kl"],
            "dev_kl_padded": base["dev_kl"],
            "dev_kl_abs_diff": abs(got["dev_kl"] - base["dev_kl"]),
            "noise_floor_grad_rel_l2": floor["grads"]["grad_rel_l2"],
            "noise_floor_teacher_token_lp_mean_abs_diff": floor["teacher"].get("teacher_token_lp_mean_abs_diff"),
        }
        lp_floor = floor["teacher"].get("teacher_token_lp_mean_abs_diff", 0.0)
        # bf16: packed must stay within the noise between two padded layouts (other batch shapes and kernels)
        ok = (
            got["teacher"].get("teacher_token_lp_mean_abs_diff", 0.0) <= max(3 * lp_floor, 0.02)
            and got["grads"]["grad_rel_l2"] <= max(3 * floor["grads"]["grad_rel_l2"], 0.02)
            and got["grads"]["grad_cosine"] >= 0.999
            and values["loss_rel_diff"]
            <= max(3 * abs(floor["loss"] - base["loss"]) / max(abs(base["loss"]), 1e-30), 0.01)
            and values["dev_kl_abs_diff"] < DEV_KL_TOLERANCE
        )
        emit("equality", "PASS" if ok else "FAIL", **values)
    del trainer, batch, results
    sync(empty=True)


# ── speed ──────────────────────────────────────────────────────────────────────


def measure(trainer, batch, mode: str, repeats: int) -> dict:
    import torch

    set_mode(trainer, mode)
    score_teacher(trainer, batch)  # warm-up (autotuning, allocator)
    train_student(trainer, batch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    teacher_s, student_s = [], []
    for _ in range(repeats):
        teacher_s.append(score_teacher(trainer, batch))
        student_s.append(train_student(trainer, batch)[0])
    return {
        "teacher_s": statistics.median(teacher_s),
        "student_s": statistics.median(student_s),
        "scoring_s": statistics.median([a + b for a, b in zip(teacher_s, student_s)]),
        "scoring_s_all": [a + b for a, b in zip(teacher_s, student_s)],
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
        "dev_kl": dev_kl(trainer, batch),
        **forwarded_tokens(trainer, batch, mode),
    }


def speed(args) -> None:
    import torch

    trainer = build_trainer(args, args.losses[0])
    common = {"student": args.student, "teacher": args.teacher, "loss": args.losses[0], "rollout": args.rollout}
    for group in args.groups:
        prompts, batch, reason = args.prompts, None, ""
        while batch is None:  # the batch: sampled once (padded scoring), the same for every mode
            try:
                set_mode(trainer, "padded")
                batch = trainer.pipeline.run(requests_for(trainer, prompts, group, args, seed=10 + group), 1.0)
            except torch.OutOfMemoryError as error:
                reason = str(error)[:300]
                trainer.opt.zero_grad(set_to_none=True)
                sync(empty=True)
                if prompts <= 1:
                    break
                prompts //= 2
        if batch is None or not batch.samples:
            emit("speed", "SKIP", **common, group=group, reason=f"rollout out of memory at 1 prompt ({reason})")
            continue
        rollout_s = batch.stats["time/rollout"] + batch.stats.get("time/sync", 0.0)
        tokens = sum(len(s.completion) for s in batch.samples)
        shape = {
            "group": group,
            "prompts": prompts,
            "completions": len(batch.samples),
            "completion_tokens": tokens,
            "prompt_tokens_mean": statistics.mean(len(s.request.prompt_ids) for s in batch.samples),
            "completion_tokens_mean": tokens / len(batch.samples),
            "rollout_s": rollout_s,
            "gpu_free_gib": free_gib(),
        }
        results = {}
        for mode in ("padded", "varlen", "tree"):
            try:
                values = measure(trainer, batch, mode, args.repeats)
            except torch.OutOfMemoryError as error:
                trainer.opt.zero_grad(set_to_none=True)
                sync(empty=True)
                emit("speed", "SKIP", **common, **shape, mode=mode, reason=f"out of memory ({str(error)[:300]})")
                continue
            values["step_s"] = rollout_s + values["scoring_s"]
            values["teacher_share_of_step"] = values["teacher_s"] / values["step_s"]
            values["teacher_share_of_scoring"] = values["teacher_s"] / values["scoring_s"]
            values["scoring_tok_s"] = tokens / values["scoring_s"]
            values["step_tok_s"] = tokens / values["step_s"]
            results[mode] = values
            emit("speed", "INFO", **common, **shape, mode=mode, **values)
        if "padded" in results:
            base = results["padded"]
            for mode in ("varlen", "tree"):
                if mode not in results:
                    continue
                got = results[mode]
                values = {
                    "group": group,
                    "mode": mode,
                    "step_time_reduction": 1 - got["step_s"] / base["step_s"],
                    "scoring_time_reduction": 1 - got["scoring_s"] / base["scoring_s"],
                    "teacher_time_reduction": 1 - got["teacher_s"] / base["teacher_s"],
                    "student_time_reduction": 1 - got["student_s"] / base["student_s"],
                    "peak_gib": got["peak_gib"],
                    "peak_gib_padded": base["peak_gib"],
                    "dev_kl_abs_diff": abs(got["dev_kl"] - base["dev_kl"]),
                }
                if mode == "tree" and group == TARGET_GROUP:
                    # pre-registered: >= 25% lower step time at group 8 with the same dev_kl; below 10%: stop
                    same = values["dev_kl_abs_diff"] < DEV_KL_TOLERANCE
                    reduction = values["step_time_reduction"]
                    verdict = (
                        "met" if reduction >= TARGET and same else "kill" if reduction < KILL or not same else "short"
                    )
                    emit("target", "PASS" if verdict == "met" else "FAIL", **common, **values, verdict=verdict,
                         target=TARGET, kill_below=KILL, dev_kl_tolerance=DEV_KL_TOLERANCE)  # fmt: skip
                else:
                    emit("speedup", "INFO", **common, **values)
        del batch
        trainer.opt.zero_grad(set_to_none=True)
        sync(empty=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=("env", "equality", "speed"))
    parser.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--work", required=True)
    parser.add_argument("--rollout", default="hf", help="rollout.backend: hf, vllm, ...")
    parser.add_argument("--losses", nargs="+", default=["full_rkl", "topk_kl"])
    parser.add_argument("--groups", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--prompts", type=int, default=8, help="prompts per step (rollout.batch_prompts)")
    parser.add_argument("--micro-seqs", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt-words", nargs=2, type=int, default=[120, 560])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.prompt_words = tuple(args.prompt_words)
    {"env": env_report, "equality": equality, "speed": speed}[args.part](args)


if __name__ == "__main__":
    sys.exit(main())
