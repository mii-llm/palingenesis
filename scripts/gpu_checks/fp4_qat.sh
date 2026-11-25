#!/usr/bin/env bash
# FP4 / FP8 quantisation-aware training: GPU checks on ONE NVIDIA A100 80GB (no FP4 / FP8 tensor cores: the
# training forward fake-quantises; vLLM serves the packed checkpoints through its weight-only kernels).
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/fp4_qat.sh
#
# Environment (all optional but PGS_PY / PGS_SRC):
#   PGS_PY       python with torch, transformers, vllm (default: python)
#   PGS_SRC      palingenesis/src (default: ../../src from this script)
#   WORK         working directory (default: a new /tmp/pgs_fp4_qat.XXXXXX); keep it to read the logs
#   NEED_GB      free space required in WORK before starting (default 8; see "Disk" below)
#   KEEP         1 keeps every checkpoint (about 30 GB for the default model instead of the peak below)
#   MODEL        default Qwen/Qwen3.5-0.8B (from the HF cache: HF_HUB_OFFLINE=1)
#   STEPS        QAT steps per format (default 300)
#   QAT_FORMATS  formats trained with QAT + KD (default "nvfp4 mxfp4")
#   EVAL_ITEMS   accuracy-proxy items (default 200)
#
# Output: one JSON line per check on stdout, {"check": ..., "status": "PASS" | "FAIL" | "SKIP" | "INFO", ...numbers};
# the last line is a summary. Logs of every step are in $WORK/logs. Expected run time on an A100 80GB: ~60-80 min
# (data generation ~5 min, 9 vLLM evaluations ~1-2 min each, 2 QAT runs ~10-15 min each, vLLM loads, benchmarks).
#
# Disk: one run's checkpoints at a time. Each run is measured (KL, accuracy, vLLM) and its weight files are then
# deleted (the files this script wrote, one by one; configs, reports, logs and results stay). The peak for a 0.8B
# model is one QAT run: final/ (the dequantized model in the run's float32, ~3.5 GB) + final-<format>/ (packed,
# ~1 GB); a PTQ run is ~1.8 GB (dequantized bf16) + ~1-1.3 GB (packed). The script prints the space it needs and
# what WORK has, and stops before any work if there is less.
#
# Checks
#   (1) ptq/<fmt>/<scales>   PTQ of MODEL (nvfp4, mxfp4, fp8, fp8_block; max scales, and mse for the FP4 formats):
#                            KL(bf16 || quantised) per token on held-out text, top-1 agreement, accuracy proxy
#                            (GSM8K test if cached, else ARC-Easy, else synthetic arithmetic; greedy, non-thinking)
#   (1b) exact/<fmt>         the dequantized export computes exactly what the fake-quantised forward computes
#   (2) qat/<fmt>            pgs train, QAT with KD to the bf16 model (forward KL), STEPS steps on self-generated
#                            text, from MSE-clipped scales, frozen at 90%: KL and accuracy vs PTQ (recovered share)
#   (3) vllm/<run>           the packed compressed-tensors checkpoint loads in vLLM (weight-only kernels on A100)
#                            and generates like its dequantized twin (accuracy within 0.05; share of identical
#                            answers and of the answer both share before they diverge). If vLLM's default kernel
#                            fails (its Humming kernels are compiled at the first forward), one retry without
#                            them (Marlin); the line says which kernels served. SKIP: no vLLM, or a GPU below the
#                            scheme's minimum compute capability (7.5).
#   (4) overhead/<mode>      forward + backward time with fake quantisation vs the plain model (bf16, 4 x 1024)
set -uo pipefail

PGS_PY=${PGS_PY:-python}
PGS_SRC=${PGS_SRC:-$(cd "$(dirname "$0")/../../src" && pwd)}
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
STEPS=${STEPS:-300}
QAT_FORMATS=${QAT_FORMATS:-"nvfp4 mxfp4"}
EVAL_ITEMS=${EVAL_ITEMS:-200}
WORK=${WORK:-$(mktemp -d /tmp/pgs_fp4_qat.XXXXXX)}
NEED_GB=${NEED_GB:-8}
KEEP=${KEEP:-0}
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
mkdir -p "$WORK/logs" "$WORK/results"
: > "$WORK/results/checks.jsonl"   # a rerun in the same WORK starts its own list
START=$(date +%s)

emit() { echo "$1" | tee -a "$WORK/results/checks.jsonl"; }

cat > "$WORK/fp4_check.py" <<'PY'
"""Helpers of scripts/gpu_checks/fp4_qat.sh (written by it into $WORK)."""

import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

TOPICS = [
    "photosynthesis", "the French Revolution", "black holes", "compound interest", "the immune system",
    "volcanoes", "machine learning", "the Roman Empire", "climate change", "the water cycle", "jazz music",
    "plate tectonics", "vaccines", "the stock market", "renewable energy", "the printing press", "honey bees",
    "quantum computing", "the human brain", "coral reefs", "the Silk Road", "electric cars", "sleep",
    "the moon landing", "inflation", "DNA", "the Olympic Games", "rainforests", "cryptography", "earthquakes",
    "the Renaissance", "antibiotics", "ocean currents", "the internet", "glaciers", "chess strategy",
    "nutrition", "the solar system", "democracy", "recycling", "the Great Wall of China", "bread baking",
    "airplanes", "memory and learning", "mushrooms", "the Industrial Revolution", "tides", "batteries",
]
HELD_OUT = [
    "lighthouses", "origami", "the Hubble telescope", "beekeeping history", "bridges", "desert ecosystems",
    "public libraries", "submarines", "wind turbines", "the Aztecs", "migratory birds", "coffee",
    "railways", "penguins", "the periodic table", "marathon training",
]
TEMPLATES = [
    "Explain {t} in a short paragraph.",
    "Give three interesting facts about {t}.",
    "Explain {t} to a ten-year-old.",
    "Write a short story that involves {t}.",
    "What are the main advantages and disadvantages of {t}?",
    "Summarize the history of {t} in a few sentences.",
    "Write a short dialogue between two friends discussing {t}.",
    "What are common misconceptions about {t}?",
]


def arithmetic_prompts(rng, n):
    out = []
    for _ in range(n):
        a, b, c = rng.randint(3, 60), rng.randint(2, 30), rng.randint(1, 99)
        out.append(
            f"A store has {a} boxes with {b} items each. It sells {c} items. "
            "How many items are left? Think step by step, then give the final number."
        )
    return out


def chat(tok, prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def vllm_engine(model, quantized=False):
    from vllm import LLM

    kwargs = dict(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        max_model_len=2048,
        enforce_eager=True,
        seed=0,
    )
    try:
        return LLM(**kwargs, limit_mm_per_prompt={"image": 0, "video": 0})
    except TypeError:
        return LLM(**kwargs)


def cmd_data(work, model, n_train, n_eval, out):
    from transformers import AutoTokenizer

    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(model)
    train_prompts = [tpl.format(t=t) for t in TOPICS for tpl in TEMPLATES] + arithmetic_prompts(rng, 400)
    rng.shuffle(train_prompts)
    held_prompts = [tpl.format(t=t) for t in HELD_OUT for tpl in TEMPLATES]
    prompts = train_prompts + held_prompts
    try:
        from vllm import SamplingParams

        llm = vllm_engine(model)
        params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=384)  # repeated prompts: new samples
        repeats = max(1, math.ceil(n_train / len(train_prompts)))
        texts = [chat(tok, p) for p in train_prompts] * repeats + [chat(tok, p) for p in held_prompts]
        outs = llm.generate(texts, params)
        completions = [o.outputs[0].text for o in outs]
        used = "vllm"
    except Exception as e:  # noqa: BLE001
        print(f"vLLM generation failed ({e!r}); HF generate", file=sys.stderr)
        completions, used = hf_generate(model, tok, prompts, 384), "hf"
        repeats = 1
    train_all = (train_prompts * repeats)[: len(completions) - len(held_prompts)]
    rows = [
        {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": c}]}
        for p, c in zip(train_all, completions)
        if c.strip()
    ][:n_train]
    held = [
        {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": c}]}
        for p, c in zip(held_prompts, completions[-len(held_prompts) :])
        if c.strip()
    ]
    Path(work, "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    Path(work, "heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held))
    items, source = eval_items(rng, n_eval)
    Path(work, "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    result = {"train_rows": len(rows), "heldout_rows": len(held), "eval_items": len(items),
              "eval_source": source, "generator": used}
    Path(out).write_text(json.dumps(result))  # a file: vLLM logs to stdout
    print(json.dumps(result))


def hf_generate(model, tok, prompts, max_new):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), 64):
        batch = tok([chat(tok, p) for p in prompts[i : i + 64]], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = m.generate(**batch, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95)
        out += tok.batch_decode(gen[:, batch.input_ids.shape[1] :], skip_special_tokens=True)
    del m
    torch.cuda.empty_cache()
    return out


def eval_items(rng, n):
    try:
        from datasets import load_dataset

        for name, config in (("openai/gsm8k", "main"), ("gsm8k", "main")):
            try:
                ds = load_dataset(name, config, split="test")
                items = [
                    {"prompt": r["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
                     "answer": r["answer"].split("####")[-1].strip().replace(",", ""), "kind": "number"}
                    for r in ds
                ]
                return items[:n], "gsm8k"
            except Exception:  # noqa: BLE001
                pass
        try:
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            items = []
            for r in ds:
                choices = "\n".join(f"{lab}. {txt}" for lab, txt in zip(r["choices"]["label"], r["choices"]["text"]))
                items.append({"prompt": f"{r['question']}\n{choices}\nAnswer with the letter of the correct option only.",
                              "answer": r["answerKey"], "kind": "letter"})
            return items[:n], "arc_easy"
        except Exception:  # noqa: BLE001
            pass
    except ImportError:
        pass
    items = []
    for _ in range(n):
        a, b, c = rng.randint(11, 99), rng.randint(3, 19), rng.randint(10, 999)
        items.append({"prompt": f"Compute {a} * {b} + {c}. Reply with the number only.",
                      "answer": str(a * b + c), "kind": "number"})
    return items, "synthetic_arithmetic"


def extract(text, kind):
    if kind == "letter":
        m = re.search(r"\b([A-E])\b", text.strip())
        return m.group(1) if m else ""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = boxed[-1] if boxed else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", cand)
    if not nums:
        return ""
    v = nums[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def cmd_acc(work, model, out, tokenizer=None):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    items = [json.loads(line) for line in Path(work, "eval.jsonl").read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(tokenizer or model)
    t0 = time.time()
    llm = vllm_engine(model)
    load = time.time() - t0
    max_tokens = 8 if items[0]["kind"] == "letter" else 512
    outs = llm.generate([chat(tok, it["prompt"]) for it in items], SamplingParams(temperature=0.0, max_tokens=max_tokens))
    texts = [o.outputs[0].text for o in outs]
    correct = [extract(t, it["kind"]) == it["answer"] for t, it in zip(texts, items)]
    result = {"accuracy": sum(correct) / len(correct), "items": len(items), "load_s": round(load, 1)}
    Path(out).write_text(json.dumps({**result, "texts": texts}))
    print(json.dumps(result))


def load_hf(path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()


def heldout_batches(tok, work, limit=128):
    rows = [json.loads(line) for line in Path(work, "heldout.jsonl").read_text().splitlines()][:limit]
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False)
        yield tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()


@torch.no_grad()
def cmd_kl(work, reference, candidates, out):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(reference)
    ref = load_hf(reference)
    results = {}
    for cand in candidates:
        model = load_hf(cand)
        cand = Path(cand).parent.name if Path(cand).name in ("dequantized", "final") else cand  # the run's name
        kl_sum = agree = tokens = 0.0
        for ids in heldout_batches(tok, work):
            lp = torch.log_softmax(ref(input_ids=ids).logits[0, :-1].float(), -1)
            lq = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
            kl_sum += float((lp.exp() * (lp - lq)).sum())
            agree += float((lp.argmax(-1) == lq.argmax(-1)).sum())
            tokens += lp.shape[0]
        results[cand] = {"kl": kl_sum / tokens, "top1_agreement": agree / tokens, "tokens": int(tokens)}
        del model
        torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(results))
    print(json.dumps(results))


@torch.no_grad()
def cmd_exact(work, model_name, fmt, dequantized_dir, out):
    """The fake-quantised forward (plugin on the bf16 model, the same arguments as pgs quantize) against the
    dequantized export loaded as a plain model: identical logits expected (the same bf16 weights)."""
    from transformers import AutoTokenizer

    from palingenesis.hooks import PluginContext
    from palingenesis.qat import FloatQAT

    tok = AutoTokenizer.from_pretrained(model_name)
    fake = load_hf(model_name)
    plugin = FloatQAT(format=fmt, scales="mse")
    ctx = PluginContext(config=None, device=torch.device("cuda"))
    plugin.transform_model(fake, ctx)
    plugin.on_train_start(fake, ctx)
    plain = load_hf(dequantized_dir)
    worst, n = 0.0, 0
    for ids in heldout_batches(tok, work, limit=8):
        a, b = fake(input_ids=ids).logits.float(), plain(input_ids=ids).logits.float()
        worst = max(worst, float((a - b).abs().max()))
        n += ids.shape[1]
    Path(out).write_text(json.dumps({"max_abs_logit_diff": worst, "tokens": n}))
    print(json.dumps({"max_abs_logit_diff": worst, "tokens": n}))


def cmd_bench(model_name, out):
    from palingenesis.hooks import PluginContext
    from palingenesis.qat import FloatQAT

    torch.manual_seed(0)
    results = {}
    modes = [("plain", None), ("nvfp4", ("nvfp4", "none")), ("mxfp4", ("mxfp4", "none")), ("fp8", ("fp8", "none")),
             ("fp8_block", ("fp8_block", "none")), ("nvfp4_w4a4", ("nvfp4", "nvfp4")), ("fp8_w8a8", ("fp8", "fp8"))]
    for name, setting in modes:
        model = load_hf(model_name).train()
        if setting is not None:
            plugin = FloatQAT(format=setting[0], activations=setting[1])
            plugin.transform_model(model, PluginContext(config=None))
        vocab = model.config.get_text_config().vocab_size
        ids = torch.randint(0, vocab, (4, 1024), device="cuda")
        times = []
        for i in range(13):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            torch.cuda.synchronize()
            if i >= 3:
                times.append(time.perf_counter() - t0)
            model.zero_grad(set_to_none=True)
        times.sort()
        results[name] = times[len(times) // 2]
        del model
        torch.cuda.empty_cache()
    base = results["plain"]
    report = {k: {"step_s": round(v, 4), "ratio": round(v / base, 3)} for k, v in results.items()}
    Path(out).write_text(json.dumps(report))
    print(json.dumps(report))


def cmd_reason(log):
    """Why vLLM failed, from its log: the engine process's exception (with the `assert` line of a bare
    AssertionError), else the last exception line."""
    lines = [re.sub(r"^\(EngineCore[^)]*\)\s?", "", ln).rstrip() for ln in Path(log).read_text(errors="replace").splitlines()]
    found = ""
    for i, line in enumerate(lines):
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception)\b", line) and "Engine core initialization failed" not in line:
            found = line
            if line.strip() == "AssertionError":
                source = [ln.strip() for ln in lines[max(0, i - 4) : i] if ln.strip().startswith("assert")]
                where = [ln.strip() for ln in lines[max(0, i - 6) : i] if ln.strip().startswith("File ")]
                found = "AssertionError: " + (source[-1] if source else "") + (f" ({where[-1]})" if where else "")
            break
    if not found:
        errors = [ln for ln in lines if re.search(r"(Error|Exception)\b", ln)]
        found = errors[-1] if errors else "no exception in the log"
    print(found.strip()[:300])


def cmd_line(check, status, *pairs):
    """One result line; pairs are key=value (values parsed as JSON when they are JSON)."""
    out = {"check": check, "status": status}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    print(json.dumps(out))


def cmd_vllm(packed_file, twin_file, name, kernels):
    """The packed checkpoint's answers (vLLM) against its dequantized twin's."""
    p = json.loads(Path(packed_file).read_text())
    try:
        d = json.loads(Path(twin_file).read_text())
    except Exception:  # noqa: BLE001 — the dequantized twin was not evaluated
        d = None
    out = {"check": f"vllm/{name}", "loaded": True, "kernels": kernels, "accuracy_packed": p["accuracy"],
           "load_s": p["load_s"]}
    if d is not None:
        pairs = list(zip(p["texts"], d["texts"]))
        shared = []
        for a, b in pairs:
            n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            shared.append(n / max(len(a), len(b), 1))
        out["accuracy_dequantized"] = d["accuracy"]
        out["identical_answers"] = round(sum(a == b for a, b in pairs) / len(pairs), 3)
        out["shared_prefix"] = round(sum(shared) / len(shared), 3)  # share of an answer before the two diverge
        out["status"] = "PASS" if abs(p["accuracy"] - d["accuracy"]) <= 0.05 else "FAIL"
    else:
        out["status"] = "PASS"
    print(json.dumps(out))


def cmd_qat_summary(results, run, fmt, acc, acc_bf16):
    def load(name):
        try:
            return json.loads(Path(results, name).read_text())
        except Exception:  # noqa: BLE001
            return {}

    def kl(name):
        return load(f"kl_{name}.json").get(name, {}).get("kl")

    q, p_max, p_mse = kl(run), kl(f"ptq_{fmt}_max"), kl(f"ptq_{fmt}_mse")
    acc = None if acc == "null" else float(acc)
    acc_bf16 = None if acc_bf16 == "null" else float(acc_bf16)
    a_max, a_mse = load(f"acc_ptq_{fmt}_max.json").get("accuracy"), load(f"acc_ptq_{fmt}_mse.json").get("accuracy")
    out = {"check": f"qat/{fmt}", "kl": q, "kl_ptq_max": p_max, "kl_ptq_mse": p_mse, "accuracy": acc,
           "accuracy_ptq_max": a_max, "accuracy_ptq_mse": a_mse, "accuracy_bf16": acc_bf16}
    if q is not None and p_max:
        out["kl_recovered_vs_ptq_max"] = round(1 - q / p_max, 3)
    if acc is not None and acc_bf16 is not None and a_max is not None and acc_bf16 > a_max:
        out["accuracy_gap_recovered_vs_ptq_max"] = round((acc - a_max) / (acc_bf16 - a_max), 3)
    seconds = load(f"{run}.time.json").get("seconds")
    if seconds is not None:
        out["train_seconds"] = seconds
    ok = q is not None and p_max is not None and q < p_max and (p_mse is None or q < p_mse)
    out["status"] = "PASS" if ok else "FAIL"
    print(json.dumps(out))


def cmd_summary(checks, seconds):
    counts = {}
    for line in Path(checks).read_text().splitlines():
        try:
            status = json.loads(line).get("status")
        except ValueError:
            status = "FAIL"  # a line that is not JSON is a broken check
        counts[status] = counts.get(status, 0) + 1
    failed = bool(counts.get("FAIL"))
    print(json.dumps({"check": "summary", "status": "FAIL" if failed else "PASS", "counts": counts,
                      "seconds": int(seconds)}))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "reason":
        raise SystemExit(cmd_reason(args[0]))
    if cmd == "line":
        raise SystemExit(cmd_line(*args))
    if cmd == "vllm":
        raise SystemExit(cmd_vllm(*args))
    if cmd == "qat_summary":
        raise SystemExit(cmd_qat_summary(*args))
    if cmd == "summary":
        cmd_summary(*args)
    if cmd == "data":
        cmd_data(args[0], args[1], int(args[2]), int(args[3]), args[4])
    elif cmd == "acc":
        cmd_acc(args[0], args[1], args[2], args[3] if len(args) > 3 else None)
    elif cmd == "kl":
        cmd_kl(args[0], args[1], args[3:], args[2])
    elif cmd == "exact":
        cmd_exact(args[0], args[1], args[2], args[3], args[4])
    elif cmd == "bench":
        cmd_bench(args[0], args[1])
    else:
        raise SystemExit(f"unknown command {cmd}")
PY

H="$PGS_PY $WORK/fp4_check.py"
R="$WORK/results"
jget() { "$PGS_PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d$2)" "$1" 2>/dev/null; }
line() { emit "$($H line "$@")"; }   # line <check> <status> key=value ... (JSON-escaped)

# Weight files of a run this script wrote, deleted one by one once the run is measured (configs, reports and
# logs stay). KEEP=1 keeps them.
drop_weights() {
    [ "$KEEP" = "1" ] && return 0
    local dir f
    for dir in "$@"; do
        [ -d "$dir" ] || continue
        for f in "$dir"/*.safetensors; do
            [ -f "$f" ] && rm -f -- "$f"
        done
    done
    return 0
}

# ── disk: what the run needs (one run's checkpoints at a time) against what WORK has
free_gb() { df -Pk "$WORK" | LC_ALL=C awk 'NR==2 {printf "%.1f", $4 / 1048576}'; }   # C locale: a decimal point
FREE_GB=$(free_gb)
echo "fp4_qat.sh: needs about ${NEED_GB} GB free in $WORK (peak: one QAT run's final/ in float32 + its packed copy); ${FREE_GB} GB free" >&2
if [ "$("$PGS_PY" -c "print(int(float('$FREE_GB') < float('$NEED_GB')))")" = "1" ]; then
    line disk FAIL work="$WORK" needed_gb="$NEED_GB" free_gb="$FREE_GB" error="not enough free space in WORK: free some, or point WORK at a larger filesystem"
    exit 1
fi
line disk INFO work="$WORK" needed_gb="$NEED_GB" free_gb="$FREE_GB" keep="$KEEP"

# ── environment
"$PGS_PY" - > "$R/env.json" 2> "$WORK/logs/env.log" <<'PY'
import json, torch
info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
if torch.cuda.is_available():
    info["gpu"] = torch.cuda.get_device_name(0)
    info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
try:
    import vllm
    info["vllm"] = vllm.__version__
except Exception as e:  # noqa: BLE001
    info["vllm"] = f"unavailable: {e!r}"
import transformers
info["transformers"] = transformers.__version__
print(json.dumps(info))
PY
line env INFO work="$WORK" model="$MODEL" info="$(cat "$R/env.json" 2>/dev/null || echo null)"
CAPABILITY=$(jget "$R/env.json" "['capability']" || echo 0)
VLLM_VERSION=$(jget "$R/env.json" "['vllm']" || echo "unavailable")

# ── data: self-generated training / held-out text, accuracy items
if ! $H data "$WORK" "$MODEL" 2400 "$EVAL_ITEMS" "$R/data.json" > "$WORK/logs/data.log" 2>&1; then
    line data FAIL error="data generation failed" log="$WORK/logs/data.log"
    exit 1
fi
line data INFO result="$(cat "$R/data.json")"

# accuracy proxy (vLLM, greedy): acc <name> <model dir> [tokenizer] -> the accuracy, or null
acc() {
    if $H acc "$WORK" "$2" "$R/acc_$1.json" ${3:-} > "$WORK/logs/acc_$1.log" 2>&1; then
        jget "$R/acc_$1.json" "['accuracy']"
    else
        echo "null"
    fi
}
# KL(bf16 || run) per token on held-out text: kl <run> <model dir>
kl() {
    $H kl "$WORK" "$MODEL" "$R/kl_$1.json" "$2" > "$WORK/logs/kl_$1.log" 2>&1
}

# ── (3) a packed checkpoint in vLLM (compressed-tensors; weight-only kernels on A100)
HUMMING="HummingFP8ScaledMMLinearKernel,HummingNvFp4LinearKernel,HummingMxFp4LinearKernel,HummingLinearKernel"
vllm_check() {  # vllm_check <name> <packed dir> <dequantized twin name>
    local name=$1 dir=$2 twin=$3 kernels="default" first second log="$WORK/logs/vllm_$1.log"
    case $VLLM_VERSION in unavailable*)
        line "vllm/$name" SKIP reason="vLLM is not installed in this environment ($VLLM_VERSION)"; return ;;
    esac
    if [ "$("$PGS_PY" -c "print(int(float('${CAPABILITY:-0}') < 7.5))")" = "1" ]; then
        line "vllm/$name" SKIP reason="compute capability $CAPABILITY: vLLM's weight-only FP4 / FP8 kernels need 7.5 or higher"
        return
    fi
    if [ ! -f "$dir/config.json" ]; then
        line "vllm/$name" FAIL error="no packed checkpoint" log="$dir"; return
    fi
    if ! $H acc "$WORK" "$dir" "$R/acc_packed_$name.json" "$MODEL" > "$log" 2>&1; then
        # vLLM's Humming kernels are compiled at the first forward (NVRTC); Marlin is the precompiled alternative
        first=$($H reason "$log")
        log="$WORK/logs/vllm_$name.marlin.log"
        if ! VLLM_DISABLED_KERNELS=$HUMMING $H acc "$WORK" "$dir" "$R/acc_packed_$name.json" "$MODEL" > "$log" 2>&1; then
            second=$($H reason "$log")
            line "vllm/$name" FAIL error="vLLM could not load or run it. Default kernels: $first. Without the Humming kernels: $second" \
                log="$WORK/logs/vllm_$name.log" log_marlin="$log"
            return
        fi
        kernels="marlin (VLLM_DISABLED_KERNELS=Humming*; the default kernels failed: $first)"
    fi
    emit "$($H vllm "$R/acc_packed_$name.json" "$R/acc_$twin.json" "$name" "$kernels")"
}

ACC_BF16=$(acc bf16 "$MODEL")
line bf16 INFO accuracy="$ACC_BF16"

# ── (1) PTQ baselines, one at a time: quantise, measure, delete the weights
for spec in nvfp4:max mxfp4:max fp8:max fp8_block:max nvfp4:mse mxfp4:mse; do
    fmt=${spec%%:*}; scales=${spec##*:}; run="ptq_${fmt}_${scales}"
    if ! $PGS_PY -m palingenesis.cli quantize --model "$MODEL" --format "$fmt" --scales "$scales" \
        --out "$WORK/$run/packed" --dequantized "$WORK/$run/dequantized" \
        > "$R/$run.quantize.json" 2> "$WORK/logs/$run.quantize.log"; then
        line "ptq/$fmt/$scales" FAIL error="pgs quantize failed" log="$WORK/logs/$run.quantize.log"
        drop_weights "$WORK/$run/packed" "$WORK/$run/dequantized"
        continue
    fi
    [ "$scales" = "max" ] || drop_weights "$WORK/$run/packed"   # only the max-scale packed copies go to vLLM
    # (1b) the dequantized export == the fake-quantised forward
    if [ "$scales" = "mse" ]; then
        if $H exact "$WORK" "$MODEL" "$fmt" "$WORK/$run/dequantized" "$R/exact_$fmt.json" > "$WORK/logs/exact_$fmt.log" 2>&1; then
            diff=$(jget "$R/exact_$fmt.json" "['max_abs_logit_diff']")
            status=$("$PGS_PY" -c "print('PASS' if float('$diff') == 0.0 else 'FAIL')")
            line "exact/$fmt" "$status" max_abs_logit_diff="$diff"
        else
            line "exact/$fmt" FAIL error="exact check crashed" log="$WORK/logs/exact_$fmt.log"
        fi
    fi
    kl "$run" "$WORK/$run/dequantized" || line "kl/$run" FAIL error="KL evaluation crashed" log="$WORK/logs/kl_$run.log"
    a=$(acc "$run" "$WORK/$run/dequantized")
    k=$(jget "$R/kl_$run.json" "['$run']['kl']" || echo null)
    agree=$(jget "$R/kl_$run.json" "['$run']['top1_agreement']" || echo null)
    err=$(tail -n 1 "$R/$run.quantize.json" | "$PGS_PY" -c "import json,sys; print(json.loads(sys.stdin.read())['relative_error_mean'])" 2>/dev/null || echo null)
    status=PASS; { [ "$a" = "null" ] || [ -z "$k" ] || [ "$k" = "null" ]; } && status=FAIL
    line "ptq/$fmt/$scales" "$status" kl="${k:-null}" top1_agreement="${agree:-null}" accuracy="$a" accuracy_bf16="$ACC_BF16" weight_rel_error="${err:-null}"
    [ "$scales" = "max" ] && vllm_check "$run" "$WORK/$run/packed" "$run"
    drop_weights "$WORK/$run/packed" "$WORK/$run/dequantized"
done

# ── (2) QAT with KD to the bf16 model, one format at a time
for fmt in $QAT_FORMATS; do
    run="qat_${fmt}"
    cat > "$WORK/$run.yaml" <<YAML
model:
  name_or_path: $MODEL
  torch_dtype: float32            # fp32 latent weights, bf16 compute (train.bf16)
  attn_implementation: sdpa
  compile: false
  use_liger_kernel: false
data:
  dataset: $WORK/train.jsonl
  streaming: false
  max_seq_length: 1024
  num_workers: 2
  packing: false
train:
  output_dir: $WORK/$run
  max_steps: $STEPS
  per_device_batch_size: 8
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-5
  min_learning_rate: 1.0e-6
  warmup_ratio: 0.05
  weight_decay: 0.0
  max_grad_norm: 1.0
  save_steps: 0                   # no resume checkpoints: the run writes final/ and final-$fmt/ only
  bf16: true
  gradient_checkpointing: none
  spike_detection: false
distill:
  enabled: true
  temperature: 1.0
  kd_weight: 1.0
  ce_weight: 0.0
  teachers:
    bf16: {model: $MODEL, dtype: bfloat16}
plugins:
  custom:
    - class: palingenesis.qat:FloatQAT
      args:
        format: $fmt
        scales: mse
        freeze_scales_at: 0.9
        metrics_every: 50
logging:
  use_wandb: false
  use_trackio: false
YAML
    t0=$(date +%s)
    if $PGS_PY -m palingenesis.cli train --config "$WORK/$run.yaml" > "$WORK/logs/$run.log" 2>&1 \
        && [ -d "$WORK/$run/final" ]; then
        echo "{\"seconds\": $(( $(date +%s) - t0 ))}" > "$R/$run.time.json"
        kl "$run" "$WORK/$run/final" || line "kl/$run" FAIL error="KL evaluation crashed" log="$WORK/logs/kl_$run.log"
        a=$(acc "$run" "$WORK/$run/final")
        emit "$($H qat_summary "$R" "$run" "$fmt" "$a" "$ACC_BF16")"
        vllm_check "$run" "$WORK/$run/final-$fmt" "$run"
    else
        reason=$(grep -E "Error|No space left" "$WORK/logs/$run.log" | tail -n 1 | head -c 300)
        line "qat/$fmt" FAIL error="pgs train failed: ${reason:-see the log}" log="$WORK/logs/$run.log" \
            free_gb="$(free_gb)"
    fi
    drop_weights "$WORK/$run/final" "$WORK/$run/final-$fmt"
done

# ── (4) step-time overhead of fake quantisation
if $H bench "$MODEL" "$R/bench.json" > "$WORK/logs/bench.log" 2>&1; then
    "$PGS_PY" - "$R/bench.json" > "$R/bench_lines.jsonl" <<'PY'
import json, sys
for mode, r in json.load(open(sys.argv[1])).items():
    if mode == "plain":
        print(json.dumps({"check": "overhead/plain", "status": "INFO", **r}))
    else:
        print(json.dumps({"check": f"overhead/{mode}", "status": "PASS" if r["ratio"] < 2.5 else "FAIL", **r}))
PY
    while read -r l; do emit "$l"; done < "$R/bench_lines.jsonl"
else
    line overhead FAIL error="benchmark crashed" log="$WORK/logs/bench.log"
fi

# ── summary (exit 1 when a check failed)
$H summary "$R/checks.jsonl" "$(( $(date +%s) - START ))" | tee -a "$R/summary.json"
exit "${PIPESTATUS[0]}"
