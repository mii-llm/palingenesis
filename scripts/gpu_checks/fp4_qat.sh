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
#   STEPS        QAT steps per format (default 1000; the training set is 8 x STEPS rows: one epoch, no row twice)
#   QAT_LR       QAT peak learning rate (default 3.0e-6, cosine to a tenth of it)
#   QAT_FORMATS  formats trained with QAT + KD (default "nvfp4 mxfp4")
#   EVAL_ITEMS   accuracy-proxy items (default 200)
#
# Output: one JSON line per check on stdout, {"check": ..., "status": "PASS" | "FAIL" | "SKIP" | "INFO", ...numbers};
# the last line is a summary. Logs of every step are in $WORK/logs. Expected run time on an A100 80GB: ~50-60 min
# (data generation ~8 min, 13 vLLM evaluations ~1-2 min each, 2 QAT runs ~8-10 min each, benchmarks).
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
#                            text, from MSE-clipped scales, frozen at 90%. Expectation (pre-registered): QAT
#                            recovers at least half of the held-out KL of the PTQ model it starts from (mse
#                            scales; KL on the answer tokens, the ones KD trains) and does not lower the accuracy
#                            proxy (within 0.05, about one standard error of a 200-item difference). The line has
#                            the numbers either way: KL on answer tokens and on all tokens, accuracy, how the
#                            answers end (boxed, cut at the length limit), the training KD at the start and the
#                            end, and the share of quantised values training changed (codes_changed).
#
# Training data: the bf16 model's own sampled answers (temperature 0.8) to prompts that are never the evaluation's:
# half reasoning questions in the evaluation's prompt format (GSM8K train split if cached, else word problems
# from several templates), half open questions on 48 topics x 16 templates. Held-out text for the KL: other
# reasoning questions and 16 other topics. (A first version trained on 2400 rows, half of them one arithmetic
# template, at 1e-5 for 300 steps: the training KD halved while the held-out KL did not move and GSM8K accuracy
# fell from 0.42 to 0.24; see the QAT guide.)
#   (3) vllm/<run>           the packed compressed-tensors checkpoint loads in vLLM (weight-only kernels on A100)
#                            and computes what its dequantized twin computes: the log-probability of every token
#                            of 20 fixed held-out texts (teacher-forced), vLLM on the packed checkpoint against
#                            transformers on the twin. PASS: mean |difference| <= max(0.03 nats, 3 x the engine
#                            noise), the engine noise being the same statistic for the twin itself in vLLM
#                            against transformers (same weights, two engines); and accuracy within 0.05. The
#                            line also has the quantised model's distance to bf16 in the same unit
#                            (logprob_gap_to_bf16: what a wrong decoding would be comparable to), the share of
#                            identical greedy answers and of an answer shared before the two diverge (greedy
#                            answers of hundreds of tokens diverge at the first near-tie: not a criterion).
#                            If vLLM's default kernel
#                            fails (its Humming kernels are compiled at the first forward), one retry without
#                            them (Marlin); the line says which kernels served. SKIP: no vLLM, or a GPU below the
#                            scheme's minimum compute capability (7.5).
#   (4) overhead/<mode>      forward + backward time with fake quantisation vs the plain model (bf16, 4 x 1024)
set -uo pipefail

PGS_PY=${PGS_PY:-python}
PGS_SRC=${PGS_SRC:-$(cd "$(dirname "$0")/../../src" && pwd)}
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
STEPS=${STEPS:-1000}
QAT_LR=${QAT_LR:-3.0e-6}
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
    "Compare {t} with something an everyday reader already knows.",
    "Write a short quiz of three questions about {t}, with answers.",
    "Describe a typical day of someone who works with {t}.",
    "Give step-by-step advice to a beginner who wants to learn about {t}.",
    "Write a formal paragraph and then a casual one about {t}.",
    "What open questions remain about {t}?",
    "List the key terms someone needs to discuss {t}, each with a one-line definition.",
    "Argue for and then against a common opinion about {t}.",
]
REASONING_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
WORD_PROBLEMS = [  # questions only: the bf16 model's own answers are the data
    "A store has {a} boxes with {b} items each. It sells {c} items. How many items are left?",
    "{n} has {a} marbles and buys {b} bags of {d} marbles each. How many marbles does {n} have now?",
    "A train travels {a} km per hour for {d} hours, then {b} km per hour for {e} hours. How far does it travel?",
    "{n} earns {a} dollars a day and spends {b} dollars a day. How much is saved after {c} days?",
    "A recipe needs {d} eggs per cake. How many eggs are needed for {a} cakes, and how many cartons of 12 is that?",
    "{a} students are split into groups of {d}. How many full groups are there and how many students are left over?",
    "A tank holds {c} liters. It is filled at {d} liters per minute. How many minutes until it is half full?",
    "{n} reads {b} pages on each of {d} days and {a} pages on the last day. How many pages in total?",
    "A shirt costs {a} dollars and is discounted by {d}0 percent. What is the price of {e} shirts?",
    "There are {a} red and {b} blue balls. {d} of each color are removed. What fraction of the rest is red?",
]
NAMES = ["Ana", "Ben", "Chen", "Dalia", "Emre", "Farah", "Goran", "Hana", "Ivo", "Jun"]


def reasoning_prompts(rng, n):
    """n distinct reasoning questions in the evaluation's prompt format, never the evaluation's own items: the
    GSM8K train split if it is cached, else word problems from several templates. (prompts, source)"""
    try:
        from datasets import load_dataset

        for name in ("openai/gsm8k", "gsm8k"):
            try:
                questions = [r["question"] for r in load_dataset(name, "main", split="train")]
                rng.shuffle(questions)
                return [q + REASONING_SUFFIX for q in questions[:n]], "gsm8k_train"
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        pass
    seen, out = set(), []
    while len(out) < n:
        text = rng.choice(WORD_PROBLEMS).format(
            a=rng.randint(12, 240), b=rng.randint(3, 60), c=rng.randint(20, 900), d=rng.randint(2, 9),
            e=rng.randint(2, 9), n=rng.choice(NAMES),
        )
        if text not in seen:
            seen.add(text)
            out.append(text + REASONING_SUFFIX)
    return out, "word_problems"


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
    # half reasoning questions (each once), half open questions (repeated prompts get new samples)
    held_reasoning = 64
    reasoning, reasoning_source = reasoning_prompts(rng, n_train // 2 + held_reasoning)
    held_prompts = [tpl.format(t=t) for t in HELD_OUT for tpl in TEMPLATES[:8]] + reasoning[:held_reasoning]
    reasoning = reasoning[held_reasoning:]
    topical = [tpl.format(t=t) for t in TOPICS for tpl in TEMPLATES]
    wanted = n_train - len(reasoning)
    topical = (topical * math.ceil(wanted / len(topical)))[:wanted]
    train_prompts = reasoning + topical
    rng.shuffle(train_prompts)
    prompts = train_prompts + held_prompts
    try:
        from vllm import SamplingParams

        llm = vllm_engine(model)
        params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=384)
        outs = llm.generate([chat(tok, p) for p in prompts], params)
        completions = [o.outputs[0].text for o in outs]
        used = "vllm"
    except Exception as e:  # noqa: BLE001
        print(f"vLLM generation failed ({e!r}); HF generate", file=sys.stderr)
        completions, used = hf_generate(model, tok, prompts, 384), "hf"
    train_all = train_prompts
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
    result = {"train_rows": len(rows), "train_reasoning_rows": len(reasoning), "train_source": reasoning_source,
              "heldout_rows": len(held), "eval_items": len(items), "eval_source": source, "generator": used}
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


def fixed_ids(tok, work, rows=20, max_length=512):
    """Token ids of the first held-out texts: the fixed prompts of the vLLM / transformers logprob comparison."""
    data = [json.loads(line) for line in Path(work, "heldout.jsonl").read_text().splitlines()][:rows]
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False) for r in data]
    return [tok(t, truncation=True, max_length=max_length).input_ids for t in texts]


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
    result = {
        "accuracy": sum(correct) / len(correct),
        "items": len(items),
        "load_s": round(load, 1),
        # how the answers end: a model that lost the format or runs to the limit shows here, not in the KL
        "boxed_share": sum("\\boxed{" in t for t in texts) / len(texts),
        "length_limit_share": sum(o.outputs[0].finish_reason == "length" for o in outs) / len(outs),
        "mean_answer_chars": round(sum(len(t) for t in texts) / len(texts)),
    }
    extra = {"texts": texts}
    try:  # teacher-forced: the log-probability vLLM gives every token of the fixed texts
        ids = fixed_ids(tok, work)
        scored = llm.generate(
            [{"prompt_token_ids": x} for x in ids], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        )
        extra["prompt_logprobs"] = [
            [step[token].logprob for token, step in zip(x[1:], o.prompt_logprobs[1:])] for x, o in zip(ids, scored)
        ]
    except Exception as e:  # noqa: BLE001 — reported by the vllm/<run> line
        extra["prompt_logprobs_error"] = repr(e)[:300]
    Path(out).write_text(json.dumps({**result, **extra}))
    print(json.dumps(result))


def load_hf(path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()


def heldout_batches(tok, work, limit=256, answer_start=False):
    """Token ids of the held-out rows; with answer_start, also the index of the answer's first token (the
    prompt up to the generation prompt is a prefix of the row's text)."""
    rows = [json.loads(line) for line in Path(work, "heldout.jsonl").read_text().splitlines()][:limit]
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()
        if not answer_start:
            yield ids
            continue
        prompt = tok(chat(tok, r["messages"][0]["content"])).input_ids
        start = len(prompt) if ids[0, : len(prompt)].tolist() == prompt else 0
        yield ids, start


@torch.no_grad()
def cmd_kl(work, reference, candidates, out):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(reference)
    ref = load_hf(reference)
    results = {}
    fixed = [torch.tensor([x], device="cuda") for x in fixed_ids(tok, work)]

    def token_logprobs(m):  # of every token of the fixed texts given the ones before it (as vLLM's prompt_logprobs)
        out = []
        for ids in fixed:
            lp = torch.log_softmax(m(input_ids=ids).logits[0, :-1].float(), -1)
            out.append(lp.gather(1, ids[0, 1:, None])[:, 0].tolist())
        return out

    ref_logprobs = token_logprobs(ref)
    for cand in candidates:
        model = load_hf(cand)
        cand = Path(cand).parent.name if Path(cand).name in ("dequantized", "final") else cand  # the run's name
        kl_sum = agree = tokens = 0.0
        answer_kl = answer_agree = answer_tokens = 0.0
        for ids, start in heldout_batches(tok, work, answer_start=True):
            lp = torch.log_softmax(ref(input_ids=ids).logits[0, :-1].float(), -1)
            lq = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            same = lp.argmax(-1) == lq.argmax(-1)
            kl_sum += float(kl.sum())
            agree += float(same.sum())
            tokens += lp.shape[0]
            first = max(start - 1, 0)  # position i predicts token i + 1: the answer's tokens from start - 1
            answer_kl += float(kl[first:].sum())
            answer_agree += float(same[first:].sum())
            answer_tokens += lp.shape[0] - first
        logprobs = token_logprobs(model)
        gap = [abs(a - b) for x, y in zip(logprobs, ref_logprobs) for a, b in zip(x, y)]
        results[cand] = {"kl": kl_sum / tokens, "top1_agreement": agree / tokens, "tokens": int(tokens),
                         # the answer's tokens only: what KD trains and what generation uses
                         "kl_answer": answer_kl / max(answer_tokens, 1),
                         "top1_agreement_answer": answer_agree / max(answer_tokens, 1),
                         "answer_tokens": int(answer_tokens),
                         "logprob_gap_to_bf16": sum(gap) / len(gap), "prompt_logprobs": logprobs}
        del model
        torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(results))
    print(json.dumps({k: {a: b for a, b in v.items() if a != "prompt_logprobs"} for k, v in results.items()}))


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


LOGPROB_TOLERANCE = 0.03  # nats: mean |difference| of a token's log-probability, packed in vLLM vs the twin in HF


def logprob_diff(a, b):
    """(mean, max) |difference| over the tokens of two lists of per-text log-probabilities."""
    if a is None or b is None or len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return None
    diffs = [abs(u - v) for x, y in zip(a, b) for u, v in zip(x, y)]
    return (sum(diffs) / len(diffs), max(diffs), len(diffs)) if diffs else None


def cmd_vllm(packed_file, twin_file, name, kernels, kl_file, twin):
    """The packed checkpoint in vLLM against its dequantized twin: token log-probabilities (the criterion),
    accuracy, greedy answers."""
    p = json.loads(Path(packed_file).read_text())
    try:
        d = json.loads(Path(twin_file).read_text())
    except Exception:  # noqa: BLE001 — the dequantized twin was not evaluated
        d = None
    try:
        hf = json.loads(Path(kl_file).read_text())[twin]
    except Exception:  # noqa: BLE001 — no transformers pass over the twin
        hf = {}
    out = {"check": f"vllm/{name}", "loaded": True, "kernels": kernels, "accuracy_packed": p["accuracy"],
           "load_s": p["load_s"]}
    packed_vs_hf = logprob_diff(p.get("prompt_logprobs"), hf.get("prompt_logprobs"))
    noise = logprob_diff((d or {}).get("prompt_logprobs"), hf.get("prompt_logprobs"))
    same_engine = logprob_diff(p.get("prompt_logprobs"), (d or {}).get("prompt_logprobs"))
    logprobs_ok = False
    if packed_vs_hf is None:
        out["logprob_error"] = (
            p.get("prompt_logprobs_error")
            or "no token log-probabilities to compare (vLLM's prompt_logprobs, or the transformers pass over the twin)"
        )
    else:
        out["logprob_mean_abs_diff"] = round(packed_vs_hf[0], 5)  # packed in vLLM vs the twin in transformers
        out["logprob_max_abs_diff"] = round(packed_vs_hf[1], 4)
        out["logprob_tokens"] = packed_vs_hf[2]
        limit = LOGPROB_TOLERANCE
        if noise is not None:  # the twin in vLLM vs the twin in transformers: the same weights in two engines
            out["logprob_engine_noise"] = round(noise[0], 5)
            limit = max(limit, 3 * noise[0])
        if same_engine is not None:  # packed vs twin, both in vLLM
            out["logprob_mean_abs_diff_in_vllm"] = round(same_engine[0], 5)
        if hf.get("logprob_gap_to_bf16") is not None:  # the quantisation itself, in the same unit
            out["logprob_gap_to_bf16"] = round(hf["logprob_gap_to_bf16"], 5)
        out["logprob_limit"] = round(limit, 5)
        logprobs_ok = packed_vs_hf[0] <= limit
    if d is not None:
        pairs = list(zip(p["texts"], d["texts"]))
        shared = []
        for a, b in pairs:
            n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            shared.append(n / max(len(a), len(b), 1))
        out["accuracy_dequantized"] = d["accuracy"]
        out["identical_answers"] = round(sum(a == b for a, b in pairs) / len(pairs), 3)
        out["shared_prefix"] = round(sum(shared) / len(shared), 3)  # share of an answer before the two diverge
        out["status"] = "PASS" if logprobs_ok and abs(p["accuracy"] - d["accuracy"]) <= 0.05 else "FAIL"
    else:
        out["status"] = "PASS" if logprobs_ok else "FAIL"
    print(json.dumps(out))


class JsonlTracker:
    """logging.trackers backend of the QAT runs: every logged metric as a line of <output_dir>/metrics.jsonl
    (the plugin's metrics are not in the console log)."""

    def __init__(self, *, output_dir, **_):
        self.file = Path(output_dir, "metrics.jsonl")
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text("")

    def log(self, metrics, step):
        row = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        with self.file.open("a") as f:
            f.write(json.dumps({"step": step, **row}) + "\n")

    def finish(self, exit_code):
        pass


def training_metrics(file):
    """From a run's metrics.jsonl: the KD loss at the start and the end (means over a tenth of the logged steps)
    and the plugin's last error / change metrics."""
    try:
        rows = [json.loads(line) for line in Path(file).read_text().splitlines() if line.strip()]
    except OSError:
        return {}
    out = {}
    kd = [next(v for k, v in r.items() if k.endswith("kd_loss")) for r in rows if any(k.endswith("kd_loss") for k in r)]
    if kd:
        tenth = max(1, len(kd) // 10)
        out["train_kd_start"] = round(sum(kd[:tenth]) / tenth, 4)
        out["train_kd_end"] = round(sum(kd[-tenth:]) / tenth, 4)
    for name in ("codes_changed", "qerr_mean", "saturated_blocks"):
        values = [v for r in rows for k, v in r.items() if k.endswith(f"float_qat/{name}")]
        if values:
            out[name] = round(values[-1], 5)
    return out


def cmd_qat_summary(results, run, fmt, acc, acc_bf16, metrics_file):
    """qat/<fmt>: the pre-registered expectation and every number behind it."""

    def load(name):
        try:
            return json.loads(Path(results, name).read_text())
        except Exception:  # noqa: BLE001
            return {}

    def kl(name, key="kl_answer"):
        return load(f"kl_{name}.json").get(name, {}).get(key)

    def answers(name):
        r = load(f"acc_{name}.json")
        return {k: r.get(k) for k in ("boxed_share", "length_limit_share", "mean_answer_chars")}

    start = f"ptq_{fmt}_mse"  # the model QAT starts from
    q, p_max, p_mse = kl(run), kl(f"ptq_{fmt}_max"), kl(start)
    acc = None if acc == "null" else float(acc)
    acc_bf16 = None if acc_bf16 == "null" else float(acc_bf16)
    a_max, a_mse = load(f"acc_ptq_{fmt}_max.json").get("accuracy"), load(f"acc_{start}.json").get("accuracy")
    out = {"check": f"qat/{fmt}", "kl_answer": q, "kl_answer_ptq_mse": p_mse, "kl_answer_ptq_max": p_max,
           "kl_all_tokens": kl(run, "kl"), "kl_all_tokens_ptq_mse": kl(start, "kl"),
           "accuracy": acc, "accuracy_ptq_mse": a_mse, "accuracy_ptq_max": a_max, "accuracy_bf16": acc_bf16,
           "answers": answers(run), "answers_ptq_mse": answers(start)}
    recovered = None
    if q is not None and p_mse:
        recovered = out["kl_recovered"] = round(1 - q / p_mse, 3)
    out.update(training_metrics(metrics_file))
    seconds = load(f"{run}.time.json").get("seconds")
    if seconds is not None:
        out["train_seconds"] = seconds
    kl_ok = recovered is not None and recovered >= 0.5
    acc_ok = acc is not None and a_mse is not None and acc >= a_mse - 0.05
    out["expectation"] = "kl_recovered >= 0.5 and accuracy >= accuracy_ptq_mse - 0.05"
    out["kl_met"], out["accuracy_met"] = kl_ok, acc_ok
    out["status"] = "PASS" if kl_ok and acc_ok else "FAIL"
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
if ! $H data "$WORK" "$MODEL" "$(( STEPS * 8 ))" "$EVAL_ITEMS" "$R/data.json" > "$WORK/logs/data.log" 2>&1; then
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
    emit "$($H vllm "$R/acc_packed_$name.json" "$R/acc_$twin.json" "$name" "$kernels" "$R/kl_$twin.json" "$twin")"
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
    ka=$(jget "$R/kl_$run.json" "['$run']['kl_answer']" || echo null)
    agree=$(jget "$R/kl_$run.json" "['$run']['top1_agreement']" || echo null)
    err=$(tail -n 1 "$R/$run.quantize.json" | "$PGS_PY" -c "import json,sys; print(json.loads(sys.stdin.read())['relative_error_mean'])" 2>/dev/null || echo null)
    status=PASS; { [ "$a" = "null" ] || [ -z "$k" ] || [ "$k" = "null" ]; } && status=FAIL
    line "ptq/$fmt/$scales" "$status" kl="${k:-null}" kl_answer="${ka:-null}" top1_agreement="${agree:-null}" accuracy="$a" accuracy_bf16="$ACC_BF16" weight_rel_error="${err:-null}"
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
  learning_rate: $QAT_LR
  min_learning_rate: $("$PGS_PY" -c "print(f'{float(\"$QAT_LR\") / 10:.3e}')")
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
        code_metrics: true          # plugins/float_qat/codes_changed: how much of the quantised model training moved
logging:
  use_wandb: false
  use_trackio: false
  trackers: ["$WORK/fp4_check.py:JsonlTracker"]   # every metric into $WORK/$run/metrics.jsonl
YAML
    t0=$(date +%s)
    if $PGS_PY -m palingenesis.cli train --config "$WORK/$run.yaml" > "$WORK/logs/$run.log" 2>&1 \
        && [ -d "$WORK/$run/final" ]; then
        echo "{\"seconds\": $(( $(date +%s) - t0 ))}" > "$R/$run.time.json"
        kl "$run" "$WORK/$run/final" || line "kl/$run" FAIL error="KL evaluation crashed" log="$WORK/logs/kl_$run.log"
        a=$(acc "$run" "$WORK/$run/final")
        emit "$($H qat_summary "$R" "$run" "$fmt" "$a" "$ACC_BF16" "$WORK/$run/metrics.jsonl")"
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
