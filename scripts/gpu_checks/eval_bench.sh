#!/usr/bin/env bash
# Benchmark of `pgs eval` against lm-evaluation-harness and lighteval: the same tasks, model, sampling, GPU and
# vLLM build in all three; wall-clock, items per second and accuracy per harness, and the agreement of pgs with
# lm-eval. One NVIDIA GPU (written for an A100 80GB; torch 2.11 cu129, vLLM 0.26), no network for models and
# datasets (HF_HUB_OFFLINE=1: they come from the Hugging Face cache); PyPI is needed once, for the two harnesses.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/eval_bench.sh [WORK]
#
# Tasks:
#   gsm8k     generation: lm-eval's own gsm8k.yaml (5-shot, greedy, 256 tokens, stop strings), 1,319 items. pgs reads
#             that very file (completion mode), so prompts, stops, filters and the metric are the same by construction;
#             lighteval runs its own gsm8k (5-shot, 256 tokens; another prompt and metric: speed only).
#   mc        loglikelihood multiple choice: MMLU (57 subjects, 5-shot, 14,042 items) when cais/mmlu is in the cache,
#             else MMLU-Pro as multiple choice (configs/eval/mmlu_pro_mc.yaml, the first MC_LIMIT questions, up to ten
#             options; the same YAML file in pgs and lm-eval). lighteval has no drop-in equivalent: not run.
#   ifeval    when google/IFEval is in the cache and lm-eval's checker dependencies install: chat template, greedy,
#             1,280 tokens, thinking off. Reported, no target (the two checkers differ in sentence splitting).
#
# Method. Every harness loads the model itself, so each (harness, task) is run twice: with 1 item (start-up: imports,
# engine load) and in full; evaluation seconds = full - start-up, measured the same way for all three. pgs builds
# every prompt of a task before it applies a limit, so its start-up run already contains its prompt building: that
# time (prepare_seconds of its report) is added back to its evaluation seconds, so that no harness has its task
# preparation left out. End-to-end seconds are reported too. All harnesses import torch and vLLM from the environment of PGS_PY:
# lm-eval and lighteval are installed into a venv under WORK that sees that environment's packages
# (--system-site-packages) and may not change any of them (pip constraints from its `pip freeze`); no other
# environment is touched. A harness that cannot be installed or run is a SKIP line with the reason.
#
# Pre-registered targets (evaluation seconds):
#   gsm8k   pgs at least 2x faster than the faster of lm-eval and lighteval
#   mc      pgs at least 3x faster than lm-eval
#   accuracy of pgs within 1 standard error of lm-eval's on gsm8k (exact_match,strict-match) and mc (acc)
#   mc      the trie's log-likelihoods equal forward passes of the same weights (see loglikelihood/trie_vs_forward)
#
# Every result is a JSON line {"check": ..., "status": "PASS" | "FAIL" | "INFO" | "SKIP", ...} on stdout, also
# collected in $WORK/results.jsonl; the exit code is 1 if any check failed. Everything is written inside WORK
# (default: a new directory under /tmp); only processes this script started are stopped. The GPU's free memory is
# read at the start and every engine is given the same share of it; a run that goes out of memory is tried once
# more with 0.7 of the share, then skipped.
#
# Options (environment):
#   MODEL=Qwen/Qwen3.5-0.8B          the checkpoint (a Hub id in the cache, or a path)
#   TASKS=gsm8k,mc,ifeval            which parts to run
#   HARNESSES=pgs,lmeval,lighteval   which harnesses
#   MC_LIMIT=2000                    MMLU-Pro fallback: the first N questions (0 = all 12,032)
#   GSM8K_LIMIT=0                    the first N GSM8K items (0 = all 1,319; smaller for a smoke test)
#   MAX_MODEL_LEN=4096               the engines' context (5-shot MMLU prompts are under 3,000 tokens)
#   MEMORY_MARGIN_GB=3               left free on the GPU beyond what others hold now
#   PGS_REPLICAS=0                   also time pgs with this many engines on the GPU (INFO lines; 0 = skip)
#   TRIE_CHECK_ITEMS=64              mc: questions whose log-likelihoods are checked against forward passes (0 = skip)
#   LMEVAL_VERSION=0.4.13  LIGHTEVAL_VERSION=0.13.0
#   LIGHTEVAL_TASKS="gsm8k|5 lighteval|gsm8k|5 lighteval|gsm8k|5|0"   task spellings tried in order
#   PART_TIMEOUT=5400                seconds per run
#   CUDA_VISIBLE_DEVICES             the GPU (default 0)
set -uo pipefail

: "${PGS_PY:?set PGS_PY to the python of an environment with the palingenesis dependencies, torch, transformers, datasets and vllm}"
: "${PGS_SRC:?set PGS_SRC to palingenesis/src}"
WORK="${1:-$(mktemp -d /tmp/pgs_eval_bench.XXXXXX)}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
TASKS="${TASKS:-gsm8k,mc,ifeval}"
HARNESSES="${HARNESSES:-pgs,lmeval,lighteval}"
MC_LIMIT="${MC_LIMIT:-2000}"
GSM8K_LIMIT="${GSM8K_LIMIT:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MEMORY_MARGIN_GB="${MEMORY_MARGIN_GB:-3}"
PGS_REPLICAS="${PGS_REPLICAS:-0}"
LMEVAL_VERSION="${LMEVAL_VERSION:-0.4.13}"
LIGHTEVAL_VERSION="${LIGHTEVAL_VERSION:-0.13.0}"
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-gsm8k|5 lighteval|gsm8k|5 lighteval|gsm8k|5|0}"
PART_TIMEOUT="${PART_TIMEOUT:-5400}"
SETSID="$(command -v setsid || true)"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# The Hugging Face cache stays where the model and the datasets are (read); what the datasets library prepares, pip's
# cache and every tool's scratch space go into WORK.
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_DATASETS_CACHE="$WORK/datasets_cache"
export TMPDIR="$WORK/tmp" TRITON_CACHE_DIR="$WORK/triton" TORCHINDUCTOR_CACHE_DIR="$WORK/inductor"
export VLLM_CACHE_ROOT="$WORK/vllm_cache" XDG_CACHE_HOME="$WORK/xdg" PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR="$WORK/pip_cache" PIP_DISABLE_PIP_VERSION_CHECK=1 NLTK_DATA="$WORK/nltk_data"
export PGS_EVAL_CACHE="$WORK/pgs_cache" RAY_TMPDIR="$WORK/tmp" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$WORK/timing"
RESULTS="$WORK/results.jsonl"
: >"$RESULTS"
STARTED=()

cleanup() {
  for pid in "${STARTED[@]:-}"; do
    [ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null
  done
  sleep 2
  for pid in "${STARTED[@]:-}"; do
    [ -n "$pid" ] && kill -KILL -- "-$pid" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

emit() { echo "$1" | tee -a "$RESULTS"; }
has() { case ",$1," in *",$2,"*) return 0 ;; *) return 1 ;; esac; }
now() { "$PGS_PY" -c 'import time; print(f"{time.time():.3f}")'; }
json_tail() { tail -c 1200 "$1" 2>/dev/null | "$PGS_PY" -c 'import json, sys; print(json.dumps(sys.stdin.read()))'; }

# run_part NAME CMD...: runs CMD in its own process group (stdout to $WORK/NAME.out, stderr to $WORK/NAME.log) and
# writes its wall-clock seconds to $WORK/timing/NAME. Returns the exit code (124: timeout); prints nothing.
run_part() {
  local name="$1"
  shift
  local log="$WORK/$name.log" t0
  mkdir -p "$(dirname "$WORK/$name")" "$(dirname "$WORK/timing/$name")"
  t0="$(now)"
  $SETSID "$@" >"$WORK/$name.out" 2>"$log" &
  local pid=$!
  STARTED+=("$pid")
  local waited=0 timed_out=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$PART_TIMEOUT" ]; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      sleep 5
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
      timed_out=1
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  wait "$pid"
  local code=$?
  kill -KILL -- "-$pid" 2>/dev/null  # whatever the command left behind in its own group
  "$PGS_PY" -c "import sys, time; print(f'{time.time() - float(sys.argv[1]):.3f}')" "$t0" >"$WORK/timing/$name"
  [ "$timed_out" -eq 1 ] && return 124
  return "$code"
}

out_of_memory() {
  grep -qiE "out of memory|OutOfMemoryError|No available memory for the cache blocks|larger than the maximum number of tokens that can be stored in KV cache|insufficient GPU memory" "$WORK/$1.log" "$WORK/$1.out" 2>/dev/null
}

# attempt NAME CMD...: CMD with the placeholder @GMU@ replaced by the GPU memory share; on out-of-memory once more
# with 0.7 of it. Emits a FAIL / SKIP line when it does not succeed. Returns 0 on success.
attempt() {
  local name="$1"
  shift
  local share="$GMU" try code
  for try in 1 2; do
    local command=()
    for arg in "$@"; do command+=("${arg//@GMU@/$share}"); done
    run_part "$name" "${command[@]}"
    code=$?
    [ "$code" -eq 0 ] && return 0
    if [ "$try" -eq 1 ] && out_of_memory "$name"; then
      share="$("$PGS_PY" -c "print(round($share * 0.7, 3))")"
      continue
    fi
    break
  done
  local status=FAIL
  { out_of_memory "$name" || [ "${name%%/*}" != "pgs" ]; } && status=SKIP  # a competitor that cannot run is skipped
  emit "{\"check\": \"run/$name\", \"status\": \"$status\", \"exit_code\": $code, \"gpu_memory_utilization\": $share, \"log_tail\": $(json_tail "$WORK/$name.log")}"
  return 1
}

# ---- environment: versions, the GPU's free memory (others may hold some), the model
"$PGS_PY" - "$MODEL" "$MEMORY_MARGIN_GB" >"$WORK/env.out" 2>"$WORK/env.log" <<'PY'
import json
import sys

model, margin = sys.argv[1], float(sys.argv[2])
line = {"check": "env", "status": "PASS"}
share = 0.0
try:
    import torch
    import vllm

    line.update(torch=torch.__version__, vllm=vllm.__version__, cuda=torch.cuda.is_available())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        share = max(0.0, min(0.9, (free / 2**30 - margin) / (total / 2**30)))
        line.update(gpu=torch.cuda.get_device_name(0), gpu_free_gb=round(free / 2**30, 1),
                    gpu_total_gb=round(total / 2**30, 1), gpu_memory_utilization=round(share, 3))
        if share < 0.1:
            line.update(status="FAIL", error="less than 10% of the GPU's memory is free")
    else:
        line.update(status="FAIL", error="CUDA is not available")
except Exception as e:  # noqa: BLE001
    line.update(status="FAIL", error=f"{type(e).__name__}: {e}")
print(json.dumps(line))
try:
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model)
    print(json.dumps({"check": "model_cache", "status": "PASS", "model": model}))
except Exception as e:  # noqa: BLE001
    print(json.dumps({"check": "model_cache", "status": "FAIL", "model": model, "error": f"{type(e).__name__}: {e}"[:400]}))
print(f"GMU={share:.3f}")
PY
grep '^{' "$WORK/env.out" | tee -a "$RESULTS"
GMU="$(grep '^GMU=' "$WORK/env.out" | cut -d= -f2)"
GMU="${GMU:-0.5}"
if grep -q '"status": "FAIL"' "$RESULTS"; then
  emit "{\"check\": \"summary\", \"status\": \"FAIL\", \"error\": \"environment\", \"work\": \"$WORK\"}"
  exit 1
fi

# ---- the competitors' venv: sees PGS_PY's packages, may not change them
VENV="$WORK/venv"
VPY="$VENV/bin/python"
LMEVAL_OK=0
LIGHTEVAL_OK=0
if has "$HARNESSES" lmeval || has "$HARNESSES" lighteval; then
  "$PGS_PY" -m venv --system-site-packages "$VENV" >"$WORK/venv.log" 2>&1
  "$PGS_PY" -m pip freeze 2>/dev/null | grep -E '^[A-Za-z0-9_.-]+==' >"$WORK/constraints.txt"
  if has "$HARNESSES" lmeval; then
    if HF_HUB_OFFLINE=0 "$VPY" -m pip install -q -c "$WORK/constraints.txt" "lm-eval==$LMEVAL_VERSION" >"$WORK/pip_lmeval.log" 2>&1 \
      && "$VPY" -c "import lm_eval" 2>>"$WORK/pip_lmeval.log"; then
      LMEVAL_OK=1
      # IFEval's checker: optional (the ifeval part is skipped for lm-eval without it)
      HF_HUB_OFFLINE=0 "$VPY" -m pip install -q -c "$WORK/constraints.txt" langdetect immutabledict nltk >>"$WORK/pip_lmeval.log" 2>&1
      HF_HUB_OFFLINE=0 "$VPY" -c "import nltk; nltk.download('punkt_tab', quiet=True)" >>"$WORK/pip_lmeval.log" 2>&1
      emit "{\"check\": \"install/lmeval\", \"status\": \"PASS\", \"version\": \"$LMEVAL_VERSION\"}"
    else
      emit "{\"check\": \"install/lmeval\", \"status\": \"SKIP\", \"version\": \"$LMEVAL_VERSION\", \"log_tail\": $(json_tail "$WORK/pip_lmeval.log")}"
    fi
  fi
  if has "$HARNESSES" lighteval; then
    # with its dependencies under the constraints; if they conflict with the environment, the package alone with the
    # dependencies its vLLM path imports
    if HF_HUB_OFFLINE=0 "$VPY" -m pip install -q -c "$WORK/constraints.txt" "lighteval==$LIGHTEVAL_VERSION" ray more_itertools >"$WORK/pip_lighteval.log" 2>&1 \
      || { HF_HUB_OFFLINE=0 "$VPY" -m pip install -q --no-deps "lighteval==$LIGHTEVAL_VERSION" >>"$WORK/pip_lighteval.log" 2>&1 \
        && HF_HUB_OFFLINE=0 "$VPY" -m pip install -q -c "$WORK/constraints.txt" ray more_itertools inspect-ai aenum termcolor pytablewriter \
          colorlog latex2sympy2_extended langcodes pycountry typer GitPython nltk rouge_score sacrebleu scikit-learn httpx openai >>"$WORK/pip_lighteval.log" 2>&1; }; then
      if "$VPY" -c "import lighteval, ray" 2>>"$WORK/pip_lighteval.log"; then
        LIGHTEVAL_OK=1
        emit "{\"check\": \"install/lighteval\", \"status\": \"PASS\", \"version\": \"$LIGHTEVAL_VERSION\"}"
      fi
    fi
    [ "$LIGHTEVAL_OK" -eq 1 ] || emit "{\"check\": \"install/lighteval\", \"status\": \"SKIP\", \"version\": \"$LIGHTEVAL_VERSION\", \"log_tail\": $(json_tail "$WORK/pip_lighteval.log")}"
  fi
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# ---- the tasks: lm-eval's own gsm8k.yaml; MMLU or the MMLU-Pro multiple-choice file; which datasets are cached
LMEVAL_TASK_DIR=""
[ "$LMEVAL_OK" -eq 1 ] && LMEVAL_TASK_DIR="$("$VPY" -c 'import os, lm_eval; print(os.path.join(os.path.dirname(lm_eval.__file__), "tasks"))' 2>/dev/null)"
GSM8K_YAML="$LMEVAL_TASK_DIR/gsm8k/gsm8k.yaml"
if [ ! -f "$GSM8K_YAML" ]; then  # lm-eval is not installed: pgs is still timed, on a copy of the same definition
  mkdir -p "$WORK/tasks/gsm8k"
  GSM8K_YAML="$WORK/tasks/gsm8k/gsm8k.yaml"
  cat >"$GSM8K_YAML" <<'YAML'
task: gsm8k
dataset_path: openai/gsm8k
dataset_name: main
output_type: generate_until
training_split: train
fewshot_split: train
test_split: test
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore: [",", "\\$", "(?s).*#### ", "\\.$"]
generation_kwargs:
  until: ["Question:", "</s>", "<|im_end|>"]
  do_sample: false
  temperature: 0.0
num_fewshot: 5
filter_list:
  - name: "strict-match"
    filter:
      - {function: "regex", regex_pattern: "#### (\\-?[0-9\\.\\,]+)"}
      - {function: "take_first"}
  - name: "flexible-extract"
    filter:
      - {function: "regex", group_select: -1, regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"}
      - {function: "take_first"}
metadata:
  version: 3.0
YAML
fi
mkdir -p "$WORK/lmeval_tasks"
cp "$REPO/configs/eval/mmlu_pro_mc.yaml" "$WORK/lmeval_tasks/mmlu_pro_mc.yaml"
"$PGS_PY" - "$GSM8K_YAML" "$WORK/lmeval_tasks/mmlu_pro_mc.yaml" >"$WORK/datasets.out" 2>"$WORK/datasets.log" <<'PY'
import json
import sys

from palingenesis.eval.config import load_task
from palingenesis.eval.tasks import build_items

for name, ref in (("gsm8k", sys.argv[1]), ("mmlu", "mmlu"), ("mmlu_pro_mc", sys.argv[2]), ("ifeval", "ifeval")):
    try:
        items = build_items(load_task(ref), limit=8, head=True)
        print(json.dumps({"check": f"dataset/{name}", "status": "PASS", "items_checked": len(items)}))
    except Exception as e:  # noqa: BLE001 — a dataset that is not in the cache: that part is skipped
        print(json.dumps({"check": f"dataset/{name}", "status": "SKIP", "error": f"{type(e).__name__}: {e}"[:400]}))
PY
grep '^{' "$WORK/datasets.out" | tee -a "$RESULTS"
cached() { grep -q "\"check\": \"dataset/$1\", \"status\": \"PASS\"" "$RESULTS"; }

MC_NAME="" MC_PGS=() MC_LM=()
if cached mmlu; then
  MC_NAME=mmlu
  MC_PGS=(--tasks mmlu)
  MC_LM=(--tasks mmlu --num_fewshot 5)
elif cached mmlu_pro_mc; then
  MC_NAME=mmlu_pro_mc
  MC_PGS=(--tasks "$WORK/lmeval_tasks/mmlu_pro_mc.yaml")
  MC_LM=(--tasks mmlu_pro_mc --include_path "$WORK/lmeval_tasks")
  [ "$MC_LIMIT" -gt 0 ] && MC_PGS+=(--limit "$MC_LIMIT" --head) && MC_LM+=(--limit "$MC_LIMIT")
fi
GSM_PGS=() GSM_LM=() GSM_LE=()
[ "$GSM8K_LIMIT" -gt 0 ] && GSM_PGS=(--limit "$GSM8K_LIMIT" --head) && GSM_LM=(--limit "$GSM8K_LIMIT") && GSM_LE=(--max-samples "$GSM8K_LIMIT")

PGS=("$PGS_PY" -m palingenesis.eval.cli --model "$MODEL" --max_model_len "$MAX_MODEL_LEN" --engine.gpu_memory_utilization @GMU@ --no_cache --json --run.resamples 2000)
LM=("$VPY" -m lm_eval --model vllm --batch_size auto --log_samples)
LM_ARGS="pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=@GMU@,max_model_len=$MAX_MODEL_LEN,seed=0"

# pair HARNESS TASK FULL_ARGS... -- START_ARGS...: the start-up run (1 item), then the full run
pair() {
  local name="$1"
  shift
  local full=() start=() seen=0
  for arg in "$@"; do
    if [ "$arg" = "--" ]; then seen=1; continue; fi
    if [ "$seen" -eq 0 ]; then full+=("$arg"); else start+=("$arg"); fi
  done
  attempt "$name.start" "${start[@]}" || return 1
  attempt "$name" "${full[@]}"
}

# ---- gsm8k
if has "$TASKS" gsm8k && cached gsm8k; then
  if has "$HARNESSES" pgs; then
    pair pgs/gsm8k "${PGS[@]}" --tasks "$GSM8K_YAML" --out "$WORK/pgs/gsm8k" ${GSM_PGS[@]+"${GSM_PGS[@]}"} -- \
      "${PGS[@]}" --tasks "$GSM8K_YAML" --out "$WORK/pgs/gsm8k.start" --limit 1 --head
    if [ "$PGS_REPLICAS" -gt 1 ]; then
      pair pgs_replicas/gsm8k "${PGS[@]}" --tasks "$GSM8K_YAML" --out "$WORK/pgs_replicas/gsm8k" --replicas "$PGS_REPLICAS" ${GSM_PGS[@]+"${GSM_PGS[@]}"} -- \
        "${PGS[@]}" --tasks "$GSM8K_YAML" --out "$WORK/pgs_replicas/gsm8k.start" --replicas "$PGS_REPLICAS" --limit 1 --head
    fi
  fi
  if has "$HARNESSES" lmeval && [ "$LMEVAL_OK" -eq 1 ]; then
    pair lmeval/gsm8k "${LM[@]}" --model_args "$LM_ARGS" --tasks gsm8k --output_path "$WORK/lmeval/gsm8k" ${GSM_LM[@]+"${GSM_LM[@]}"} -- \
      "${LM[@]}" --model_args "$LM_ARGS" --tasks gsm8k --output_path "$WORK/lmeval/gsm8k.start" --limit 1
  fi
  if has "$HARNESSES" lighteval && [ "$LIGHTEVAL_OK" -eq 1 ]; then
    LE_ARGS="model_name=$MODEL,dtype=bfloat16,gpu_memory_utilization=@GMU@,max_model_length=$MAX_MODEL_LEN"
    LE_SPEC=""
    for spec in $LIGHTEVAL_TASKS; do  # the task's spelling changed between releases: the first one that runs
      if run_part "lighteval/gsm8k.start" "$VENV/bin/lighteval" vllm "${LE_ARGS//@GMU@/$GMU}" "$spec" --output-dir "$WORK/lighteval/gsm8k.start" --max-samples 1; then
        LE_SPEC="$spec"
        break
      fi
    done
    if [ -n "$LE_SPEC" ]; then
      attempt lighteval/gsm8k "$VENV/bin/lighteval" vllm "$LE_ARGS" "$LE_SPEC" --output-dir "$WORK/lighteval/gsm8k" ${GSM_LE[@]+"${GSM_LE[@]}"}
    else
      emit "{\"check\": \"run/lighteval/gsm8k\", \"status\": \"SKIP\", \"error\": \"no task spelling ran\", \"log_tail\": $(json_tail "$WORK/lighteval/gsm8k.start.log")}"
    fi
  fi
fi

# ---- multiple choice by loglikelihood
if has "$TASKS" mc && [ -n "$MC_NAME" ]; then
  if has "$HARNESSES" pgs; then
    pair pgs/mc "${PGS[@]}" ${MC_PGS[@]+"${MC_PGS[@]}"} --out "$WORK/pgs/mc" -- \
      "${PGS[@]}" --tasks "${MC_PGS[1]}" --out "$WORK/pgs/mc.start" --limit 1 --head
  fi
  # the trie's log-likelihoods against plain forward passes of the same weights (transformers, no vLLM): the first
  # TRIE_CHECK_ITEMS questions, every choice. Pre-registered: the same answer on >= 95% of them, and log-likelihoods
  # within 0.25 nats (two bf16 implementations) on >= 95% of the choices.
  if has "$HARNESSES" pgs && [ "${TRIE_CHECK_ITEMS:-64}" -gt 0 ] && [ -f "$WORK/pgs/mc/summary.json" ]; then
    if attempt pgs/mc_forward "$PGS_PY" -m palingenesis.eval.cli --model "$MODEL" --tasks "${MC_PGS[1]}" --backend hf \
      --max_model_len "$MAX_MODEL_LEN" --no_cache --json --limit "${TRIE_CHECK_ITEMS:-64}" --head --out "$WORK/pgs/mc_forward"; then
      "$PGS_PY" - "$WORK/pgs/mc" "$WORK/pgs/mc_forward" <<'PY' | tee -a "$RESULTS"
import glob
import json
import sys


def samples(directory):
    (path,) = glob.glob(f"{directory}/*.samples.jsonl")
    return {r["id"]: r for r in map(json.loads, open(path))}


trie, forward = samples(sys.argv[1]), samples(sys.argv[2])
shared = sorted(set(trie) & set(forward))
pairs = [(a, b) for i in shared for a, b in zip(trie[i]["loglikelihoods"], forward[i]["loglikelihoods"])]
close = sum(abs(a - b) <= 0.25 for a, b in pairs) / max(1, len(pairs))
same = sum(trie[i]["completion"] == forward[i]["completion"] for i in shared) / max(1, len(shared))
ok = bool(shared) and close >= 0.95 and same >= 0.95
print(json.dumps({"check": "loglikelihood/trie_vs_forward", "status": "PASS" if ok else "FAIL", "items": len(shared),
                  "choices": len(pairs), "within_0.25_nats": round(close, 4), "same_answer": round(same, 4),
                  "max_abs_difference": round(max((abs(a - b) for a, b in pairs), default=0.0), 4),
                  "requests_per_item": round(sum(trie[i]["requests"] for i in shared) / max(1, len(shared)), 2)}))
PY
    fi
  fi
  if has "$HARNESSES" lmeval && [ "$LMEVAL_OK" -eq 1 ]; then
    pair lmeval/mc "${LM[@]}" --model_args "$LM_ARGS" ${MC_LM[@]+"${MC_LM[@]}"} --output_path "$WORK/lmeval/mc" -- \
      "${LM[@]}" --model_args "$LM_ARGS" ${MC_LM[@]+"${MC_LM[@]}"} --output_path "$WORK/lmeval/mc.start" --limit 1
  fi
elif has "$TASKS" mc; then
  emit '{"check": "run/mc", "status": "SKIP", "error": "neither cais/mmlu nor TIGER-Lab/MMLU-Pro is in the cache"}'
fi

# ---- ifeval (chat template, thinking off, greedy)
if has "$TASKS" ifeval && cached ifeval; then
  if has "$HARNESSES" pgs; then
    pair pgs/ifeval "${PGS[@]}" --tasks ifeval --modes plain --temperature 0 --out "$WORK/pgs/ifeval" -- \
      "${PGS[@]}" --tasks ifeval --modes plain --temperature 0 --out "$WORK/pgs/ifeval.start" --limit 1 --head
  fi
  if has "$HARNESSES" lmeval && [ "$LMEVAL_OK" -eq 1 ] && "$VPY" -c "import langdetect, immutabledict, nltk" 2>/dev/null; then
    pair lmeval/ifeval "${LM[@]}" --model_args "$LM_ARGS,enable_thinking=False" --apply_chat_template --tasks ifeval --output_path "$WORK/lmeval/ifeval" -- \
      "${LM[@]}" --model_args "$LM_ARGS,enable_thinking=False" --apply_chat_template --tasks ifeval --output_path "$WORK/lmeval/ifeval.start" --limit 1
  fi
fi

# ---- the report: seconds, items per second and accuracy per harness; speed-ups and agreement against the targets
"$PGS_PY" - "$WORK" "$MC_NAME" <<'PY' | tee -a "$RESULTS"
import glob
import json
import os
import sys

from palingenesis.eval import conformance

work, mc_name = sys.argv[1], sys.argv[2]
TARGETS = {"gsm8k": 2.0, "mc": 3.0}
LM_TASK = {"gsm8k": ("gsm8k", "exact_match,strict-match"), "mc": (mc_name, "acc"), "ifeval": ("ifeval", "prompt_level_strict_acc")}


def seconds(name):
    try:
        return float(open(os.path.join(work, "timing", name)).read())
    except (OSError, ValueError):
        return None


def pgs_result(harness, task):
    path = os.path.join(work, harness, task, "summary.json")
    if not os.path.isfile(path):
        return None
    report = json.load(open(path))
    result = report["results"][0]
    mean = result.get("mean_score") or {}
    return {"items": result["items"], "prepare_seconds": round(report.get("prepare_seconds", 0.0), 2), "accuracy": mean.get("estimate"), "stderr": result.get("stderr"),
            "internal_seconds": round(result["seconds"], 2), "tokens_per_second": round(result["tokens_per_second"]),
            "time_to_first_result": result.get("time_to_first_result"), "engine_requests": result.get("engine_requests"),
            "gpu_util_mean": result.get("gpu_util_mean"), "results": result.get("results"),
            "metrics": {k: v["mean"] for k, v in (result.get("metrics") or {}).items()}}


def lmeval_result(task):
    name, metric = LM_TASK[task]
    aggregate, docs = conformance.lmeval_results(os.path.join(work, "lmeval", task), name, metric)
    items = None
    for path in glob.glob(os.path.join(work, "lmeval", task, "**", "results*.json"), recursive=True):
        counts = json.load(open(path)).get("n-samples") or {}
        items = sum(v.get("effective", 0) for v in counts.values()) or None
    return None if aggregate is None and not docs else {"items": items or len(docs), "accuracy": aggregate, "docs": docs}


def lighteval_result(task):
    for path in sorted(glob.glob(os.path.join(work, "lighteval", task, "**", "results*.json"), recursive=True)):
        results = json.load(open(path)).get("results") or {}
        for key, metrics in results.items():
            if "gsm8k" in key:
                value = next((v for k, v in metrics.items() if "stderr" not in k and isinstance(v, int | float)), None)
                return {"items": None, "accuracy": value, "metric": next(iter(metrics), None)}
    return None


for task in ("gsm8k", "mc", "ifeval"):
    rows = {}
    for harness, read in (("pgs", lambda t: pgs_result("pgs", t)), ("pgs_replicas", lambda t: pgs_result("pgs_replicas", t)),
                          ("lmeval", lmeval_result), ("lighteval", lighteval_result)):
        full, start = seconds(f"{harness}/{task}"), seconds(f"{harness}/{task}.start")
        result = read(task) if full is not None else None
        if result is None or full is None or start is None:
            continue
        docs = result.pop("docs", None)
        evaluation = max(full - start + (result.get("prepare_seconds") or 0.0), 1e-3)
        items = result.get("items") or (rows.get("lmeval") or rows.get("pgs") or {}).get("items")
        rows[harness] = {**result, "items": items, "wall_seconds": round(full, 1), "startup_seconds": round(start, 1),
                         "eval_seconds": round(evaluation, 1), "items_per_second": round(items / evaluation, 2) if items else None}
        rows[harness]["_docs"] = docs
    if not rows:
        continue
    for harness, row in rows.items():
        line = {"check": f"bench/{task}/{harness}", "status": "INFO", **{k: v for k, v in row.items() if not k.startswith("_")}}
        if task == "mc":
            line["mc_task"] = mc_name
        print(json.dumps(line))
    if "pgs" not in rows:
        continue
    competitors = {h: r["eval_seconds"] for h, r in rows.items() if h in ("lmeval", "lighteval")}
    if competitors:
        fastest = min(competitors, key=competitors.get)
        speedup = competitors[fastest] / rows["pgs"]["eval_seconds"]
        line = {"check": f"speedup/{task}", "pgs_eval_seconds": rows["pgs"]["eval_seconds"], "fastest_competitor": fastest,
                "competitor_eval_seconds": competitors[fastest], "speedup": round(speedup, 2),
                "end_to_end_speedup": round(rows[fastest]["wall_seconds"] / rows["pgs"]["wall_seconds"], 2)}
        if task in TARGETS:
            line.update(status="PASS" if speedup >= TARGETS[task] else "FAIL", target=TARGETS[task])
        else:
            line["status"] = "INFO"
        print(json.dumps(line))
    if "lmeval" in rows and rows["pgs"].get("results"):
        name, metric = LM_TASK[task]
        ours = conformance.pgs_scores(rows["pgs"]["results"], metric)
        report = conformance.compare(ours, rows["lmeval"]["accuracy"], rows["lmeval"]["_docs"] or {}, max_se=1.0)
        gated = task in TARGETS
        print(json.dumps({"check": f"agreement/{task}", "status": ("PASS" if report["within"] else "FAIL") if gated else "INFO",
                          "metric": metric, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in report.items()}}))
PY

failed=$(grep -c '"status": "FAIL"' "$RESULTS")
skipped_count=$(grep -c '"status": "SKIP"' "$RESULTS")
status=PASS
[ "$failed" -eq 0 ] || status=FAIL
emit "{\"check\": \"summary\", \"status\": \"$status\", \"failed\": $failed, \"skipped\": $skipped_count, \"work\": \"$WORK\", \"seconds\": $SECONDS}"
[ "$failed" -eq 0 ]
