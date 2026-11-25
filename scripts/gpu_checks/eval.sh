#!/usr/bin/env bash
# GPU check of `pgs eval`: GSM8K test (1,319 items) on one model in both thinking modes, k = 1, through vLLM, and
# a `pgs compare` of the two modes. One NVIDIA GPU (written for an A100), vLLM installed, no network: the model and
# the datasets come from the Hugging Face cache.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/eval.sh [WORK]
#
# Pre-registered: each mode's evaluation (generation and scoring of the 1,319 items, engine start excluded) takes
# under MAX_SECONDS (600). Accuracy is reported, with its interval; it has no target.
#
# Every result is a JSON line {"check": ..., "status": "PASS" | "FAIL" | "INFO" | "SKIP", ...} on stdout, also
# collected in $WORK/results.jsonl; the exit code is 1 if any check failed. Everything is written inside WORK
# (default: a new directory under /tmp); only processes this script started are stopped.
#
# Options (environment):
#   MODEL=Qwen/Qwen3.5-0.8B          the checkpoint (a Hub id in the cache, or a path)
#   MODES=plain,think  K=1           thinking modes (each a separate result) and samples per item
#   MAX_SECONDS=600                  the per-mode limit
#   GPU_MEMORY_UTILIZATION=0.5       vLLM's share of the GPU (a shared machine: leave room for others)
#   MAX_MODEL_LEN=4096               the engine's context (GSM8K prompts are short; max_tokens is 2048)
#   EXTRA_TASKS=math500              further tasks, reported without a target when their dataset is in the cache
#                                    ("" to skip)
#   EVAL_ARGS="--temperature 0"      further `pgs eval` arguments (default: the model's generation_config sampling)
#   CUDA_VISIBLE_DEVICES             the GPU (default 0)
#   REQUIRE_GPU=0                    a dry run of the plumbing without CUDA / vLLM (with EVAL_ARGS="--backend hf
#                                    --max_tokens 8" and a tiny MODEL): the environment line is INFO, not FAIL
set -uo pipefail

: "${PGS_PY:?set PGS_PY to the python of an environment with the palingenesis dependencies, torch, transformers, datasets and vllm}"
: "${PGS_SRC:?set PGS_SRC to palingenesis/src}"
WORK="${1:-$(mktemp -d /tmp/pgs_eval.XXXXXX)}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
MODES="${MODES:-plain,think}"
K="${K:-1}"
MAX_SECONDS="${MAX_SECONDS:-600}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
EXTRA_TASKS="${EXTRA_TASKS-math500}"
EVAL_ARGS="${EVAL_ARGS:-}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
SETSID="$(command -v setsid || true)"  # each part in its own process group (absent on macOS: dry runs only)
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# The Hugging Face cache stays where the model and the datasets are (read only: huggingface_hub's default is
# $XDG_CACHE_HOME/huggingface, so it is pinned before XDG_CACHE_HOME moves into WORK). The datasets library's own
# cache of prepared tables moves into WORK: a dataset that is only in the Hub cache is read from its files there.
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_DATASETS_CACHE="$WORK/datasets_cache"
export TMPDIR="$WORK/tmp" TRITON_CACHE_DIR="$WORK/triton" TORCHINDUCTOR_CACHE_DIR="$WORK/inductor"
export VLLM_CACHE_ROOT="$WORK/vllm_cache" XDG_CACHE_HOME="$WORK/xdg" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
RESULTS="$WORK/results.jsonl"
: >"$RESULTS"
STARTED=()  # process-group leaders this script started

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

json_tail() {  # the last 800 bytes of a file as a JSON string
  tail -c 800 "$1" | "$PGS_PY" -c 'import json, sys; print(json.dumps(sys.stdin.read()))'
}

# run_part NAME TIMEOUT_S CMD...: runs CMD in its own process group, stdout to $WORK/NAME.out and stderr to
# $WORK/NAME.log; a timeout or a non-zero exit is a FAIL line. Returns the command's exit code (124: timeout).
run_part() {
  local name="$1" limit="$2"
  shift 2
  local log="$WORK/$name.log" start=$SECONDS
  $SETSID "$@" >"$WORK/$name.out" 2>"$log" &
  local pid=$!
  STARTED+=("$pid")
  local waited=0 timed_out=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$limit" ]; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      sleep 5
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
      timed_out=1
      break
    fi
    sleep 2
    waited=$((waited + 2))
  done
  wait "$pid"
  local code=$?
  kill -KILL -- "-$pid" 2>/dev/null  # whatever the command left behind in its group
  if [ "$timed_out" -eq 1 ]; then
    emit "{\"check\": \"$name\", \"status\": \"FAIL\", \"error\": \"timeout after ${limit}s\", \"log_tail\": $(json_tail "$log")}"
    return 124
  fi
  if [ "$code" -ne 0 ]; then
    local source="$log"
    [ -s "$source" ] || source="$WORK/$name.out"
    emit "{\"check\": \"$name\", \"status\": \"FAIL\", \"exit_code\": $code, \"log_tail\": $(json_tail "$source")}"
    return "$code"
  fi
  emit "{\"check\": \"$name\", \"status\": \"PASS\", \"wall_seconds\": $((SECONDS - start))}"
}

# ---- environment: CUDA, vLLM, the model and the datasets in the cache
"$PGS_PY" - "$MODEL" "gsm8k" "$EXTRA_TASKS" "$REQUIRE_GPU" >"$WORK/env.out" 2>"$WORK/env.log" <<'PY'
import json
import sys

model, required, extra = sys.argv[1], sys.argv[2].split(","), [t for t in sys.argv[3].split(",") if t]
bad = "FAIL" if sys.argv[4] == "1" else "INFO"  # REQUIRE_GPU=0: a dry run of the plumbing
line = {"check": "env", "status": "PASS"}
try:
    import torch

    line.update(torch=torch.__version__, cuda=torch.cuda.is_available())
    import vllm

    line.update(vllm=vllm.__version__)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        line.update(gpu=torch.cuda.get_device_name(0), gpu_free_gb=round(free / 2**30, 1), gpu_total_gb=round(total / 2**30, 1))
    else:
        line.update(status=bad, error="CUDA is not available")
except Exception as e:  # noqa: BLE001
    line.update(status=bad, error=f"{type(e).__name__}: {e}")
print(json.dumps(line))
try:
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model)
    print(json.dumps({"check": "model_cache", "status": "PASS", "model": model}))
except Exception as e:  # noqa: BLE001
    print(json.dumps({"check": "model_cache", "status": "FAIL", "model": model, "error": f"{type(e).__name__}: {e}"[:400]}))
from palingenesis.eval.config import load_task
from palingenesis.eval.tasks import build_items

for name in required + extra:
    try:
        items = build_items(load_task(name))
        print(json.dumps({"check": f"dataset/{name}", "status": "PASS", "items": len(items)}))
    except Exception as e:  # noqa: BLE001 — an extra task whose dataset is not cached is skipped, not failed
        status = "FAIL" if name in required else "SKIP"
        print(json.dumps({"check": f"dataset/{name}", "status": status, "error": f"{type(e).__name__}: {e}"[:400]}))
PY
env_code=$?
grep '^{' "$WORK/env.out" | tee -a "$RESULTS"
if [ "$env_code" -ne 0 ]; then
  emit "{\"check\": \"env\", \"status\": \"FAIL\", \"exit_code\": $env_code, \"log_tail\": $(json_tail "$WORK/env.log")}"
fi

# ---- the evaluation: one engine, every mode (each mode's own seconds are in its summary)
TASKS="gsm8k"
for task in ${EXTRA_TASKS//,/ }; do
  grep -q "\"check\": \"dataset/$task\", \"status\": \"PASS\"" "$RESULTS" && TASKS="$TASKS,$task"
done
if ! grep -q '"status": "FAIL"' "$RESULTS"; then
  # shellcheck disable=SC2086
  run_part pgs_eval $((MAX_SECONDS * 6 + 900)) "$PGS_PY" -m palingenesis.eval.cli --model "$MODEL" --tasks "$TASKS" \
    --modes "$MODES" --k "$K" --out "$WORK/eval" --max_model_len "$MAX_MODEL_LEN" \
    --engine.gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" --json $EVAL_ARGS
fi

# ---- the report: accuracy (no target) and seconds (pre-registered limit) per task and mode
if [ -f "$WORK/eval/summary.json" ]; then
  "$PGS_PY" - "$WORK/eval/summary.json" "$MAX_SECONDS" <<'PY' | tee -a "$RESULTS"
import json
import sys

report, limit = json.load(open(sys.argv[1])), float(sys.argv[2])
for r in report["results"]:
    gated = r["task"] == "gsm8k"  # the pre-registered check; other tasks are reported
    mean = r.get("mean_score") or {}
    line = {
        "check": f"eval/{r['task']}/{r['mode']}",
        "status": "INFO",
        "model": report["model"],
        "items": r["items"],
        "k": r["k"],
        "accuracy": mean.get("estimate"),
        "accuracy_lo": mean.get("lo"),
        "accuracy_hi": mean.get("hi"),
        "wilson": [r["wilson"]["lo"], r["wilson"]["hi"]] if "wilson" in r else None,
        "seconds": round(r["seconds"], 1),
        "tokens_per_second": round(r["tokens_per_second"]),
        "completion_tokens_mean": round(r["completion_tokens"]["mean"], 1),
        "truncated_share": r["truncated_share"],
        "loop_share": r["loop_share"],
        "unscored_samples": r["unscored_samples"],
        "sampling": r["sampling"],
    }
    if gated:
        problems = []
        if r["seconds"] >= limit:
            problems.append(f"{r['seconds']:.0f}s is not under the pre-registered {limit:.0f}s")
        if r["items"] != 1319:
            problems.append(f"{r['items']} scored items, expected 1319")
        line.update(status="FAIL" if problems else "PASS", limit_seconds=limit)
        if problems:
            line["error"] = "; ".join(problems)
    print(json.dumps(line))
PY
fi

# ---- pgs compare of the modes (paired by item): the difference with its interval, McNemar at k = 1
FIRST="${MODES%%,*}"
for mode in ${MODES//,/ }; do
  [ "$mode" = "$FIRST" ] && continue
  a="$WORK/eval/gsm8k.$FIRST.jsonl" b="$WORK/eval/gsm8k.$mode.jsonl"
  [ -f "$a" ] && [ -f "$b" ] || continue
  if "$PGS_PY" -m palingenesis.agent_tooling.compare "$a" "$b" --strict --json >"$WORK/compare_$mode.json" 2>"$WORK/compare_$mode.log"; then
    "$PGS_PY" - "$WORK/compare_$mode.json" "$FIRST" "$mode" <<'PY' | tee -a "$RESULTS"
import json
import sys

report = json.load(open(sys.argv[1]))
c = report["comparison"]
m = c["mean_score"]
print(json.dumps({
    "check": f"compare/gsm8k/{sys.argv[2]}_vs_{sys.argv[3]}",
    "status": "PASS",
    "paired_items": c["paired_items"],
    "a": m["a"],
    "b": m["b"],
    "diff": m["diff"],
    "diff_lo": m["lo"],
    "diff_hi": m["hi"],
    "p_permutation": m["p_permutation"],
    "mcnemar": c.get("mcnemar"),
}))
PY
  else
    emit "{\"check\": \"compare/gsm8k/${FIRST}_vs_$mode\", \"status\": \"FAIL\", \"log_tail\": $(json_tail "$WORK/compare_$mode.log")}"
  fi
done

failed=$(grep -c '"status": "FAIL"' "$RESULTS")
skipped_count=$(grep -c '"status": "SKIP"' "$RESULTS")
status=PASS
[ "$failed" -eq 0 ] || status=FAIL
emit "{\"check\": \"summary\", \"status\": \"$status\", \"failed\": $failed, \"skipped\": $skipped_count, \"work\": \"$WORK\", \"seconds\": $SECONDS}"
[ "$failed" -eq 0 ]
