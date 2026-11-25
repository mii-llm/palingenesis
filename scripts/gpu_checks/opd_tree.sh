#!/usr/bin/env bash
# GPU checks of train.packing in on-policy distillation (`pgs distill`): padded / varlen / tree scoring.
# One NVIDIA GPU (written for an A100 80GB, torch 2.11 cu129, flash-linear-attention installed), models in the
# Hugging Face cache, no network.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/opd_tree.sh [WORK]
#
# Every result is a JSON line {"check": ..., "status": "PASS" | "FAIL" | "INFO" | "SKIP", ...} on stdout, also
# collected in $WORK/results.jsonl; the exit code is 1 if any check failed. Everything is written inside WORK
# (default: a new directory under /tmp); only processes this script started are stopped.
#
# Parts:
#   equality  on the real models (bf16), one sampled batch: teacher log-probs / top-k / hidden states, loss, dev_kl
#             and student gradients, varlen and tree against padded (full_rkl with the fused head kernel, topk_kl)
#   speed     step time (rollout + teacher scoring + student forward / backward / optimizer), tokens/s, the teacher's
#             share and peak memory for padded / varlen / tree at group sizes 4 / 8 / 16, median of 3; the line
#             {"check": "target"} judges the pre-registered target at group 8: >= 25% lower step time with the same
#             dev_kl (|difference| < 0.01) is "met" (PASS); below 10% is "kill", in between "short" (both FAIL)
#
# Memory: the GPU's free memory is read at the start (someone else may hold part of it) and sets the prompts per
# step; a batch that does not fit is retried smaller, then reported as SKIP with the reason.
#
# Options (environment):
#   STUDENT=Qwen/Qwen3.5-0.8B   TEACHER=Qwen/Qwen3.5-2B (falls back to the student as its own teacher when the
#                               teacher is not in the cache)
#   ROLLOUT=hf                  rollout.backend of the measured steps (vllm: the engine a real run uses; the step's
#                               rollout share, and so the target, depend on it)
#   GROUPS_="4 8 16"  PROMPTS (prompts per step; default by free memory: 8 / 4 / 2)  MAX_NEW_TOKENS=384  REPEATS=3
#   SKIP="equality speed"       parts to leave out
#   CUDA_VISIBLE_DEVICES        the GPU (default 0)
set -uo pipefail

: "${PGS_PY:?set PGS_PY to the python of an environment with the palingenesis dependencies, torch and transformers}"
: "${PGS_SRC:?set PGS_SRC to palingenesis/src}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-$(mktemp -d /tmp/pgs_opd_tree.XXXXXX)}"
mkdir -p "$WORK"
STUDENT="${STUDENT:-Qwen/Qwen3.5-0.8B}"
TEACHER="${TEACHER:-Qwen/Qwen3.5-2B}"
ROLLOUT="${ROLLOUT:-hf}"
SKIP="${SKIP:-}"
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# The Hugging Face cache stays where the models are (pinned before XDG_CACHE_HOME moves into WORK).
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export TMPDIR="$WORK/tmp" TRITON_CACHE_DIR="$WORK/triton" TORCHINDUCTOR_CACHE_DIR="$WORK/inductor"
export VLLM_CACHE_ROOT="$WORK/vllm_cache" XDG_CACHE_HOME="$WORK/xdg"
mkdir -p "$TMPDIR"
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

skipped() { [[ " $SKIP " == *" $1 "* ]]; }

json_tail() {  # the last 600 bytes of a file as a JSON string
  tail -c 600 "$1" | "$PGS_PY" -c 'import json, sys; print(json.dumps(sys.stdin.read()))'
}

# run_part NAME TIMEOUT_S CMD...: runs CMD in its own process group, its JSON lines to stdout and RESULTS, the rest
# to $WORK/NAME.log; a timeout or a crash without a JSON line is a FAIL line
run_part() {
  local name="$1" limit="$2"
  shift 2
  local log="$WORK/$name.log" start=$SECONDS
  setsid "$@" >"$log.out" 2>"$log" &
  local pid=$!
  STARTED+=("$pid")
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$limit" ]; then
      kill -TERM -- "-$pid" 2>/dev/null
      sleep 5
      kill -KILL -- "-$pid" 2>/dev/null
      echo "{\"check\": \"$name\", \"status\": \"FAIL\", \"error\": \"timeout after ${limit}s\"}" | tee -a "$RESULTS"
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done
  wait "$pid"
  local code=$?
  grep '^{' "$log.out" | tee -a "$RESULTS"
  if [ "$code" -ne 0 ] && [ "$waited" -lt "$limit" ]; then
    local tail_ source="$log"
    [ -s "$source" ] || source="$log.out"
    tail_=$(json_tail "$source")
    echo "{\"check\": \"$name\", \"status\": \"FAIL\", \"exit_code\": $code, \"log_tail\": $tail_}" | tee -a "$RESULTS"
  fi
  echo "{\"check\": \"$name/elapsed\", \"status\": \"INFO\", \"seconds\": $((SECONDS - start))}" >>"$RESULTS"
}

cached() {  # whether a model is a local directory or in the Hugging Face cache
  [ -e "$1" ] || "$PGS_PY" -c 'import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_files_only=True, allow_patterns=["config.json"])' "$1" >/dev/null 2>&1
}

if ! cached "$STUDENT"; then
  echo "{\"check\": \"model_cache\", \"status\": \"FAIL\", \"model\": \"$STUDENT\", \"error\": \"not in the Hugging Face cache (HF_HOME / HF_HUB_CACHE)\"}" |
    tee -a "$RESULTS"
  exit 1
fi
if ! cached "$TEACHER"; then
  echo "{\"check\": \"model_cache\", \"status\": \"INFO\", \"model\": \"$TEACHER\", \"note\": \"not in the cache: the student is its own teacher\"}" |
    tee -a "$RESULTS"
  TEACHER="$STUDENT"
fi

# free memory now (another user's processes may hold part of the GPU): fewer prompts per step when little is free
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null | head -n 1 | tr -d ' ')
FREE_MIB="${FREE_MIB:-0}"
if [ -z "${PROMPTS:-}" ]; then
  if [ "$FREE_MIB" -ge 56000 ]; then PROMPTS=8; elif [ "$FREE_MIB" -ge 28000 ]; then PROMPTS=4; else PROMPTS=2; fi
fi
echo "{\"check\": \"gpu_memory\", \"status\": \"INFO\", \"free_mib\": $FREE_MIB, \"prompts_per_step\": $PROMPTS}" | tee -a "$RESULTS"

CHECK=("$PGS_PY" "$HERE/opd_tree_check.py")
COMMON=(--student "$STUDENT" --teacher "$TEACHER" --work "$WORK" --rollout "$ROLLOUT" --prompts "$PROMPTS"
  --max-new-tokens "${MAX_NEW_TOKENS:-384}" --repeats "${REPEATS:-3}")

run_part env 600 "${CHECK[@]}" env "${COMMON[@]}"
skipped equality || run_part equality 3600 "${CHECK[@]}" equality "${COMMON[@]}"
# shellcheck disable=SC2086
skipped speed || run_part speed 7200 "${CHECK[@]}" speed "${COMMON[@]}" --losses full_rkl --groups ${GROUPS_:-4 8 16}

failed=$(grep -c '"status": "FAIL"' "$RESULTS")
skipped_count=$(grep -c '"status": "SKIP"' "$RESULTS")
status=PASS
[ "$failed" -eq 0 ] || status=FAIL
echo "{\"check\": \"summary\", \"status\": \"$status\", \"failed\": $failed, \"skipped\": $skipped_count, \"work\": \"$WORK\", \"seconds\": $SECONDS}" |
  tee -a "$RESULTS"
[ "$failed" -eq 0 ]
