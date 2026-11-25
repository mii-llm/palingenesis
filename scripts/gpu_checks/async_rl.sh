#!/usr/bin/env bash
# GPU check of asynchronous RL: staleness 0 / 1 / 4 / 8 x objective (masked_is, decoupled) on a small math run, on
# the colocated in-process vLLM engine and on a vLLM server on the same GPU (the stand-in for engines on other
# nodes), plus the per-token in-flight paths (rollout.inflight tag / resume, rollout.stale mask, rollout.sync
# publish) and the fail-safe weight push protocol (checksummed buckets, restore, a Gloo link per engine, a lost
# worker) between processes of this machine.
# One NVIDIA GPU (written for an A100 80GB), the model in the Hugging Face cache, no network.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src \
#   DATA=data/gsm8k_train.jsonl EVAL_DATA=data/gsm8k_test.jsonl bash scripts/gpu_checks/async_rl.sh [WORK]
#
# Every result is a JSON line {"check": ..., "status": "PASS" | "FAIL" | "KILL" | "INFO" | "SKIP", ...} on stdout,
# also collected in $WORK/results.jsonl; the exit code is 1 if any check failed. Everything is written inside WORK
# (default: a new directory under /tmp); only processes this script started are stopped.
#
# Pre-registered (see async_rl_check.py): decoupled at staleness 8 gains >= 1.2 x the evaluation reward per minute
# of wall-clock of the on-policy run and ends within 1 point of its final reward (PASS); KILL when masked_is at
# staleness 8 already does. One seed: read the verdict against the standard error the line reports.
#
# Options (environment):
#   MODEL=Qwen/Qwen3.5-0.8B  STEPS=40  BATCH_PROMPTS=32  GROUP_SIZE=8  EVAL_SAMPLES=512  LR=  SEED=0
#   STALENESS=0,1,4,8  LOSSES=masked_is,decoupled  PROXIMAL=step|reference
#   PARTS=colocated,server,inflight   the training arms; SKIP="protocol train" leaves a whole part out
#   CONFIG=configs/rl_math.yaml       the base config (model, data and train sections are overridden)
#   CUDA_VISIBLE_DEVICES              the GPU (default 0)
#
# Not covered here (needs more than one GPU or node): the NCCL link itself (rollout.push_link nccl: two GPUs),
# pushes over a real network, many engines per trainer rank, and a worker lost on another node.
set -uo pipefail

: "${PGS_PY:?set PGS_PY to the python of an environment with the palingenesis dependencies, torch, transformers and vllm}"
: "${PGS_SRC:?set PGS_SRC to palingenesis/src}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${1:-$(mktemp -d /tmp/pgs_async_rl.XXXXXX)}"
mkdir -p "$WORK"
SKIP="${SKIP:-}"
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export TMPDIR="$WORK/tmp" TRITON_CACHE_DIR="$WORK/triton" TORCHINDUCTOR_CACHE_DIR="$WORK/inductor"
export VLLM_CACHE_ROOT="$WORK/vllm_cache" XDG_CACHE_HOME="$WORK/xdg"
mkdir -p "$TMPDIR"
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

skipped() { [[ " $SKIP " == *" $1 "* ]]; }

# run_part NAME TIMEOUT_S CMD...: CMD in its own process group; its JSON lines to stdout and RESULTS, the rest to
# $WORK/NAME.log; a timeout or a crash is a FAIL line
run_part() {
  local name="$1" limit="$2"
  shift 2
  local log="$WORK/$name.log"
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
  if [ "$code" -ne 0 ] && [ "$waited" -lt "$limit" ] && ! grep -q '"status": "FAIL"' "$log.out"; then
    local tail_
    tail_=$(tail -c 600 "$log" | "$PGS_PY" -c 'import json, sys; print(json.dumps(sys.stdin.read()))')
    echo "{\"check\": \"$name\", \"status\": \"FAIL\", \"exit_code\": $code, \"log_tail\": $tail_}" | tee -a "$RESULTS"
  fi
}

# 1. the push protocol between processes of this machine (CPU tensors: checksums, restore, the Gloo link and its
#    fall back to TCP, a lost and restarted worker): the repository's own tests of it
if ! skipped protocol; then
  if "$PGS_PY" -c 'import pytest' 2>/dev/null && [ -f "$REPO/tests/test_rl_pool_safe.py" ]; then
    run_part protocol 1800 env CUDA_VISIBLE_DEVICES="" "$PGS_PY" -m pytest -q -p no:cacheprovider \
      "$REPO/tests/test_rl_pool_safe.py" "$REPO/tests/test_rl_async.py"
    if tail -n 3 "$WORK/protocol.log.out" | grep -q ' passed' && ! tail -n 3 "$WORK/protocol.log.out" | grep -q 'failed'; then
      echo '{"check": "protocol", "status": "PASS"}' | tee -a "$RESULTS"
    else
      echo '{"check": "protocol", "status": "FAIL", "log": "'"$WORK"'/protocol.log.out"}' | tee -a "$RESULTS"
    fi
  else
    echo '{"check": "protocol", "status": "SKIP", "reason": "pytest or the tests are not here"}' | tee -a "$RESULTS"
  fi
fi

# 2. the training arms
if ! skipped train; then
  : "${DATA:?set DATA to the training rows (JSONL with a prompt and an answer column, e.g. GSM8K)}"
  : "${EVAL_DATA:?set EVAL_DATA to the held-out rows}"
  MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
  if [ ! -e "$MODEL" ] && ! "$PGS_PY" -c 'import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_files_only=True, allow_patterns=["config.json"])' "$MODEL" >/dev/null 2>&1; then
    echo "{\"check\": \"model_cache\", \"status\": \"FAIL\", \"model\": \"$MODEL\", \"error\": \"not in the Hugging Face cache\"}" |
      tee -a "$RESULTS"
  else
    arms=$(echo "${STALENESS:-0,1,4,8}" | tr ',' '\n' | wc -l)
    run_part train $((5400 * (2 * arms + 6))) "$PGS_PY" "$HERE/async_rl_check.py" --work "$WORK/arms" \
      --config "${CONFIG:-$REPO/configs/rl_math.yaml}" --data "$DATA" --eval-data "$EVAL_DATA" --model "$MODEL" \
      --steps "${STEPS:-40}" --batch-prompts "${BATCH_PROMPTS:-32}" --group-size "${GROUP_SIZE:-8}" \
      --eval-samples "${EVAL_SAMPLES:-512}" --learning-rate "${LR:-0}" --seed "${SEED:-0}" \
      --staleness "${STALENESS:-0,1,4,8}" --losses "${LOSSES:-masked_is,decoupled}" \
      --proximal "${PROXIMAL:-step}" --parts "${PARTS:-colocated,server,inflight}"
  fi
fi

failed=$(grep -c '"status": "FAIL"' "$RESULTS")
killed=$(grep -c '"status": "KILL"' "$RESULTS")
status=PASS
[ "$killed" -eq 0 ] || status=KILL
[ "$failed" -eq 0 ] || status=FAIL
echo "{\"check\": \"summary\", \"status\": \"$status\", \"failed\": $failed, \"work\": \"$WORK\", \"seconds\": $SECONDS}" |
  tee -a "$RESULTS"
[ "$failed" -eq 0 ]
