#!/usr/bin/env bash
# GPU checks of the RL ports: train.packing (padded / varlen / tree) and the rollout concurrency controller.
# One NVIDIA GPU (written for an A100 80GB), models in the Hugging Face cache, no network.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/rl_packing.sh [WORK]
#
# Every result is a JSON line {"check": ..., "status": "PASS" | "FAIL" | "INFO", ...} on stdout, also collected in
# $WORK/results.jsonl; the exit code is 1 if any check failed. Everything is written inside WORK (default: a new
# directory under /tmp); only processes this script started are stopped.
#
# Options (environment):
#   MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B"   equality and throughput models
#   SMOKE_MODEL=Qwen/Qwen3.5-0.8B  SMOKE_STEPS=20                    the `pgs rl` smoke runs (one per packing mode)
#   SKIP="equality throughput smoke concurrency"                      parts to leave out
#   EQUALITY_GROUPS=4  THROUGHPUT_GROUPS=8                            groups per scenario (x 8 rollouts)
#   CUDA_VISIBLE_DEVICES                                              the GPU (default 0)
set -uo pipefail

: "${PGS_PY:?set PGS_PY to the python of a Python environment with the palingenesis dependencies, torch, transformers and vllm}"
: "${PGS_SRC:?set PGS_SRC to palingenesis/src}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-$(mktemp -d /tmp/pgs_rl_packing.XXXXXX)}"
mkdir -p "$WORK"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3.5-0.8B}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
SKIP="${SKIP:-}"
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
    local tail_
    tail_=$(json_tail "$log")
    echo "{\"check\": \"$name\", \"status\": \"FAIL\", \"exit_code\": $code, \"log_tail\": $tail_}" | tee -a "$RESULTS"
  fi
  echo "{\"check\": \"$name/elapsed\", \"status\": \"INFO\", \"seconds\": $((SECONDS - start))}" >>"$RESULTS"
}

CHECK=("$PGS_PY" "$HERE/rl_packing_check.py")
DEVICE_ARGS=(--device "${CHECK_DEVICE:-cuda:0}")  # CHECK_DEVICE=cpu: a dry run of the plumbing on tiny models

run_part env 600 "${CHECK[@]}" env

for model in $MODELS; do
  skipped equality && break
  tag="${model//\//_}"
  run_part "equality_$tag" 2400 "${CHECK[@]}" equality --model "$model" --groups "${EQUALITY_GROUPS:-4}" "${DEVICE_ARGS[@]}"
done

for model in $MODELS; do
  skipped throughput && break
  tag="${model//\//_}"
  for scenario in rl branching; do
    run_part "throughput_${tag}_$scenario" 3600 "${CHECK[@]}" throughput --model "$model" --scenario "$scenario" \
      --groups "${THROUGHPUT_GROUPS:-8}" --repeats 3 "${DEVICE_ARGS[@]}"
  done
done

if ! skipped smoke; then
  for mode in padded varlen tree; do
    config=$("$PGS_PY" "$HERE/rl_packing_check.py" smoke-config --work "$WORK" --mode "$mode" --model "$SMOKE_MODEL" \
      --steps "$SMOKE_STEPS")
    run_part "smoke_run_$mode" 3600 "$PGS_PY" -m palingenesis.rl.trainer --config "$config"
    "${CHECK[@]}" smoke-report --log "$WORK/smoke_run_$mode.log" --mode "$mode" --steps "$SMOKE_STEPS" | tee -a "$RESULTS"
  done
fi

if ! skipped concurrency; then
  if [ -f "$HERE/rl_concurrency.py" ]; then
    run_part concurrency 2400 "$PGS_PY" "$HERE/rl_concurrency.py" --work "$WORK/concurrency"
  else
    echo '{"check": "concurrency", "status": "INFO", "skipped": "scripts/gpu_checks/rl_concurrency.py not found"}' |
      tee -a "$RESULTS"
  fi
fi

failed=$(grep -c '"status": "FAIL"' "$RESULTS")
status=PASS
[ "$failed" -eq 0 ] || status=FAIL
echo "{\"check\": \"summary\", \"status\": \"$status\", \"failed\": $failed, \"work\": \"$WORK\", \"seconds\": $SECONDS}" |
  tee -a "$RESULTS"
[ "$failed" -eq 0 ]
