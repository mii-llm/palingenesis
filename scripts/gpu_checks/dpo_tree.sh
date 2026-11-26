#!/usr/bin/env bash
# GPU checks of dpo.packing (padded / varlen / tree): equality with the padded [2P, S] rows on real models, micro-step
# speed and memory at prompt shares of 50 / 80 / 95%, and a short `pgs train` run per layout.
# One NVIDIA GPU (written for an A100 80GB; torch with flash-linear-attention installed), models in the Hugging Face
# cache, no network.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/dpo_tree.sh [WORK]
#
# Every result is a JSON line {"check": ..., "status": "PASS" | "FAIL" | "INFO" | "SKIP", ...} on stdout, also collected
# in $WORK/results.jsonl; the exit code is 1 if any check failed. Everything is written inside WORK (default: a new
# directory under /tmp); only processes this script started are stopped. The GPU's free memory is read before each
# part loads anything; a part that runs out of memory retries with smaller micro-batches, then reports SKIP.
#
# Pre-registered (the `speedup` lines): tree >= 1.5x the fastest padded layout at prompt shares >= 70% is PASS;
# below 1.2x is FAIL (the feature is dropped); in between, INFO with target_met false. Equality was pre-registered
# as a gradient cosine >= 0.999; in bf16 two padded layouts of the same pairs differ by more than that, so the
# `equality` lines are judged against that measured floor, and a float32 pass against its own (float32) floor, with absolute gradient guards.
#
# Options (environment):
#   MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B"     equality and throughput models
#   SHARES="0.5 0.8 0.95"                          prompt shares of the throughput pairs
#   PADDED_TOKENS=32768                            padded tokens of a throughput micro-batch ([2P, S]; sets P)
#   CHECKPOINTING=selective                        train.gradient_checkpointing of the measured policy
#   SMOKE_MODEL=Qwen/Qwen3.5-0.8B  SMOKE_STEPS=20  the `pgs train` smoke runs (padded and tree)
#   FP32_MODELS="Qwen/Qwen3.5-0.8B"  FP32_PAIRS=1  the float32 equality pass (bounds relative to the float32 floor)
#   SKIP="equality equality_fp32 throughput compiled smoke"   parts to leave out
#   CUDA_VISIBLE_DEVICES                           the GPU (default 0)
set -uo pipefail

: "${PGS_PY:?set PGS_PY to the python of an environment with the palingenesis dependencies, torch and transformers}"
: "${PGS_SRC:?set PGS_SRC to palingenesis/src}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-$(mktemp -d /tmp/pgs_dpo_tree.XXXXXX)}"
mkdir -p "$WORK"
MODELS="${MODELS:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B}"
SHARES="${SHARES:-0.5 0.8 0.95}"
PADDED_TOKENS="${PADDED_TOKENS:-32768}"
CHECKPOINTING="${CHECKPOINTING:-selective}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3.5-0.8B}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
SKIP="${SKIP:-}"
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# The Hugging Face cache stays where the models are: huggingface_hub's default is $XDG_CACHE_HOME/huggingface, so
# it is pinned before XDG_CACHE_HOME moves into WORK (compiled kernels go there; the models do not).
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export TMPDIR="$WORK/tmp" TRITON_CACHE_DIR="$WORK/triton" TORCHINDUCTOR_CACHE_DIR="$WORK/inductor"
export XDG_CACHE_HOME="$WORK/xdg"
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

CHECK=("$PGS_PY" "$HERE/dpo_tree_check.py")
DEVICE_ARGS=(--device "${CHECK_DEVICE:-cuda:0}")  # CHECK_DEVICE=cpu: a dry run of the plumbing on a tiny model

run_part env 600 "${CHECK[@]}" env "${DEVICE_ARGS[@]}"

# the models must be in the Hugging Face cache (no network): say which are missing once, before every part fails
if [ "${CHECK_DEVICE:-cuda:0}" != "cpu" ]; then
  for model in $MODELS $SMOKE_MODEL; do
    if [ ! -e "$model" ] && ! "$PGS_PY" -c 'import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_files_only=True, allow_patterns=["config.json"])' "$model" >/dev/null 2>&1; then
      hub="${HF_HUB_CACHE:-$HF_HOME/hub}"
      echo "{\"check\": \"model_cache\", \"status\": \"FAIL\", \"model\": \"$model\", \"hub_cache\": \"$hub\", \"error\": \"not in the Hugging Face cache (HF_HOME / HF_HUB_CACHE)\"}" |
        tee -a "$RESULTS"
    fi
  done
fi

for model in $MODELS; do
  skipped equality && break
  tag="${model//\//_}"
  run_part "equality_$tag" 2400 "${CHECK[@]}" equality --model "$model" --pairs "${EQUALITY_PAIRS:-6}" \
    --checkpointing "$CHECKPOINTING" "${DEVICE_ARGS[@]}"
done
# float32, no autocast: bounds relative to the float32 floor, absolute gradient guards (no bf16 noise to hide a real error); packed attention has no flash
# kernel there and runs on an explicit [N, N] mask, so few pairs, on the small model
for model in ${FP32_MODELS-Qwen/Qwen3.5-0.8B}; do
  skipped equality && break
  skipped equality_fp32 && break
  [ "${CHECK_DEVICE:-cuda:0}" = "cpu" ] && break
  tag="${model//\//_}"
  run_part "equality_fp32_$tag" 2400 "${CHECK[@]}" equality --model "$model" --pairs "${FP32_PAIRS:-1}" \
    --precision float32 --checkpointing "$CHECKPOINTING" "${DEVICE_ARGS[@]}"
done

for model in $MODELS; do
  skipped throughput && break
  tag="${model//\//_}"
  for share in $SHARES; do
    run_part "throughput_${tag}_$share" 2400 "${CHECK[@]}" throughput --model "$model" --share "$share" \
      --padded_tokens "$PADDED_TOKENS" --repeats 3 --checkpointing "$CHECKPOINTING" "${DEVICE_ARGS[@]}"
    # the compiled padded baseline (model.compile, which dpo.packing refuses), in a process of its own
    if ! skipped compiled && [ "${CHECK_DEVICE:-cuda:0}" != "cpu" ]; then
      run_part "compiled_${tag}_$share" 3600 "${CHECK[@]}" throughput --model "$model" --share "$share" --compiled \
        --padded_tokens "$PADDED_TOKENS" --repeats 3 --checkpointing "$CHECKPOINTING" "${DEVICE_ARGS[@]}"
    fi
  done
done
skipped throughput || "${CHECK[@]}" verdict --results "$RESULTS" | tee -a "$RESULTS"

if ! skipped smoke && [ "${CHECK_DEVICE:-cuda:0}" != "cpu" ]; then
  for mode in padded tree; do
    config=$("${CHECK[@]}" smoke-config --work "$WORK" --mode "$mode" --model "$SMOKE_MODEL" --steps "$SMOKE_STEPS")
    run_part "smoke_run_$mode" 3600 "$PGS_PY" -m palingenesis.train --config "$config"
    "${CHECK[@]}" smoke-report --log "$WORK/smoke_run_$mode.log" --mode "$mode" --steps "$SMOKE_STEPS" | tee -a "$RESULTS"
  done
fi

# (a part that exits non-zero after reporting its own FAIL lines adds a wrapper line: failures are counted by check)
failed=$(grep -c '"status": "FAIL"' "$RESULTS")
skipped_count=$(grep -c '"status": "SKIP"' "$RESULTS")
status=PASS
[ "$failed" -eq 0 ] || status=FAIL
echo "{\"check\": \"summary\", \"status\": \"$status\", \"failed\": $failed, \"skipped\": $skipped_count, \"work\": \"$WORK\", \"seconds\": $SECONDS}" |
  tee -a "$RESULTS"
[ "$failed" -eq 0 ]
