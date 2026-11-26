#!/usr/bin/env bash
# GPU check of asynchronous in-training evaluation (eval_tasks.placement async): training never slows down or
# stops because of its evaluator. pgs train SFT runs of one model, the same seed, with and without an evaluator.
# Written for one A100 80GB machine; the evaluator's place follows what the machine offers:
#   two MIG instances   trainer on the first, evaluator on the second (eval_tasks.devices: [1])
#   two or more GPUs    trainer on the first, evaluator on the second (CUDA_VISIBLE_DEVICES split)
#   one GPU, no MIG     the evaluator runs the hf engine on the CPU (eval_tasks.service: auto): every cost on the
#                       trainer's side (snapshot, staging, sender thread, results) is the same; only the
#                       evaluator's own speed differs. Sharing the GPU is refused by design.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src DATA=sft.jsonl EVAL_DATA=gsm8k_test.jsonl \
#   bash scripts/gpu_checks/async_eval.sh [WORK]
#
# DATA: chat rows (messages) for SFT; EVAL_DATA: rows with a question column (EVAL_PROMPT_FIELD, default
# question) and a numeric `answer`. Every result is a JSON line {"check", "status": PASS | FAIL | INFO, ...} on
# stdout and in $WORK/results.jsonl; exit code 1 if any check failed. Everything is written inside WORK (default:
# a new directory under /tmp); only processes this script started are stopped; at most ~20 GB of disk (the
# checkpoint arm keeps one checkpoint, and its size is checked).
#
# Pre-registered criteria: see async_eval_check.py (step time within 2 % of no evaluation, identical losses,
# refusal before any model load, an evaluator out of memory never stops training, disk <= 20 GB).
#
# Options (environment):
#   CONFIG=configs/qwen35_08b.yaml  MODEL=Qwen/Qwen3.5-0.8B  STEPS=300  EVERY=50  EVAL_LIMIT=50  EVAL_MAX_TOKENS=256
#   ARMS="none none2 push checkpoint oom refusal real_oom inline"   (real_oom: MIG / two GPUs only)
#   CUDA_VISIBLE_DEVICES   the GPUs (or MIG instances) the check may use
#
# Not covered here (needs more than one node): pushes over a real network next to FSDP's NCCL traffic
# (eval_tasks.push_gbps), service.json discovery across nodes, EVAL_NODES in train_slurm.sh, a lost evaluator node,
# and devices: next to torchrun ranks and remote rollout engines on an 8-GPU node.
set -uo pipefail

: "${PGS_PY:?set PGS_PY to the python of an environment with the palingenesis dependencies, torch, transformers and vllm}"
: "${PGS_SRC:?set PGS_SRC to palingenesis/src}"
: "${DATA:?set DATA to an SFT jsonl (messages)}"
: "${EVAL_DATA:?set EVAL_DATA to a jsonl of questions with a numeric answer}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export WORK="${1:-$(mktemp -d /tmp/pgs_async_eval.XXXXXX)}"
mkdir -p "$WORK"
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CONFIG="${CONFIG:-$REPO/configs/qwen35_08b.yaml}" MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
export STEPS="${STEPS:-300}" EVERY="${EVERY:-50}" DATA EVAL_DATA
# TRITON_CACHE_AUTOTUNING: every arm reuses the first run's autotune choices (a near-tie on a new shape would
# otherwise pick another kernel in another run and end bit-identical losses at that step)
export TMPDIR="$WORK/tmp" TRITON_CACHE_DIR="$WORK/triton" TORCHINDUCTOR_CACHE_DIR="$WORK/inductor" TRITON_CACHE_AUTOTUNING=1
export VLLM_CACHE_ROOT="$WORK/vllm_cache"
mkdir -p "$TMPDIR"
RESULTS="$WORK/results.jsonl"
: >"$RESULTS"
STARTED=()
CHECK="$HERE/async_eval_check.py"

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

eval "$("$PGS_PY" "$CHECK" topology)"   # MODE TRAIN_DEV EVAL_DEV
export MODE
echo "{\"check\": \"topology\", \"status\": \"INFO\", \"mode\": \"$MODE\", \"train\": \"$TRAIN_DEV\", \"evaluator\": \"$EVAL_DEV\"}" | tee -a "$RESULTS"
ARMS="${ARMS:-none none2 push checkpoint oom refusal real_oom inline}"

background() {  # background NAME ENV... -- CMD...: in its own process group, logged to $WORK/NAME.log
  local name="$1"
  shift
  setsid env "$@" >"$WORK/$name.log" 2>&1 &
  STARTED+=("$!")
  echo "$!"
}

train_arm() {  # train_arm ARM [ENV=VALUE ...]: one pgs train run; exit code in $WORK/ARM.exit
  local arm="$1"
  shift
  local visible="$TRAIN_DEV"
  if [ "$MODE" != cpu ] && [[ "$arm" == push || "$arm" == oom || "$arm" == real_oom ]]; then
    visible="$TRAIN_DEV,$EVAL_DEV"  # rank 0 takes the first; eval_tasks.devices: [1] is the second
  fi
  "$PGS_PY" "$CHECK" config "$arm" "$WORK/$arm.yaml" || return 1
  local started
  started=$(date +%s)
  env CUDA_VISIBLE_DEVICES="$visible" "$@" "$PGS_PY" -m palingenesis.train --config "$WORK/$arm.yaml" >"$WORK/$arm.log" 2>&1
  echo $? >"$WORK/$arm.exit"
  echo "$(( $(date +%s) - started ))" >"$WORK/$arm.seconds"
}

evaluator() {  # evaluator ARM MODE [ENV...]: pgs eval serve / watch for the run of ARM, on EVAL_DEV (cpu: none)
  local arm="$1" kind="$2"
  shift 2
  mkdir -p "$WORK/$arm"
  if [ "$kind" = serve ]; then
    background "$arm.evaluator" CUDA_VISIBLE_DEVICES="$EVAL_DEV" "$@" "$PGS_PY" -m palingenesis.eval.service serve --dir "$WORK/$arm" --host 127.0.0.1
  else
    background "$arm.evaluator" CUDA_VISIBLE_DEVICES="$EVAL_DEV" "$@" "$PGS_PY" -m palingenesis.eval.service watch "$WORK/$arm" --poll 5
  fi
}

for arm in $ARMS; do
  case "$arm" in
    none | none2 | inline)
      train_arm "$arm"
      ;;
    push | oom)
      extra=()
      [ "$arm" = oom ] && extra=(PGS_TEST_EVAL_OOM=always)
      if [ "$MODE" = cpu ]; then
        evaluator "$arm" serve "${extra[@]}" >/dev/null
        train_arm "$arm"
      else
        train_arm "$arm" "${extra[@]}"   # the trainer starts the evaluator (devices: [1]); it inherits the variable
      fi
      ;;
    checkpoint)
      evaluator "$arm" watch >/dev/null
      ( max=0; while [ ! -f "$WORK/$arm.exit" ]; do
          kb=$(du -sk "$WORK/$arm" 2>/dev/null | cut -f1); [ "${kb:-0}" -gt "$max" ] && max=$kb
          echo "$max" >"$WORK/$arm.disk_max_kb"; sleep 5; done ) &
      STARTED+=("$!")
      train_arm "$arm"
      sleep 6
      ;;
    refusal)
      started=$(date +%s)
      "$PGS_PY" "$CHECK" config refusal "$WORK/refusal.yaml"
      env CUDA_VISIBLE_DEVICES="$TRAIN_DEV" "$PGS_PY" -m palingenesis.train --config "$WORK/refusal.yaml" >"$WORK/refusal.log" 2>&1
      echo $? >"$WORK/refusal.exit"
      echo "$(( $(date +%s) - started ))" >"$WORK/refusal.seconds"
      ;;
    real_oom)
      if [ "$MODE" = cpu ]; then
        echo '{"check": "oom/real_oom", "status": "SKIP", "reason": "needs a second GPU or MIG instance"}' | tee -a "$RESULTS"
        continue
      fi
      # most of the evaluator's GPU held by another process: its engine cannot start
      background real_oom.hog CUDA_VISIBLE_DEVICES="$EVAL_DEV" "$PGS_PY" "$CHECK" hog "${HOG_GB:-30}" >/dev/null
      sleep 20
      train_arm real_oom
      ;;
    *) echo "{\"check\": \"$arm\", \"status\": \"FAIL\", \"error\": \"unknown arm\"}" | tee -a "$RESULTS" ;;
  esac
done

"$PGS_PY" "$CHECK" verdict "$WORK" | tee -a "$RESULTS"
code=${PIPESTATUS[0]}
echo "results: $RESULTS (logs in $WORK)"
exit "$code"
