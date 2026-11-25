#!/usr/bin/env bash
# GPU checks of the SGLang rollout backend (rollout.backend sglang / sglang_server) on one NVIDIA GPU.
#
#   PGS_SRC=/path/to/palingenesis/src SGLANG_VENV=/scratch/sglang-venv WORKDIR=/scratch/sglang-checks \
#   VLLM_PYTHON=/path/to/pgs/.venv/bin/python  bash scripts/gpu_checks/sglang.sh
#
# Required:  PGS_SRC      the palingenesis src directory to test (put on PYTHONPATH; nothing is installed from it)
#            SGLANG_VENV  a virtual environment this script creates (or reuses) for SGLang; no other is modified
#            WORKDIR      every temporary file, cache, log and result goes here
# Optional:  VLLM_PYTHON  the python of an environment with vLLM (palingenesis[vllm]) for the parity and
#                         throughput comparisons (default: <repo>/.venv/bin/python when it imports vllm; else skipped)
#            SGLANG_VERSION (0.5.17: the last release built on torch 2.11), CUDA_INDEX (cu129),
#            MODEL_DENSE (Qwen/Qwen3-0.6B), MODEL_HYBRID (Qwen/Qwen3.5-0.8B), GPU (0), SKIP_INSTALL=1,
#            FLASHINFER_JIT_CACHE=1 (also install flashinfer's prebuilt kernels, several GB: for a machine without
#            a CUDA toolkit, where its just-in-time build fails), LOG_TAIL (40: lines of a failed step's log printed)
#
# Prints one JSON line per check ({"check", "status": PASS|FAIL, numbers...}), also in $WORKDIR/results.jsonl:
#   1 generation and log-prob parity with vLLM (each model)   2 weight updates (colocated; server tensor and disk)
#   3 sleep / wake memory release                              4 10-step pgs rl and pgs distill smokes
#   5 decode throughput vs vLLM at an equal batch (median of 3)
# Models come from the Hugging Face cache (HF_HUB_OFFLINE=1, read only). Every step runs in a process group of its
# own, which is killed when the step ends; at the end no process started by this run is left, on the GPU or off it
# (every process carries a marker in its environment; checked, "cleanup" line). A failed step prints the end of its
# log to stderr; the full logs are in $WORKDIR/logs. Nothing is written outside WORKDIR and SGLANG_VENV (caches,
# bytecode and temporary files included); /dev/shm is not used beyond Python's own semaphores. Exit code 1 when a
# check failed.
# Takes about 45-60 minutes on an A100 80GB, plus ~10 minutes for the first install.

set -uo pipefail

: "${PGS_SRC:?set PGS_SRC to the palingenesis src directory}"
: "${SGLANG_VENV:?set SGLANG_VENV to the virtual environment to create for SGLang}"
: "${WORKDIR:?set WORKDIR to a directory for temporary files and results}"
PGS_SRC="$(cd "$PGS_SRC" && pwd)"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.17}"
CUDA_INDEX="${CUDA_INDEX:-cu129}"
MODEL_DENSE="${MODEL_DENSE:-Qwen/Qwen3-0.6B}"
MODEL_HYBRID="${MODEL_HYBRID:-Qwen/Qwen3.5-0.8B}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKS="$HERE/sglang_checks.py"

mkdir -p "$WORKDIR"/{tmp,cache,logs,out}
WORKDIR="$(cd "$WORKDIR" && pwd)"
if [ "${#WORKDIR}" -gt 80 ]; then  # SGLang's processes talk over unix sockets in TMPDIR: a path limit of 107 bytes
  echo "WORKDIR is too long a path (${#WORKDIR} > 80 characters) for the unix sockets in $WORKDIR/tmp" >&2
  exit 2
fi
RESULTS="$WORKDIR/results.jsonl"
: > "$RESULTS"

# Everything temporary inside WORKDIR; the Hugging Face cache stays where the models are.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$WORKDIR/tmp" XDG_CACHE_HOME="$WORKDIR/cache"
export TRITON_CACHE_DIR="$WORKDIR/cache/triton" TORCHINDUCTOR_CACHE_DIR="$WORKDIR/cache/inductor"
export FLASHINFER_WORKSPACE_BASE="$WORKDIR/cache" VLLM_CACHE_ROOT="$WORKDIR/cache/vllm"
export CUDA_CACHE_PATH="$WORKDIR/cache/nv" UV_CACHE_DIR="$WORKDIR/cache/uv" PIP_CACHE_DIR="$WORKDIR/cache/pip"
export HF_DATASETS_CACHE="$WORKDIR/cache/datasets" PYTHONPYCACHEPREFIX="$WORKDIR/cache/pycache"
export MPLCONFIGDIR="$WORKDIR/cache/mpl" NUMBA_CACHE_DIR="$WORKDIR/cache/numba" TORCH_HOME="$WORKDIR/cache/torch"
export WANDB_MODE=disabled WANDB_DIR="$WORKDIR/tmp"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="$PGS_SRC" PYTHONUNBUFFERED=1
export PGS_SGLANG_CHECKS_RUN="$WORKDIR"  # marks every process of this run (the final leftover check reads it)

emit() { echo "$1" | tee -a "$RESULTS"; }
report() {  # <check name for failures> <report arguments...>
  local name="$1" line
  shift
  line="$("$SG" "$CHECKS" report "$@" 2>> "$WORKDIR/logs/report.log")"
  if [ -n "$line" ]; then emit "$line"; else fail_line "$name" "no result (see $WORKDIR/logs/report.log)"; fi
}
fail_line() {  # check, reason
  emit "$(printf '{"check": "%s", "status": "FAIL", "reason": "%s"}' "$1" "$2")"
}

GROUPS_FILE="$WORKDIR/tmp/process_groups"
: > "$GROUPS_FILE"
cleanup() {
  while read -r group; do
    [ -n "$group" ] || continue
    kill -TERM -- "-$group" 2>/dev/null
  done < "$GROUPS_FILE"
  sleep 3
  while read -r group; do
    [ -n "$group" ] || continue
    kill -KILL -- "-$group" 2>/dev/null
  done < "$GROUPS_FILE"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

# run_step <name> <timeout seconds> <command...>: in a session of its own (its process group is killed at the
# end, whatever the command leaves running), output in logs/<name>.log. Returns the command's exit code.
run_step() {
  local name="$1" limit="$2"
  shift 2
  echo "[sglang-checks] $(date +%H:%M:%S) $name" >&2
  : > "$GROUPS_FILE.$name"  # its time: server logs written after it belong to this step
  setsid bash -c 'echo $$ >> "$0"; exec timeout --kill-after=30 "$1" "${@:2}"' "$GROUPS_FILE" "$limit" "$@" \
    > "$WORKDIR/logs/$name.log" 2>&1 &
  local pid=$!
  wait "$pid"
  local code=$?
  kill -KILL -- "-$pid" 2>/dev/null  # the step's leftovers (engine schedulers, servers)
  if [ "$code" -ne 0 ]; then
    echo "[sglang-checks] $name exited with $code (124 / 137: its ${limit}s limit); the end of $WORKDIR/logs/$name.log:" >&2
    tail -n "${LOG_TAIL:-40}" "$WORKDIR/logs/$name.log" | sed 's/^/    | /' >&2
    local extra
    for extra in "$WORKDIR"/tmp/server_*.log "$WORKDIR"/tmp/smoke_*/*_run/sglang*.log; do  # a launched server's log
      [ -f "$extra" ] && [ "$extra" -nt "$GROUPS_FILE.$name" ] || continue
      echo "[sglang-checks] the end of $extra:" >&2
      tail -n "${LOG_TAIL:-40}" "$extra" | sed 's/^/    | /' >&2
    done
  fi
  rm -f "$GROUPS_FILE.$name"
  return "$code"
}

# ------------------------------------------------------------------ environment

SG="$SGLANG_VENV/bin/python"
if [ "${SKIP_INSTALL:-0}" != "1" ] && [ "$(cat "$SGLANG_VENV/.pgs-sglang" 2>/dev/null)" != "$SGLANG_VERSION-$CUDA_INDEX" ]; then
  install() {
    set -e
    if command -v uv >/dev/null; then
      [ -x "$SG" ] || uv venv --python 3.12 "$SGLANG_VENV"
      pipi() { uv pip install --python "$SG" "$@"; }
      pipr() {  # reinstall: <index url> <no-deps 0|1> <packages...>
        local index="$1" nodeps="$2" flags=()
        shift 2
        for p in "$@"; do flags+=(--reinstall-package "${p%%[=<>[]*}"); done
        [ "$nodeps" = 1 ] && flags+=(--no-deps)
        uv pip install --python "$SG" "${flags[@]}" --index-url "$index" "$@"
      }
      pipu() { uv pip uninstall --python "$SG" "$@"; }
    else
      [ -x "$SG" ] || python3 -m venv "$SGLANG_VENV"
      "$SG" -m pip install --upgrade pip
      pipi() { "$SG" -m pip install "$@"; }
      pipr() {
        local index="$1" nodeps="$2" flags=(--force-reinstall)
        shift 2
        [ "$nodeps" = 1 ] && flags+=(--no-deps)
        "$SG" -m pip install "${flags[@]}" --index-url "$index" "$@"
      }
      pipu() { "$SG" -m pip uninstall -y "$@"; }
    fi
    version() { "$SG" -c "from importlib.metadata import version; print(version('$1'))" 2>/dev/null; }
    freeze() { "$SG" -c "from importlib.metadata import distributions as d; print('\n'.join(sorted(f'{x.metadata[\"Name\"]}=={x.version}' for x in d())))"; }
    pipi "sglang==$SGLANG_VERSION"
    # SGLang's PyPI wheels target CUDA 13. For a CUDA 12.9 driver its install guide swaps in the cu129 builds of
    # torch, sglang-kernel and sgl-deep-gemm, and its Dockerfile the CUDA 12 flavours of flashinfer and cutlass.
    if [ "$CUDA_INDEX" != "cu130" ]; then
      local kernel gemm flashinfer cu13
      local cutlass
      cutlass="$(version nvidia-cutlass-dsl || true)"
      kernel="$(version sglang-kernel)"
      gemm="$(version sgl-deep-gemm || true)"
      flashinfer="$(version flashinfer-python)"
      cu13="$(freeze | awk -F'==' '$1 ~ /-cu13$/ {print $1}')"
      [ -z "$cu13" ] || pipu $cu13
      pipr "https://download.pytorch.org/whl/$CUDA_INDEX" 0 torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0
      pipr "https://docs.sglang.ai/whl/$CUDA_INDEX/" 1 "sglang-kernel==$kernel"
      [ -z "$gemm" ] || pipr "https://docs.sglang.ai/whl/$CUDA_INDEX/" 1 "sgl-deep-gemm==$gemm"
      freeze | grep -iE '^(torch|torchvision|torchaudio|transformers)==' > "$WORKDIR/tmp/constraints.txt" || true
      pipi -c "$WORKDIR/tmp/constraints.txt" "flashinfer-python[cu12]==$flashinfer" "cuda-python>=12,<13" \
        "nvidia-cutlass-dsl${cutlass:+==$cutlass}"
    fi
    if [ "${FLASHINFER_JIT_CACHE:-0}" = "1" ]; then  # prebuilt kernels (SGLang's Dockerfile: INSTALL_FLASHINFER_JIT_CACHE)
      pipr "https://flashinfer.ai/whl/$CUDA_INDEX" 1 "flashinfer-jit-cache==$(version flashinfer-python)" \
        || echo "flashinfer-jit-cache was not installed: flashinfer builds its kernels at first use"
    fi
    # palingenesis' own dependencies, without letting anything move torch or transformers
    freeze | grep -iE '^(torch|torchvision|torchaudio|transformers)==' > "$WORKDIR/tmp/constraints.txt" || true
    pipi -c "$WORKDIR/tmp/constraints.txt" "datasets>=4.0" "accelerate>=1.10" pyyaml safetensors math-verify \
      flash-linear-attention psutil
    echo "$SGLANG_VERSION-$CUDA_INDEX" > "$SGLANG_VENV/.pgs-sglang"
  }
  if ! (install) > "$WORKDIR/logs/install.log" 2>&1; then
    fail_line "0_install" "installing sglang==$SGLANG_VERSION failed: see $WORKDIR/logs/install.log"
    exit 1
  fi
fi
{
  echo "[sglang-checks] WORKDIR $WORKDIR, SGLANG_VENV $SGLANG_VENV, PGS_SRC $PGS_SRC, GPU $CUDA_VISIBLE_DEVICES"
  df -h "$WORKDIR" /dev/shm 2>/dev/null | sed 's/^/    | /'
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/    | /'
} >&2
environment="$("$SG" - 2> "$WORKDIR/logs/environment.log" <<'EOF'
import json, torch
from importlib.metadata import PackageNotFoundError, version

def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

ok = torch.cuda.is_available()
import sglang  # noqa: F401 (a broken install fails here, not in the first check)
from sglang.srt.server_args import ServerArgs  # noqa: F401
print(json.dumps({"check": "0_environment", "status": "PASS" if ok else "FAIL", "sglang": version("sglang"),
                  "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": version("transformers"),
                  "sglang_kernel": installed("sglang-kernel"), "flashinfer": installed("flashinfer-python"),
                  "torch_memory_saver": installed("torch_memory_saver"),
                  "gpu": torch.cuda.get_device_name(0) if ok else None,
                  "free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 1) if ok else None}))
EOF
)"
if [ -z "$environment" ]; then
  fail_line "0_environment" "the SGLang environment does not import: see $WORKDIR/logs/environment.log"
  tail -n "${LOG_TAIL:-40}" "$WORKDIR/logs/environment.log" | sed 's/^/    | /' >&2
  exit 1
fi
emit "$environment"
case "$environment" in *'"status": "FAIL"'*) echo "[sglang-checks] torch sees no GPU" >&2; exit 1 ;; esac

if [ -z "${VLLM_PYTHON:-}" ] && [ -x "$PGS_SRC/../.venv/bin/python" ] \
  && "$PGS_SRC/../.venv/bin/python" -c "import vllm" >/dev/null 2>&1; then
  VLLM_PYTHON="$PGS_SRC/../.venv/bin/python"
fi
VLLM_PYTHON="${VLLM_PYTHON:-}"

OUT="$WORKDIR/out"

# ------------------------------------------------------------------ 1 parity

for model in "$MODEL_DENSE" "$MODEL_HYBRID"; do
  tag="$(basename "$model")"
  run_step "gen_sglang_$tag" 1200 "$SG" "$CHECKS" generate --engine sglang --model "$model" --out "$OUT/gen_sglang_$tag.json"
  inputs=("$OUT/gen_sglang_$tag.json")
  if [ -n "$VLLM_PYTHON" ]; then
    run_step "gen_vllm_$tag" 1200 "$VLLM_PYTHON" "$CHECKS" generate --engine vllm --model "$model" --out "$OUT/gen_vllm_$tag.json" \
      && inputs+=("$OUT/gen_vllm_$tag.json")
  fi
  if [ -f "$OUT/gen_sglang_$tag.json" ] \
    && run_step "score_$tag" 1200 "$SG" "$CHECKS" score --model "$model" --inputs "${inputs[@]}" --out "$OUT/score_$tag.json"; then
    report 1_generation_logprob_parity --check parity --inputs "$OUT/score_$tag.json"
  else
    fail_line "1_generation_logprob_parity" "$model: see $WORKDIR/logs/gen_sglang_$tag.log / score_$tag.log"
  fi
done

# ------------------------------------------------------------------ 2 weight updates

for spec in "colocated:$MODEL_DENSE" "colocated:$MODEL_HYBRID" "tensor:$MODEL_DENSE" "disk:$MODEL_DENSE"; do
  mode="${spec%%:*}" model="${spec#*:}" tag="$(basename "${spec#*:}")"
  if run_step "update_${mode}_$tag" 1500 "$SG" "$CHECKS" update --mode "$mode" --model "$model" --workdir "$WORKDIR/tmp" \
    --out "$OUT/update_${mode}_$tag.json"; then
    report "2_weight_update_$mode" --check update --inputs "$OUT/update_${mode}_$tag.json"
  else
    fail_line "2_weight_update_$mode" "$model: see $WORKDIR/logs/update_${mode}_$tag.log"
  fi
done

# ------------------------------------------------------------------ 3 sleep / wake

for model in "$MODEL_DENSE" "$MODEL_HYBRID"; do
  tag="$(basename "$model")"
  if run_step "sleep_$tag" 1200 "$SG" "$CHECKS" sleep --model "$model" --memory 0.5 --out "$OUT/sleep_$tag.json"; then
    report 3_sleep_wake --check sleep --inputs "$OUT/sleep_$tag.json"
  else
    fail_line "3_sleep_wake" "$model: see $WORKDIR/logs/sleep_$tag.log"
  fi
done

# ------------------------------------------------------------------ 4 smokes

"$SG" "$CHECKS" smoke-config --kind rl --model "$MODEL_HYBRID" --workdir "$WORKDIR/tmp/smoke_rl" --out "$OUT/rl.yaml"
run_step smoke_rl 2400 "$SG" -m palingenesis.rl.trainer --config "$OUT/rl.yaml"
code=$?
report 4_smoke_rl --check smoke_rl --exit-code "$code" --inputs "$WORKDIR/tmp/smoke_rl/rl_run/metrics.jsonl"

"$SG" "$CHECKS" smoke-config --kind distill --model "$MODEL_DENSE" --teacher "$MODEL_DENSE" \
  --workdir "$WORKDIR/tmp/smoke_distill" --out "$OUT/distill.yaml"
run_step smoke_distill 2400 "$SG" -m palingenesis.opd.trainer --config "$OUT/distill.yaml"
code=$?
report 4_smoke_distill --check smoke_distill --exit-code "$code" \
  --inputs "$WORKDIR/tmp/smoke_distill/distill_run/metrics.jsonl"

# ------------------------------------------------------------------ 5 throughput

tag="$(basename "$MODEL_DENSE")"
run_step "throughput_sglang_$tag" 1200 "$SG" "$CHECKS" throughput --engine sglang --model "$MODEL_DENSE" --out "$OUT/tp_sglang.json"
inputs=("$OUT/tp_sglang.json")
if [ -n "$VLLM_PYTHON" ]; then
  run_step "throughput_vllm_$tag" 1200 "$VLLM_PYTHON" "$CHECKS" throughput --engine vllm --model "$MODEL_DENSE" \
    --out "$OUT/tp_vllm.json" && inputs+=("$OUT/tp_vllm.json")
fi
if [ -f "$OUT/tp_sglang.json" ]; then
  report 5_throughput --check throughput --inputs "${inputs[@]}"
else
  fail_line "5_throughput" "see $WORKDIR/logs/throughput_sglang_$tag.log"
fi

# ------------------------------------------------------------------ cleanup

cleanup
# Processes of this run still alive: every one it started has the marker in its environment (this shell and its
# subshells do not: it was exported after they started; the tr and grep of the scan itself are skipped by name).
# A launched server ends itself within seconds of its parent's death (SIGTERM, then SGLang's own shutdown), so
# wait before counting.
marked() {
  local pid name
  for pid in /proc/[0-9]*; do
    pid="${pid#/proc/}"
    if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "PGS_SGLANG_CHECKS_RUN=$WORKDIR"; then
      name="$(cat "/proc/$pid/comm" 2>/dev/null)" || continue
      case "$name" in tr | grep | cat | "") continue ;; esac
      echo "$pid"
    fi
  done
}
for _ in $(seq 30); do
  [ -n "$(marked)" ] || break
  sleep 1
done
left=0
for pid in $(marked); do
  echo "[sglang-checks] left over: $pid $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-200)" >&2
  kill -KILL "$pid" 2>/dev/null
  left=$((left + 1))
done
sleep 2
on_gpu=0
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "PGS_SGLANG_CHECKS_RUN=$WORKDIR"; then
    on_gpu=$((on_gpu + 1))
  fi
done
status=PASS
[ "$left" -eq 0 ] && [ "$on_gpu" -eq 0 ] || status=FAIL
emit "$(printf '{"check": "cleanup", "status": "%s", "leftover_processes_killed": %d, "still_on_gpu": %d, "results": "%s"}' \
  "$status" "$left" "$on_gpu" "$RESULTS")"
failed="$(grep -c '"status": "FAIL"' "$RESULTS")"
echo "[sglang-checks] $(grep -c '"status": "PASS"' "$RESULTS") PASS, $failed FAIL (results: $RESULTS, logs: $WORKDIR/logs)" >&2
[ "$failed" -eq 0 ]
