#!/usr/bin/env bash
# GPU checks of the SGLang rollout backend (rollout.backend sglang / sglang_server) on one NVIDIA GPU.
#
#   PGS_SRC=/path/to/palingenesis/src SGLANG_VENV=/scratch/sglang-venv WORKDIR=/scratch/sglang-checks \
#   VLLM_PYTHON=/path/to/pgs/.venv/bin/python  bash scripts/gpu_checks/sglang.sh
#
# Required:  PGS_SRC      the palingenesis src directory to test (put on PYTHONPATH; nothing is installed from it)
#            SGLANG_VENV  a virtual environment this script creates (or reuses) for SGLang; no other is modified.
#                         About 9 GB (it is memory on /dev/shm), 10 GB free asked for; uv's cache goes next to it
#                         (<SGLANG_VENV>.uv-cache: hard links of the same files; UV_CACHE=<dir> to move it)
#            WORKDIR      every temporary file, cache, log and result goes here
# Optional:  VLLM_PYTHON  the python of an environment with vLLM (palingenesis[vllm]) for the parity and
#                         throughput comparisons (default: <repo>/.venv/bin/python when it imports vllm; else skipped)
#            SGLANG_VERSION (0.5.17: the last release built on torch 2.11), CUDA_INDEX (cu129: the PyTorch index
#            name of the CUDA build every package is installed for; the driver must support it), MIN_FREE_GB (10),
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
# log to stderr; the full logs are in $WORKDIR/logs. Nothing is written outside WORKDIR, SGLANG_VENV and the uv
# cache next to it (caches, bytecode, temporary files and the kernels SGLang builds at first use included: every
# cache variable points into WORKDIR/cache and the steps run with HOME=WORKDIR/home; the "no_writes_in_home" line
# checks ~/.cache, ~/.tilelang, ~/.triton, ~/.deep_gemm and ~/.nv for new files); /dev/shm is not used beyond Python's own semaphores. Exit code 1 when a
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
export CUDA_CACHE_PATH="$WORKDIR/cache/nv" PIP_CACHE_DIR="$WORKDIR/cache/pip"
# uv's cache: next to the environment by default, so its files are hard links of the environment's (no second copy,
# and an install after the environment was removed downloads nothing). UV_CACHE=<dir on a disk> keeps it out of RAM
# when SGLANG_VENV is on /dev/shm (a copy then, still no download).
export UV_CACHE_DIR="${UV_CACHE:-$SGLANG_VENV.uv-cache}" UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export UV_PYTHON_INSTALL_DIR="$SGLANG_VENV.python"  # only used when the machine has no Python 3.12
export HF_DATASETS_CACHE="$WORKDIR/cache/datasets" PYTHONPYCACHEPREFIX="$WORKDIR/cache/pycache"
export MPLCONFIGDIR="$WORKDIR/cache/mpl" NUMBA_CACHE_DIR="$WORKDIR/cache/numba" TORCH_HOME="$WORKDIR/cache/torch"
export WANDB_MODE=disabled WANDB_DIR="$WORKDIR/tmp"
# SGLang's just-in-time kernel builds. Their defaults are in the home directory, whatever XDG_CACHE_HOME says:
#   sgl_kernel's JIT (tvm-ffi)   TVM_FFI_CACHE_DIR, default ~/.cache/tvm-ffi/sgl_kernel_jit_*
#   tilelang                     TILELANG_CACHE_DIR, default ~/.tilelang/cache
#   flashinfer                   FLASHINFER_WORKSPACE_BASE (above), default ~/.cache/flashinfer
#   triton                       TRITON_CACHE_DIR (above) and TRITON_HOME, default ~/.triton
#   DeepGEMM                     DG_JIT_CACHE_DIR, default ~/.deep_gemm
#   flash-attn cute DSL          FLASH_ATTENTION_CUTE_DSL_CACHE_DIR, default $TMPDIR/<user>/...
export TVM_FFI_CACHE_DIR="$WORKDIR/cache/tvm-ffi" TILELANG_CACHE_DIR="$WORKDIR/cache/tilelang"
export TRITON_HOME="$WORKDIR/cache/triton-home" DG_JIT_CACHE_DIR="$WORKDIR/cache/deep_gemm"
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="$WORKDIR/cache/flash-attn-cute"
# Whatever else writes under ~ (a library that knows only HOME): after the install everything runs with HOME in
# WORKDIR. The Hugging Face cache keeps its real place (HF_HOME above, resolved from the real home).
REAL_HOME="$HOME"
STEP_HOME="$WORKDIR/home"
mkdir -p "$STEP_HOME"
HOME_WATCH=("$REAL_HOME/.cache" "$REAL_HOME/.tilelang" "$REAL_HOME/.triton" "$REAL_HOME/.deep_gemm" "$REAL_HOME/.nv")
START_MARK="$WORKDIR/tmp/started"
: > "$START_MARK"
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
INSTALL_LOG="$WORKDIR/logs/install.log"
SGLANG_INDEX="https://docs.sglang.ai/whl/$CUDA_INDEX/"
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_INDEX"
kib() { du -sk "$@" 2>/dev/null | tail -n 1 | cut -f1; }  # with several paths: their total, hard links counted once
gib() { LC_ALL=C awk -v k="${1:-0}" 'BEGIN {printf "%.1f", k / 1048576}'; }

# SGLang's PyPI wheel declares the CUDA 13 builds of torch and of its kernel packages. Installing it and swapping
# packages afterwards downloads both sets, and every later resolution wants the CUDA 13 libraries back. So the
# environment is resolved once, with SGLang's own requirements replaced for $CUDA_INDEX (sglang_checks.py
# install-plan writes the replacements: torch / torchaudio / torchvision from PyTorch's $CUDA_INDEX index, the
# [cu12] extras, cuda-python below 13, sglang-kernel and sgl-deep-gemm from SGLang's $CUDA_INDEX index) together
# with palingenesis' own dependencies; the result is checked for CUDA 13 packages before anything large is
# downloaded, and `uv pip sync` makes the environment exactly that list (leftovers of an earlier attempt go).
install() {
  local plan="$WORKDIR/tmp/install" uv="" have free existed=0
  [ ! -x "$SG" ] || existed=1
  local indexes=(--extra-index-url "$TORCH_INDEX" --extra-index-url "$SGLANG_INDEX" --index-strategy unsafe-best-match)
  mkdir -p "$plan" "$(dirname "$SGLANG_VENV")" "$UV_CACHE_DIR" || return 1
  if command -v uv > /dev/null 2>&1; then
    uv="$(command -v uv)"
  else  # no uv on this machine: one for this environment, outside its site-packages (pip sync would remove it)
    uv="$SGLANG_VENV/.uv/bin/uv"
    if [ ! -x "$uv" ]; then
      [ -x "$SG" ] || python3 -m venv "$SGLANG_VENV" || { echo "INSTALL: neither uv nor python3 -m venv works"; return 1; }
      "$SG" -m pip install --no-cache-dir --quiet --target "$SGLANG_VENV/.uv" uv \
        || { echo "INSTALL: could not install uv with pip (install uv, or check the network)"; return 1; }
    fi
  fi
  echo "INSTALL: $("$uv" --version), cache $UV_CACHE_DIR, link mode $UV_LINK_MODE"
  if [ "$existed" = 1 ]; then
    # an environment without this script's mark: an install that did not finish, or another version. Its packages
    # (and the pip cache an earlier version of this script kept in WORKDIR) go first: their space is needed.
    echo "INSTALL: $SGLANG_VENV holds $(gib "$(kib "$SGLANG_VENV")") GB of an unfinished or other install: emptying it"
    [ ! -d "$PIP_CACHE_DIR" ] || "$SG" -m pip cache purge > /dev/null 2>&1 || true
    : > "$plan/empty.txt"
    "$uv" pip sync "$plan/empty.txt" --python "$SG" --allow-empty-requirements --quiet \
      || echo "INSTALL: could not empty it (continuing)"
  fi
  have="$(kib "$SGLANG_VENV" "$UV_CACHE_DIR")"
  free="$(df -Pk "$(dirname "$SGLANG_VENV")" | awk 'NR == 2 {print $4}')"
  echo "INSTALL: $(gib "$free") GB free for $SGLANG_VENV, $(gib "${have:-0}") GB already there (with the uv cache)"
  if [ "$(( ${free:-0} + ${have:-0} ))" -lt "$(( ${MIN_FREE_GB:-10} * 1048576 ))" ]; then
    echo "INSTALL: not enough space: the environment takes about 9 GB (MIN_FREE_GB=${MIN_FREE_GB:-10} changes the limit)"
    return 1
  fi
  [ -x "$SG" ] || "$uv" venv --python 3.12 "$SGLANG_VENV" || { echo "INSTALL: uv could not create $SGLANG_VENV"; return 1; }
  "$SG" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' \
    || { echo "INSTALL: $SG is not Python 3.12 (sglang $SGLANG_VERSION ships cp312 wheels): remove $SGLANG_VENV"; return 1; }
  "$SG" "$CHECKS" install-plan --sglang-version "$SGLANG_VERSION" --cuda "$CUDA_INDEX" --workdir "$plan" \
    || { echo "INSTALL: could not read sglang==$SGLANG_VERSION's requirements from PyPI"; return 1; }
  echo "INSTALL: requirements:"; sed 's/^/    /' "$plan/requirements.in"
  echo "INSTALL: replacing SGLang's own for $CUDA_INDEX:"; sed 's/^/    /' "$plan/overrides.txt"
  "$uv" pip compile "$plan/requirements.in" --override "$plan/overrides.txt" --python "$SG" "${indexes[@]}" \
    --no-header --no-annotate --quiet -o "$plan/lock.txt" \
    || { echo "INSTALL: no set of packages satisfies sglang==$SGLANG_VERSION on $CUDA_INDEX (see above)"; return 1; }
  "$SG" "$CHECKS" install-verify --lock-only --lock "$plan/lock.txt" --sglang-version "$SGLANG_VERSION" \
    --cuda "$CUDA_INDEX" || { echo "INSTALL: the resolution is not a $CUDA_INDEX one; nothing was installed"; return 1; }
  echo "INSTALL: $(grep -c '==' "$plan/lock.txt") packages; torch and SGLang's builds:"
  grep -E '^(torch|torchaudio|torchvision|sglang|sglang-kernel|sgl-deep-gemm|flashinfer-python|transformers)==' \
    "$plan/lock.txt" | sed 's/^/    /'
  "$uv" pip sync "$plan/lock.txt" --python "$SG" "${indexes[@]}" \
    || { echo "INSTALL: installing the packages failed (space? network?)"; return 1; }
  if [ "${FLASHINFER_JIT_CACHE:-0}" = "1" ]; then  # prebuilt kernels (SGLang's Dockerfile: INSTALL_FLASHINFER_JIT_CACHE)
    "$uv" pip install --python "$SG" --no-deps --index-url "https://flashinfer.ai/whl/$CUDA_INDEX" \
      "flashinfer-jit-cache==$(sed -n 's/^flashinfer-python==//p' "$plan/lock.txt")" \
      || echo "INSTALL: flashinfer-jit-cache was not installed: flashinfer builds its kernels at first use"
  fi
  # the environment must import and be the CUDA build asked for, on this machine's driver
  "$SG" "$CHECKS" install-verify --lock "$plan/lock.txt" --sglang-version "$SGLANG_VERSION" --cuda "$CUDA_INDEX" \
    || { echo "INSTALL: the installed environment failed its check (install-verify lines above)"; return 1; }
  cp "$plan/lock.txt" "$SGLANG_VENV/.pgs-sglang.lock"
  echo "$SGLANG_VERSION-$CUDA_INDEX" > "$SGLANG_VENV/.pgs-sglang"
}

if [ "${SKIP_INSTALL:-0}" != "1" ] && [ "$(cat "$SGLANG_VENV/.pgs-sglang" 2>/dev/null)" != "$SGLANG_VERSION-$CUDA_INDEX" ]; then
  echo "[sglang-checks] $(date +%H:%M:%S) installing sglang==$SGLANG_VERSION ($CUDA_INDEX) into $SGLANG_VENV" >&2
  rm -f "$SGLANG_VENV/.pgs-sglang"
  install > "$INSTALL_LOG" 2>&1
  code=$?
  grep -a '^INSTALL: ' "$INSTALL_LOG" | sed 's/^/    | /' >&2
  if [ "$code" -ne 0 ]; then
    echo "[sglang-checks] the install failed; the end of $INSTALL_LOG:" >&2
    grep -a -v '^INSTALL: ' "$INSTALL_LOG" | tail -n "${LOG_TAIL:-40}" | cut -c1-400 | sed 's/^/    | /' >&2
    reason="$(grep -a '^INSTALL: ' "$INSTALL_LOG" | tail -n 1 | sed 's/^INSTALL: //; s/["\\]/ /g')"
    fail_line "0_install" "sglang==$SGLANG_VERSION ($CUDA_INDEX): ${reason:-see $INSTALL_LOG}"
    exit 1
  fi
fi
export HOME="$STEP_HOME"  # from here on (the install above used the real one; its directories are all explicit)
# The environment's size: with SGLANG_VENV on a RAM file system (/dev/shm) this is memory. The uv cache holds the
# same files (hard links when it is on the venv's file system: counted once in the total).
venv_kib="$(kib "$SGLANG_VENV")"
total_kib="$(kib "$SGLANG_VENV" "$UV_CACHE_DIR")"
status=PASS
[ "$(cat "$SGLANG_VENV/.pgs-sglang" 2>/dev/null)" = "$SGLANG_VERSION-$CUDA_INDEX" ] || [ "${SKIP_INSTALL:-0}" = "1" ] || status=FAIL
emit "$(printf '{"check": "0_install", "status": "%s", "sglang": "%s", "cuda_index": "%s", "venv_gib": %s, "venv_and_uv_cache_gib": %s, "venv": "%s", "uv_cache": "%s"}' \
  "$status" "$SGLANG_VERSION" "$CUDA_INDEX" "$(gib "${venv_kib:-0}")" "$(gib "${total_kib:-0}")" "$SGLANG_VENV" "$UV_CACHE_DIR")"
echo "[sglang-checks] $SGLANG_VENV: $(gib "${venv_kib:-0}") GB; with the uv cache $UV_CACHE_DIR: $(gib "${total_kib:-0}") GB" \
  "(the same files when hard-linked; to free the space remove both)" >&2
if [ "${total_kib:-0}" -gt "$(( ${MAX_ENV_GB:-9} * 1048576 ))" ]; then
  echo "[sglang-checks] that is more than ${MAX_ENV_GB:-9} GB: if the two are on different file systems the cache is" \
    "a second copy (UV_CACHE=<a directory on a disk> keeps it out of /dev/shm)" >&2
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
# Nothing of this run under the real home: files that appeared since it started in the caches SGLang and its
# kernels would use there. The Hugging Face cache is left out (HF_HOME, read only here; other jobs of this user
# may download into it), and so is the uv cache when it was put there. Files under ~/.cache that belong to none of
# SGLang's libraries (~/.nv, ~/.cache/vllm, ...) are counted apart: another job of the same user may have written
# them, as it may ~/.triton (the list says which files).
written="$WORKDIR/logs/written_in_home.txt"
KERNEL_CACHES='/\.cache/(tvm-ffi|tilelang|flashinfer|deep_gemm|sglang|flash_attention_cute_dsl_cache)(/|$)|/\.(tilelang|triton|deep_gemm)/'
for dir in "${HOME_WATCH[@]}"; do
  [ -d "$dir" ] || continue
  find "$dir" -xdev \( -path "$HF_HOME" -o -path "$REAL_HOME/.cache/huggingface" -o -path "$UV_CACHE_DIR" \) -prune \
    -o -type f -newer "$START_MARK" -print 2>/dev/null
done > "$written"
ours="$(grep -c -E "$KERNEL_CACHES" "$written")"
others="$(( $(wc -l < "$written") - ours ))"
status=PASS
[ "$ours" -eq 0 ] || status=FAIL
emit "$(printf '{"check": "no_writes_in_home", "status": "%s", "new_kernel_cache_files": %d, "other_new_files_under_cache": %d, "list": "%s"}' \
  "$status" "$ours" "$others" "$written")"
if [ "$ours" -ne 0 ]; then
  echo "[sglang-checks] written under $REAL_HOME by this run (first 20 of $ours):" >&2
  grep -E "$KERNEL_CACHES" "$written" \
    | head -n 20 | sed 's/^/    | /' >&2
fi
failed="$(grep -c '"status": "FAIL"' "$RESULTS")"
echo "[sglang-checks] $(grep -c '"status": "PASS"' "$RESULTS") PASS, $failed FAIL (results: $RESULTS, logs: $WORKDIR/logs)" >&2
[ "$failed" -eq 0 ]
