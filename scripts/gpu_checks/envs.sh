#!/usr/bin/env bash
# GPU check of the environment catalogue: `pgs env run` for every built-in environment with one model.
# One NVIDIA GPU (written for an A100 80GB), the model in the Hugging Face cache, no network.
#
#   PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/envs.sh [WORK]
#
# Every result is a JSON line {"check": ..., "status": "PASS" | "FAIL" | "INFO" | "SKIP", ...} on stdout, also collected
# in $WORK/results.jsonl; the exit code is 1 if any check failed. Everything is written inside WORK (default: a new
# directory under /tmp); only processes this script started are stopped.
#
#   audit/<env>   the CPU audit of the copy under test (pgs env test): oracle paid, degenerate policies not
#   run/<env>     pgs env run <env> --model MODEL --k K: pass@1, pass@K, mean reward, episodes/s, how episodes ended.
#                 PASS means the run completed and scored its episodes (no rollout errors, at most 5% unscored);
#                 the pass rates are the measurement, not a criterion. Per-row results: $WORK/<env>.jsonl
#                 (readable by pgs compare).
#   table         the pass@1 / pass@K / episodes/s of every environment, as one line
#
# Options (environment):
#   MODEL=Qwen/Qwen3.5-0.8B   K=4   ROWS=64   TEMPERATURE=0.7   THINKING=off (off | on | default)
#   ENVS="math-boxed wordle ..."     the environments to run (default: every built-in with generated rows)
#   SANDBOX=docker | process         where the code environments run model-written programs. Default: docker when
#                                    the daemon answers, else those environments are skipped. "process" runs the
#                                    programs on this machine with resource limits and NO isolation: your decision.
#   HARBOR_TASKS=/path/to/tasks      also run the harbor entry on these task directories (needs Docker)
#   GPU_MEMORY=0.3                   vLLM's gpu_memory_utilization (a 0.8B model needs little; raise for larger ones)
#   BACKEND=vllm                     the rollout engine (vllm | sglang | hf)
#   MAX_MODEL_LEN=16384  TIMEOUT=2400 (seconds per environment)  CUDA_VISIBLE_DEVICES (default 0)
#   SKIP="audit run"                 parts to leave out
# Takes about 30-40 minutes on an A100 with the defaults (most of it engine starts: one per environment).
set -uo pipefail

: "${PGS_PY:?set PGS_PY to the python of an environment with the palingenesis dependencies, torch, transformers and vllm}"
: "${PGS_SRC:?set PGS_SRC to palingenesis/src}"
WORK="${1:-$(mktemp -d /tmp/pgs_envs.XXXXXX)}"
mkdir -p "$WORK"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
K="${K:-4}"
ROWS="${ROWS:-64}"
TEMPERATURE="${TEMPERATURE:-0.7}"
THINKING="${THINKING:-off}"
BACKEND="${BACKEND:-vllm}"
GPU_MEMORY="${GPU_MEMORY:-0.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TIMEOUT="${TIMEOUT:-2400}"
SKIP="${SKIP:-}"
export PYTHONPATH="$PGS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# The Hugging Face cache stays where the models are; compiled kernels and vLLM's cache go into WORK.
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
emit() { echo "$1" | tee -a "$RESULTS"; }
PGS=("$PGS_PY" -m palingenesis.cli)

# run_limited NAME TIMEOUT_S CMD...: CMD in its own process group, stdout to $WORK/NAME.out, stderr to $WORK/NAME.log;
# returns CMD's exit code, 124 after a timeout
run_limited() {
  local name="$1" limit="$2"
  shift 2
  setsid "$@" >"$WORK/$name.out" 2>"$WORK/$name.log" &
  local pid=$!
  STARTED+=("$pid")
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$limit" ]; then
      kill -TERM -- "-$pid" 2>/dev/null
      sleep 5
      kill -KILL -- "-$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 2
    waited=$((waited + 2))
  done
  wait "$pid"
}

json_tail() { tail -c 600 "$1" | "$PGS_PY" -c 'import json, sys; print(json.dumps(sys.stdin.read()))'; }

emit "{\"check\": \"setup\", \"status\": \"INFO\", \"model\": \"$MODEL\", \"k\": $K, \"rows\": $ROWS, \"temperature\": $TEMPERATURE, \"thinking\": \"$THINKING\", \"backend\": \"$BACKEND\", \"work\": \"$WORK\"}"

if [ ! -e "$MODEL" ] && ! "$PGS_PY" -c 'import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_files_only=True, allow_patterns=["config.json"])' "$MODEL" >/dev/null 2>&1; then
  emit "{\"check\": \"model_cache\", \"status\": \"FAIL\", \"model\": \"$MODEL\", \"hub_cache\": \"${HF_HUB_CACHE:-$HF_HOME/hub}\", \"error\": \"not in the Hugging Face cache (HF_HOME / HF_HUB_CACHE)\"}"
fi

ENVS="${ENVS:-$("$PGS_PY" -c '
from palingenesis.rl.catalog import registry
print(" ".join(s.name for s in registry.specs() if s.datasets[s.default_dataset].kind == "generator"))')}"
[ -n "${HARBOR_TASKS:-}" ] && ENVS="$ENVS harbor"

SANDBOX="${SANDBOX:-}"
if [ -z "$SANDBOX" ] && docker info >/dev/null 2>&1; then SANDBOX=docker; fi

# ---- the CPU audit of this copy: a verifier with a hole would make the pass rates below meaningless
if ! skipped audit; then
  for env in $ENVS; do
    data=()
    [ "$env" = harbor ] && data=(--dataset "$HARBOR_TASKS")
    CUDA_VISIBLE_DEVICES="" run_limited "audit_$env" 900 "${PGS[@]}" env test "$env" --n 32 --json ${data[@]+"${data[@]}"}
    code=$?
    "$PGS_PY" - "$env" "$code" "$WORK/audit_$env.out" <<'PY' | tee -a "$RESULTS"
import json, sys
env, code, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
line = {"check": f"audit/{env}", "status": "PASS" if code == 0 else "FAIL"}
try:
    report = json.load(open(path))
    oracle = report.get("oracle") or {}
    line.update(rows=report.get("rows"), oracle_passed=oracle.get("passed"), steps_per_s=round(oracle.get("steps_per_s", 0.0), 1),
                degenerate_passed={k: v["passed"] for k, v in (report.get("degenerate") or {}).items() if v["passed"]},
                failures=report.get("failures", [])[:3])
except Exception as error:
    line.update(status="FAIL", error=f"no audit report: {error}", exit_code=code)
print(json.dumps(line))
PY
  done
fi

# ---- pgs env run for each environment
if ! skipped run; then
  for env in $ENVS; do
    extra=()
    kind=$("$PGS_PY" -c 'import sys
from palingenesis.rl.catalog import get
print(get(sys.argv[1]).sandbox)' "$env")
    if [ "$kind" = process ]; then
      if [ -z "$SANDBOX" ]; then
        emit "{\"check\": \"run/$env\", \"status\": \"SKIP\", \"reason\": \"runs model-written code: no Docker daemon here; SANDBOX=process runs it with resource limits and no isolation\"}"
        continue
      fi
      extra+=(--sandbox "$SANDBOX")
    fi
    if [ "$env" = harbor ]; then
      if ! docker info >/dev/null 2>&1; then
        emit "{\"check\": \"run/harbor\", \"status\": \"SKIP\", \"reason\": \"needs a Docker daemon\"}"
        continue
      fi
      extra+=(--dataset "$HARBOR_TASKS" --max_concurrent 16)
    fi
    start=$SECONDS
    run_limited "run_$env" "$TIMEOUT" "${PGS[@]}" env run "$env" --model "$MODEL" --k "$K" --n "$ROWS" \
      --temperature "$TEMPERATURE" --thinking "$THINKING" --backend "$BACKEND" --gpu_memory_utilization "$GPU_MEMORY" \
      --max_model_len "$MAX_MODEL_LEN" --output "$WORK/$env.jsonl" --json ${extra[@]+"${extra[@]}"}
    code=$?
    tail_=$(json_tail "$WORK/run_$env.log")
    "$PGS_PY" - "$env" "$code" "$WORK/run_$env.out" "$K" "$((SECONDS - start))" "$tail_" <<'PY' | tee -a "$RESULTS"
import json, sys
env, code, path, k, wall, tail = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
line = {"check": f"run/{env}", "status": "FAIL", "exit_code": code, "wall_s": wall}
summary = None
for text in open(path, errors="replace"):
    if text.startswith('{"env"'):
        summary = json.loads(text)
if code == 124:
    line["error"] = "timeout"
elif summary is None:
    line.update(error="no summary line", log_tail=json.loads(tail))
else:
    episodes, errors = summary["episodes"], summary["finishes"].get("error", 0.0)
    ok = code == 0 and episodes == summary["rows"] * k and errors == 0 and summary["unscored"] <= 0.05 * episodes
    line.update(
        status="PASS" if ok else "FAIL", rows=summary["rows"], episodes=episodes, unscored=summary["unscored"],
        **{"pass@1": summary["pass@1"], f"pass@{k}": summary[f"pass@{k}"]}, mean_reward=summary["mean_reward"],
        episodes_per_s=round(summary["episodes_per_s"], 2), finishes=summary["finishes"],
        tool_error_rate=round(summary["tool_error_rate"], 3), completion_tokens=round(summary["completion_tokens"], 1),
    )
print(json.dumps(line))
PY
  done
  "$PGS_PY" - "$RESULTS" "$K" <<'PY' | tee -a "$RESULTS"
import json, sys
k = sys.argv[2]
rows = [json.loads(line) for line in open(sys.argv[1]) if line.startswith('{"check": "run/')]
table = {r["check"][4:]: {"pass@1": r["pass@1"], f"pass@{k}": r[f"pass@{k}"], "episodes_per_s": r["episodes_per_s"]}
         for r in rows if "pass@1" in r}
print(json.dumps({"check": "table", "status": "INFO", "environments": table}))
PY
fi

failed=$(grep -c '"status": "FAIL"' "$RESULTS")
skipped_count=$(grep -c '"status": "SKIP"' "$RESULTS")
status=PASS
[ "$failed" -eq 0 ] || status=FAIL
emit "{\"check\": \"summary\", \"status\": \"$status\", \"failed\": $failed, \"skipped\": $skipped_count, \"work\": \"$WORK\", \"seconds\": $SECONDS}"
[ "$failed" -eq 0 ]
