# Step 0 helpers (sourced by run_all.sh). Real pinned Heretic 171bb98b, tiny-random/qwen3.8, CPU, Linux/WSL2.
# Needs a workspace provisioned by tests/realtools/provision_cpu.py with the model and datasets cached.
set -u
WS=${REALTOOLS_WS:?set REALTOOLS_WS to a provisioned workspace}
ENV=$(ls -d "$WS"/tools/envs/heretic-* | head -1)
HERETIC=$ENV/bin/heretic
PY=$ENV/bin/python
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=$HERE/base.toml
SNAP=$(ls -d "$WS"/cache/hf/hub/models--tiny-random--qwen3.8/snapshots/* | head -1)
S0=${STEP0_DIR:-$HOME/step0}
TOOL=$HERE/study.py

# mkcfg DIR N_TRIALS MODEL_ACTION CHECKPOINT_ACTION SAVE_DIR TRIAL_INDEX
mkcfg() {
  mkdir -p "$1/checkpoints"
  ln -sfn "$SNAP" "$1/model"
  sed -e "s|^n_trials = .*|n_trials = $2|" \
      -e "s|^n_startup_trials = .*|n_startup_trials = 2|" \
      -e "s|^model_action = .*|model_action = \"$3\"|" \
      -e "s|^checkpoint_action = .*|checkpoint_action = \"$4\"|" \
      -e "s|^save_directory = .*|save_directory = \"$5\"|" \
      -e "s|^trial_index = .*|trial_index = $6|" \
      -e "s|^seed = .*|seed = 1234|" \
      "$BASE" > "$1/config.toml"
}

# run_heretic DIR LOG [SIGNAL TRIAL]: optionally send SIGNAL to the process group once trial TRIAL starts.
run_heretic() {
  local dir=$1 log=$2 sig=${3:-} at=${4:-}
  ( cd "$dir" && exec env HF_HOME="$WS/cache/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
      COLUMNS=400 NO_COLOR=1 TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 \
      setsid "$HERETIC" </dev/null >"$log" 2>&1 ) &
  local pid=$!
  if [ -n "$sig" ]; then
    until grep -q "Running trial $at of" "$log" 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 0.5; done
    sleep 1
    kill -"$sig" -- -"$(ps -o pgid= -p "$(pgrep -f "$HERETIC" | head -1)" | tr -d ' ')"
  fi
  wait $pid; echo "exit=$?"
}

inspect() { "$PY" "$TOOL" inspect "$1/checkpoints/model.jsonl"; }
patch() { local d=$1; shift; "$PY" "$TOOL" patch "$d/checkpoints/model.jsonl" "$@"; }
whash() { (cd "$1" 2>/dev/null && sha256sum *.safetensors | sha256sum | cut -c1-16) || echo "no-weights"; }
engine_lines() { grep -E "Resuming|Running trial|Restoring model|Model saved|rror" "$1"; }
same_trials() {
  python3 -c "import json,sys; b,a=(json.load(open(p)) for p in sys.argv[1:]); print('trials', b['n_trials_recorded'], '->', a['n_trials_recorded'], '| identical trial set:', b['trials']==a['trials'], '| finished', b['finished'], '->', a['finished'], '| states', a['states'])" "$1" "$2"
}
