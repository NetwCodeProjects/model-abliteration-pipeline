#!/usr/bin/env bash
# Project A entry point. All logic lives in the Python CLI (src/model_pipeline/cli.py).
#
# Python selection (no surprise installs):
#   1. the project control venv (.venv) if it exists;
#   2. --help/--version/--list-stages: any system python3 >= 3.10 (stdlib only);
#   3. --dry-run and inspection flags: a system python3 that already has PyYAML (python3-yaml);
#   4. anything else bootstraps .venv first (uv sync --frozen, or venv + pip with hash-locked
#      requirements) after confirmation (--yes or an interactive 'y'). Nothing is curl'ed into a shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ARGS=("$@")

has_arg() {
  local wanted="$1" a
  for a in "${ARGS[@]+"${ARGS[@]}"}"; do
    [ "$a" = "$wanted" ] && return 0
  done
  return 1
}

venv_python() {
  if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"; return 0; fi
  if [ -x "$VENV/Scripts/python.exe" ]; then echo "$VENV/Scripts/python.exe"; return 0; fi
  return 1
}

find_system_python() {
  # $1: optional extra import that must succeed (e.g. "yaml")
  local c check="import sys${1:+, $1}; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
  for c in python3 python3.13 python3.12 python3.11 python3.10 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "$check" >/dev/null 2>&1; then
      command -v "$c"
      return 0
    fi
  done
  return 1
}

bootstrap() {
  local py="$1"
  if ! has_arg --yes && ! has_arg -y; then
    if [ -t 0 ]; then
      printf 'The control environment (%s) is missing. Create it now (project-local, hash-locked)? [y/N] ' "$VENV" >&2
      read -r reply
      case "$reply" in y|Y|yes|YES) ;; *) echo "ERROR [E_NOT_CONFIRMED] bootstrap declined" >&2; exit 15 ;; esac
    else
      echo "ERROR [E_CONFIRMATION_REQUIRED] control environment missing; re-run with --yes to create $VENV" >&2
      exit 15
    fi
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "Bootstrapping $VENV with uv (uv.lock, frozen)" >&2
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --no-dev --no-install-project --project "$ROOT" >&2
  elif "$py" -c 'import ensurepip, venv' >/dev/null 2>&1; then
    echo "Bootstrapping $VENV with venv + pip (requirements/control.txt, --require-hashes)" >&2
    "$py" -m venv "$VENV"
    "$(venv_python)" -m pip install --quiet --require-hashes --no-deps -r "$ROOT/requirements/control.txt" >&2
  else
    echo "ERROR [E_BOOTSTRAP_IMPOSSIBLE] neither 'uv' nor Python venv/ensurepip is available." >&2
    echo "  hint: sudo apt-get install -y python3-venv   (or install uv from https://docs.astral.sh/uv/)" >&2
    exit 5
  fi
}

if PY="$(venv_python)"; then
  exec "$PY" -m model_pipeline "$@"
fi

SYS_PY="$(find_system_python)" || { echo "ERROR [E_PYTHON_MISSING] Python >= 3.10 not found (sudo apt-get install -y python3)" >&2; exit 5; }

if has_arg --help || has_arg -h || has_arg --version || has_arg --list-stages; then
  exec "$SYS_PY" -m model_pipeline "$@"
fi

if has_arg --bootstrap; then
  bootstrap "$SYS_PY"
  echo "Control environment ready: $(venv_python)" >&2
  exit 0
fi

INSPECT=0
for flag in --dry-run --print-config --list-runs --status --cleanup; do
  if has_arg "$flag"; then INSPECT=1; fi
done
if [ "$INSPECT" = 1 ] && ! has_arg --confirm-cleanup; then
  if YAML_PY="$(find_system_python yaml)"; then
    exec "$YAML_PY" -m model_pipeline "$@"
  fi
  echo "ERROR [E_CONTROL_ENV_MISSING] this command needs PyYAML and the control environment does not exist yet." >&2
  echo "  hint: sudo apt-get install -y python3-yaml, or create .venv with: ./main.sh --bootstrap --yes" >&2
  exit 5
fi

bootstrap "$SYS_PY"
exec "$(venv_python)" -m model_pipeline "$@"
