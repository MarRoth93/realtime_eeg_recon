#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the Python interpreter, in priority order:
#   1. MARYAM_DEMO_PYTHON override
#   2. the uv-managed project venv (uv sync --all-extras -> .venv/)
#   3. the legacy conda BCI env
#   4. whatever `python` is on PATH
PYTHON_BIN="${MARYAM_DEMO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="/home/psycontrol/miniforge3/envs/BCI/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_live_demo.py" "${1:-$PROJECT_ROOT/config/lab_demo.json}"
