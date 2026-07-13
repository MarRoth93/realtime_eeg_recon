#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="/home/psycontrol/miniforge3/envs/BCI/bin/python"
PYTHON_BIN="${MARYAM_DEMO_PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_live_demo.py" "${1:-$PROJECT_ROOT/config/lab_demo.json}"
