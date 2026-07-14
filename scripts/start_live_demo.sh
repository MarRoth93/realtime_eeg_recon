#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the Python interpreter, in priority order:
#   1. MARYAM_DEMO_PYTHON override
#   2. the uv-managed project venv (uv sync --all-extras -> .venv/)
#   3. a BCI env under the active conda installation
#   4. the legacy fixed BCI env
#   5. the active conda environment
#   6. whatever `python` is on PATH
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi

PYTHON_BIN=""
for candidate in \
  "${MARYAM_DEMO_PYTHON:-}" \
  "$PROJECT_ROOT/.venv/bin/python" \
  "${CONDA_BASE:+$CONDA_BASE/envs/BCI/bin/python}" \
  "/home/psycontrol/miniforge3/envs/BCI/bin/python" \
  "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" \
  "$(command -v python 2>/dev/null || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No usable Python interpreter found. Set MARYAM_DEMO_PYTHON." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_live_demo.py" "${1:-$PROJECT_ROOT/config/lab_demo.json}"
