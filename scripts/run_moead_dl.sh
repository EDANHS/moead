#!/usr/bin/env bash
# Simple Linux-only launcher: activate venv (one-line) and run the Python script

set -euo pipefail

# First arg: venv dir (default 'venv'), rest forwarded to Python script
VENV_DIR="${1:-venv}"
shift || true

# Create venv if it doesn't exist (keeps script simple)
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi

# Activate venv in a single line (POSIX-compatible for Linux)
. "$VENV_DIR/bin/activate"

# --- Configuración fácil (edita estos valores) -----------------
# Pon aquí los valores que quieras usar por defecto.
USE_GPU=0            # 0 = CPU, 1 = GPU
N_GENERATIONS=5
H_DIVISIONS=3
N_NEIGHBORS=2
N_R=1
LOG_FILE="moead_dl_log.json"
CHECKPOINT_FILE="moead_dl_checkpoint.pkl"
# ---------------------------------------------------------------

CMD=(python Scripts/run_moead_dl.py)
if [ "$USE_GPU" -eq 1 ]; then
  CMD+=(--use-gpu)
fi
CMD+=(--n_generations "$N_GENERATIONS")
CMD+=(--h_divisions "$H_DIVISIONS")
CMD+=(--n_neighbors "$N_NEIGHBORS")
CMD+=(--n_r "$N_R")
CMD+=(--log "$LOG_FILE")
CMD+=(--checkpoint "$CHECKPOINT_FILE")

echo "Ejecutando: ${CMD[*]}"
"${CMD[@]}"
