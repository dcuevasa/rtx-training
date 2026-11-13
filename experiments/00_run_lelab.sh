#!/usr/bin/env bash
# =====================================================================
# LeLab Web UI (Inside rtx-lerobot Docker Container)
# Serves the LeLab React frontend + FastAPI backend on port 8000.
#
# Usage:
#   1. Foreground interactive:
#      PORT=8000 ./rtx-training/docker/run.sh bash /workspace/rtx-training/experiments/00_run_lelab.sh
#
#   2. Detached background daemon:
#      DETACH=1 NAME=lelab PORT=8000 ./rtx-training/docker/run.sh bash /workspace/rtx-training/experiments/00_run_lelab.sh
# =====================================================================
set -euo pipefail

LELAB_PORT="${LELAB_PORT:-8000}"
LELAB_HOST="${LELAB_HOST:-0.0.0.0}"

# Ensure websockets and editable LeLab are present in the persistent /cache/home/.local
if ! python3 -c "import websockets" >/dev/null 2>&1; then
  echo "Installing websockets into persistent user cache..."
  pip install --user "websockets>=15.0.1"
fi

if ! python3 -c "import lelab" >/dev/null 2>&1; then
  echo "Installing LeLab (editable, no-deps) into persistent user cache..."
  pip install --no-deps --user -e /workspace/leLab
fi

if [[ ! -f "/workspace/leLab/frontend/dist/index.html" ]]; then
  echo "ERROR: /workspace/leLab/frontend/dist/index.html not found."
  echo "Build frontend/dist via node:22-alpine on the host first."
  exit 1
fi

echo "=========================================================="
echo " Launching LeLab Web Interface (Inside Docker)"
echo " Listen Address: http://${LELAB_HOST}:${LELAB_PORT}"
echo " Local browser:  http://localhost:${LELAB_PORT}"
echo " Remote SSH:     ssh -L ${LELAB_PORT}:localhost:${LELAB_PORT} <user>@<host-ip>"
echo "=========================================================="

exec python3 -m lelab.scripts.lelab \
  --host "$LELAB_HOST" \
  --port "$LELAB_PORT" \
  --no-open \
  "$@"
