#!/usr/bin/env bash
# Launch any command inside the rtx-lerobot container (nothing is installed on the host).
#
# Usage:
#   ./run.sh                       # interactive shell
#   ./run.sh lerobot-train ...     # one-off command
#   DETACH=1 NAME=act_overfit ./run.sh bash /workspace/rtx-training/experiments/03_act_overfit_sanity_check.sh
#
# Mounts:
#   <Tesis root>             -> /workspace           (code, scripts, outputs)
#   ~/.cache/rtx-docker      -> /cache               (HF hub/datasets, torch, ultralytics weights)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TESIS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-rtx-lerobot:0.6.1}"
NAME="${NAME:-rtx-lerobot-$(date +%H%M%S)}"
TESIS_DIR="${TESIS_DIR:-$DEFAULT_TESIS_DIR}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/rtx-docker}"
mkdir -p "$CACHE_DIR"/{huggingface,torch,ultralytics,home}

# Credentials live in ~/.bashrc (export HF_TOKEN=..., export HF_USER=..., export WANDB_API_KEY=...).
# Non-interactive shells do not source .bashrc, so pick them up explicitly if present.
if [[ -f "$HOME/.bashrc" ]]; then
  eval "$(grep -E '^export (HF_TOKEN|HF_USER|WANDB_API_KEY)=' "$HOME/.bashrc" || true)"
fi

TTY_FLAGS=()
[[ -t 0 && -t 1 ]] && TTY_FLAGS=(-it)
DETACH_FLAGS=()
[[ "${DETACH:-0}" == "1" ]] && DETACH_FLAGS=(-d) && TTY_FLAGS=()

PORT_FLAGS=()
if [[ -n "${PORT:-}" ]]; then
  for p in ${PORT//,/ }; do
    if [[ "$p" =~ : ]]; then
      PORT_FLAGS+=(-p "$p")
    else
      PORT_FLAGS+=(-p "$p:$p")
    fi
  done
fi

exec docker run --rm "${TTY_FLAGS[@]}" "${DETACH_FLAGS[@]}" "${PORT_FLAGS[@]}" \
  --name "$NAME" \
  --device nvidia.com/gpu=all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/cache/home \
  -e PATH="/cache/home/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e HF_TOKEN="${HF_TOKEN:-}" -e HF_USER="${HF_USER:-}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e PYTHONPATH="/workspace/leLab:/workspace/lerobot-vision-detector:/workspace/lerobot_policy_modular_vla:/workspace/FLAG-Embodied-data:/workspace/Grounded-Segment-Anything:/workspace/Grounded-Segment-Anything/GroundingDINO:/workspace/Grounded-Segment-Anything/segment_anything${PYTHONPATH:+:$PYTHONPATH}" \
  -v "$TESIS_DIR":/workspace \
  -v "$CACHE_DIR":/cache \
  -w /workspace \
  "$IMAGE" "${@:-bash}"
