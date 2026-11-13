#!/usr/bin/env bash
# =====================================================================
# Phase 2: SAM 2 Video Visual Prompt Annotation (Dual Mode: CLI & Web UI)
# Resolves top-down perspective classification failures and tracks target
# objects seamlessly through physical gripper occlusion.
# Runs inside the rtx-lerobot Docker container.
#
# Usage:
#   1. Headless CLI mode (Default):
#      ./rtx-training/docker/run.sh bash /workspace/rtx-training/experiments/02_augment_visual_prompt_sam2.sh
#
#   2. Interactive Web UI mode (Port 7860):
#      PORT=7860 ./rtx-training/docker/run.sh bash /workspace/rtx-training/experiments/02_augment_visual_prompt_sam2.sh ui
# =====================================================================
set -euo pipefail

MODE_TYPE="${1:-cli}"
INPUT_REPO="${2:-bendca61/so101-surface-transfer-v1-pruned}"
OUTPUT_REPO="${3:-bendca61/so101-surface-transfer-sam2-v1}"
OVERLAY_MODE="${4:-both}"  # 'both', 'bbox', or 'mask'

if [[ "$MODE_TYPE" == "ui" || "$MODE_TYPE" == "--ui" || "$MODE_TYPE" == "web" ]]; then
  echo "=========================================================="
  echo " Launching SAM 2 Interactive Web UI (Inside Docker)"
  echo " Address:       http://0.0.0.0:7860"
  echo " Local browser: http://localhost:7860"
  echo " Remote SSH:    ssh -L 7860:localhost:7860 <user>@<host-ip>"
  echo "=========================================================="
  exec python /workspace/lerobot-vision-detector/scripts/sam2_webui.py
fi

echo "=========================================================="
echo " Starting SAM 2 Headless CLI Video Augmentation (Inside Docker)"
echo " Input Repo:    $INPUT_REPO"
echo " Output Repo:   $OUTPUT_REPO"
echo " Overlay Mode:  $OVERLAY_MODE"
echo " Target Camera: cam_high (cam_wrist untouched)"
echo " Model:         SAM 2.1 Base (configs/sam2.1/sam2.1_hiera_b+.yaml)"
echo " Auto-prompt:   Enabled (Surface / Object centroid heuristic)"
echo "=========================================================="

python /workspace/lerobot-vision-detector/scripts/sam2_annotate.py \
  --input_repo "$INPUT_REPO" \
  --output_repo "$OUTPUT_REPO" \
  --cameras cam_high \
  --mode "$OVERLAY_MODE" \
  --checkpoint /workspace/sam2.1_b.pt \
  --auto_prompt \
  --push_to_hub

echo "=========================================================="
echo " SAM 2 Video visual prompt augmentation completed successfully!"
echo " Augmented Dataset: https://huggingface.co/datasets/$OUTPUT_REPO"
echo "=========================================================="
