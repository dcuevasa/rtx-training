#!/usr/bin/env bash
# =====================================================================
# Phase 1: Prune Dataset (bendca61/so101-surface-transfer-v1)
# Drops depth video streams and trims 13D EE state down to 6-DoF joints.
# Runs inside the rtx-lerobot Docker container.
# =====================================================================
set -euo pipefail

INPUT_REPO="${1:-bendca61/so101-surface-transfer-v1}"
OUTPUT_REPO="${2:-bendca61/so101-surface-transfer-v1-pruned}"
LOCAL_CACHE="/cache/temp_prune_surface_transfer"

echo "=========================================================="
echo " Starting Dataset Prune Pipeline (Inside Docker)"
echo " Input Repo:  $INPUT_REPO"
echo " Output Repo: $OUTPUT_REPO"
echo " Temp Cache:  $LOCAL_CACHE"
echo "=========================================================="

# Clean previous temp if any
rm -rf "$LOCAL_CACHE" 2>/dev/null || true

python /workspace/FLAG-Embodied-data/prune_dataset.py \
  --input_repo "$INPUT_REPO" \
  --output_repo "$OUTPUT_REPO" \
  --local_dir "$LOCAL_CACHE" \
  --drop_depth \
  --drop_ee

echo "=========================================================="
echo " Prune completed successfully!"
echo " Uploaded to: https://huggingface.co/datasets/$OUTPUT_REPO"
echo "=========================================================="
