#!/usr/bin/env bash
# =====================================================================
# Phase 5: SmolVLA (450M) Policy Fine-Tuning
# Fine-tunes SmolVLA base on the visual-prompted dataset on RTX 5080 (16 GB VRAM).
# Designed for safe deployment within the 8 GB VRAM budget of RTX 5070.
# =====================================================================
set -euo pipefail

DATASET_REPO="${1:-bendca61/so101-surface-transfer-v1-prompted}"
STEPS="${2:-20000}"
BATCH_SIZE="${3:-8}"
NUM_WORKERS="${4:-8}"
OUTPUT_DIR="/workspace/outputs/train/smolvla_surface_transfer"
JOB_NAME="smolvla_surface_transfer"

echo "=========================================================="
echo " Starting SmolVLA Fine-Tuning Pipeline (Inside Docker)"
echo " Pretrained Base: lerobot/smolvla_base"
echo " Dataset Repo:    $DATASET_REPO"
echo " Steps:           $STEPS"
echo " Batch Size:      $BATCH_SIZE"
echo " Num Workers:     $NUM_WORKERS"
echo " Dtype:           bfloat16"
echo " Device:          cuda (NVIDIA RTX 5080 sm_120)"
echo " Output Dir:      $OUTPUT_DIR"
echo "=========================================================="

rm -rf "$OUTPUT_DIR" 2>/dev/null || true

lerobot-train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --policy.path="lerobot/smolvla_base" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.dtype=bfloat16 \
  --rename_map='{"observation.images.cam_high": "observation.images.image"}' \
  --policy.input_features='{"observation.images.image": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.state": {"type": "STATE", "shape": [6]}}' \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "=========================================================="
echo " SmolVLA Fine-Tuning completed successfully!"
echo " Checkpoints saved in: $OUTPUT_DIR"
echo "=========================================================="
