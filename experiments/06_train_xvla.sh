#!/usr/bin/env bash
# =====================================================================
# Phase 5b: X-VLA (0.9B, Florence-2 + Soft Prompts) Policy Fine-Tuning
# Fine-tunes lerobot/xvla-base on the visual-prompted dataset on RTX 5080 (16 GB VRAM).
# Uses bfloat16, frozen Florence-2 vision tower, and action_mode=auto (6-DoF padded to 20D).
# =====================================================================
set -euo pipefail

DATASET_REPO="${1:-bendca61/so101-surface-transfer-v1-gsam-obscure}"
STEPS="${2:-20000}"
BATCH_SIZE="${3:-8}"
NUM_WORKERS="${4:-8}"
OUTPUT_DIR="${5:-/workspace/outputs/train/xvla_surface_transfer}"
JOB_NAME="$(basename "$OUTPUT_DIR")"

echo "=========================================================="
echo " Starting X-VLA Fine-Tuning Pipeline (Inside Docker)"
echo " Pretrained Base: lerobot/xvla-base"
echo " Dataset Repo:    $DATASET_REPO"
echo " Steps:           $STEPS"
echo " Batch Size:      $BATCH_SIZE"
echo " Num Workers:     $NUM_WORKERS"
echo " Dtype:           bfloat16 (freeze_vision_encoder=true)"
echo " Device:          cuda (NVIDIA RTX 5080 sm_120)"
echo " Output Dir:      $OUTPUT_DIR"
echo "=========================================================="

rm -rf "$OUTPUT_DIR" 2>/dev/null || true

lerobot-train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --policy.path="lerobot/xvla-base" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.use_amp=false \
  --policy.action_mode=auto \
  --policy.max_action_dim=20 \
  --policy.freeze_vision_encoder=true \
  --rename_map='{"observation.images.cam_high": "observation.images.image", "observation.images.cam_wrist": "observation.images.image2"}' \
  --policy.input_features='{"observation.images.image": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.images.image2": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.state": {"type": "STATE", "shape": [6]}}' \
  --policy.output_features='{"action": {"type": "ACTION", "shape": [6]}}' \
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
echo " X-VLA Fine-Tuning completed successfully!"
echo " Checkpoints saved in: $OUTPUT_DIR"
echo "=========================================================="
