#!/usr/bin/env bash
# =====================================================================
# Phase 4: Full Multi-Episode ACT Policy Training
# Trains ACT across all 15 surface transfer episodes using the
# visual-prompted dataset (or pruned baseline) on RTX 5080 (16 GB VRAM).
# =====================================================================
set -euo pipefail

DATASET_REPO="${1:-bendca61/so101-surface-transfer-v1-prompted}"
STEPS="${2:-30000}"
BATCH_SIZE="${3:-16}"
NUM_WORKERS="${4:-8}"
OUTPUT_DIR="/workspace/outputs/train/act_surface_transfer_full"
JOB_NAME="act_surface_transfer_full"

echo "=========================================================="
echo " Starting Full ACT Policy Training (Inside Docker)"
echo " Dataset Repo: $DATASET_REPO"
echo " Total Steps:  $STEPS"
echo " Batch Size:   $BATCH_SIZE"
echo " Num Workers:  $NUM_WORKERS"
echo " Device:       cuda (NVIDIA RTX 5080 sm_120)"
echo " Output Dir:   $OUTPUT_DIR"
echo "=========================================================="

rm -rf "$OUTPUT_DIR" 2>/dev/null || true

lerobot-train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.kl_weight=10.0 \
  --policy.optimizer_lr=1e-5 \
  --policy.push_to_hub=false \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[6]},"observation.images.cam_high":{"type":"VISUAL","shape":[3,480,640]},"observation.images.cam_wrist":{"type":"VISUAL","shape":[3,480,640]}}' \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false

echo "=========================================================="
echo " Full ACT Training completed successfully!"
echo " Checkpoints saved in: $OUTPUT_DIR"
echo "=========================================================="
