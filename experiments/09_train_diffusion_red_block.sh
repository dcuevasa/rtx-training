#!/usr/bin/env bash
# =====================================================================
# Diffusion Policy (~260M) Training on Red Block Blue -> Pink (74 eps)
# Dual-camera (cam_high + cam_wrist), 6-DoF, unmasked, DDIM (16 steps)
# for real-time closed-loop deployment on RTX 5070 (8 GB VRAM).
# =====================================================================
set -euo pipefail

DATASET_REPO="${1:-bendca61/so101-red-block-blue-to-pink-v1-pruned}"
STEPS="${2:-30000}"
BATCH_SIZE="${3:-16}"
NUM_WORKERS="${4:-8}"
OUTPUT_DIR="${5:-/workspace/outputs/train/diffusion_red_block_blue_to_pink_30k}"
JOB_NAME="$(basename "$OUTPUT_DIR")"

echo "=========================================================="
echo " Starting Diffusion Policy Training Pipeline (Inside Docker)"
echo " Dataset Repo:    $DATASET_REPO"
echo " Steps:           $STEPS"
echo " Batch Size:      $BATCH_SIZE"
echo " Num Workers:     $NUM_WORKERS"
echo " Horizon / Exec:  64 / 32 (n_obs_steps=2, DDIM 16 steps)"
echo " Device:          cuda (NVIDIA RTX 5080 sm_120, AMP FP16)"
echo " Output Dir:      $OUTPUT_DIR"
echo "=========================================================="

rm -rf "$OUTPUT_DIR" 2>/dev/null || true

lerobot-train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}' \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=16 \
  --policy.push_to_hub=false \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[6]},"observation.images.cam_high":{"type":"VISUAL","shape":[3,480,640]},"observation.images.cam_wrist":{"type":"VISUAL","shape":[3,480,640]}}' \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=200 \
  --wandb.enable=false

echo "=========================================================="
echo " Diffusion Policy training completed successfully!"
echo " Checkpoints saved in: $OUTPUT_DIR"
echo "=========================================================="
