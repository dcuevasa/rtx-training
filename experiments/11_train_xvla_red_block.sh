#!/usr/bin/env bash
# =====================================================================
# Red-Block X-VLA (0.9B, Florence-2 + Soft Prompts + 24L DiT) Specialist
# Fine-tunes lerobot/xvla-base on the 74-episode red-block dataset
# on RTX 5080 (16 GB VRAM) with dual-view cameras (image + image2).
# =====================================================================
set -euo pipefail

DATASET_REPO="${1:-bendca61/so101-red-block-blue-to-pink-v1-pruned}"
STEPS="${2:-25000}"
BATCH_SIZE="${3:-16}"
NUM_WORKERS="${4:-8}"
OUTPUT_DIR="${5:-/workspace/outputs/train/xvla_red_block_25k_bs16}"
JOB_NAME="$(basename "$OUTPUT_DIR")"

echo "=========================================================="
echo " Starting X-VLA Red-Block Specialist Fine-Tuning (Docker)"
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
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}' \
  --policy.path="lerobot/xvla-base" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.use_amp=false \
  --policy.action_mode=auto \
  --policy.max_action_dim=20 \
  --policy.freeze_vision_encoder=true \
  --policy.scheduler_decay_steps="$STEPS" \
  --rename_map='{"observation.images.cam_high": "observation.images.image", "observation.images.cam_wrist": "observation.images.image2"}' \
  --policy.input_features='{"observation.images.image": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.images.image2": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.state": {"type": "STATE", "shape": [6]}}' \
  --policy.output_features='{"action": {"type": "ACTION", "shape": [6]}}' \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=200 \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "=========================================================="
echo " X-VLA Red-Block Fine-Tuning completed successfully!"
echo " Checkpoints saved in: $OUTPUT_DIR"
echo "=========================================================="
