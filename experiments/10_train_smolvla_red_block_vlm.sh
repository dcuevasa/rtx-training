#!/usr/bin/env bash
# =====================================================================
# Red-Block Unfrozen-VLM SmolVLA (450M) Specialist Fine-Tuning
# Fine-tunes lerobot/smolvla_base on the 74-episode red-block dataset
# with train_expert_only=false (unfreezing connector + VLM layers 0..14
# + action expert = 306.47M trainable params) while keeping SigLIP frozen.
# =====================================================================
set -euo pipefail

DATASET_REPO="${1:-bendca61/so101-red-block-blue-to-pink-v1-pruned}"
STEPS="${2:-25000}"
BATCH_SIZE="${3:-16}"
NUM_WORKERS="${4:-8}"
OUTPUT_DIR="${5:-/workspace/outputs/train/smolvla_red_block_vlm_25k_bs16}"
JOB_NAME="$(basename "$OUTPUT_DIR")"

echo "=========================================================="
echo " Starting Unfrozen-VLM SmolVLA Fine-Tuning (Inside Docker)"
echo " Pretrained Base: lerobot/smolvla_base"
echo " Dataset Repo:    $DATASET_REPO"
echo " Steps:           $STEPS"
echo " Batch Size:      $BATCH_SIZE"
echo " Num Workers:     $NUM_WORKERS"
echo " Trainable:       VLM connector + LLM(0..14) + Expert (306.47M)"
echo " Dtype:           bfloat16 (use_amp=false, lr=5e-5)"
echo " Device:          cuda (NVIDIA RTX 5080 sm_120)"
echo " Output Dir:      $OUTPUT_DIR"
echo "=========================================================="

rm -rf "$OUTPUT_DIR" 2>/dev/null || true

lerobot-train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}' \
  --policy.path="lerobot/smolvla_base" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_decay_steps="$STEPS" \
  --rename_map='{"observation.images.cam_high": "observation.images.camera1", "observation.images.cam_wrist": "observation.images.camera2"}' \
  --policy.input_features='{"observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.state": {"type": "STATE", "shape": [6]}}' \
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
echo " Unfrozen-VLM SmolVLA Fine-Tuning completed successfully!"
echo " Checkpoints saved in: $OUTPUT_DIR"
echo "=========================================================="
