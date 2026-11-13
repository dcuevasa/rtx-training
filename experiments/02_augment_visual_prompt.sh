#!/usr/bin/env bash
# =====================================================================
# Phase 2: Offline Visual Prompt Annotation (lerobot-vision-detector)
# Generates bounding box overlays on cam_high for the target object (cup/mug)
# using YOLO11x on the pruned dataset.
# Runs inside the rtx-lerobot Docker container.
# =====================================================================
set -euo pipefail

INPUT_REPO="${1:-bendca61/so101-surface-transfer-v1-pruned}"
OUTPUT_REPO="${2:-bendca61/so101-surface-transfer-v1-prompted}"
TARGET_OBJECTS="${3:-cup}"
MODEL_NAME="${4:-yolo11x.pt}"
MODE="${5:-bbox}"   # 'bbox' or 'seg'

echo "=========================================================="
echo " Starting Visual Prompt Augmentation Pipeline (Inside Docker)"
echo " Input Repo:      $INPUT_REPO"
echo " Output Repo:     $OUTPUT_REPO"
echo " Target Objects:  $TARGET_OBJECTS"
echo " Vision Model:    $MODEL_NAME"
echo " Mode:            $MODE"
echo " Target Camera:   cam_high (cam_wrist remains raw pixel)"
echo "=========================================================="

python -m lerobot_vision_detector.cli.augment \
  --input_repo "$INPUT_REPO" \
  --output_repo "$OUTPUT_REPO" \
  --cameras cam_high \
  --mode "$MODE" \
  --target_objects "$TARGET_OBJECTS" \
  --model_name "$MODEL_NAME" \
  --conf_threshold 0.25 \
  --device cuda \
  --batch_size 16 \
  --push_to_hub \
  --overwrite

echo "=========================================================="
echo " Visual prompt augmentation completed successfully!"
echo " Augmented Dataset: https://huggingface.co/datasets/$OUTPUT_REPO"
echo "=========================================================="
