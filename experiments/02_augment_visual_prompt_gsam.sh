#!/usr/bin/env bash
# =====================================================================
# Phase 2: Offline Visual Prompt Mask Annotation with Grounded-SAM
# Generates segmentation mask overlays on cam_high for the target object (cup/mug)
# using Grounded-SAM (Grounding DINO + SAM ViT-Base) on the pruned dataset.
# The robot and all other non-target entities are completely excluded.
# Runs inside the rtx-lerobot Docker container.
# =====================================================================
set -euo pipefail

INPUT_REPO="${1:-bendca61/so101-surface-transfer-v1-pruned}"
OUTPUT_REPO="${2:-bendca61/so101-surface-transfer-v1-gsam-masks}"
TARGET_OBJECTS="${3:-mug, bowl, remote, bottle, cellphone}"
SAM_VERSION="${4:-vit_b}"
MODE="${5:-seg}"       # 'seg' for pure instance mask overlays
STRIDE="${6:-2}"     # Detection stride: 2 runs detection on every 2nd frame (15 Hz) & propagates masks (2x speedup)

MAX_EPISODES_FLAG=""
if [[ -n "${MAX_EPISODES:-}" ]]; then
  MAX_EPISODES_FLAG="--max_episodes $MAX_EPISODES"
fi

echo "=========================================================="
echo " Starting Grounded-SAM Mask Augmentation Pipeline (High Throughput)"
echo " Input Repo:      $INPUT_REPO"
echo " Output Repo:     $OUTPUT_REPO"
echo " Candidate Objs:  $TARGET_OBJECTS (Single target per episode from prompt, no robot)"
echo " Vision Model:    Grounded-SAM (Grounding DINO + SAM $SAM_VERSION)"
echo " Modality:        $MODE (Instance Segmentation Masks)"
echo " Target Camera:   cam_high (cam_wrist remains raw pixel)"
echo " Keyframe Stride: $STRIDE (Propagates masks smoothly between keyframes)"
echo "=========================================================="

python -m lerobot_vision_detector.cli.augment \
  --input_repo "$INPUT_REPO" \
  --output_repo "$OUTPUT_REPO" \
  --cameras cam_high \
  --detector_type grounded_sam \
  --sam_version "$SAM_VERSION" \
  --mode "$MODE" \
  --target_objects "$TARGET_OBJECTS" \
  --target_from_prompt \
  --conf_threshold 0.25 \
  --text_threshold 0.25 \
  --stride "$STRIDE" \
  --device cuda \
  --batch_size 16 \
  $MAX_EPISODES_FLAG \
  --push_to_hub \
  --overwrite

echo "=========================================================="
echo " Grounded-SAM mask augmentation completed successfully!"
echo " Augmented Dataset: https://huggingface.co/datasets/$OUTPUT_REPO"
echo "=========================================================="
