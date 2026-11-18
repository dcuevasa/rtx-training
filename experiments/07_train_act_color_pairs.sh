#!/usr/bin/env bash
# =====================================================================
# Phase 7: Per-Color-Combination Dual-Camera ACT Specialists (Init-5)
# Trains dedicated dual-camera ACT policies for every color transfer
# pair (pink, yellow, blue) -> (pink, yellow, blue) using the initial-5
# masked dataset (`bendca61/so101-surface-transfer-v1-gsam-obscure-init5`)
# and uploads each to Hugging Face Hub (`bendca61/act-so101-init5-<c1>-to-<c2>`).
#
# Usage (inside Docker):
#   bash 07_train_act_color_pairs.sh all
#   bash 07_train_act_color_pairs.sh pink yellow "[0,19,23,33,45,53]" 3000 8 4
# =====================================================================
set -euo pipefail

DATASET_REPO="${DATASET_REPO:-bendca61/so101-surface-transfer-v1-gsam-obscure-init5}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/outputs/bendca61_so101-surface-transfer-v1-gsam-obscure-init5}"
HF_OWNER="${HF_OWNER:-bendca61}"

train_and_upload_pair() {
  local src_color="$1"
  local dst_color="$2"
  local episodes_json="$3"
  local steps="${4:-3000}"
  local batch_size="${5:-8}"
  local num_workers="${6:-4}"

  local pair_slug="${src_color}-to-${dst_color}"
  local job_name="act_init5_${src_color}_to_${dst_color}"
  local output_dir="/workspace/outputs/train/${job_name}"
  local log_file="/workspace/outputs/train/${job_name}.log"
  local hf_repo="${HF_OWNER}/act-so101-init5-${pair_slug}"

  mkdir -p /workspace/outputs/train
  rm -rf "$output_dir" 2>/dev/null || true

  echo "==========================================================" | tee "$log_file"
  echo " Training ACT Specialist: ${src_color} -> ${dst_color}" | tee -a "$log_file"
  echo " Dataset:  $DATASET_REPO (root=$DATASET_ROOT)" | tee -a "$log_file"
  echo " Episodes: $episodes_json | Steps: $steps | Batch: $batch_size | Workers: $num_workers" | tee -a "$log_file"
  echo " HF Repo:  $hf_repo" | tee -a "$log_file"
  echo "==========================================================" | tee -a "$log_file"

  lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.episodes="$episodes_json" \
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
    --output_dir="$output_dir" \
    --job_name="$job_name" \
    --batch_size="$batch_size" \
    --num_workers="$num_workers" \
    --steps="$steps" \
    --save_freq="$steps" \
    --log_freq=100 \
    --wandb.enable=false >>"$log_file" 2>&1

  python3 /workspace/rtx-training/tools/upload_checkpoint.py \
    --repo_id "$hf_repo" \
    --run_dir "$output_dir" \
    --step last >>"$log_file" 2>&1

  echo "Completed and uploaded: https://huggingface.co/${hf_repo}" | tee -a "$log_file"
}

if [[ "${1:-all}" == "all" ]]; then
  echo "Running Batch 1 (parallel): pink->yellow & pink->blue"
  train_and_upload_pair pink yellow "[0,19,23,33,45,53]" 3000 8 4 &
  p1=$!
  train_and_upload_pair pink blue "[2,6,15,37,47]" 3000 8 4 &
  p2=$!
  wait "$p1"
  wait "$p2"

  echo "Running Batch 2 (parallel): yellow->pink & yellow->blue"
  train_and_upload_pair yellow pink "[1,3,7,8,17,18,20,24,32]" 3000 8 4 &
  p1=$!
  train_and_upload_pair yellow blue "[9,11,22,29,31,35,43,46]" 3000 8 4 &
  p2=$!
  wait "$p1"
  wait "$p2"

  echo "Running Batch 3 (parallel): blue->pink & blue->yellow"
  train_and_upload_pair blue pink "[5,21,30,34,36,52]" 3000 8 4 &
  p1=$!
  train_and_upload_pair blue yellow "[4,10,16,26,42,44,48]" 3000 8 4 &
  p2=$!
  wait "$p1"
  wait "$p2"

  echo "Running Batch 4 (same-color combinations): pink->pink, yellow->yellow, blue->blue"
  train_and_upload_pair pink pink "[14,25,40,51]" 1500 8 4 &
  p1=$!
  train_and_upload_pair yellow yellow "[13,27,41,50]" 1500 8 4 &
  p2=$!
  wait "$p1"
  wait "$p2"

  train_and_upload_pair blue blue "[12,28,38,39,49]" 1500 8 4

  echo "=========================================================="
  echo " All 9 ACT color-combination specialists trained & uploaded!"
  echo "=========================================================="
else
  if [[ $# -lt 3 ]]; then
    echo "Usage: $0 all | <src_color> <dst_color> <episodes_json> [steps] [batch_size] [num_workers]" >&2
    exit 1
  fi
  train_and_upload_pair "$1" "$2" "$3" "${4:-3000}" "${5:-8}" "${6:-4}"
fi
