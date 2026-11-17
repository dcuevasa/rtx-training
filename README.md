# rtx-training

Containerized LeRobot (`0.6.1`) data-processing and policy-training pipeline configured for an **x86_64** desktop workstation with an **NVIDIA GeForce RTX 5080 (16 GB GDDR7 VRAM, Blackwell `sm_120`)**, **no `sudo`**, and **zero host installs**.

## Quick Start

1. **Build the Docker image** (`rtx-lerobot:0.6.1`):
   ```bash
   docker build -t rtx-lerobot:0.6.1 /home/dcuevas/Documentos/Tesis/rtx-training/docker
   ```

2. **Run the environment & dataset smoke test**:
   ```bash
   ./rtx-training/docker/run.sh python3 /workspace/rtx-training/tools/smoke_test_env.py
   ```

3. **Run an experiment script**:
   ```bash
   # Phase 2b: Fast initial-N-frame hybrid dataset splicing (obscure frames 0..4, raw 5..end)
   ./rtx-training/docker/run.sh \
     python3 /workspace/lerobot-vision-detector/scripts/mix_initial_frames_dataset.py \
       --initial_frames 5 --overwrite --push_to_hub

   # Phase 3: ACT single-episode overfit check
   ./rtx-training/docker/run.sh bash /workspace/rtx-training/experiments/03_act_overfit_sanity_check.sh

   # Phase 4: Full dual-camera ACT training on init5 dataset
   ./rtx-training/docker/run.sh \
     bash /workspace/rtx-training/experiments/04_train_act_full.sh \
     bendca61/so101-surface-transfer-v1-gsam-obscure-init5 5000 16 8 /workspace/outputs/train/act_init5_5k

   # Phase 5a: SmolVLA (450M) fine-tuning on init5 dataset
   DETACH=1 NAME=smolvla_train ./rtx-training/docker/run.sh \
     bash /workspace/rtx-training/experiments/05_train_smolvla.sh \
     bendca61/so101-surface-transfer-v1-gsam-obscure-init5 5000 8 8 /workspace/outputs/train/smolvla_init5_5k

   # Phase 5b: X-VLA (879M) fine-tuning on init5 dataset
   DETACH=1 NAME=xvla_train ./rtx-training/docker/run.sh \
     bash /workspace/rtx-training/experiments/06_train_xvla.sh \
     bendca61/so101-surface-transfer-v1-gsam-obscure-init5 5000 8 8 /workspace/outputs/train/xvla_init5_5k
   ```

`docker/run.sh` automatically resolves the parent `Tesis` directory to `/workspace` (override with `TESIS_DIR=...`), mounts `~/.cache/rtx-docker` to `/cache`, passes GPU access via `--device nvidia.com/gpu=all`, and forwards `HF_TOKEN`, `HF_USER`, and `WANDB_API_KEY` from `~/.bashrc`.
