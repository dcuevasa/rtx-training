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
   ./rtx-training/docker/run.sh bash /workspace/rtx-training/experiments/03_act_overfit_sanity_check.sh
   ```

`docker/run.sh` automatically resolves the parent `Tesis` directory to `/workspace` (override with `TESIS_DIR=...`), mounts `~/.cache/rtx-docker` to `/cache`, passes GPU access via `--device nvidia.com/gpu=all`, and forwards `HF_TOKEN`, `HF_USER`, and `WANDB_API_KEY` from `~/.bashrc`.
