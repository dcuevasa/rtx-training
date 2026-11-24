# LeRobot training image for the local RTX 5080 workstation (x86_64, sm_120, 16 GB VRAM, CUDA 13).
# Base: NGC PyTorch (torch 2.9 built for sm_120/121, flash-attn, transformer-engine).
# Rule: NOTHING is installed on the host (no sudo). All deps live in this image.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    HF_LEROBOT_HOME=/cache/huggingface/lerobot \
    TORCH_HOME=/cache/torch \
    YOLO_CONFIG_DIR=/cache/ultralytics

# ffmpeg (shared libs) is required by pyav video decoding of LeRobot v3 AV1 videos
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev \
        git git-lfs libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Freeze the NGC-provided torch stack so pip can never replace it with a PyPI wheel.
RUN pip list --format=freeze 2>/dev/null | \
        grep -iE '^(torch|torchvision|torchaudio|triton|flash[_-]attn|transformer[_-]engine|numpy)==' \
        > /etc/pip/ngc-freeze.txt && cat /etc/pip/ngc-freeze.txt && \
    cat /etc/pip/ngc-freeze.txt >> /etc/pip/constraint.txt

ARG LEROBOT_VERSION=0.6.1
# Core training stack + VLA extras (smolvla, pi, xvla, peft). No hardware extras: training-only container.
RUN pip install "lerobot[training,smolvla,pi,xvla,peft]==${LEROBOT_VERSION}" && \
    pip install "ultralytics>=8.3.0" && \
    pip uninstall -y torchcodec || true && \
    python -c "import torch, lerobot; print('torch', torch.__version__, 'lerobot', lerobot.__version__)"

# SAM 2, Gradio, and WebSockets for Web UIs (SAM 2 Annotator & LeLab)
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install --no-build-isolation "sam2" hydra-core iopath "gradio>=5.0.0" "websockets>=15.0.1" && \
    python -c "import sam2, gradio, websockets; from sam2.build_sam import build_sam2_video_predictor; print('SAM 2, Gradio, and WebSockets installed successfully!')"

# Grounded-SAM dependencies (GroundingDINO & Segment Anything)
RUN pip install --no-cache-dir addict yapf timm supervision && \
    python -c "import addict, yapf, timm, supervision; print('Grounded-SAM dependencies installed successfully!')"

# Diffusion Policy dependency (diffusers, constrained against NGC torch stack)
RUN pip install --no-cache-dir -c /etc/pip/constraint.txt "diffusers>=0.27.2" && \
    python -c "import diffusers; print('diffusers', diffusers.__version__)"

WORKDIR /workspace
