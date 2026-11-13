#!/usr/bin/env python3
"""
Upload trained LeRobot policy checkpoints to Hugging Face Hub.

Usage examples (inside Docker via run.sh):
  # 1. Upload the latest checkpoint of the current run to Hugging Face Hub:
  ./rtx-training/docker/run.sh python3 /workspace/rtx-training/tools/upload_checkpoint.py \
      --repo_id bendca61/smolvla-so101-surface-transfer

  # 2. Upload a specific step (e.g. step 35,000):
  ./rtx-training/docker/run.sh python3 /workspace/rtx-training/tools/upload_checkpoint.py \
      --repo_id bendca61/smolvla-so101-surface-transfer \
      --step 35000

  # 3. Upload ALL checkpoints (saved into checkpoints/<step>/ inside the repo):
  ./rtx-training/docker/run.sh python3 /workspace/rtx-training/tools/upload_checkpoint.py \
      --repo_id bendca61/smolvla-so101-surface-transfer \
      --all-checkpoints
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DEFAULT_BASE = Path("/cache/home/.cache/huggingface/lerobot/outputs/train")


def find_latest_run_dir(base_dir: Path) -> Path:
    runs = sorted(base_dir.glob("*/run"), key=os.path.getmtime, reverse=True)
    if not runs:
        # Check without /run suffix
        runs = sorted(base_dir.glob("*"), key=os.path.getmtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No training runs found in {base_dir}")
    return runs[0]


def main():
    parser = argparse.ArgumentParser(description="Upload LeRobot policy checkpoints to Hugging Face Hub")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo ID, e.g. 'bendca61/smolvla-so101-surface-transfer'")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to run output directory (default: most recent run)")
    parser.add_argument("--step", type=str, default="last", help="Step to upload (e.g. '35000', 'last'; default: 'last')")
    parser.add_argument("--all-checkpoints", action="store_true", help="Upload all intermediate checkpoints to the repo")
    parser.add_argument("--private", action="store_true", help="Make model repository private")
    args = parser.parse_args()

    api = HfApi()
    whoami = api.whoami()
    print(f"Authenticated as Hugging Face user: {whoami['name']}")

    run_path = Path(args.run_dir) if args.run_dir else find_latest_run_dir(DEFAULT_BASE)
    checkpoints_dir = run_path / "checkpoints" if (run_path / "checkpoints").exists() else run_path

    if not checkpoints_dir.exists():
        sys.exit(f"Error: Checkpoints directory not found at {checkpoints_dir}")

    print(f"Reading checkpoints from: {checkpoints_dir}")
    create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    if args.all_checkpoints:
        steps = sorted([p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name != "last" and (p / "pretrained_model").exists()])
        if not steps:
            sys.exit(f"No checkpoint folders with pretrained_model found in {checkpoints_dir}")

        print(f"Found {len(steps)} checkpoints to upload:")
        for s in steps:
            print(f" -> Uploading step {s.name} ...")
            api.upload_folder(
                folder_path=str(s / "pretrained_model"),
                repo_id=args.repo_id,
                path_in_repo=f"checkpoints/{s.name}",
                repo_type="model",
                commit_message=f"Upload checkpoint step {s.name}",
            )

        # Also upload latest/last to the root of the repo so it can be loaded directly
        last_dir = checkpoints_dir / "last" / "pretrained_model"
        if not last_dir.exists() and steps:
            last_dir = steps[-1] / "pretrained_model"

        if last_dir.exists():
            print("Uploading default deployable model to repository root...")
            api.upload_folder(
                folder_path=str(last_dir),
                repo_id=args.repo_id,
                repo_type="model",
                commit_message="Set latest checkpoint as root policy",
            )
    else:
        # Upload single step
        step_str = f"{int(args.step):06d}" if args.step.isdigit() else args.step
        target_dir = checkpoints_dir / step_str / "pretrained_model"

        if not target_dir.exists():
            # Try raw step without leading zeros
            target_dir = checkpoints_dir / args.step / "pretrained_model"

        if not target_dir.exists():
            available = [p.name for p in checkpoints_dir.iterdir() if p.is_dir()]
            sys.exit(f"Error: Step '{args.step}' not found in {checkpoints_dir}. Available: {available}")

        print(f"Uploading checkpoint '{step_str}' from {target_dir} to https://huggingface.co/{args.repo_id} ...")
        api.upload_folder(
            folder_path=str(target_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload policy checkpoint step {step_str}",
        )

    print(f"\n Successfully uploaded to: https://huggingface.co/{args.repo_id}")
    print(f"You can load this policy anywhere with:\n  LeRobotPolicy.from_pretrained('{args.repo_id}')")


if __name__ == "__main__":
    main()
