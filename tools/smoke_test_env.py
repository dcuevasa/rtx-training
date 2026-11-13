"""Smoke test: env + dataset loading + video decoding throughput inside rtx-lerobot container."""
import time
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
for m in ["lerobot", "torchcodec", "av", "transformers", "peft", "ultralytics", "cv2", "sam2", "gradio"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s}", getattr(mod, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(f"{m:12s} IMPORT FAIL: {type(e).__name__}: {str(e)[:200]}")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

REPO = "bendca61/so101-surface-transfer-v1"
meta = LeRobotDatasetMetadata(REPO)
print("episodes", meta.total_episodes, "frames", meta.total_frames, "fps", meta.fps)
print("tasks:\n", meta.tasks)
eps = meta.episodes
cols = [c for c in eps.column_names if c in ("episode_index", "length", "tasks")]
for row in eps.select_columns(cols).to_list():
    print(row)

for backend in ["torchcodec", "pyav"]:
    try:
        ds = LeRobotDataset(REPO, episodes=[0], video_backend=backend)
        t = time.time()
        n = 32
        for i in range(n):
            item = ds[i * 5]
        dt = time.time() - t
        keys = {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")}
        print(f"[{backend}] OK {n/dt:.1f} samples/s  keys={keys}")
    except Exception as e:  # noqa: BLE001
        print(f"[{backend}] FAIL {type(e).__name__}: {str(e)[:300]}")
