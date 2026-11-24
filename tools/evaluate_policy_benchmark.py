#!/usr/bin/env python3
"""
Comprehensive Thesis Benchmark & Diagnostic Suite for SO-101 LeRobot Policies.

Evaluates ACT, Diffusion Policy, SmolVLA, and X-VLA checkpoints uniformly on:
1. Kinematic Phase Segmentation (Initiation, Transit, Pre-Grasp, Grasp, Place)
2. 6-DoF Per-Joint Error Decomposition (shoulder_pan .. gripper)
3. Open-Loop Horizon Decay Curve (cumulative & pointwise error at k=1..50)
4. Trajectory Smoothness (d1, d2, d3 vs. human GT) & Chunk-Boundary Jump
   (Raw vs. Option 1 Smoothing w=5, k=10)
5. Multi-Phase Camera & State Attribution (cam_high vs. cam_wrist vs. state)
6. Spatial Batch Generalization (Batch 1 vs. Batch 2) & Photometric Robustness
7. Physical Rollout Telemetry Analysis (e.g., eval-act-so101-red-block-blue-to-pink-v1)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
PROXIMAL_IDX = [0, 1, 2]
DISTAL_IDX = [3, 4]
GRIPPER_IDX = 5


def load_policy_and_processors(
    pretrained_path: str,
    device: str = "cuda",
    revision: str | None = None,
):
    """Load policy, config, preprocessor, and postprocessor from local path or HF Hub."""
    cfg = PreTrainedConfig.from_pretrained(pretrained_path, revision=revision)
    cfg.device = device
    # Keep use_amp consistent with device
    if device == "cpu":
        cfg.use_amp = False

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_path, config=cfg, revision=revision)
    policy.to(device)
    policy.eval()

    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return cfg, policy, pre, post


def resolve_camera_keys(dataset: LeRobotDataset) -> tuple[str, str]:
    """Identify high and wrist camera keys present in dataset features."""
    keys = list(dataset.meta.camera_keys)
    high_key = next((k for k in keys if "high" in k or "camera1" in k or k.endswith(".image")), keys[0])
    wrist_key = next((k for k in keys if "wrist" in k or "camera2" in k or "image2" in k), keys[-1])
    return high_key, wrist_key


def extract_episode_bounds(dataset: LeRobotDataset) -> dict[int, tuple[int, int]]:
    """Return {ep_idx: (from_idx, to_idx)} from LeRobotDataset metadata."""
    bounds: dict[int, tuple[int, int]] = {}
    ep_meta = dataset.meta.episodes
    # v3.0 HuggingFace Dataset of episode metadata
    for row in ep_meta:
        ep_idx = int(row["episode_index"])
        bounds[ep_idx] = (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
    return bounds


def segment_episode_kinematics(states: np.ndarray, actions: np.ndarray) -> dict[str, int | bool]:
    """
    Detect kinematic milestones for an episode:
    - t_move_start: first moving frame (|action[t+1] - action[t]|_max > 0.5 deg)
    - t_grasp: steepest gripper closing onset in first 75% of episode
    - phases: init, transit, pregrasp, grasp, place + contiguous boundary anchor
    """
    n = len(actions)
    diffs = np.abs(np.diff(actions, axis=0)).max(axis=1)
    moving = np.where(diffs > 0.5)[0]
    t_move_start = int(moving[0]) if len(moving) > 0 else min(15, max(0, n // 10))
    t_move_start = min(t_move_start, max(0, n - 40))

    # Detect grasp closure: steepest 5-frame drop in gripper channel where min < 18 deg
    grip = actions[:, GRIPPER_IDX]
    search_lo = min(max(t_move_start + 15, 10), max(0, n - 30))
    search_hi = min(max(search_lo + 10, int(0.75 * n)), max(search_lo + 1, n - 10))
    if search_hi - search_lo > 6:
        drops = grip[search_lo + 5 : search_hi + 5] - grip[search_lo:search_hi]
        best_rel = int(np.argmin(drops))
        t_grasp = search_lo + best_rel
    else:
        t_grasp = min(max(n // 2, t_move_start + 20), max(0, n - 15))

    # Count distinct open->closed transitions (<12 after >20) for retry detection
    closed = grip < 14.0
    cycles = int(np.sum((~closed[:-1]) & closed[1:]))

    t_pregrasp = max(t_move_start, t_grasp - 15)
    if t_pregrasp - t_move_start >= 20:
        t_transit = (t_move_start + t_pregrasp) // 2
    else:
        t_transit = min(n - 2, t_move_start + 10)

    t_grasp_close = max(0, min(n - 20, t_grasp - 5))
    t_place = max(min(n - 35, (t_grasp + n) // 2), min(n - 2, t_grasp + 15))
    t_boundary_a = max(t_move_start, t_pregrasp - 20)
    t_boundary_b = min(n - 2, t_boundary_a + 20)

    return {
        "length": n,
        "t_move_start": t_move_start,
        "t_transit": t_transit,
        "t_pregrasp": t_pregrasp,
        "t_grasp": t_grasp,
        "t_grasp_close": t_grasp_close,
        "t_place": t_place,
        "t_boundary_a": t_boundary_a,
        "t_boundary_b": t_boundary_b,
        "gripper_cycles": cycles,
        "is_retry_ep": cycles > 1,
    }


def apply_option1_smoothing(
    chunk: np.ndarray,
    prev_action: np.ndarray | None = None,
    smooth_window: int = 5,
    blend_steps: int = 10,
) -> np.ndarray:
    """Replicate runtime Option 1 triangular smoothing + boundary blending."""
    out = chunk.copy()
    t_len = len(out)
    if smooth_window >= 3 and t_len >= 3:
        weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32) if smooth_window == 5 else np.ones(smooth_window, dtype=np.float32)
        weights /= weights.sum()
        pad = len(weights) // 2
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.zeros_like(out)
        for j in range(out.shape[1]):
            smoothed[:, j] = np.convolve(padded[:, j], weights, mode="valid")
        out = smoothed

    if prev_action is not None and blend_steps > 0:
        k = min(blend_steps, t_len)
        alphas = np.linspace(1.0 / (k + 1), 1.0, k, dtype=np.float32)[:, None]
        out[:k] = (1.0 - alphas) * prev_action[None, :] + alphas * out[:k]
    return out


def predict_chunk_at_index(
    dataset: LeRobotDataset,
    ep_from: int,
    idx: int,
    cfg: PreTrainedConfig,
    policy: torch.nn.Module,
    pre: Any,
    post: Any,
    high_key: str,
    wrist_key: str,
    seed: int = 42,
    override_high: torch.Tensor | None = None,
    override_wrist: torch.Tensor | None = None,
    state_delta_deg: np.ndarray | None = None,
    brightness_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Run deterministic policy chunk prediction at global frame `idx`, returning (chunk [T, 6], latency_ms)."""
    n_obs = max(1, int(getattr(cfg, "n_obs_steps", 1)))
    frame_indices = [max(ep_from, idx - (n_obs - 1 - offset)) for offset in range(n_obs)]

    processed_frames = []
    for f_idx in frame_indices:
        item = dataset[f_idx]
        obs: dict[str, Any] = {}
        for k, v in item.items():
            if k == "action":
                continue
            if isinstance(v, torch.Tensor):
                obs[k] = v.clone()
            else:
                obs[k] = v

        if override_high is not None:
            obs[high_key] = override_high.clone()
        if override_wrist is not None:
            obs[wrist_key] = override_wrist.clone()
        if brightness_scale is not None:
            obs[high_key] = torch.clamp(obs[high_key] * brightness_scale, 0.0, 1.0)
            obs[wrist_key] = torch.clamp(obs[wrist_key] * brightness_scale, 0.0, 1.0)
        if state_delta_deg is not None and "observation.state" in obs:
            obs["observation.state"] = obs["observation.state"] + torch.tensor(
                state_delta_deg, dtype=obs["observation.state"].dtype
            )

        proc = pre(obs)
        processed_frames.append(proc)

    if n_obs == 1:
        batch = processed_frames[0]
    else:
        batch = {}
        for k in processed_frames[0]:
            v0 = processed_frames[0][k]
            if isinstance(v0, torch.Tensor) and k.startswith("observation."):
                # proc[k] has leading batch dim 1; stack across dim=1 -> (1, n_obs, ...)
                batch[k] = torch.stack([pf[k] for pf in processed_frames], dim=1)
            else:
                batch[k] = v0

    policy.reset()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        raw_chunk = policy.predict_action_chunk(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    unnorm = post(raw_chunk)
    if isinstance(unnorm, torch.Tensor):
        arr = unnorm[0].detach().float().cpu().numpy()
    else:
        arr = np.asarray(unnorm)[0]
    # Trim padded dimensions (e.g. X-VLA internal 20D) to actual 6 DoF if needed
    return arr[:, :6].astype(np.float32), dt_ms


def compute_smoothness(traj: np.ndarray) -> dict[str, float]:
    """Compute velocity d1, acceleration d2, and jerk d3 norms (mean L1 across 6 joints)."""
    d1 = np.abs(np.diff(traj, axis=0)).mean() if len(traj) > 1 else 0.0
    d2 = np.abs(np.diff(traj, n=2, axis=0)).mean() if len(traj) > 2 else 0.0
    d3 = np.abs(np.diff(traj, n=3, axis=0)).mean() if len(traj) > 3 else 0.0
    return {"d1": float(d1), "d2": float(d2), "d3": float(d3)}


def analyze_rollout_dataset(repo_id: str) -> dict[str, Any]:
    """Compute hardware tracking lag, FPS, smoothness, and gripper stats on a rollout dataset."""
    ds = LeRobotDataset(repo_id, video_backend="pyav")
    hf = ds.hf_dataset
    states = np.asarray(hf["observation.state"], dtype=np.float32)
    actions = np.asarray(hf["action"], dtype=np.float32)
    timestamps = np.asarray(hf["timestamp"], dtype=np.float32)

    dt = np.diff(timestamps)
    dt_pos = dt[dt > 0]
    tracking_lag = np.abs(actions - states)
    smooth_action = compute_smoothness(actions)
    smooth_state = compute_smoothness(states)

    grip_act = actions[:, GRIPPER_IDX]
    grip_state = states[:, GRIPPER_IDX]
    cycles_cmd = int(np.sum((~(grip_act[:-1] < 14.0)) & (grip_act[1:] < 14.0)))
    cycles_meas = int(np.sum((~(grip_state[:-1] < 14.0)) & (grip_state[1:] < 14.0)))

    return {
        "repo_id": repo_id,
        "num_frames": int(len(actions)),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "fps_median": float(1.0 / np.median(dt_pos)),
        "fps_mean": float(1.0 / np.mean(dt_pos)),
        "dt_p95_ms": float(np.percentile(dt_pos * 1000.0, 95)),
        "dt_max_ms": float(np.max(dt_pos * 1000.0)),
        "tracking_lag_mean_deg": float(tracking_lag.mean()),
        "tracking_lag_per_joint_deg": {JOINTS[i]: float(tracking_lag[:, i].mean()) for i in range(6)},
        "action_smoothness": smooth_action,
        "state_smoothness": smooth_state,
        "gripper_cycles_commanded": cycles_cmd,
        "gripper_cycles_measured": cycles_meas,
    }


def evaluate_model(
    label: str,
    pretrained_path: str,
    dataset: LeRobotDataset,
    ep_bounds: dict[int, tuple[int, int]],
    all_actions: np.ndarray,
    all_states: np.ndarray,
    episode_ids: list[int],
    high_key: str,
    wrist_key: str,
    device: str = "cuda",
    revision: str | None = None,
    run_perturbations: bool = True,
    max_perturb_eps: int = 12,
) -> dict[str, Any]:
    """Run full thesis benchmark on one model over `episode_ids`."""
    print(f"\n>>> Benchmarking [{label}] ({pretrained_path}) on {len(episode_ids)} episodes...", flush=True)
    cfg, policy, pre, post = load_policy_and_processors(pretrained_path, device=device, revision=revision)
    chunk_cap = int(getattr(cfg, "n_action_steps", None) or getattr(cfg, "chunk_size", 50))
    if hasattr(cfg, "chunk_size") and getattr(cfg, "chunk_size") is not None:
        eval_horizon = min(int(cfg.chunk_size), 50)
    else:
        eval_horizon = min(chunk_cap, 50)

    phase_names = ["init", "transit", "pregrasp", "grasp_close", "place"]
    phase_chunk_errs: dict[str, list[float]] = {p: [] for p in phase_names}
    phase_step0_errs: dict[str, list[float]] = {p: [] for p in phase_names}
    phase_joint_errs: dict[str, list[np.ndarray]] = {p: [] for p in phase_names}
    horizon_step_errs: dict[int, list[float]] = {k: [] for k in [1, 5, 10, 15, 20, 25, 30, 32, 50]}
    horizon_point_errs: dict[int, list[float]] = {k: [] for k in [1, 5, 10, 15, 20, 25, 30, 32, 50]}

    gt_d1, gt_d2, gt_d3 = [], [], []
    raw_d1, raw_d2, raw_d3 = [], [], []
    opt1_d1, opt1_d2, opt1_d3 = [], [], []
    jump_raw_l1, jump_raw_max, jump_opt1_l1, jump_opt1_max = [], [], [], []

    start_max_deltas, end_deltas = [], []
    frozen_start_count = 0
    timing_errs, missed_closure, total_closure_checks = [], 0, 0
    latencies_ms = []

    batch1_errs, batch2_errs = [], []
    batch1_pregrasp, batch2_pregrasp = [], []

    # Perturbation trackers across phases
    perturb_metrics: dict[str, dict[str, list[float]]] = {
        ph: {k: [] for k in ["swap_high", "swap_wrist", "zero_high", "zero_wrist", "state_2deg"]}
        for ph in ["transit", "pregrasp", "grasp"]
    }
    bright_up_errs, bright_dn_errs = [], []

    for ep_pos, ep_id in enumerate(episode_ids):
        ep_from, ep_to = ep_bounds[ep_id]
        ep_len = ep_to - ep_from
        ep_act = all_actions[ep_from:ep_to]
        ep_st = all_states[ep_from:ep_to]
        seg = segment_episode_kinematics(ep_st, ep_act)

        anchors = {
            "init": ep_from + int(seg["t_move_start"]),
            "transit": ep_from + int(seg["t_transit"]),
            "pregrasp": ep_from + int(seg["t_pregrasp"]),
            "grasp_close": ep_from + int(seg["t_grasp_close"]),
            "place": ep_from + int(seg["t_place"]),
        }

        ep_all_chunk_errs = []
        ep_pre_chunk = None

        for ph, g_idx in anchors.items():
            local_t = g_idx - ep_from
            avail = min(eval_horizon, ep_len - local_t)
            if avail < 5:
                continue
            pred, dt_ms = predict_chunk_at_index(
                dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=42 + ep_pos
            )
            if not (ep_pos == 0 and ph == "init"):
                latencies_ms.append(dt_ms)
            h = min(len(pred), avail)
            pred_h = pred[:h]
            gt_h = ep_act[local_t : local_t + h]
            diff = np.abs(pred_h - gt_h)

            mean_err = float(diff.mean())
            step0_err = float(diff[0].mean())
            joint_err = diff.mean(axis=0)

            phase_chunk_errs[ph].append(mean_err)
            phase_step0_errs[ph].append(step0_err)
            phase_joint_errs[ph].append(joint_err)
            ep_all_chunk_errs.append(mean_err)

            for k in horizon_step_errs:
                if h >= k:
                    horizon_step_errs[k].append(float(diff[:k].mean()))
                    horizon_point_errs[k].append(float(diff[k - 1].mean()))

            sm_gt = compute_smoothness(gt_h)
            sm_raw = compute_smoothness(pred_h)
            sm_opt1 = compute_smoothness(apply_option1_smoothing(pred_h))
            gt_d1.append(sm_gt["d1"])
            gt_d2.append(sm_gt["d2"])
            gt_d3.append(sm_gt["d3"])
            raw_d1.append(sm_raw["d1"])
            raw_d2.append(sm_raw["d2"])
            raw_d3.append(sm_raw["d3"])
            opt1_d1.append(sm_opt1["d1"])
            opt1_d2.append(sm_opt1["d2"])
            opt1_d3.append(sm_opt1["d3"])

            if ph == "init":
                excursion = float(np.abs(pred_h - ep_st[local_t][None, :]).max())
                start_max_deltas.append(excursion)
                if excursion < 10.0:
                    frozen_start_count += 1
            elif ph == "place":
                end_deltas.append(float(np.abs(pred_h[-1] - pred_h[0]).mean()))
            elif ph == "pregrasp":
                ep_pre_chunk = pred_h
                # Gripper closure timing check inside pregrasp chunk (t_grasp - 15 -> 15 frames ahead)
                if h >= 18:
                    total_closure_checks += 1
                    gt_g = gt_h[:, GRIPPER_IDX]
                    pr_g = pred_h[:, GRIPPER_IDX]
                    gt_close_idx = int(np.argmin(np.diff(gt_g[:20]))) if len(gt_g) > 5 else 15
                    # Predicted closure: steepest negative drop or crossing below 18 deg when starting > 18
                    closed_pts = np.where(pr_g < 18.0)[0]
                    if len(closed_pts) > 0:
                        pr_close_idx = int(closed_pts[0])
                        timing_errs.append(abs(pr_close_idx - gt_close_idx))
                    else:
                        missed_closure += 1

        if ep_all_chunk_errs:
            ep_mean = float(np.mean(ep_all_chunk_errs))
            if ep_id <= 50:
                batch1_errs.append(ep_mean)
                if phase_chunk_errs["pregrasp"]:
                    batch1_pregrasp.append(phase_chunk_errs["pregrasp"][-1])
            else:
                batch2_errs.append(ep_mean)
                if phase_chunk_errs["pregrasp"]:
                    batch2_pregrasp.append(phase_chunk_errs["pregrasp"][-1])

        # Contiguous chunk boundary jump (step 20 transition near grasp approach)
        idx_a = ep_from + int(seg["t_boundary_a"])
        idx_b = ep_from + int(seg["t_boundary_b"])
        step_gap = idx_b - idx_a
        if step_gap >= 10:
            pred_a, _ = predict_chunk_at_index(dataset, ep_from, idx_a, cfg, policy, pre, post, high_key, wrist_key, seed=42 + ep_pos)
            pred_b, _ = predict_chunk_at_index(dataset, ep_from, idx_b, cfg, policy, pre, post, high_key, wrist_key, seed=42 + ep_pos)
            if len(pred_a) >= step_gap:
                last_a = pred_a[step_gap - 1]
                jump_raw = np.abs(pred_b[0] - last_a)
                jump_raw_l1.append(float(jump_raw.mean()))
                jump_raw_max.append(float(jump_raw.max()))

                sm_a = apply_option1_smoothing(pred_a)
                last_sm_a = sm_a[step_gap - 1]
                sm_b = apply_option1_smoothing(pred_b, prev_action=last_sm_a)
                jump_sm = np.abs(sm_b[0] - last_sm_a)
                jump_opt1_l1.append(float(jump_sm.mean()))
                jump_opt1_max.append(float(jump_sm.max()))

        # Multi-phase camera & state attribution + brightness robustness
        if run_perturbations and ep_pos < max_perturb_eps:
            other_ep = episode_ids[(ep_pos + max(1, len(episode_ids) // 3)) % len(episode_ids)]
            o_from, o_to = ep_bounds[other_ep]
            o_seg = segment_episode_kinematics(all_states[o_from:o_to], all_actions[o_from:o_to])
            o_anchors = {
                "transit": o_from + int(o_seg["t_transit"]),
                "pregrasp": o_from + int(o_seg["t_pregrasp"]),
                "grasp": o_from + int(o_seg["t_grasp"]),
            }
            cur_anchors = {
                "transit": ep_from + int(seg["t_transit"]),
                "pregrasp": ep_from + int(seg["t_pregrasp"]),
                "grasp": ep_from + int(seg["t_grasp"]),
            }

            for ph_p in ["transit", "pregrasp", "grasp"]:
                g_idx = cur_anchors[ph_p]
                o_idx = o_anchors[ph_p]
                base_pred, _ = predict_chunk_at_index(dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=100 + ep_pos)
                o_item = dataset[o_idx]
                c_item = dataset[g_idx]

                pred_sh, _ = predict_chunk_at_index(dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=100 + ep_pos, override_high=o_item[high_key])
                pred_sw, _ = predict_chunk_at_index(dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=100 + ep_pos, override_wrist=o_item[wrist_key])
                pred_zh, _ = predict_chunk_at_index(dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=100 + ep_pos, override_high=torch.zeros_like(c_item[high_key]))
                pred_zw, _ = predict_chunk_at_index(dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=100 + ep_pos, override_wrist=torch.zeros_like(c_item[wrist_key]))
                pred_st, _ = predict_chunk_at_index(dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=100 + ep_pos, state_delta_deg=np.full(6, 2.0, dtype=np.float32))

                perturb_metrics[ph_p]["swap_high"].append(float(np.abs(pred_sh - base_pred).mean()))
                perturb_metrics[ph_p]["swap_wrist"].append(float(np.abs(pred_sw - base_pred).mean()))
                perturb_metrics[ph_p]["zero_high"].append(float(np.abs(pred_zh - base_pred).mean()))
                perturb_metrics[ph_p]["zero_wrist"].append(float(np.abs(pred_zw - base_pred).mean()))
                perturb_metrics[ph_p]["state_2deg"].append(float(np.abs(pred_st - base_pred).mean()))

                if ph_p == "pregrasp" and ep_pre_chunk is not None:
                    local_pg = g_idx - ep_from
                    h_pg = min(len(ep_pre_chunk), ep_len - local_pg)
                    gt_pg = ep_act[local_pg : local_pg + h_pg]
                    pred_up, _ = predict_chunk_at_index(dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=42 + ep_pos, brightness_scale=1.25)
                    pred_dn, _ = predict_chunk_at_index(dataset, ep_from, g_idx, cfg, policy, pre, post, high_key, wrist_key, seed=42 + ep_pos, brightness_scale=0.75)
                    bright_up_errs.append(float(np.abs(pred_up[:h_pg] - gt_pg).mean()))
                    bright_dn_errs.append(float(np.abs(pred_dn[:h_pg] - gt_pg).mean()))

    # Aggregate results
    all_phase_errs = [v for vals in phase_chunk_errs.values() for v in vals]
    all_step0_errs = [v for vals in phase_step0_errs.values() for v in vals]
    all_joint_mat = np.stack([j for vals in phase_joint_errs.values() for j in vals], axis=0)

    def summarize(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
        arr = np.asarray(vals, dtype=np.float32)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
        }

    phase_summary = {}
    for ph in phase_names:
        j_arr = np.stack(phase_joint_errs[ph], axis=0) if phase_joint_errs[ph] else np.zeros((1, 6))
        phase_summary[ph] = {
            "chunk_L1": summarize(phase_chunk_errs[ph]),
            "step0_L1": summarize(phase_step0_errs[ph]),
            "proximal_L1": float(j_arr[:, PROXIMAL_IDX].mean()),
            "distal_L1": float(j_arr[:, DISTAL_IDX].mean()),
            "gripper_L1": float(j_arr[:, GRIPPER_IDX].mean()),
        }

    camera_attribution = {}
    for ph_p, mdict in perturb_metrics.items():
        sh = float(np.mean(mdict["swap_high"])) if mdict["swap_high"] else float("nan")
        sw = float(np.mean(mdict["swap_wrist"])) if mdict["swap_wrist"] else float("nan")
        zh = float(np.mean(mdict["zero_high"])) if mdict["zero_high"] else float("nan")
        zw = float(np.mean(mdict["zero_wrist"])) if mdict["zero_wrist"] else float("nan")
        st = float(np.mean(mdict["state_2deg"])) if mdict["state_2deg"] else float("nan")
        wrist_share = float(100.0 * sw / (sh + sw)) if (sh + sw) > 0 and not np.isnan(sh) else float("nan")
        ratio = float(sh / sw) if sw > 0 and not np.isnan(sw) else float("nan")
        camera_attribution[ph_p] = {
            "swap_high": sh,
            "swap_wrist": sw,
            "high_to_wrist_ratio": ratio,
            "wrist_reliance_pct": wrist_share,
            "zero_high": zh,
            "zero_wrist": zw,
            "state_2deg_sens": st,
        }

    res = {
        "label": label,
        "pretrained_path": pretrained_path,
        "policy_type": cfg.type,
        "chunk_size": eval_horizon,
        "n_obs_steps": int(getattr(cfg, "n_obs_steps", 1)),
        "num_episodes": len(episode_ids),
        "device": device,
        "inference_latency_ms": {
            "mean": float(np.mean(latencies_ms)) if latencies_ms else float("nan"),
            "p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else float("nan"),
        },
        "overall": {
            "chunk_L1": summarize(all_phase_errs),
            "step0_L1": summarize(all_step0_errs),
            "per_joint_mean_L1": {JOINTS[i]: float(all_joint_mat[:, i].mean()) for i in range(6)},
            "per_joint_p90_L1": {JOINTS[i]: float(np.percentile(all_joint_mat[:, i], 90)) for i in range(6)},
            "start_max_delta_min": float(min(start_max_deltas)) if start_max_deltas else float("nan"),
            "start_max_delta_mean": float(np.mean(start_max_deltas)) if start_max_deltas else float("nan"),
            "frozen_start_rate_pct": float(100.0 * frozen_start_count / max(1, len(start_max_deltas))),
            "end_delta_mean": float(np.mean(end_deltas)) if end_deltas else float("nan"),
            "gripper_timing_err_frames": float(np.mean(timing_errs)) if timing_errs else float("nan"),
            "missed_closure_rate_pct": float(100.0 * missed_closure / max(1, total_closure_checks)),
        },
        "phases": phase_summary,
        "horizon_decay": {
            str(k): {
                "cum_L1": float(np.mean(v)),
                "point_L1": float(np.mean(horizon_point_errs[k])),
            }
            for k, v in horizon_step_errs.items()
            if len(v) > 0
        },
        "smoothness": {
            "gt": {"d1": float(np.mean(gt_d1)), "d2": float(np.mean(gt_d2)), "d3": float(np.mean(gt_d3))},
            "raw": {"d1": float(np.mean(raw_d1)), "d2": float(np.mean(raw_d2)), "d3": float(np.mean(raw_d3))},
            "opt1": {"d1": float(np.mean(opt1_d1)), "d2": float(np.mean(opt1_d2)), "d3": float(np.mean(opt1_d3))},
            "boundary_jump_raw_l1": float(np.mean(jump_raw_l1)) if jump_raw_l1 else float("nan"),
            "boundary_jump_raw_max": float(np.mean(jump_raw_max)) if jump_raw_max else float("nan"),
            "boundary_jump_opt1_l1": float(np.mean(jump_opt1_l1)) if jump_opt1_l1 else float("nan"),
            "boundary_jump_opt1_max": float(np.mean(jump_opt1_max)) if jump_opt1_max else float("nan"),
        },
        "camera_attribution": camera_attribution,
        "generalization": {
            "batch1_ep0_50_chunk_L1": float(np.mean(batch1_errs)) if batch1_errs else float("nan"),
            "batch2_ep51_73_chunk_L1": float(np.mean(batch2_errs)) if batch2_errs else float("nan"),
            "batch1_pregrasp_L1": float(np.mean(batch1_pregrasp)) if batch1_pregrasp else float("nan"),
            "batch2_pregrasp_L1": float(np.mean(batch2_pregrasp)) if batch2_pregrasp else float("nan"),
            "pregrasp_bright_up25_L1": float(np.mean(bright_up_errs)) if bright_up_errs else float("nan"),
            "pregrasp_bright_dn25_L1": float(np.mean(bright_dn_errs)) if bright_dn_errs else float("nan"),
        },
    }

    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def format_markdown_report(results: list[dict[str, Any]], rollout_stats: dict[str, Any] | None) -> str:
    """Format benchmark JSON results into readable GitHub-flavored Markdown tables."""
    lines = ["# SO-101 Multi-Policy Thesis Benchmark Report\n"]

    lines.append("## 1. Overall Trajectory Accuracy, Timing & Latency\n")
    lines.append("| Model | Type (`H`) | Chunk L1 (mean/med/p90) | Step0 L1 | Pre-Grasp L1 | Distal Wrist L1 | Gripper Timing Err | Missed / Frozen | Latency |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        ov = r["overall"]
        pg = r["phases"]["pregrasp"]
        lines.append(
            f"| **{r['label']}** | `{r['policy_type']} ({r['chunk_size']})` | "
            f"`{ov['chunk_L1']['mean']:.2f}°` / `{ov['chunk_L1']['median']:.2f}°` / `{ov['chunk_L1']['p90']:.2f}°` | "
            f"`{ov['step0_L1']['mean']:.2f}°` | `{pg['chunk_L1']['mean']:.2f}°` | `{pg['distal_L1']:.2f}°` | "
            f"`{ov['gripper_timing_err_frames']:.1f} f` | `{ov['missed_closure_rate_pct']:.1f}%` / `{ov['frozen_start_rate_pct']:.1f}%` | `{r['inference_latency_ms']['mean']:.1f} ms` (`{r.get('device','cuda')}`) |"
        )

    lines.append("\n## 2. Per-Joint 6-DoF Mean L1 Error (°)\n")
    lines.append("| Model | `shoulder_pan` | `shoulder_lift` | `elbow_flex` | `wrist_flex` | `wrist_roll` | `gripper` |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        pj = r["overall"]["per_joint_mean_L1"]
        lines.append(
            f"| **{r['label']}** | " + " | ".join(f"`{pj[j]:.2f}°`" for j in JOINTS) + " |"
        )

    lines.append("\n## 3. Smoothness & 20-Step Chunk Boundary Jump (Raw vs. Option 1 Smoothing)\n")
    lines.append("| Model | GT `d2` | Raw `d1 / d2 / d3` | Option-1 `d1 / d2` | Boundary Jump Raw (`L1 / max`) | Boundary Jump Opt-1 (`L1 / max`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        sm = r["smoothness"]
        lines.append(
            f"| **{r['label']}** | `{sm['gt']['d2']:.2f}` | "
            f"`{sm['raw']['d1']:.2f} / {sm['raw']['d2']:.2f} / {sm['raw']['d3']:.2f}` | "
            f"`{sm['opt1']['d1']:.2f} / {sm['opt1']['d2']:.2f}` | "
            f"`{sm['boundary_jump_raw_l1']:.2f}° / {sm['boundary_jump_raw_max']:.2f}°` | "
            f"`{sm['boundary_jump_opt1_l1']:.2f}° / {sm['boundary_jump_opt1_max']:.2f}°` |"
        )

    lines.append("\n## 4. Multi-Phase Visual & State Attribution (`Pre-Grasp t_grasp-15` & `Grasp t_grasp`)\n")
    lines.append("| Model | Transit Wrist % (`high/wrist`) | Pre-Grasp `swap_high / swap_wrist` (`ratio`) | Pre-Grasp Wrist Share | Grasp `swap_high / swap_wrist` | Grasp Wrist Share | `state +2°` Sens |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        ca = r["camera_attribution"]
        tr, pg, gr = ca["transit"], ca["pregrasp"], ca["grasp"]
        lines.append(
            f"| **{r['label']}** | `{tr['wrist_reliance_pct']:.1f}%` (`{tr['swap_high']:.2f}°/{tr['swap_wrist']:.2f}°`) | "
            f"`{pg['swap_high']:.2f}° / {pg['swap_wrist']:.2f}°` (`{pg['high_to_wrist_ratio']:.2f}×`) | **`{pg['wrist_reliance_pct']:.1f}%`** | "
            f"`{gr['swap_high']:.2f}° / {gr['swap_wrist']:.2f}°` | **`{gr['wrist_reliance_pct']:.1f}%`** | `{pg['state_2deg_sens']:.2f}°` |"
        )

    lines.append("\n## 5. Open-Loop Horizon Decay (Cumulative Chunk L1° over `n_action_steps = k`)\n")
    ks = ["1", "5", "10", "15", "20", "30", "32", "50"]
    lines.append("| Model | " + " | ".join(f"`k={k}`" for k in ks) + " |")
    lines.append("| :--- | " + " | ".join(":---:" for _ in ks) + " |")
    for r in results:
        hd = r["horizon_decay"]
        vals = [f"`{hd[k]['cum_L1']:.2f}°`" if k in hd else "—" for k in ks]
        lines.append(f"| **{r['label']}** | " + " | ".join(vals) + " |")

    lines.append("\n## 6. Spatial Layout Shift (`Batch 1 ep 0..50` vs `Batch 2 ep 51..73`) & Brightness Robustness\n")
    lines.append("| Model | Batch 1 Chunk L1 | Batch 2 Chunk L1 | Batch 1 Pre-Grasp | Batch 2 Pre-Grasp | Bright `+25%` | Bright `-25%` |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        g = r["generalization"]
        lines.append(
            f"| **{r['label']}** | `{g['batch1_ep0_50_chunk_L1']:.2f}°` | `{g['batch2_ep51_73_chunk_L1']:.2f}°` | "
            f"`{g['batch1_pregrasp_L1']:.2f}°` | `{g['batch2_pregrasp_L1']:.2f}°` | "
            f"`{g['pregrasp_bright_up25_L1']:.2f}°` | `{g['pregrasp_bright_dn25_L1']:.2f}°` |"
        )

    if rollout_stats is not None:
        lines.append("\n## 7. Physical Hardware Rollout Telemetry\n")
        lines.append(f"- **Dataset:** `{rollout_stats['repo_id']}` (`{rollout_stats['num_frames']}` frames in `{rollout_stats['duration_s']:.1f} s`)\n")
        lines.append(f"- **Control Rate:** `{rollout_stats['fps_median']:.2f} Hz` median / `{rollout_stats['fps_mean']:.2f} Hz` mean (`dt p95 = {rollout_stats['dt_p95_ms']:.1f} ms`, `max = {rollout_stats['dt_max_ms']:.1f} ms`)\n")
        lines.append(f"- **Servo Tracking Lag (`|action - state|` mean):** `{rollout_stats['tracking_lag_mean_deg']:.2f}°` | Joint lags: `{rollout_stats['tracking_lag_per_joint_deg']}`\n")
        lines.append(f"- **Smoothness (`d1/d2`):** Commanded `{rollout_stats['action_smoothness']['d1']:.2f} / {rollout_stats['action_smoothness']['d2']:.2f}` | Measured state `{rollout_stats['state_smoothness']['d1']:.2f} / {rollout_stats['state_smoothness']['d2']:.2f}`\n")
        lines.append(f"- **Gripper Cycles:** `{rollout_stats['gripper_cycles_commanded']}` commanded, `{rollout_stats['gripper_cycles_measured']}` measured\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SO-101 Multi-Policy Thesis Benchmark Suite")
    parser.add_argument("--dataset", required=True, help="Dataset repo ID on HF Hub")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of 'label=pretrained_path' entries (e.g., act_v2=bendca61/act-so101-red-block-blue-to-pink)",
    )
    parser.add_argument("--episodes", type=int, default=24, help="Number of evenly sampled episodes (or 0 for all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-perturbations", action="store_true", help="Skip camera/state/brightness ablations")
    parser.add_argument("--max-perturb-eps", type=int, default=12)
    parser.add_argument("--rollout-repo", type=str, default=None, help="Optional physical rollout dataset repo to analyze")
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--out-md", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}...", flush=True)
    dataset = LeRobotDataset(args.dataset, video_backend="pyav")
    ep_bounds = extract_episode_bounds(dataset)
    hf = dataset.hf_dataset
    all_actions = np.asarray(hf["action"], dtype=np.float32)
    all_states = np.asarray(hf["observation.state"], dtype=np.float32)
    high_key, wrist_key = resolve_camera_keys(dataset)
    print(f"Dataset loaded: {len(ep_bounds)} episodes, {len(all_actions)} frames, cameras=({high_key}, {wrist_key})", flush=True)

    sorted_eps = sorted(ep_bounds.keys())
    if args.episodes <= 0 or args.episodes >= len(sorted_eps):
        eval_eps = sorted_eps
    else:
        idx = np.linspace(0, len(sorted_eps) - 1, args.episodes, dtype=int)
        eval_eps = [sorted_eps[i] for i in idx]

    results = []
    for spec in args.models:
        if "=" not in spec:
            sys.exit(f"Invalid --models entry '{spec}', expected 'label=path'")
        label, path = spec.split("=", 1)
        res = evaluate_model(
            label=label,
            pretrained_path=path,
            dataset=dataset,
            ep_bounds=ep_bounds,
            all_actions=all_actions,
            all_states=all_states,
            episode_ids=eval_eps,
            high_key=high_key,
            wrist_key=wrist_key,
            device=args.device,
            run_perturbations=not args.no_perturbations,
            max_perturb_eps=args.max_perturb_eps,
        )
        results.append(res)

    rollout_stats = analyze_rollout_dataset(args.rollout_repo) if args.rollout_repo else None

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"dataset": args.dataset, "episodes": eval_eps, "models": results, "rollout": rollout_stats}, indent=2))

    md = format_markdown_report(results, rollout_stats)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    print("\n" + md)
    print(f"\nSaved JSON -> {out_json}\nSaved MD   -> {out_md}")


if __name__ == "__main__":
    main()
