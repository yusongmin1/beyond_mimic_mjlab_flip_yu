#!/usr/bin/env python
"""Batch-export every g1_tracking run to a self-contained motion-ONNX.

For each run directory under logs/rsl_rl/g1_tracking/:
- pick the newest checkpoint (model_*.pt, or policy.pt if newer);
- rebuild actor MLP (154-512-256-128-29, ELU) + EmpiricalNormalization
  (rsl_rl 4.0.1: (x-mean)/(std+1e-2)) from actor_state_dict;
- load the run's motion npz (params/env.yaml motion_file, fallback: infer
  from run name / embedded TorchScript motion);
- embed joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/
  body_ang_vel_w and export (obs,time_step)->7 outputs ONNX;
- attach the same metadata dict as the training exporter
  (joint_names mjlab order, action_scale, default_joint_pos, ...).

Output: <run_dir>/exported/policy.onnx
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn

MJLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACKING_DIR = os.path.join(MJLAB_ROOT, "logs", "rsl_rl", "g1_tracking")
MOTIONS = os.path.join(MJLAB_ROOT, "motions")

# mjlab G1 joint order (== MuJoCo g1.xml tree order) — same as the sim2sim script.
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
N = len(JOINT_NAMES)

# mjlab G1 action scale (0.25 * effort/kp), mjlab joint order.
ACTION_SCALE = (
    [0.518, 0.330, 0.518, 0.330, 0.560, 0.560] * 2   # legs L/R
    + [0.518, 0.560, 0.560]                          # waist yaw/roll/pitch
    + [0.560, 0.560, 0.560, 0.560, 0.560, 0.128, 0.128]   # left arm
    + [0.560, 0.560, 0.560, 0.560, 0.560, 0.128, 0.128]   # right arm
)
DEFAULT_JOINT_POS = (
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0] * 2 + [0.0, 0.0, 0.0]
    + [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
    + [0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
)
OBS_NAMES = ["command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions"]
BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
ANCHOR = "torso_link"
# npz body index (mjcf tree order) for each of the 14 tracking bodies.
NPZ_BODY_IDX = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]


class NormalizedActor(nn.Module):
    """actor MLP + obs normalizer, exactly rsl_rl 4.0.1 semantics."""

    def __init__(self, sd: dict):
        super().__init__()
        layers = []
        i = 0
        while f"mlp.{i}.weight" in sd:
            layers.append(
                nn.Linear(sd[f"mlp.{i}.weight"].shape[1], sd[f"mlp.{i}.weight"].shape[0])
            )
            if f"mlp.{i + 2}.weight" in sd:
                layers.append(nn.ELU())
            i += 2
        self.mlp = nn.Sequential(*layers)
        mlp_sd = {k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")}
        self.mlp.load_state_dict(mlp_sd)
        self.register_buffer("mean", sd["obs_normalizer._mean"].clone())
        self.register_buffer("std", sd["obs_normalizer._std"].clone())

    def forward(self, x):
        return self.mlp((x - self.mean) / (self.std + 1e-2))


class MotionModel(nn.Module):
    def __init__(self, actor: NormalizedActor, motion_npz: dict):
        super().__init__()
        self.policy = actor
        self.register_buffer("joint_pos", torch.tensor(np.asarray(motion_npz["joint_pos"], np.float32)))
        self.register_buffer("joint_vel", torch.tensor(np.asarray(motion_npz["joint_vel"], np.float32)))
        self.register_buffer(
            "body_pos_w", torch.tensor(np.asarray(motion_npz["body_pos_w"], np.float32)[:, NPZ_BODY_IDX])
        )
        self.register_buffer(
            "body_quat_w", torch.tensor(np.asarray(motion_npz["body_quat_w"], np.float32)[:, NPZ_BODY_IDX])
        )
        self.register_buffer(
            "body_lin_vel_w", torch.tensor(np.asarray(motion_npz["body_lin_vel_w"], np.float32)[:, NPZ_BODY_IDX])
        )
        self.register_buffer(
            "body_ang_vel_w", torch.tensor(np.asarray(motion_npz["body_ang_vel_w"], np.float32)[:, NPZ_BODY_IDX])
        )

    def forward(self, obs, time_step):
        t = torch.clamp(time_step.long().squeeze(-1), max=self.joint_pos.shape[0] - 1)
        return (
            self.policy(obs),
            self.joint_pos[t], self.joint_vel[t],
            self.body_pos_w[t], self.body_quat_w[t],
            self.body_lin_vel_w[t], self.body_ang_vel_w[t],
        )


def csv_str(arr) -> str:
    return ",".join(f"{x:.6f}" if isinstance(x, (int, float, np.floating)) else str(x) for x in arr)


def find_motion(run_dir: str, run_name: str) -> str:
    # 1) params/env.yaml motion_file
    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if os.path.isfile(env_yaml):
        txt = open(env_yaml).read()
        m = re.search(r"motion_file:\s*(\S+)", txt)
        if m:
            p = m.group(1).strip("'\"")
            if os.path.isfile(p):
                return p
            # remote path (hyqf machine) — remap into local motions/
            base = os.path.basename(p)
            for sub in ("", "chunwan", "omnixtreme", "mimickit"):
                cand = os.path.join(MOTIONS, sub, base) if sub else os.path.join(MOTIONS, base)
                if os.path.isfile(cand):
                    return cand
    # 2) infer from run name
    for pat, sub in [
        (r"((?:88|90)_\d+_stageii_unitree_g1(?:_fixed)?(?:_with_prep)?)", "chunwan"),
        (r"(g1_motion_long_\d+(?:_with_prep)?)", "omnixtreme"),
        (r"(g1_(?:kick_combo|spinkick|cartwheel2?))", "mimickit"),
        (r"(dance1_subject2)", ""),
    ]:
        m = re.search(pat, run_name)
        if m:
            stem = m.group(1)
            cand = os.path.join(MOTIONS, sub, stem + ".npz") if sub else os.path.join(MOTIONS, stem + ".npz")
            if os.path.isfile(cand):
                return cand
    # 3) motion embedded in policy.pt (fallback verification path)
    raise FileNotFoundError(f"no motion npz found for {run_name}")


def export_run(run_dir: str, force: bool) -> str:
    run_name = os.path.basename(os.path.normpath(run_dir))
    out_dir = os.path.join(run_dir, "exported")
    out_path = os.path.join(out_dir, "policy.onnx")
    if os.path.isfile(out_path) and not force:
        return f"[skip] {run_name} (exported/policy.onnx exists)"

    ckpts = sorted(
        glob.glob(os.path.join(run_dir, "model_*.pt")),
        key=lambda p: int(re.search(r"model_(\d+)", p).group(1)),
    )
    policy_pt = os.path.join(run_dir, "policy.pt")
    # Prefer rsl_rl checkpoints (30-body npz path). TorchScript policy.pt is fallback only.
    if ckpts:
        ckpt_path = ckpts[-1]
    elif os.path.isfile(policy_pt):
        ckpt_path = policy_pt
    else:
        return f"[fail] {run_name}: no checkpoint"

    motion_path = find_motion(run_dir, run_name)
    motion = dict(np.load(motion_path, allow_pickle=True))

    if ckpt_path.endswith("policy.pt"):
        # TorchScript: rebuild normalizer+actor from buffers, motion from embedded arrays
        ts = torch.jit.load(ckpt_path, map_location="cpu")
        sd = {k: v.clone() for k, v in ts.state_dict().items() if k.startswith("policy.")}
        sd = {k[len("policy.") :]: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if k.startswith("mlp.") or k.startswith("obs_normalizer.")}
        actor = NormalizedActor(sd)
        motion = {
            "joint_pos": ts.joint_pos.numpy(),
            "joint_vel": ts.joint_vel.numpy(),
            "body_pos_w": ts.body_pos_w.numpy(),
            "body_quat_w": ts.body_quat_w.numpy(),
            "body_lin_vel_w": ts.body_lin_vel_w.numpy(),
            "body_ang_vel_w": ts.body_ang_vel_w.numpy(),
        }
        model = MotionModelIdentity(actor, motion)
        motion_path = ckpt_path  # for log line
    else:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["actor_state_dict"]
        actor = NormalizedActor(sd)
        model = MotionModel(actor, motion)

    os.makedirs(out_dir, exist_ok=True)
    obs = torch.zeros(1, int(sd["mlp.0.weight"].shape[1]))
    ts_in = torch.zeros(1, 1)
    torch.onnx.export(
        model, (obs, ts_in), out_path,
        export_params=True, opset_version=11,
        input_names=["obs", "time_step"],
        output_names=["actions", "joint_pos", "joint_vel", "body_pos_w",
                      "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
        dynamic_axes={},
    )

    # metadata (same keys as training exporter)
    meta = {
        "run_path": run_name,
        "joint_names": JOINT_NAMES,
        "joint_stiffness": [1.0] * N,
        "joint_damping": [0.0] * N,
        "default_joint_pos": DEFAULT_JOINT_POS,
        "command_names": "motion",
        "observation_names": OBS_NAMES,
        "observation_history_lengths": [1] * len(OBS_NAMES),
        "action_scale": ACTION_SCALE,
        "anchor_body_name": ANCHOR,
        "body_names": BODY_NAMES,
    }
    m = onnx.load(out_path)
    for k, v in meta.items():
        e = onnx.StringStringEntryProto()
        e.key, e.value = k, v if isinstance(v, str) else csv_str(v)
        m.metadata_props.append(e)
    onnx.save(m, out_path)

    T = motion["joint_pos"].shape[0]
    return f"[ ok ] {run_name}: {os.path.basename(ckpt_path)} + {os.path.basename(motion_path)} (T={T}) -> exported/policy.onnx"


class MotionModelIdentity(MotionModel):
    """For motion dicts whose body arrays are already 14-body tracking order."""

    def __init__(self, actor, motion_npz):
        # bypass the NPZ_BODY_IDX slicing by padding a mapping-friendly dict
        full = {
            "joint_pos": motion_npz["joint_pos"],
            "joint_vel": motion_npz["joint_vel"],
            "body_pos_w": motion_npz["body_pos_w"],
            "body_quat_w": motion_npz["body_quat_w"],
            "body_lin_vel_w": motion_npz["body_lin_vel_w"],
            "body_ang_vel_w": motion_npz["body_ang_vel_w"],
        }
        nn.Module.__init__(self)
        self.policy = actor
        for name in ("joint_pos", "joint_vel"):
            self.register_buffer(name, torch.tensor(np.asarray(full[name], np.float32)))
        for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            self.register_buffer(name, torch.tensor(np.asarray(full[name], np.float32)))


def main() -> None:
    force = "--force" in sys.argv
    # Optional positional log dirs (default: g1_tracking). Example:
    #   python scripts/export_g1_tracking_onnx.py logs/rsl_rl/g1_fall_recovery
    #   python scripts/export_g1_tracking_onnx.py --force logs/rsl_rl/g1_fall_recovery
    dirs = [
        os.path.abspath(a)
        for a in sys.argv[1:]
        if a != "--force" and not a.startswith("-")
    ]
    if not dirs:
        dirs = [TRACKING_DIR]
    for tracking_dir in dirs:
        runs = sorted(d for d in glob.glob(os.path.join(tracking_dir, "*/")) if os.path.isdir(d))
        print(f"found {len(runs)} runs in {tracking_dir}")
        for run_dir in runs:
            try:
                print(export_run(run_dir, force), flush=True)
            except Exception as e:
                print(
                    f"[fail] {os.path.basename(os.path.normpath(run_dir))}: {e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
