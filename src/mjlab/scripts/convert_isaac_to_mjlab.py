"""Convert Isaac(Sim/Lab)-ordered G1 motion .npz to mjlab joint order.

Input npz (IsaacLab convention, e.g. retargeted OmniControl / PhysX FK):
  fps, joint_pos (T,29), joint_vel, body_pos_w (T,30,3),
  body_quat_w (T,30,4), body_lin_vel_w, body_ang_vel_w

IsaacLab G1 joints follow the PhysX/USD BFS tree order (L/R interleaved):
  L/R hip_pitch, waist_yaw, L/R hip_roll, waist_roll, L/R hip_yaw, ...
Body order is interleaved too. mjlab uses the SDK order (left leg, right
leg, waist, left arm, right arm). Joint values and quaternions (wxyz, Z-up)
are compatible — only the ordering differs.

What this script does:
  1. Permute joint_pos / joint_vel columns: isaac_col[k] -> mjlab order.
  2. Re-run MuJoCo FK per frame (root from body_pos_w/body_quat_w[:,0],
     which need no conversion) to rebuild body_pos_w / body_quat_w /
     body_lin_vel_w / body_ang_vel_w in mjlab body order (pelvis, L leg,
     R leg, waist, L arm, R arm).
  3. Save as mjlab npz, same keys as convert_chunwan.py.

Usage:
  uv run python -m mjlab.scripts.convert_isaac_to_mjlab \
      --input motion.npz --output motions/g1_motion_long.npz
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import Entity

from .convert_100style import _build_g1_model, _finite_diff, _quat_diff_ang_vel
from .convert_mimickit import G1_JOINT_NAMES

# IsaacLab / PhysX BFS joint order (npz column -> joint name).
ISAAC_G1_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)

# mjlab joint k takes Isaac npz column PERM[k].
PERM: tuple[int, ...] = tuple(
  ISAAC_G1_JOINT_NAMES.index(name) for name in G1_JOINT_NAMES
)


def convert_npz(
  in_path: Path,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addrs: list[int],
  body_ids: list[int],
) -> dict[str, np.ndarray]:
  d = np.load(in_path)
  joint_pos = d["joint_pos"][:, PERM].astype(np.float32)
  joint_vel = d["joint_vel"][:, PERM].astype(np.float32)
  root_pos = d["body_pos_w"][:, 0, :].astype(np.float64)
  root_quat = d["body_quat_w"][:, 0, :].astype(np.float64)
  root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
  fps = float(d["fps"][0])
  dt = 1.0 / fps

  t = joint_pos.shape[0]
  body_pos = np.zeros((t, len(body_ids), 3), dtype=np.float32)
  body_quat = np.zeros((t, len(body_ids), 4), dtype=np.float32)
  for i in tqdm(range(t), desc=in_path.name, unit="frame", leave=False):
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos[i]
    data.qpos[3:7] = root_quat[i]
    for j, adr in enumerate(joint_qpos_addrs):
      data.qpos[adr] = joint_pos[i, j]
    mujoco.mj_forward(model, data)
    for bi, bid in enumerate(body_ids):
      body_pos[i, bi] = data.xpos[bid]
      body_quat[i, bi] = data.xquat[bid]

  body_lin_vel = _finite_diff(body_pos.reshape(t, -1), dt).reshape(
    t, len(body_ids), 3
  )
  body_ang_vel = np.stack(
    [_quat_diff_ang_vel(body_quat[:, bi], dt) for bi in range(len(body_ids))],
    axis=1,
  ).astype(np.float32)

  return {
    "fps": np.array([fps], dtype=np.float64),
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "body_lin_vel_w": body_lin_vel.astype(np.float32),
    "body_ang_vel_w": body_ang_vel,
  }


def main(
  input: str = "motion.npz",
  output: str = "motions/g1_motion_long.npz",
  overwrite: bool = False,
) -> None:
  """Convert one Isaac-ordered G1 npz to mjlab order."""
  in_path, out_path = Path(input), Path(output)
  if not in_path.exists():
    raise FileNotFoundError(in_path)
  if out_path.exists() and not overwrite:
    raise FileExistsError(f"{out_path} exists (use --overwrite)")
  out_path.parent.mkdir(parents=True, exist_ok=True)

  model, joint_qpos_addrs, body_ids = _build_g1_model()
  data = mujoco.MjData(model)

  motion = convert_npz(
    in_path=in_path,
    model=model,
    data=data,
    joint_qpos_addrs=joint_qpos_addrs,
    body_ids=body_ids,
  )
  np.savez_compressed(out_path, **motion)
  print(f"[OK] {in_path} -> {out_path} ({motion['joint_pos'].shape[0]} frames)")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
