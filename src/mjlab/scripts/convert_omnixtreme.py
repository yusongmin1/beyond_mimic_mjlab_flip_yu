"""Convert OmniXtreme long G1 npz to per-motion mjlab npz clips.

Source npz (7468 frames @ 50 Hz) is already in mjlab G1 joint order with
wxyz quaternions — verified by rendering; joints are kept as-is.

This script:
  1. Re-runs MuJoCo FK to regenerate body_pos_w / body_quat_w in mjlab
     body order, plus finite-diff velocities.
  2. Segments the long sequence into per-motion clips at low-energy
     boundaries and writes one npz per motion.
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

from .convert_100style import _finite_diff, _quat_diff_ang_vel
from .render_g1_motions import G1_JOINT_NAMES


def build_model() -> tuple[mujoco.MjModel, list[int], list[int]]:
  robot = Entity(get_g1_robot_cfg())
  while robot.spec.keys:
    robot.spec.delete(robot.spec.keys[0])
  model = robot.spec.compile()
  addrs = []
  for name in G1_JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    addrs.append(int(model.jnt_qposadr[jid]))
  body_ids = list(range(1, model.nbody))
  return model, addrs, body_ids


def convert(
  src: Path, model: mujoco.MjModel, data: mujoco.MjData,
  joint_qpos_addrs: list[int], body_ids: list[int],
) -> dict[str, np.ndarray]:
  d = np.load(src)
  fps = float(d["fps"][0])
  joint_pos = d["joint_pos"].astype(np.float32)
  joint_vel = d["joint_vel"].astype(np.float32)
  root_pos = d["body_pos_w"][:, 0, :].astype(np.float64)
  root_quat = d["body_quat_w"][:, 0, :].astype(np.float64)
  root_quat /= np.linalg.norm(root_quat, axis=1, keepdims=True)

  t = joint_pos.shape[0]
  body_pos = np.zeros((t, len(body_ids), 3), dtype=np.float32)
  body_quat = np.zeros((t, len(body_ids), 4), dtype=np.float32)
  for i in range(t):
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos[i]
    data.qpos[3:7] = root_quat[i]
    for j, adr in enumerate(joint_qpos_addrs):
      data.qpos[adr] = joint_pos[i, j]
    mujoco.mj_forward(model, data)
    for bi, bid in enumerate(body_ids):
      body_pos[i, bi] = data.xpos[bid]
      body_quat[i, bi] = data.xquat[bid]

  # Root z is kept as-is: the source renders correctly already.
  dt = 1.0 / fps
  joint_vel = _finite_diff(joint_pos, dt).astype(np.float32)
  body_lin_vel = _finite_diff(
    body_pos.reshape(t, -1), dt).reshape(t, len(body_ids), 3)
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


def segment(
  motion: dict[str, np.ndarray], fps: float,
  min_pause_s: float = 0.25, min_len_s: float = 1.0,
  smooth_s: float = 0.15, rel_thresh: float = 0.4, merge_gap_s: float = 0.4,
) -> list[tuple[int, int]]:
  """Split at smoothed-energy pauses; returns (start, end) index pairs.

  A pause = smoothed energy (joint speed + root speed) below
  ``rel_thresh * median`` for at least ``min_pause_s``. Pauses closer than
  ``merge_gap_s`` are treated as one; the cut is the pause midpoint.
  """
  speed = np.linalg.norm(motion["joint_vel"], axis=1)
  dt = 1.0 / fps
  root_speed = np.linalg.norm(
    _finite_diff(motion["body_pos_w"][:, 0], dt), axis=1)
  energy = speed + root_speed

  k = max(int(smooth_s * fps), 1)
  energy = np.convolve(energy, np.ones(k) / k, mode="same")
  low = energy < rel_thresh * np.median(energy)

  # Runs of consecutive low-energy frames.
  edges = np.diff(low.astype(int))
  starts = np.where(edges == 1)[0] + 1
  ends = np.where(edges == -1)[0] + 1
  if low[0]:
    starts = np.r_[0, starts]
  if low[-1]:
    ends = np.r_[ends, len(low)]

  min_pause = int(min_pause_s * fps)
  pauses = [
    (s, e) for s, e in zip(starts, ends)
    if e - s >= min_pause
  ]
  # Merge pauses separated by less than merge_gap_s.
  merge_gap = int(merge_gap_s * fps)
  merged: list[list[int]] = []
  for s, e in pauses:
    if merged and s - merged[-1][1] < merge_gap:
      merged[-1][1] = e
    else:
      merged.append([s, e])
  # Cut at each pause midpoint.
  cuts = [0] + [(s + e) // 2 for s, e in merged] + [len(energy)]

  min_len = int(min_len_s * fps)
  return [
    (s, e) for s, e in zip(cuts[:-1], cuts[1:]) if e - s >= min_len
  ]


def main(
  input_npz: str = "motions/g1_motion_long.npz",
  output_dir: str = "motions/omnixtreme",
  min_pause_s: float = 0.25,
) -> None:
  """Convert long npz and split into per-motion clips."""
  src = Path(input_npz)
  out_root = Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)

  model, addrs, body_ids = build_model()
  data = mujoco.MjData(model)
  motion = convert(src, model, data, addrs, body_ids)
  fps = float(motion["fps"][0])

  # Full converted sequence.
  np.savez_compressed(out_root / "g1_motion_long.npz", **motion)

  segs = segment(motion, fps, min_pause_s=min_pause_s)
  for k, (s, e) in enumerate(tqdm(segs, desc="omnixtreme", unit="clip")):
    clip = {key: arr[s:e] for key, arr in motion.items()}
    if clip["fps"].shape[0] == 0:  # guard: never emit empty fps
      clip["fps"] = np.array([fps], dtype=np.float64)
    np.savez_compressed(
      out_root / f"g1_motion_long_{k:02d}.npz", **clip)
  print(
    f"[INFO]: full + {len(segs)} clips -> {out_root.resolve()} "
    f"(boundaries: {segs})"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
