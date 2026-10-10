"""Convert chunwan-flip G1 CSVs (SDK joint order) to mjlab .npz motions.

Input CSV format (no header, one frame per row):
  cols 0-2   root position x, y, z (Z-up world)
  cols 3-6   root quaternion qx, qy, qz, qw
  cols 7-35  29 joint positions (rad), Unitree G1 SDK order

The G1 SDK order matches the mjlab G1 joint order used in convert_mimickit
(hip/ankle legs, waist, arms), so joints map 1:1.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg

from .convert_100style import (
  _build_g1_model,
  _finite_diff,
  _quat_diff_ang_vel,
)


def convert_csv(
  csv_path: Path,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addrs: list[int],
  body_ids: list[int],
  input_fps: float,
  output_fps: float,
) -> dict[str, np.ndarray]:
  raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
  if raw.ndim == 1:
    raw = raw[None, :]
  if raw.shape[1] != 36:
    raise ValueError(
      f"{csv_path.name}: expected 36 columns (3 pos + 4 quat + 29 joints), "
      f"got {raw.shape[1]}"
    )

  t_in = raw.shape[0]
  # Resample input_fps → output_fps: frame count scales by output/input.
  t_out = int(round((t_in - 1) * output_fps / input_fps)) + 1
  src_idx = np.arange(t_out) * (t_in - 1) / max(t_out - 1, 1)
  i0 = np.floor(src_idx).astype(int).clip(0, t_in - 1)
  i1 = np.ceil(src_idx).astype(int).clip(0, t_in - 1)
  w = (src_idx - i0)[:, None]

  def lerp_rows(arr: np.ndarray) -> np.ndarray:
    return (arr[i0] * (1 - w) + arr[i1] * w).astype(np.float32)

  root_pos = lerp_rows(raw[:, 0:3])
  quat_xyzw = lerp_rows(raw[:, 3:7])
  # Ensure stable hemisphere before slerp-ish blend, then normalize.
  sign = np.where((quat_xyzw * np.roll(quat_xyzw, 1, axis=0)).sum(-1, keepdims=True) < 0, -1.0, 1.0)
  sign[0] = 1.0
  quat_xyzw = quat_xyzw * sign
  quat_xyzw /= np.linalg.norm(quat_xyzw, axis=-1, keepdims=True).clip(min=1e-8)
  root_quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]  # xyzw → wxyz
  joint_pos = lerp_rows(raw[:, 7:36])

  dt = 1.0 / output_fps
  joint_vel = _finite_diff(joint_pos, dt)

  # Ground alignment: GMR retarget leaves a per-clip constant vertical
  # offset (SMPL-X pelvis vs G1 pelvis mismatch). Robust baseline:
  # mean height of contact frames (lowest foot origin < 0.15 m) when such
  # frames are common, else the global minimum. Align baseline to
  # FOOT_ORIGIN_HEIGHT so the true sole (≈0.035 m below origin) hits z=0.
  FOOT_ORIGIN_HEIGHT = 0.0353  # ankle_roll body origin height at stance.
  foot_body_ids = [5, 6, 11, 12]  # ankle/foot bodies in G1 FK order.
  probe = np.linspace(0, joint_pos.shape[0] - 1, num=25).astype(int)
  lows = []
  for k in probe:
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos[k]
    data.qpos[3:7] = root_quat_wxyz[k]
    for j, adr in enumerate(joint_qpos_addrs):
      data.qpos[adr] = joint_pos[k, j]
    mujoco.mj_forward(model, data)
    lows.append(min(data.xpos[bid][2] for bid in foot_body_ids))
  lows = np.asarray(lows)
  contact = lows[lows < 0.15]
  base = contact.mean() if len(contact) >= 0.3 * len(lows) else lows.min()
  offset = base - FOOT_ORIGIN_HEIGHT
  if offset > 0:
    root_pos[:, 2] -= offset

  t = joint_pos.shape[0]
  body_pos = np.zeros((t, len(body_ids), 3), dtype=np.float32)
  body_quat = np.zeros((t, len(body_ids), 4), dtype=np.float32)

  for i in range(t):
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos[i]
    data.qpos[3:7] = root_quat_wxyz[i]
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
    "fps": np.array([output_fps], dtype=np.float64),
    "joint_pos": joint_pos,
    "joint_vel": joint_vel.astype(np.float32),
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "body_lin_vel_w": body_lin_vel.astype(np.float32),
    "body_ang_vel_w": body_ang_vel,
  }


def main(
  input_dir: str = "/home/zju/Downloads/chunwan-flip-main/data",
  output_dir: str = "motions/chunwan",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  skip_existing: bool = True,
) -> None:
  """Batch-convert chunwan-flip CSVs to mjlab .npz under output_dir."""
  root = Path(input_dir)
  out_root = Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)

  csvs = sorted(root.glob("*.csv"))
  if not csvs:
    raise FileNotFoundError(f"No CSVs found under {root}")

  model, joint_qpos_addrs, body_ids = _build_g1_model()
  data = mujoco.MjData(model)

  converted = 0
  skipped = 0
  for csv_path in tqdm(csvs, desc="chunwan", unit="clip"):
    out_path = out_root / f"{csv_path.stem}.npz"
    if skip_existing and out_path.exists():
      skipped += 1
      continue
    motion = convert_csv(
      csv_path=csv_path,
      model=model,
      data=data,
      joint_qpos_addrs=joint_qpos_addrs,
      body_ids=body_ids,
      input_fps=input_fps,
      output_fps=output_fps,
    )
    np.savez_compressed(out_path, **motion)
    converted += 1

  print(f"[INFO]: Converted {converted}, skipped {skipped}, out={out_root.resolve()}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
