#!/usr/bin/env python
"""Render a fallen-init ``.pth`` pool to an MP4 (one pose per frame).

Example::

  MUJOCO_GL=egl python scripts/viz_fallen_init_pth.py \\
    src/mjlab/tasks/tracking/assets/robot_init_states_g1_user.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MJLAB_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_XML = os.path.join(
  _MJLAB_ROOT,
  "src",
  "mjlab",
  "asset_zoo",
  "robots",
  "unitree_g1",
  "xmls",
  "g1_scene.xml",
)

JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]


def main() -> None:
  parser = argparse.ArgumentParser(description="Visualize fallen-init .pth as MP4.")
  parser.add_argument(
    "pth",
    type=str,
    nargs="?",
    default=os.path.join(
      _MJLAB_ROOT,
      "src/mjlab/tasks/tracking/assets/robot_init_states_g1_user.pth",
    ),
    help="Path to robot_init_states_*.pth",
  )
  parser.add_argument("--out", type=str, default=None, help="Output mp4 path.")
  parser.add_argument("--xml", type=str, default=_DEFAULT_XML)
  parser.add_argument("--fps", type=float, default=12.0, help="Video FPS.")
  parser.add_argument("--hold", type=int, default=1, help="Repeat each pose N frames.")
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument(
    "--every",
    type=int,
    default=1,
    help="Subsample poses (every N-th). Default: all.",
  )
  parser.add_argument(
    "--center_xy",
    action="store_true",
    default=True,
    help="Snap root x,y to 0 for a fixed camera (default on).",
  )
  parser.add_argument("--no_center_xy", action="store_true")
  args = parser.parse_args()
  if args.no_center_xy:
    args.center_xy = False

  if not os.path.isfile(args.pth):
    raise SystemExit(f"[ERROR] not found: {args.pth}")
  out = args.out or os.path.splitext(os.path.abspath(args.pth))[0] + ".mp4"

  import mujoco

  data_pt = torch.load(args.pth, map_location="cpu", weights_only=False)
  dof_pos = data_pt["dof_pos"].numpy().astype(np.float64)
  if "robot_root_states_wxyz" in data_pt:
    root = data_pt["robot_root_states_wxyz"].numpy().astype(np.float64)
    quat_wxyz = root[:, 3:7]
  else:
    root = data_pt["robot_root_states_xyzw"].numpy().astype(np.float64)
    # xyzw -> wxyz
    q = root[:, 3:7]
    quat_wxyz = np.concatenate([q[:, 3:4], q[:, 0:3]], axis=1)
  pos = root[:, :3].copy()
  n = dof_pos.shape[0]
  if dof_pos.shape[1] != len(JOINT_NAMES):
    raise SystemExit(
      f"[ERROR] dof dim {dof_pos.shape[1]} != {len(JOINT_NAMES)}"
    )

  model = mujoco.MjModel.from_xml_path(args.xml)
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
  data = mujoco.MjData(model)
  free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
  free_adr = model.jnt_qposadr[free_jid]
  qpos_adr = []
  for name in JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise SystemExit(f"[ERROR] joint missing in xml: {name}")
    qpos_adr.append(model.jnt_qposadr[jid])
  qpos_adr = np.asarray(qpos_adr)

  # Camera looking at origin from a slight elevation (fallen poses near ground).
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.lookat[:] = np.array([0.0, 0.0, 0.25])
  cam.distance = 2.4
  cam.elevation = -20.0
  cam.azimuth = 135.0

  renderer = mujoco.Renderer(model, height=args.height, width=args.width)

  try:
    import imageio.v2 as imageio
  except ImportError:
    import imageio

  indices = list(range(0, n, max(1, args.every)))
  print(
    f"[viz] {args.pth}: {n} poses -> {len(indices)} frames "
    f"(hold={args.hold}, fps={args.fps}) -> {out}"
  )

  writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8)
  try:
    for k, i in enumerate(indices):
      data.qpos[:] = 0.0
      data.qvel[:] = 0.0
      xy = pos[i, :2].copy()
      if args.center_xy:
        xy[:] = 0.0
      data.qpos[free_adr : free_adr + 3] = [xy[0], xy[1], pos[i, 2]]
      data.qpos[free_adr + 3 : free_adr + 7] = quat_wxyz[i]
      data.qpos[qpos_adr] = dof_pos[i]
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, camera=cam)
      frame = renderer.render()
      for _ in range(max(1, args.hold)):
        writer.append_data(frame)
      if (k + 1) % 50 == 0 or k + 1 == len(indices):
        print(f"  {k + 1}/{len(indices)}", flush=True)
  finally:
    writer.close()
    renderer.close()

  print(f"[ok] wrote {out}")


if __name__ == "__main__":
  main()
