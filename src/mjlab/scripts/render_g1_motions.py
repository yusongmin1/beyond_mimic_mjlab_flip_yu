"""Render G1 motion .npz files to MP4 videos with MuJoCo offscreen renderer.

Replays joint_pos trajectories frame-by-frame through MuJoCo forward
kinematics and writes one MP4 per clip. Works for any G1 npz under
``motions/`` (mimickit / chunwan / amass_hard / 100style formats — all share
29 joints in mjlab G1 order).
"""

from __future__ import annotations

from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
import tyro

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import Entity

G1_JOINT_NAMES = (
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
)


def build_g1_model(width: int = 640, height: int = 360) -> tuple[mujoco.MjModel, list[int]]:
  """Compile standalone G1 model with a ground plane; return model + addrs."""
  robot = Entity(get_g1_robot_cfg())
  while robot.spec.keys:
    robot.spec.delete(robot.spec.keys[0])
  spec = robot.spec
  # Add a ground plane at z=0 so foot clearance is visible in videos.
  spec.worldbody.add_geom(
    name="ground_plane",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),  # Infinite plane (x/y halfsize 0).
    rgba=(0.55, 0.6, 0.65, 1.0),
  )
  model = spec.compile()
  model.vis.global_.offwidth = width
  model.vis.global_.offheight = height
  joint_qpos_addrs: list[int] = []
  for name in G1_JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise ValueError(f"Joint '{name}' not found in G1 model.")
    joint_qpos_addrs.append(int(model.jnt_qposadr[jid]))
  return model, joint_qpos_addrs


def render_npz(
  npz_path: Path,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addrs: list[int],
  out_path: Path,
  width: int,
  height: int,
  camera: str,
  follow: bool,
  every_nth: int,
) -> None:
  d = np.load(npz_path)
  joint_pos = d["joint_pos"]
  root_pos = d["body_pos_w"][:, 0, :]
  root_quat = d["body_quat_w"][:, 0, :]
  fps = float(d["fps"][0]) / every_nth

  renderer = mujoco.Renderer(model, height=height, width=width)
  cam = mujoco.MjvCamera()
  frames: list[np.ndarray] = []

  for i in range(0, joint_pos.shape[0], every_nth):
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos[i]
    data.qpos[3:7] = root_quat[i]
    for j, adr in enumerate(joint_qpos_addrs):
      data.qpos[adr] = joint_pos[i, j]
    mujoco.mj_forward(model, data)

    if follow:
      # Chase camera: fixed offset behind/above, lookat follows the root.
      cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      cam.lookat[:] = root_pos[i]
      cam.distance, cam.azimuth, cam.elevation = 4.0, 130.0, -15.0
      renderer.update_scene(data, camera=cam)
    else:
      renderer.update_scene(data, camera=camera)
    frames.append(renderer.render())

  out_path.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(out_path), frames, fps=round(fps))
  renderer.close()


def main(
  input_dir: str = "motions",
  output_dir: str = "videos/g1_motions",
  width: int = 1280,
  height: int = 720,
  camera: str = "track",
  follow: bool = True,
  every_nth: int = 1,
  skip_existing: bool = True,
  pattern: str = "*.npz",
) -> None:
  """Render all G1 motion npz under input_dir (recursively) to MP4.

  Only clips with 29 joints are rendered (G1); others are skipped.
  Output mirrors the input subdirectory layout under output_dir.

  Args:
    input_dir: Root directory to search for .npz motions.
    output_dir: Directory to write .mp4 videos.
    width: Video width in pixels.
    height: Video height in pixels.
    camera: MuJoCo camera name (use "track" with follow=True).
    follow: Track the root body with a fixed-offset chase camera.
    every_nth: Render every Nth frame (1 = full fps).
    skip_existing: Skip videos that already exist.
    pattern: Glob pattern for npz files.
  """
  root = Path(input_dir)
  out_root = Path(output_dir)

  model, joint_qpos_addrs = build_g1_model(width=width, height=height)
  data = mujoco.MjData(model)

  npzs = sorted(p for p in root.rglob(pattern))
  rendered = skipped = passed = 0
  for npz_path in npzs:
    try:
      n_joints = np.load(npz_path)["joint_pos"].shape[1]
    except Exception as e:  # noqa: BLE001
      print(f"[WARN] {npz_path}: cannot read ({e}); skipped")
      passed += 1
      continue
    if n_joints != 29:
      passed += 1
      continue

    rel = npz_path.relative_to(root)
    out_path = out_root / rel.with_suffix(".mp4")
    if skip_existing and out_path.exists():
      skipped += 1
      continue
    print(f"[RENDER] {npz_path} -> {out_path}")
    render_npz(
      npz_path=npz_path,
      model=model,
      data=data,
      joint_qpos_addrs=joint_qpos_addrs,
      out_path=out_path,
      width=width,
      height=height,
      camera=camera,
      follow=follow,
      every_nth=every_nth,
    )
    rendered += 1

  print(
    f"[INFO]: rendered={rendered} skipped={skipped} non-g1={passed} "
    f"out={out_root.resolve()}"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
