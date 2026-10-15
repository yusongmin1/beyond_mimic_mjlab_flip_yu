"""Render G1 motion .npz files to MP4 videos with MuJoCo offscreen renderer.

Replays joint_pos trajectories frame-by-frame through the FULL tracking
scene (same terrain entity, lighting, and camera as convert_mimickit's
--render output, e.g. motions/g1_cartwheel2.mp4): bright directional light,
blue checkered floor, close ASSET_ROOT-tracking camera
(distance=2.0, elevation=-5, azimuth=20).
"""

from __future__ import annotations

from pathlib import Path

import mediapy as media
import numpy as np
import torch
import tyro

import mjlab
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

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


def _make_scene(output_fps: int, device: str) -> tuple[Scene, Simulation]:
  """Same tracking scene as convert_mimickit._make_scene (g1 branch)."""
  from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
  )

  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return scene, sim


def render_npz(
  npz_path: Path,
  sim: Simulation,
  scene: Scene,
  renderer: OffscreenRenderer,
  out_path: Path,
  every_nth: int,
) -> None:
  d = np.load(npz_path)
  joint_pos = d["joint_pos"]
  root_pos = d["body_pos_w"][:, 0, :]
  root_quat = d["body_quat_w"][:, 0, :]
  fps = float(d["fps"][0]) / every_nth

  robot_entity = scene["robot"]
  robot_joint_indexes = robot_entity.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]

  frames: list[np.ndarray] = []
  scene.reset()
  for i in range(0, joint_pos.shape[0], every_nth):
    root_states = robot_entity.data.default_root_state.clone()
    root_states[:, 0:3] = torch.as_tensor(root_pos[i], device=sim.device)
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = torch.as_tensor(root_quat[i], device=sim.device)
    robot_entity.write_root_state_to_sim(root_states)

    jp = robot_entity.data.default_joint_pos.clone()
    jv = robot_entity.data.default_joint_vel.clone()
    jp[:, robot_joint_indexes] = torch.as_tensor(joint_pos[i], device=sim.device)
    robot_entity.write_joint_state_to_sim(jp, jv)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    renderer.update(sim.data)
    frames.append(renderer.render())

  out_path.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(out_path), frames, fps=round(fps))


def main(
  input_dir: str = "motions",
  output_dir: str = "videos/g1_motions",
  width: int = 640,
  height: int = 480,
  every_nth: int = 1,
  skip_existing: bool = True,
  pattern: str = "*.npz",
  device: str = "cuda:0",
) -> None:
  """Render all G1 motion npz under input_dir (recursively) to MP4.

  Uses the full tracking scene renderer (same visuals as the
  g1_cartwheel2.mp4 reference): bright light, blue checkered floor,
  close tracking camera.

  Args:
    input_dir: Root directory to search for .npz motions.
    output_dir: Directory to write .mp4 videos.
    width: Video width in pixels.
    height: Video height in pixels.
    every_nth: Render every Nth frame (1 = full fps).
    skip_existing: Skip videos that already exist.
    pattern: Glob pattern for npz files.
    device: Compute device for the scene.
  """
  root = Path(input_dir)
  out_root = Path(output_dir)

  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA unavailable, falling back to CPU.")
    device = "cpu"

  scene, sim = _make_scene(output_fps=50, device=device)
  viewer_cfg = ViewerConfig(
    height=height,
    width=width,
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    distance=2.0,
    elevation=-5.0,
    azimuth=20,
  )
  renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
  renderer.initialize()

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
      sim=sim,
      scene=scene,
      renderer=renderer,
      out_path=out_path,
      every_nth=every_nth,
    )
    rendered += 1

  renderer.close()
  print(
    f"[INFO]: rendered={rendered} skipped={skipped} non-g1={passed} "
    f"out={out_root.resolve()}"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
