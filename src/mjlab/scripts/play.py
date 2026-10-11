"""Script to play RL agent with RSL-RL."""

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# Hand-region bodies for contact printing (G1 wrist links; the rubber hand
# collision geoms were removed, wrist capsules are the remaining colliders).
HAND_BODY_PATTERN = r"(left|right)_wrist_(pitch|roll|yaw)_link"


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  keep_randomization: bool = False
  """If True, use training env cfg (domain randomization on) and default to 50 envs."""
  print_hand_contact: bool = False
  """Print hand/wrist-ground contact links and force magnitudes during play."""
  hand_contact_bodies: str = HAND_BODY_PATTERN
  """Regex of body names counted as 'hand region' for contact printing."""
  hand_contact_min_force: float = 1.0
  """Minimum contact force magnitude (N) to report."""
  export_torchscript: bool = False
  """If True, export TorchScript ``policy.pt`` after loading the checkpoint."""
  export_torchscript_path: str | None = None
  """Output path for TorchScript export. Default: ``<checkpoint_dir>/policy.pt``."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


class HandContactPrinter:
  """Wraps an env-like object; after each step prints hand-region ground contacts.

  Reads the 'hand_ground_contact' ContactSensor (body-mode primaries, one
  MuJoCo sensor per body) and prints per-body force magnitudes above the
  threshold. Passthrough for get_observations/step/reset/close; all other
  attribute access delegates to the wrapped env.
  """

  def __init__(
    self,
    env,
    body_names: list[str],
    min_force: float,
    print_interval_s: float = 0.5,
  ):
    self._env = env
    self._body_names = body_names
    self._min_force = min_force
    self._print_interval_s = print_interval_s
    self._last_print_time = 0.0

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  def get_observations(self):
    return self._env.get_observations()

  def reset(self):
    return self._env.reset()

  def close(self) -> None:
    self._env.close()

  def step(self, actions):
    ret = self._env.step(actions)
    try:
      self._print_hand_contacts()
    except Exception as e:  # noqa: BLE001 — printing must never kill play.
      print(f"[hand-contact] WARN: contact read failed: {e}")
    return ret

  def _print_hand_contacts(self) -> None:
    scene = self._env.unwrapped.scene
    sensor = scene["hand_ground_contact"]
    data = sensor.data
    if data.found is None or data.force is None:
      return

    found = data.found  # [B, N]
    force = data.force  # [B, N, 3]
    force_mag = torch.linalg.vector_norm(force, dim=-1)  # [B, N]

    # Only inspect env 0 (play focuses on the tracked env; printing all envs
    # would spam at 50 envs).
    hits = (found[0] > 0) & (force_mag[0] > self._min_force)
    if not bool(hits.any()):
      return

    now = time.time()
    if now - self._last_print_time < self._print_interval_s:
      return
    self._last_print_time = now

    parts = []
    for idx in torch.nonzero(hits, as_tuple=False).flatten().tolist():
      parts.append(
        f"{self._body_names[idx]}: {force_mag[0, idx].item():.1f} N"
      )
    print(f"[hand-contact] {'; '.join(parts)}")


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=not cfg.keep_randomization)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.keep_randomization:
    # Keep training-time DR, but allow long interactive sessions.
    env_cfg.episode_length_s = int(1e9)
    print("[INFO]: Domain randomization kept enabled")

  # Resolve num_envs: keep_randomization defaults to 50 so DR is visible.
  num_envs = cfg.num_envs
  if num_envs is None and cfg.keep_randomization:
    num_envs = 50
  if num_envs is not None:
    env_cfg.scene.num_envs = num_envs

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    elif cfg.wandb_run_path is None:
      # 未指定 checkpoint 时，按 run 目录名字母序、再按 checkpoint 编号排序，
      # 取最后一个（通常是最新时间戳 run 下、迭代数最大的 model_*.pt）。
      candidates = sorted(
        (p for p in log_root_path.glob("*/model_*.pt") if p.exists()),
        key=lambda p: (p.parent.name, f"{p.name:0>30}"),
      )
      if not candidates:
        raise FileNotFoundError(
          f"No checkpoints found under {log_root_path}. "
          "Provide --checkpoint-file or --wandb-run-path."
        )
      resume_path = candidates[-1]
      print(
        f"[INFO]: Auto-selected checkpoint (name-sorted): {resume_path} "
        f"({len(candidates)} available)"
      )
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )

  if cfg.print_hand_contact:
    # Resolve hand-region bodies against the robot entity and add a
    # body-vs-terrain ContactSensor (one primary per body → per-link force).
    entities = env_cfg.scene.entities or {}
    robot_names = [n for n, e in entities.items() if "robot" in n.lower()] or list(
      entities.keys()
    )
    if not robot_names:
      raise ValueError("--print-hand-contact requires a robot entity in the scene.")
    robot_name = "robot" if "robot" in entities else robot_names[0]
    robot_spec = entities[robot_name].spec_fn()
    all_bodies = [b.name.split("/")[-1] for b in robot_spec.bodies[1:]]
    import re as _re  # noqa: PLC0415

    hand_bodies = [b for b in all_bodies if _re.fullmatch(cfg.hand_contact_bodies, b)]
    if not hand_bodies:
      raise ValueError(
        f"--print-hand-contact: no bodies match '{cfg.hand_contact_bodies}'. "
        f"Available: {all_bodies}"
      )
    hand_sensor = ContactSensorCfg(
      name="hand_ground_contact",
      primary=ContactMatch(
        mode="body", pattern=tuple(hand_bodies), entity=robot_name
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
    )
    env_cfg.scene.sensors = tuple(env_cfg.scene.sensors or ()) + (hand_sensor,)
    print(
      f"[INFO]: Hand contact printing on: {hand_bodies} "
      f"(threshold {cfg.hand_contact_min_force} N)"
    )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  print(
    f"[INFO]: Playing with {env.num_envs} env(s); "
    "viewer overlay should show Env 1/N (zoom out if robots look stacked)."
  )

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if cfg.print_hand_contact:
    env = HandContactPrinter(
      env,
      body_names=hand_bodies,
      min_force=cfg.hand_contact_min_force,
    )

  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    if cfg.export_torchscript:
      from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

      if not isinstance(runner, MotionTrackingOnPolicyRunner):
        raise RuntimeError(
          "--export-torchscript is only supported for tracking runners."
        )
      assert resume_path is not None
      out = (
        Path(cfg.export_torchscript_path)
        if cfg.export_torchscript_path is not None
        else resume_path.parent / "policy.pt"
      )
      runner.export_policy_to_torchscript(str(out.parent), out.name)
      print(f"[INFO]: TorchScript exported to {out}")

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
