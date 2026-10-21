"""Fall-and-recovery training utilities (normal / fallen env splitting).

Ported and adapted from:
- AMP_mjlab ``src/tasks/amp_loco/mdp/{events,curriculums,terminations}.py``:
  delayed termination (recovery window), upward recovery assist force with a
  decay curriculum.
- KungFuAthleteBot ``unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py``:
  fallen-state initialization from a pre-collected ``.pth`` pool
  (``robot_init_states_*.pth``, diverse pelvis orientations near the ground).

Env grouping (AMP_mjlab-style two groups, re-rolled on every episode reset):
- 60% normal tracking envs (reference-state init, immediate terminations).
- 40% fall-recovery envs (``recovery_env_ratio=0.4``; re-sampled each reset).
  Within the recovery group, ``fallen_init_ratio`` (default 1.0) start lying
  on the ground from the ``.pth`` pool; if set below 1.0, the remainder
  start from the motion reference and only enter the get-up window after
  tracking failure.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBaseCfg
from mjlab.managers.termination_manager import TerminationManager
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class RecoveryTerminationManager(TerminationManager):
  """TerminationManager with a recovery window for a subset of envs.

  For recovery envs, a termination signal starts a counter instead of
  resetting immediately. The env is given ``max_recovery_steps`` env steps
  (the "get up" window). The reset signal is released only when the
  termination condition has persisted (cumulatively, via the counter) for
  the whole window. If the env recovers on its own (termination condition
  clears), the counter resets.

  The recovery set is NOT static: the mask is re-read from the motion
  command every step (``recovery_mask_fn``), which re-rolls the AMP-style
  60% normal / 40% fall-recovery split on every reset. Adapted from
  AMP_mjlab ``DelayedTerminationManager``.
  """

  def __init__(
    self,
    base: TerminationManager,
    recovery_mask_fn: Callable[[], torch.Tensor],
    max_recovery_steps: int,
  ) -> None:
    # Steal all internal state from the base manager (avoid re-init).
    self.__dict__.update(base.__dict__)
    self._recovery_mask_fn = recovery_mask_fn  # () -> (num_envs,) bool
    self._recovery_env_mask = torch.zeros(
      base.num_envs if hasattr(base, "num_envs") else 0,
      dtype=torch.bool,
    )
    self._recovery_counters = None  # Lazily created on first compute.
    # True on the single step the delayed-recovery window expires (failed get-up).
    # Cleared every compute; rewards read it after terminations run.
    self._recovery_window_expired = None
    self._max_recovery_steps = max_recovery_steps

  def compute(self) -> torch.Tensor:
    # Refresh the recovery set from the command (re-rolled every reset).
    mask = self._recovery_mask_fn()
    device = mask.device
    if (
      self._recovery_counters is None
      or self._recovery_counters.device != device
      or self._recovery_counters.shape != mask.shape
    ):
      self._recovery_counters = torch.zeros_like(mask, dtype=torch.long)
      self._recovery_window_expired = torch.zeros_like(mask)
    assert self._recovery_window_expired is not None
    self._recovery_window_expired.zero_()
    self._recovery_env_mask = mask
    # Envs that left the recovery group: drop their stale counters.
    self._recovery_counters[~mask] = 0

    dones = super().compute()  # fills _truncated_buf, _terminated_buf

    if self._max_recovery_steps <= 0:
      return dones

    # For recovery envs that just got a done signal, increment counter.
    rec_and_done = mask & dones
    self._recovery_counters[rec_and_done] += 1

    # Recovery envs whose counter hasn't reached threshold: suppress reset.
    not_ready = rec_and_done & (self._recovery_counters < self._max_recovery_steps)
    self._terminated_buf[not_ready] = False

    # Recovery envs whose counter reached threshold: allow reset, clear counter.
    # Flag expire for the reward term (counters are zeroed below, so rewards
    # cannot re-read counter==max on this frame).
    ready = rec_and_done & (self._recovery_counters >= self._max_recovery_steps)
    self._recovery_window_expired[ready] = True
    self._recovery_counters[ready] = 0

    # Clear counters for recovery envs that are NOT done (recovered on their own).
    self._recovery_counters[mask & ~dones] = 0

    return self._truncated_buf | self._terminated_buf


def install_recovery_termination(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
  recovery_env_ratio: float = 0.4,
  max_recovery_steps: int = 100,
) -> None:
  """Startup event: wrap the termination manager with a recovery window.

  The recovery set is dynamic: the manager re-reads the command-side
  ``is_recovery_env`` mask every step, and the command re-rolls the
  AMP-style two-group split on every reset. With
  ``recovery_env_ratio=0.4`` this yields 60% normal / 40% fall-recovery
  per reset (membership re-sampled each episode; no fixed env ids).
  Exhausting the delayed window (default 100 steps / 2 s; total down time
  with the 2 s tolerant phase ≈ 4 s) sets ``_recovery_window_expired`` for
  ``penalty_recovery_window_timeout``.
  """
  del env_ids  # Unused (startup).
  if not 0.0 <= recovery_env_ratio <= 1.0:
    raise ValueError(f"recovery_env_ratio must be in [0, 1], got {recovery_env_ratio}.")
  if recovery_env_ratio <= 0.0 or max_recovery_steps <= 0:
    return
  if isinstance(env.termination_manager, RecoveryTerminationManager):
    return

  def mask_fn() -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    mask = getattr(command, "is_recovery_env", None)
    if mask is None:
      return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return mask.bool()

  env.termination_manager = RecoveryTerminationManager(
    base=env.termination_manager,
    recovery_mask_fn=mask_fn,
    max_recovery_steps=max_recovery_steps,
  )
  print(
    f"[recovery] Installed RecoveryTerminationManager: dynamic mask from "
    f"'{command_name}' (~{int(recovery_env_ratio * 100)}% recovery envs, "
    f"re-rolled every reset) with a {max_recovery_steps}-step recovery window."
  )


class RecoveryFallenPool:
  """Singleton pool of fallen init states loaded from a ``.pth`` file.

  File format (KungFuAthleteBot convention):
  - ``dof_pos``: (N, num_dof) joint positions.
  - ``robot_root_states_xyzw``: (N, 13) [pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)].
  - ``robot_root_states_wxyz``: (N, 13) same with quat in wxyz order.

  The joint order of the pool matches the g1 MuJoCo XML order (verified
  identical between KungFuAthleteBot and this repo), i.e. the mjlab joint
  order used by the tracking MotionCommand.
  """

  _instance: RecoveryFallenPool | None = None
  _data: dict[str, torch.Tensor] | None = None
  _diverse_idxs: list[int] | None = None

  @classmethod
  def get(cls) -> RecoveryFallenPool:
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def init(self, init_pos_file: str, num_diverse: int = 2048) -> None:
    if not os.path.isfile(init_pos_file):
      raise FileNotFoundError(
        f"Fallen-init pool file not found: {init_pos_file}"
      )
    if self._data is not None:
      return
    import numpy as np

    data = torch.load(init_pos_file, map_location="cpu")
    self._data = {
      "dof_pos": data["dof_pos"].float(),
      "robot_root_states_xyzw": data["robot_root_states_xyzw"].float(),
    }
    quats_xyzw = self._data["robot_root_states_xyzw"][:, 3:7]
    self._diverse_idxs = self._select_most_diverse_quaternions(quats_xyzw, num_diverse)
    n = self._data["dof_pos"].shape[0]
    print(
      f"[recovery] Loaded fallen-init pool: {n} states ({len(self._diverse_idxs)} "
      f"diverse) from {init_pos_file}"
    )

  @staticmethod
  def _select_most_diverse_quaternions(
    quats: torch.Tensor, num_select: int
  ) -> list[int]:
    """Greedy farthest-point sampling on unit quaternions (KungFuAthleteBot)."""
    from math import floor

    n = quats.shape[0]
    if num_select >= n or num_select <= 1:
      return list(range(min(n, max(num_select, 1))))

    selected_idx = [floor(torch.randint(0, n, (1,)).item())]

    for _ in range(num_select - 1):
      remaining_idx = list(set(range(n)) - set(selected_idx))
      remaining_quats = quats[remaining_idx]  # (M,4)
      selected_quats = quats[selected_idx]  # (len(selected_idx),4)

      sim = torch.abs(remaining_quats @ selected_quats.T)  # |dot product|
      dist = 1 - sim
      min_dist, _ = dist.min(dim=1)

      next_idx_in_remaining = min_dist.argmax().item()
      selected_idx.append(remaining_idx[next_idx_in_remaining])

    return selected_idx

  def sample(
    self, num_samples: int, device: str, zero_velocity: bool = False
  ) -> dict[str, torch.Tensor]:
    """Sample fallen states from the diverse pool."""
    assert self._data is not None, "Call init() before sample()."
    pool = torch.tensor(self._diverse_idxs, dtype=torch.long)
    ids = pool[torch.randint(0, pool.numel(), (num_samples,))]
    root_xyzw = self._data["robot_root_states_xyzw"][ids].to(device)
    dof_pos = self._data["dof_pos"][ids].to(device)
    dof_vel = torch.zeros_like(dof_pos)
    if zero_velocity:
      root_xyzw = root_xyzw.clone()
      root_xyzw[:, 7:13] = 0.0
    return {"root_xyzw": root_xyzw, "dof_pos": dof_pos, "dof_vel": dof_vel}


class apply_recovery_assist_force:
  """Interval-every-step event: upward assist force for recovering envs.

  Register with ``mode="interval"`` and
  ``interval_range_s=(env.step_dt, env.step_dt)`` so it fires every env step.

  Adapted from AMP_mjlab ``apply_recovery_assist_force``, but default trigger
  is step 1 (immediately when failure is announced / delay window opens).
  When a recovery env's counter first hits ``trigger_recovery_steps``, sample
  ``apply_prob`` (default 0.8, same as training) and (if accepted) apply a
  constant world-+Z force for ``duration_steps`` env steps.

  ``current_force`` starts at ``initial_force`` (250 N). Training decays it via
  the ``recovery_assist_force`` curriculum; play keeps the initial value (no
  curriculum). At F == 0 no forces are applied.
  """

  @dataclass
  class VizCfg:
    """Arrow visualization for the active upward assist wrench."""

    force_rgba: tuple[float, float, float, float] = (0.15, 0.85, 0.35, 0.95)
    torque_rgba: tuple[float, float, float, float] = (0.95, 0.45, 0.15, 0.95)
    force_scale: float = 0.004  # meters per Newton
    torque_scale: float = 0.02  # meters per N·m
    width: float = 0.02
    min_force: float = 1.0
    min_torque: float = 0.5

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._viz_cfg: apply_recovery_assist_force.VizCfg = cfg.params.get(
      "viz_cfg", apply_recovery_assist_force.VizCfg()
    )
    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )
    # Curriculum writes this; play keeps the initial value (no decay).
    self.current_force: float = float(cfg.params.get("initial_force", 250.0))
    self._duration_steps: int = int(cfg.params.get("duration_steps", 25))
    self._active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    self._remaining = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    trigger_recovery_steps: int = 1,
    apply_prob: float = 0.8,
    initial_force: float = 250.0,
    duration_steps: int = 25,
    viz_cfg: VizCfg | None = None,
  ) -> None:
    del env_ids, asset_cfg, initial_force, duration_steps, viz_cfg

    tm = env.termination_manager
    if not isinstance(tm, RecoveryTerminationManager):
      return

    counters = tm._recovery_counters
    recovery_mask = tm._recovery_env_mask
    if counters is None:
      return

    # Clear when recovery ends (recovered or window expired and reset).
    clear = self._active & (counters == 0)
    if clear.any():
      self._clear_force(clear.nonzero(as_tuple=False).squeeze(-1))

    # Tick duration for still-active assists.
    if self._active.any():
      self._remaining[self._active] -= 1
      expired = self._active & (self._remaining <= 0)
      if expired.any():
        self._clear_force(expired.nonzero(as_tuple=False).squeeze(-1))

    if self.current_force <= 0.0:
      if self._active.any():
        self.clear_all()
      return

    # Decide once when the counter first reaches the trigger step.
    candidates = recovery_mask & (counters == trigger_recovery_steps) & (~self._active)
    if candidates.any():
      cand_ids = candidates.nonzero(as_tuple=False).squeeze(-1)
      accept = torch.rand(len(cand_ids), device=self._device) < apply_prob
      trigger_ids = cand_ids[accept]
      if len(trigger_ids) > 0:
        self._active[trigger_ids] = True
        self._remaining[trigger_ids] = self._duration_steps

    if not self._active.any():
      return

    active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
    self._write_upward_force(active_ids, self.current_force)

  def _write_upward_force(self, env_ids: torch.Tensor, force_n: float) -> None:
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    forces = torch.zeros((n, self._num_bodies, 3), device=self._device)
    torques = torch.zeros_like(forces)
    forces[..., 2] = force_n  # world +Z
    self._asset.write_external_wrench_to_sim(
      forces, torques, env_ids=env_ids, body_ids=self._body_ids
    )

  def _clear_force(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    zeros = torch.zeros((len(env_ids), self._num_bodies, 3), device=self._device)
    self._asset.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
    )
    self._active[env_ids] = False
    self._remaining[env_ids] = 0

  def clear_all(self) -> None:
    if self._active.any():
      self._clear_force(self._active.nonzero(as_tuple=False).squeeze(-1))

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, slice):
      active = self._active.nonzero(as_tuple=False).squeeze(-1)
    else:
      active = env_ids[self._active[env_ids]]
    if len(active) > 0:
      self._clear_force(active)
    self._active[env_ids] = False
    self._remaining[env_ids] = 0

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw force (green) and torque (orange) arrows at assisted body COMs."""
    if not self._active.any():
      return
    viz = self._viz_cfg
    min_force_sq = viz.min_force * viz.min_force
    min_torque_sq = viz.min_torque * viz.min_torque
    wrench = self._asset.data.body_external_wrench  # (E, B, 6)
    com_pos = self._asset.data.body_com_pos_w
    body_ids = self._body_ids
    if isinstance(body_ids, slice):
      body_ids = list(range(wrench.shape[1]))
    elif not isinstance(body_ids, list):
      body_ids = list(body_ids)

    for env_idx in visualizer.get_env_indices(self._num_envs):
      if not bool(self._active[env_idx]):
        continue
      for body_i in body_ids:
        force = wrench[env_idx, body_i, :3]
        torque = wrench[env_idx, body_i, 3:6]
        start_np = com_pos[env_idx, body_i].detach().cpu().numpy()
        if (force * force).sum().item() >= min_force_sq:
          force_np = force.detach().cpu().numpy()
          visualizer.add_arrow(
            start=start_np,
            end=start_np + force_np * viz.force_scale,
            color=viz.force_rgba,
            width=viz.width,
          )
        if (torque * torque).sum().item() >= min_torque_sq:
          torque_np = torque.detach().cpu().numpy()
          visualizer.add_arrow(
            start=start_np,
            end=start_np + torque_np * viz.torque_scale,
            color=viz.torque_rgba,
            width=viz.width,
          )


def recovery_assist_force_at_iteration(
  initial_force: float,
  iteration: int,
  force_decay: float = 20.0,
  decay_interval_iters: int = 500,
) -> float:
  """Training curriculum force at a given PPO iteration (same schedule as train)."""
  stages = max(0, int(iteration)) // decay_interval_iters
  return max(0.0, float(initial_force) - force_decay * stages)


def set_recovery_assist_force_for_iteration(
  env: ManagerBasedRlEnv,
  iteration: int,
  event_name: str = "recovery_assist_force",
  force_decay: float = 20.0,
  decay_interval_iters: int = 500,
) -> float | None:
  """Freeze assist force to the training value at ``iteration`` (for play).

  Returns the applied force, or None if the event is not registered.
  """
  try:
    term_cfg = env.event_manager.get_term_cfg(event_name)
  except ValueError:
    return None
  func = term_cfg.func
  if not hasattr(func, "current_force"):
    return None
  initial_force = float(term_cfg.params.get("initial_force", 250.0))
  force = recovery_assist_force_at_iteration(
    initial_force, iteration, force_decay, decay_interval_iters
  )
  prev = float(func.current_force)
  func.current_force = force
  if force <= 0.0 and prev > 0.0 and hasattr(func, "clear_all"):
    func.clear_all()
  print(
    f"[recovery_assist_force] play iter={iteration} force={force:.1f}N "
    f"(same schedule as train: -{force_decay}N every {decay_interval_iters} iters, "
    f"initial={initial_force:.1f}N)"
  )
  return force


def recovery_assist_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  event_name: str = "recovery_assist_force",
  initial_force: float = 250.0,
  force_decay: float = 20.0,
  decay_interval_iters: int = 500,
  steps_per_iter: int = 24,
) -> torch.Tensor:
  """Curriculum: decay the recovery upward-assist force over PPO iterations.

  AMP_mjlab ``recovery_assist_force`` port. Force schedule:

    F(iter) = max(0, initial_force - force_decay * floor(iter / decay_interval_iters))

  where ``iter = common_step_counter // steps_per_iter`` (one PPO iteration).
  Writes F into the assist event's ``current_force`` and returns it so the
  CurriculumManager logs ``Curriculum/recovery_assist_force``.
  """
  del env_ids  # Unused.
  iteration = env.common_step_counter // steps_per_iter
  force = recovery_assist_force_at_iteration(
    initial_force, iteration, force_decay, decay_interval_iters
  )

  try:
    term_cfg = env.event_manager.get_term_cfg(event_name)
  except ValueError:
    return torch.tensor([force])

  func = term_cfg.func
  if hasattr(func, "current_force"):
    prev = float(func.current_force)
    func.current_force = force
    if force <= 0.0 and prev > 0.0 and hasattr(func, "clear_all"):
      func.clear_all()
    if abs(prev - force) > 1e-6:
      print(
        f"[recovery_assist_force] iter={iteration} force={force:.1f}N "
        f"(decay {force_decay}N every {decay_interval_iters} iters)"
      )

  return torch.tensor([force])


def action_rate_l2_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  reward_name: str = "action_rate_l2",
  initial_weight: float = -0.025,
  weight_decay: float = 0.005,
  min_weight: float = -0.1,
  decay_interval_iters: int = 500,
  steps_per_iter: int = 24,
) -> torch.Tensor:
  """Curriculum: anneal ``action_rate_l2`` weight for fall-recovery training.

  Schedule (weight is negative; more negative = stronger penalty)::

    w(iter) = max(min_weight, initial_weight - weight_decay * floor(iter / decay_interval_iters))

  Defaults: start at -0.025, subtract 0.005 every 500 PPO iters, floor at -0.1.
  """
  del env_ids  # Unused.
  iteration = env.common_step_counter // steps_per_iter
  stages = iteration // decay_interval_iters
  weight = max(min_weight, initial_weight - weight_decay * stages)

  try:
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
  except ValueError:
    return torch.tensor([weight])

  prev = float(term_cfg.weight)
  term_cfg.weight = weight
  if abs(prev - weight) > 1e-9:
    print(
      f"[action_rate_l2] iter={iteration} weight={weight:.4f} "
      f"(decay {weight_decay} every {decay_interval_iters} iters, min {min_weight})"
    )
  return torch.tensor([weight])


class TolerantTermination:
  """Class-based termination with a tolerant window for fallen/recovery envs.

  For normal envs: terminates immediately when any sub-term triggers (same as
  the stock tracking terminations). For recovery envs (mask from the command
  term): terminates only when the sub-terms have triggered continuously for
  ``bad_tracking_time_threshold_s`` worth of env steps, giving the fallen
  robot a window to get up before the episode is cut.

  Adapted from KungFuAthleteBot ``TolerantTermination``. The ``reset`` hook
  keeps per-env counters in sync with episode resets.
  """

  def __init__(
    self,
    bad_tracking_time_threshold_s: float,
    command_name: str,
    terms: list,
  ):
    self.bad_tracking_time_threshold_s = bad_tracking_time_threshold_s
    self.command_name = command_name
    self.terms = terms
    self.bad_tracking_episode_length: torch.Tensor | None = None
    self.last_triggered_terms: dict[str, torch.Tensor] = {}
    self._env: ManagerBasedRlEnv | None = None
    self.final_trigger_mask: torch.Tensor | None = None

  def __call__(self, env: ManagerBasedRlEnv):
    import math

    self._env = env
    if self.bad_tracking_episode_length is None:
      self.bad_tracking_episode_length = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long,
      )

    current_combined_bad = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool,
    )
    command = env.command_manager.get_term(self.command_name)

    for term_name, func, params in self.terms:
      raw_bad = func(env, **params)
      current_combined_bad |= raw_bad
      self.last_triggered_terms[term_name] = raw_bad

    bad_tracking_max_episode_length = math.ceil(
      self.bad_tracking_time_threshold_s / env.step_dt
    )

    self.bad_tracking_episode_length = torch.where(
      current_combined_bad,
      self.bad_tracking_episode_length + 1,
      torch.zeros_like(self.bad_tracking_episode_length),
    )

    is_recovery = command.is_recovery_env if hasattr(command, "is_recovery_env") else None
    if is_recovery is None:
      tolerant = current_combined_bad
    else:
      tolerant = (
        self.bad_tracking_episode_length >= bad_tracking_max_episode_length
      )
    bad_tracking_terminate = torch.where(
      is_recovery,
      tolerant,
      current_combined_bad,
    ) if is_recovery is not None else tolerant
    self.final_trigger_mask = bad_tracking_terminate
    return bad_tracking_terminate

  def reset(self, env_ids=None):
    if self.bad_tracking_episode_length is None or self._env is None:
      return

    if env_ids is None:
      self.bad_tracking_episode_length.zero_()
      mask = self.final_trigger_mask
    else:
      self.bad_tracking_episode_length[env_ids] = 0
      mask = self.final_trigger_mask[env_ids]

    for name, value in self.last_triggered_terms.items():
      self._env.extras[f"Episode_Termination/TolerantTermination/{name}"] = (
        torch.count_nonzero(
          (value if env_ids is None else value[env_ids]) & mask
        ).item()
      )
