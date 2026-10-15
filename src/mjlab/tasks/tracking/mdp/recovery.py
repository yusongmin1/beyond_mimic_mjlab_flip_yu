"""Fall-and-recovery training utilities (normal / fallen env splitting).

Ported and adapted from:
- AMP_mjlab ``src/tasks/amp_loco/mdp/{events,curriculums,terminations}.py``:
  delayed termination (recovery window), upward recovery assist force with a
  decay curriculum.
- KungFuAthleteBot ``unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py``:
  fallen-state initialization from a pre-collected ``.pth`` pool
  (``robot_init_states_*.pth``, diverse pelvis orientations near the ground).

Env grouping (per episode reset):
- 50% normal tracking envs (reference-state init, standard terminations).
- 50% recovery envs, of which:
  - half are "fall-window" envs: they start from the motion reference but get
    a delayed-termination recovery window after tracking failure (fall), so
    the policy must get up;
  - half are "fallen-init" envs: initialized lying on the ground from the
    ``.pth`` pool, also with the recovery window (tolerant terminations).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.termination_manager import TerminationManager
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class RecoveryTerminationManager(TerminationManager):
  """TerminationManager with a recovery window for a subset of envs.

  For recovery envs, a termination signal starts a counter instead of
  resetting immediately. The env is given ``max_recovery_steps`` env steps
  (the "get up" window). The reset signal is released only when the
  termination condition has persisted (cumulatively, via the counter) for
  the whole window. If the env recovers on its own (termination condition
  clears), the counter resets.

  Adapted from AMP_mjlab ``DelayedTerminationManager``.
  """

  def __init__(
    self,
    base: TerminationManager,
    recovery_env_mask: torch.Tensor,
    max_recovery_steps: int,
  ) -> None:
    # Steal all internal state from the base manager (avoid re-init).
    self.__dict__.update(base.__dict__)
    self._recovery_env_mask = recovery_env_mask  # (num_envs,) bool
    self._recovery_counters = torch.zeros_like(recovery_env_mask, dtype=torch.long)
    self._max_recovery_steps = max_recovery_steps

  def compute(self) -> torch.Tensor:
    dones = super().compute()  # fills _truncated_buf, _terminated_buf

    if self._max_recovery_steps <= 0:
      return dones

    # For recovery envs that just got a done signal, increment counter.
    rec_and_done = self._recovery_env_mask & dones
    self._recovery_counters[rec_and_done] += 1

    # Recovery envs whose counter hasn't reached threshold: suppress reset.
    not_ready = rec_and_done & (self._recovery_counters < self._max_recovery_steps)
    self._terminated_buf[not_ready] = False

    # Recovery envs whose counter reached threshold: allow reset, clear counter.
    ready = rec_and_done & (self._recovery_counters >= self._max_recovery_steps)
    self._recovery_counters[ready] = 0

    # Clear counters for recovery envs that are NOT done (recovered on their own).
    self._recovery_counters[self._recovery_env_mask & ~dones] = 0

    return self._truncated_buf | self._terminated_buf


def install_recovery_termination(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  recovery_env_ratio: float = 0.5,
  max_recovery_steps: int = 100,
) -> None:
  """Startup event: wrap the termination manager with a recovery window.

  The mask is static for the whole training run (as in AMP_mjlab): the same
  envs keep the recovery window. The normal/recovery identity per env only
  matters inside a single episode, and the reset sampler re-rolls the
  fall-type split (fall-window vs fallen-init) on every reset.
  """
  del env_ids  # Unused (startup).
  if not 0.0 <= recovery_env_ratio <= 1.0:
    raise ValueError(f"recovery_env_ratio must be in [0, 1], got {recovery_env_ratio}.")
  num_rec = int(env.num_envs * recovery_env_ratio)
  if num_rec <= 0 or max_recovery_steps <= 0:
    return
  if isinstance(env.termination_manager, RecoveryTerminationManager):
    return
  mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  indices = torch.randperm(env.num_envs, device=env.device)[:num_rec]
  mask[indices] = True
  env.termination_manager = RecoveryTerminationManager(
    base=env.termination_manager,
    recovery_env_mask=mask,
    max_recovery_steps=max_recovery_steps,
  )
  print(
    f"[recovery] Installed RecoveryTerminationManager: {num_rec}/{env.num_envs} envs "
    f"with a {max_recovery_steps}-step recovery window."
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


def apply_recovery_assist_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  trigger_recovery_steps: int = 1,
  apply_prob: float = 0.8,
  initial_force: float = 250.0,
  duration_steps: int = 25,
) -> None:
  """Interval-every-step event: upward assist force for recovering envs.

  Should be registered with ``mode="interval"`` and
  ``interval_range_s=(env.step_dt, env.step_dt)`` so it fires every env step.

  Adapted from AMP_mjlab ``apply_recovery_assist_force``, but default trigger
  is step 1 (immediately when failure is announced / delay window opens),
  not the window midpoint — mid-window assist often leaves too little time
  to stand. When a recovery env's counter first hits
  ``trigger_recovery_steps``, sample ``apply_prob`` and (if accepted) apply
  a constant world-+Z force on the selected body for ``duration_steps`` env
  steps (or until the env recovers / resets).

  The applied force is ``state["force"]`` (AMP_mjlab's ``current_force``),
  decayed by the ``recovery_assist_force`` curriculum term; before the
  curriculum first runs it equals ``initial_force`` (also used at play).
  At F == 0 no forces are applied (fully weaned).
  """
  del env_ids  # Interval events receive per-env ids; we manage actives globally.
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  num_bodies = len(body_ids) if isinstance(body_ids, list) else asset.num_bodies

  tm = env.termination_manager
  if not isinstance(tm, RecoveryTerminationManager):
    return

  state = _assist_state(env)
  active, remaining = state["active"], state["remaining"]

  # Current curriculum force: written by the recovery_assist_force
  # curriculum term (training); falls back to initial_force before the
  # curriculum first runs and in play mode (AMP_mjlab keeps the same
  # convention via the event's ``current_force`` attribute).
  current_force = (
    state["force"] if state["force"] is not None else initial_force
  )

  counters = tm._recovery_counters
  recovery_mask = tm._recovery_env_mask

  # Clear when recovery ends (recovered or window expired and reset).
  clear = active & (counters == 0)
  if clear.any():
    _write_upward_force(env, asset, body_ids, num_bodies, clear.nonzero().squeeze(-1), 0.0)
    active[clear] = False
    remaining[clear] = 0

  # Tick duration for still-active assists.
  if active.any():
    remaining[active] -= 1
    expired = active & (remaining <= 0)
    if expired.any():
      _write_upward_force(env, asset, body_ids, num_bodies, expired.nonzero().squeeze(-1), 0.0)
      active[expired] = False
      remaining[expired] = 0

  if current_force <= 0.0:
    if active.any():
      ids = active.nonzero().squeeze(-1)
      _write_upward_force(env, asset, body_ids, num_bodies, ids, 0.0)
      active[ids] = False
    return

  # Decide once when the counter first reaches the trigger step.
  candidates = recovery_mask & (counters == trigger_recovery_steps) & (~active)
  if candidates.any():
    cand_ids = candidates.nonzero().squeeze(-1)
    accept = torch.rand(len(cand_ids), device=env.device) < apply_prob
    trigger_ids = cand_ids[accept]
    if len(trigger_ids) > 0:
      active[trigger_ids] = True
      remaining[trigger_ids] = duration_steps

  if not active.any():
    return

  active_ids = active.nonzero().squeeze(-1)
  _write_upward_force(env, asset, body_ids, num_bodies, active_ids, current_force)


def _assist_state(env: ManagerBasedRlEnv) -> dict:
  """Lazily create per-env assist bookkeeping on the event function state."""
  state = getattr(apply_recovery_assist_force, "_state_" + str(id(env)), None)
  if state is None:
    state = {
      "active": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
      "remaining": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
      "force": None,  # Current curriculum force (N); set by the curriculum.
    }
    setattr(apply_recovery_assist_force, "_state_" + str(id(env)), state)
  return state


def recovery_assist_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  initial_force: float = 250.0,
  force_decay: float = 20.0,
  decay_interval_iters: int = 500,
  steps_per_iter: int = 24,
) -> torch.Tensor:
  """Curriculum: decay the recovery upward-assist force over PPO iterations.

  AMP_mjlab ``recovery_assist_force`` port. Force schedule:

    F(iter) = max(0, initial_force - force_decay * floor(iter / decay_interval_iters))

  where ``iter = common_step_counter // steps_per_iter`` (one PPO iteration).
  Writes F into the assist event shared state (the event reads it every
  step) and returns it so the CurriculumManager logs
  ``Curriculum/recovery_assist_force`` into the per-iteration block.
  """
  del env_ids  # Unused.
  iteration = env.common_step_counter // steps_per_iter
  stages = iteration // decay_interval_iters
  force = max(0.0, initial_force - force_decay * stages)

  state = _assist_state(env)
  prev = state["force"]
  state["force"] = force
  if force <= 0.0 and prev is not None and prev > 0.0:
    # Fully weaned: clear any in-flight assists.
    active = state["active"]
    if active.any():
      asset = env.scene["robot"]
      active_ids = active.nonzero(as_tuple=False).squeeze(-1)
      _write_upward_force(env, asset, None, asset.num_bodies, active_ids, 0.0)
      active[active_ids] = False
      state["remaining"][active_ids] = 0
  if prev is None or abs(prev - force) > 1e-6:
    print(
      f"[recovery_assist_force] iter={iteration} force={force:.1f}N "
      f"(decay {force_decay}N every {decay_interval_iters} iters)"
    )

  return torch.tensor([force])


def _write_upward_force(
  env: ManagerBasedRlEnv,
  asset: Entity,
  body_ids,
  num_bodies: int,
  env_ids: torch.Tensor,
  force_n: float,
) -> None:
  if len(env_ids) == 0:
    return
  n = len(env_ids)
  forces = torch.zeros((n, num_bodies, 3), device=env.device)
  torques = torch.zeros_like(forces)
  if force_n != 0.0:
    forces[..., 2] = force_n  # world +Z
  asset.write_external_wrench_to_sim(
    forces, torques, env_ids=env_ids, body_ids=body_ids
  )


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
