"""Useful methods for MDP rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for being alive."""
  return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize terminated episodes that don't correspond to episodic timeouts."""
  return env.termination_manager.terminated.float()


def joint_torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint torques applied on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.actuator_force), dim=1)


def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint velocities on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def action_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the acceleration of the actions using L2 squared kernel."""
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.sum(torch.square(action_acc), dim=1)


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits."""
  asset: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  out_of_limits = -(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


class posture:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del std  # Unused.
    asset: Entity = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


class electrical_power_cost:
  """Penalize electrical power consumption of actuators."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    joint_ids, _ = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    actuator_ids, _ = asset.find_actuators(
      cfg.params["asset_cfg"].joint_names,
    )
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self._actuator_ids = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, self._actuator_ids]
    qd = asset.data.joint_vel[:, self._joint_ids]
    mech = tau * qd
    mech_pos = torch.clamp(mech, min=0.0)  # Don't penalize regen.
    return torch.sum(mech_pos, dim=1)


def negative_joint_power_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  power_deadband: float = 0.0,
  penalty_scale: float = 1.0,
) -> torch.Tensor:
  """Penalize excessive regenerative braking (negative mechanical power).

  OmniXtreme power-safety term (arXiv:2602.23843, eq. 3 / appendix V-K):
  for each selected joint, compute the instantaneous mechanical power
  P = tau * qd from the applied actuator force and joint velocity, and
  penalize the negative power beyond a deadband with an L2-squared kernel:

      sum_j (max(-P_j - power_deadband, 0) / penalty_scale)^2

  Braking within the deadband is free; large regenerative braking spikes
  (which can trigger motor overcurrent / power protection on hardware) are
  strongly discouraged. Use with a negative weight. Expects one actuator per
  selected joint (asset_cfg should resolve matching joint/actuator ids).
  """
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.actuator_ids]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  power = tau * qd
  excess = torch.clamp(-power - power_deadband, min=0.0) / max(penalty_scale, 1.0e-6)
  return torch.sum(torch.square(excess), dim=1)


class negative_knee_power_l2:
  """OmniXtreme negative-power penalty aligned joint<->actuator by name.

  Same math as :func:`negative_joint_power_l2`, but resolves the actuator
  controlling each selected joint by name at init time. This guarantees
  tau/qd column alignment even when the entity's actuator insertion order
  differs from its joint tree order (e.g. G1 motor-group ordering).
  Params: asset_cfg (joint_names selects the joints), power_deadband [W],
  penalty_scale [W] (normalizer K in the paper). Use with a negative weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]

    joint_ids, joint_names = asset.find_joints(
      asset_cfg.joint_names, preserve_order=True
    )
    if not joint_ids:
      raise ValueError(
        "negative_knee_power_l2: no joints matched "
        f"'{asset_cfg.joint_names}' on entity '{asset_cfg.name}'."
      )

    actuator_index = {name: i for i, name in enumerate(asset.actuator_names)}
    missing = [n for n in joint_names if n not in actuator_index]
    if missing:
      raise ValueError(
        f"negative_knee_power_l2: joints without a matching actuator: {missing}. "
        "This term expects one actuator per selected joint."
      )

    self._asset_name = asset_cfg.name
    self._joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    self._actuator_ids = torch.tensor(
      [actuator_index[n] for n in joint_names], dtype=torch.long, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    power_deadband: float = 0.0,
    penalty_scale: float = 1.0,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, self._actuator_ids]
    qd = asset.data.joint_vel[:, self._joint_ids]
    power = tau * qd
    excess = torch.clamp(-power - power_deadband, min=0.0) / max(penalty_scale, 1.0e-6)
    return torch.sum(torch.square(excess), dim=1)


class joint_force_exceedance_l2:
  """Penalize actuator (joint) force magnitude beyond a threshold.

  NOTE: for distal joints driven by small motors, ``actuator_force`` is
  clamped by the trapezoidal torque-speed envelope (e.g. G1 wrists cap at
  8.6 / 31.9 N·m), so a threshold above those limits can never trigger. To
  penalize large hand-ground contact loads use
  :func:`contact_force_exceedance` with a ContactSensor instead.

  For each selected joint, resolves the controlling actuator by name at init
  time (guaranteeing force-column alignment even when actuator insertion
  order differs from joint tree order). Penalizes the excess of the
  instantaneous actuator force magnitude beyond ``force_threshold`` with an
  L2-squared kernel:

      sum_j (max(|tau_j| - force_threshold, 0))^2

  Params: asset_cfg (joint_names selects the joints), force_threshold.
  Use with a negative weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]

    joint_ids, joint_names = asset.find_joints(
      asset_cfg.joint_names, preserve_order=True
    )
    if not joint_ids:
      raise ValueError(
        "joint_force_exceedance_l2: no joints matched "
        f"'{asset_cfg.joint_names}' on entity '{asset_cfg.name}'."
      )

    actuator_index = {name: i for i, name in enumerate(asset.actuator_names)}
    missing = [n for n in joint_names if n not in actuator_index]
    if missing:
      raise ValueError(
        f"joint_force_exceedance_l2: joints without a matching actuator: {missing}. "
        "This term expects one actuator per selected joint."
      )

    self._asset_name = asset_cfg.name
    self._actuator_ids = torch.tensor(
      [actuator_index[n] for n in joint_names], dtype=torch.long, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 0.0,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    force = asset.data.actuator_force[:, self._actuator_ids]
    excess = (force.abs() - force_threshold).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def contact_force_exceedance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalize external contact force magnitude on sensor bodies beyond a threshold.

  Reads a ContactSensor (fields must include ``force``; e.g. body-mode
  primaries vs the terrain with ``reduce="maxforce"``) and returns, per env,
  the summed linear excess of the per-body contact force magnitude over
  ``force_threshold``:

      sum_b max(||F_b|| - force_threshold, 0)

  Intended for hand/wrist-ground contact safety: penalizes contact forces
  beyond a threshold with weight ``-0.01`` (units: penalty per newton of
  excess, per step). Unlike actuator-force penalties this is NOT clamped by
  the motor torque envelope, so it reflects the true external impact load.
  """
  from mjlab.sensor import ContactSensor  # noqa: PLC0415 — avoid cycle.

  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    return torch.zeros(env.num_envs, device=env.device)
  force_mag = torch.linalg.vector_norm(force, dim=-1)  # [B, N]
  excess = (force_mag - force_threshold).clamp(min=0.0)
  return torch.sum(excess, dim=1)


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
