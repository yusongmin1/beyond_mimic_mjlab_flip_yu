from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch

from mjlab.sensor import ContactSensor, JointTorqueSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_anchor_linear_velocity_z_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track world-frame z linear velocity of the motion anchor (current frame)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(
    command.anchor_lin_vel_w[:, 2] - command.robot_anchor_lin_vel_w[:, 2]
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_angular_velocity_x_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track world-frame x angular velocity of the motion anchor (current frame)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(
    command.anchor_ang_vel_w[:, 0] - command.robot_anchor_ang_vel_w[:, 0]
  )
  return torch.exp(-error / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def head_contact_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Binary cost: 1.0 when the head geom touches anything (ground)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found.squeeze(-1) > 0.0).float()


def penalty_relative_root_orientation(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Penalize pelvis orientation error relative to the reference.

  Encourages rolling the pelvis upright first (instead of flailing)
  during stand-up. KungFuAthleteBot ``penalty_relative_root_orientation``.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(
    command.body_quat_relative_w[:, command.root_index],
    command.robot_body_quat_w[:, command.root_index],
  ) ** 2
  return error.squeeze(-1)


def penalty_xy_rate_before_stand(
  env: ManagerBasedRlEnv,
  command_name: str,
  stand_threshold: float,
) -> torch.Tensor:
  """Penalize horizontal anchor drift while the robot is still fallen.

  Stand-up-gated: the penalty is only active while the shoulders' height
  error w.r.t. the reference exceeds ``stand_threshold`` (i.e. the robot
  has not stood up yet). Encourages getting up in place instead of
  crawling/scooting away. KungFuAthleteBot ``penalty_xy_rate_before_stand``.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.norm(
    command.prev_anchor_pos[..., :2] - command.robot_anchor_pos_w[..., :2], dim=1
  )
  diff = torch.norm(
    (
      command.body_pos_relative_w[:, command.shoulders_indexes, 2]
      - command.robot_body_pos_w[:, command.shoulders_indexes, 2]
    ),
    dim=-1,
  )
  return torch.where(
    diff > stand_threshold, error, torch.tensor(0.0, device=error.device, dtype=error.dtype)
  )


def reward_reference_height_before_stand(
  env: ManagerBasedRlEnv,
  command_name: str,
  stand_threshold: float,
  std: float = 0.25,
) -> torch.Tensor:
  """Reward matching the reference root height while still fallen.

  Stand-up-gated (same gate as ``penalty_xy_rate_before_stand``): active
  only while the shoulders' height error w.r.t. the reference exceeds
  ``stand_threshold`` (i.e. the robot is lying down / getting up).
  Rewards the pelvis height tracking the reference so the robot rises
  toward the reference posture instead of staying flat on the ground.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  diff = torch.norm(
    (
      command.body_pos_relative_w[:, command.shoulders_indexes, 2]
      - command.robot_body_pos_w[:, command.shoulders_indexes, 2]
    ),
    dim=-1,
  )
  error_sq = torch.square(
    command.body_pos_relative_w[:, command.root_index, 2]
    - command.robot_body_pos_w[:, command.root_index, 2]
  ).squeeze(-1)
  reward = torch.exp(-error_sq / std**2)
  return torch.where(
    diff > stand_threshold, reward, torch.tensor(0.0, device=reward.device, dtype=reward.dtype)
  )


def motion_joint_torque_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  filter_mode: Literal["mean", "last"],
  std: float,
) -> torch.Tensor:
  if std <= 0.0:
    raise ValueError(f"motion_joint_torque_error_exp expects std > 0, got {std}.")

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, JointTorqueSensor):
    raise TypeError(
      f"Sensor '{sensor_name}' must be JointTorqueSensor, got {type(sensor).__name__}."
    )

  ref_tau = command.joint_tau
  act_tau = sensor.get_torque(filter_mode)
  tau_limit = sensor.data.joint_torque_limit

  if ref_tau.shape != act_tau.shape:
    raise ValueError(
      "motion_joint_torque_error_exp shape mismatch between reference and actual "
      "torque: "
      f"{tuple(ref_tau.shape)} vs {tuple(act_tau.shape)}."
    )
  if tau_limit.shape != act_tau.shape:
    raise ValueError(
      "motion_joint_torque_error_exp shape mismatch between actual torque and torque "
      "limits: "
      f"{tuple(act_tau.shape)} vs {tuple(tau_limit.shape)}."
    )

  err = (ref_tau - act_tau) / tau_limit
  err_mse = torch.mean(err * err, dim=-1)
  return torch.exp(-err_mse / std**2)

