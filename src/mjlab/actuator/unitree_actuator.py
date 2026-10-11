"""Unitree motor actuator with trapezoidal torque-speed envelope.

Implements the Unitree motor model from unitree_rl_lab
(``unitree_actuators.py``): a PD actuator whose torque is clipped by a
trapezoidal torque-speed envelope, plus static/dynamic joint friction.

Envelope definition (per joint, torque vs |velocity|):

        Torque Limit, N·m
            ^
    Y2─────|
           |──────────────Y1
           |              │\\
           |              │ \\
           |              │  \\
    -------+--------------|------> velocity: rad/s
                          X1   X2

  - Y1: peak torque, torque and speed in the same direction.
  - Y2: peak torque, torque and speed in opposite directions.
  - X1: maximum speed at full torque (T-N curve knee point).
  - X2: no-load speed.

  - Fs: static friction coefficient.
  - Fd: dynamic friction coefficient.
  - Va: velocity at which the friction is fully activated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

UnitreeCfgT = TypeVar("UnitreeCfgT", bound="UnitreeActuatorCfg")


@dataclass
class TorqueSpeedEnvelope:
  """Trapezoidal torque-speed corner points (rad/s, N·m)."""

  x1: float
  """Maximum speed at full torque (knee point), rad/s."""

  x2: float
  """No-load speed, rad/s."""

  y1: float
  """Peak torque with torque and speed in the same direction, N·m."""

  y2: float
  """Peak torque with torque and speed in opposite directions, N·m."""


@dataclass(kw_only=True)
class UnitreeActuatorCfg(IdealPdActuatorCfg):
  """Ideal PD actuator with Unitree torque-speed envelope and friction.

  effort_limit is used as the MuJoCo-side forcerange safety net and should
  be set to the largest envelope torque (typically y2).
  """

  x1: float
  """Maximum speed at full torque (T-N curve knee point), rad/s."""

  x2: float
  """No-load speed, rad/s."""

  y1: float
  """Peak torque, torque and speed in the same direction, N·m."""

  y2: float | None = None
  """Peak torque, torque and speed in opposite directions, N·m.

  If None, defaults to y1.
  """

  fs: float = 0.0
  """Static friction coefficient, N·m."""

  fd: float = 0.0
  """Dynamic friction coefficient, N·m·s/rad."""

  va: float = 0.01
  """Velocity at which the friction is fully activated, rad/s."""

  def __post_init__(self) -> None:
    if self.y2 is None:
      self.y2 = self.y1
    super().__post_init__()

  @property
  def envelope(self) -> TorqueSpeedEnvelope:
    assert self.y2 is not None
    return TorqueSpeedEnvelope(self.x1, self.x2, self.y1, self.y2)

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> UnitreeActuator:
    return UnitreeActuator(self, entity, target_ids, target_names)


def clip_unitree_effort(
  effort: torch.Tensor,
  joint_vel: torch.Tensor,
  x1: torch.Tensor,
  x2: torch.Tensor,
  y1: torch.Tensor,
  y2: torch.Tensor,
) -> torch.Tensor:
  """Clip PD torque with the direction-dependent Unitree envelope.

  Uses peak y1 when torque and velocity point the same way (driving),
  y2 otherwise (braking). Beyond x1 the limit tapers linearly to zero at x2.
  """
  same_direction = (joint_vel * effort) > 0
  peak = torch.where(same_direction, y1, y2)
  denom = torch.clamp(x2 - x1, min=1.0e-6)
  over = torch.clamp(joint_vel.abs() - x1, min=0.0)
  slope = peak / denom
  tapered = torch.clamp(peak - slope * over, min=0.0)
  limit = torch.where(joint_vel.abs() < x1, peak, tapered)
  return torch.clamp(effort, -limit, limit)


class UnitreeActuator(IdealPdActuator[UnitreeCfgT], Generic[UnitreeCfgT]):
  """PD actuator with Unitree torque-speed envelope clipping and friction."""

  def __init__(
    self,
    cfg: UnitreeCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._joint_vel: torch.Tensor | None = None
    self._envelope_x1: torch.Tensor | None = None
    self._envelope_x2: torch.Tensor | None = None
    self._envelope_y1: torch.Tensor | None = None
    self._envelope_y2: torch.Tensor | None = None
    self._friction_static: torch.Tensor | None = None
    self._friction_dynamic: torch.Tensor | None = None
    self._activation_vel: torch.Tensor | None = None

  def initialize(
    self,
    mj_model,
    model,
    data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    num_envs = data.nworld
    num_joints = len(self._target_names)
    cfg = self.cfg
    assert cfg.y2 is not None

    self._joint_vel = torch.zeros(num_envs, num_joints, device=device)
    self._envelope_x1 = torch.full(
      (num_envs, num_joints), cfg.x1, device=device
    )
    self._envelope_x2 = torch.full(
      (num_envs, num_joints), cfg.x2, device=device
    )
    self._envelope_y1 = torch.full(
      (num_envs, num_joints), cfg.y1, device=device
    )
    self._envelope_y2 = torch.full(
      (num_envs, num_joints), cfg.y2, device=device
    )
    self._friction_static = torch.full((num_envs, num_joints), cfg.fs, device=device)
    self._friction_dynamic = torch.full((num_envs, num_joints), cfg.fd, device=device)
    self._activation_vel = torch.full((num_envs, num_joints), cfg.va, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._joint_vel is not None
    self._joint_vel[:] = cmd.vel
    effort = super().compute(cmd)
    assert self._friction_static is not None
    assert self._friction_dynamic is not None
    assert self._activation_vel is not None
    # Subtract the friction model from the commanded torque.
    return effort - (
      self._friction_static * torch.tanh(cmd.vel / self._activation_vel)
      + self._friction_dynamic * cmd.vel
    )

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self._joint_vel is not None
    assert self._envelope_x1 is not None
    assert self._envelope_x2 is not None
    assert self._envelope_y1 is not None
    assert self._envelope_y2 is not None
    return clip_unitree_effort(
      effort,
      self._joint_vel,
      self._envelope_x1,
      self._envelope_x2,
      self._envelope_y1,
      self._envelope_y2,
    )
