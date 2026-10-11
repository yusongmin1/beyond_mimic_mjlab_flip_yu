"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import UnitreeActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia_from_two_stage_planetary
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Rotor inertias / gears for the two-stage planetary gearboxes.
# Reflected (rotor) inertia at the joint: J2 + J1*i2^2 + Jr*(i1*i2)^2.
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

##
# Torque-speed envelope presets (from unitree_rl_lab ``unitree_actuators.py``).
# Trapezoid: full peak torque up to x1, then linear taper to zero at x2.
# y1: peak torque with torque and speed in the same direction (driving).
# y2: peak torque with torque and speed in opposite directions (braking).
# fs/fd: static/dynamic joint friction; va: friction activation velocity.
##

ENVELOPE_N7520_22P5 = (14.5, 22.7, 111.0, 131.0)
ENVELOPE_N7520_22P5_FRICTION = (2.4, 0.24)
ENVELOPE_N7520_14P3 = (22.63, 35.52, 71.0, 83.3)
ENVELOPE_N7520_14P3_FRICTION = (1.6, 0.16)
ENVELOPE_N5020_16 = (30.86, 40.13, 24.8, 31.9)
ENVELOPE_N5020_16_FRICTION = (0.6, 0.06)
ENVELOPE_W4010_25 = (15.3, 24.76, 4.8, 8.6)
ENVELOPE_W4010_25_FRICTION = (0.6, 0.06)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


def _unitree_actuator_cfg(
  envelope: tuple[float, float, float, float],
  friction: tuple[float, float],
  *,
  target_names_expr: tuple[str, ...],
  armature: float,
  stiffness: float,
  damping: float,
  torque_scale: float = 1.0,
) -> UnitreeActuatorCfg:
  """Build a UnitreeActuatorCfg from a motor envelope preset.

  Args:
    envelope: (x1, x2, y1, y2) trapezoid corners, rad/s and N·m.
    friction: (fs, fd) static/dynamic friction coefficients.
    target_names_expr: Joints driven by this motor group.
    armature: Reflected rotor inertia set on the joints.
    stiffness: PD proportional gain.
    damping: PD derivative gain.
    torque_scale: Scale on torque-class params for parallel motor pairs
      (e.g. 2× N5020 on waist pitch/roll and ankles); x1/x2 are unchanged
      since parallel motors share the same speed range.
  """
  x1, x2, y1, y2 = envelope
  fs, fd = friction
  return UnitreeActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    # MuJoCo-side safety net; the runtime envelope is the real limit.
    effort_limit=y2 * torque_scale,
    armature=armature,
    x1=x1,
    x2=x2,
    y1=y1 * torque_scale,
    y2=y2 * torque_scale,
    fs=fs * torque_scale,
    fd=fd * torque_scale,
  )


G1_ACTUATOR_5020 = _unitree_actuator_cfg(
  ENVELOPE_N5020_16,
  ENVELOPE_N5020_16_FRICTION,
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  armature=ARMATURE_5020,
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
)
G1_ACTUATOR_7520_14 = _unitree_actuator_cfg(
  ENVELOPE_N7520_14P3,
  ENVELOPE_N7520_14P3_FRICTION,
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  armature=ARMATURE_7520_14,
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
)
G1_ACTUATOR_7520_22 = _unitree_actuator_cfg(
  ENVELOPE_N7520_22P5,
  ENVELOPE_N7520_22P5_FRICTION,
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  armature=ARMATURE_7520_22,
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
)
G1_ACTUATOR_4010 = _unitree_actuator_cfg(
  ENVELOPE_W4010_25,
  ENVELOPE_W4010_25_FRICTION,
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  armature=ARMATURE_4010,
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
# Torque-class envelope params (y1/y2/fs/fd) scale by 2; x1/x2 are unchanged.
G1_ACTUATOR_WAIST = _unitree_actuator_cfg(
  ENVELOPE_N5020_16,
  ENVELOPE_N5020_16_FRICTION,
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  armature=ARMATURE_5020 * 2,
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  torque_scale=2.0,
)
G1_ACTUATOR_ANKLE = _unitree_actuator_cfg(
  ENVELOPE_N5020_16,
  ENVELOPE_N5020_16_FRICTION,
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  armature=ARMATURE_5020 * 2,
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  torque_scale=2.0,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, UnitreeActuatorCfg)
  e = a.effort_limit  # Envelope braking peak y2 (incl. parallel-motor scale).
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
