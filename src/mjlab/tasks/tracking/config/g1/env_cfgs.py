"""Unitree G1 flat tracking environment configurations."""

import os
from typing import Literal

from mjlab.actuator import DelayedActuatorCfg
from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, JointTorqueSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.tracking.tracking_env_cfg_dance import make_dance_env_cfg

TauMode = Literal["off", "reward", "critic", "actor_critic"]
TauFilter = Literal["mean", "last"]

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
_FALLEN_INIT_PTH = os.path.abspath(
  os.path.join(_ASSETS_DIR, "robot_init_states_g1_8192.pth")
)


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  tau_mode: TauMode = "off",
  tau_filter: TauFilter = "mean",
  play: bool = False,
  enable_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration.

  Args:
    has_state_estimation: Keep privileged state terms in the actor obs.
    tau_mode: Torque obs/reward mode.
    tau_filter: Torque sensor window filter.
    play: Disable episode timeout / DR used during interactive play.
    enable_actuator_delay: Wrap actuators with 0–3 physics-step delay
      (independent lag per joint).
  """
  cfg = make_tracking_env_cfg()
  return _apply_g1_overrides(
    cfg,
    has_state_estimation=has_state_estimation,
    tau_mode=tau_mode,
    tau_filter=tau_filter,
    play=play,
    enable_actuator_delay=enable_actuator_delay,
    # enable_actuator_delay=False,
  )


def unitree_g1_flat_dance_env_cfg(
  has_state_estimation: bool = True,
  tau_mode: TauMode = "off",
  tau_filter: TauFilter = "mean",
  play: bool = False,
  enable_actuator_delay: bool = True,
  enable_fall_recovery: bool = False,
  recovery_env_ratio: float = 0.5,
  fallen_init_ratio: float = 0.5,
  max_recovery_steps: int = 100,
  bad_tracking_time_threshold_s: float = 2.0,
  recovery_assist_initial_force: float = 250.0,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 dance tracking configuration.

  Same G1 overrides as the tracking task, but built on the dance base
  config (tracking_env_cfg_dance.py) with its relaxed anchor rewards.

  Args:
    enable_fall_recovery: Split envs into normal / recovery groups.
      recovery_env_ratio of envs get a get-up window after falling
      (delayed termination + tolerant tracking termination); within that
      group fallen_init_ratio start lying on the ground from a
      pre-collected pool (KungFuAthleteBot robot_init_states format).
    recovery_env_ratio: Fraction of envs in the recovery group (0.5 => 50%).
    fallen_init_ratio: Fraction of the recovery group that is initialized
      fallen (0.5 => half of the recovery envs; the rest fall during the
      episode and get the window).
    max_recovery_steps: Recovery window length in env steps (second stage).
    bad_tracking_time_threshold_s: Tolerant-termination window for
      recovery envs (seconds of continuous bad tracking). Together with
      max_recovery_steps the total get-up window is
      bad_tracking_time_threshold_s + max_recovery_steps * step_dt
      (default 2s + 2s = 4s).
    recovery_assist_initial_force: Initial upward assist force (N) on the
      torso while a recovery env is down; decays to 0 over training.
  """
  cfg = make_dance_env_cfg()
  return _apply_g1_overrides(
    cfg,
    has_state_estimation=has_state_estimation,
    tau_mode=tau_mode,
    tau_filter=tau_filter,
    play=play,
    enable_actuator_delay=enable_actuator_delay,
    enable_fall_recovery=enable_fall_recovery,
    recovery_env_ratio=recovery_env_ratio,
    fallen_init_ratio=fallen_init_ratio,
    max_recovery_steps=max_recovery_steps,
    bad_tracking_time_threshold_s=bad_tracking_time_threshold_s,
    recovery_assist_initial_force=recovery_assist_initial_force,
  )


def _apply_g1_overrides(
  cfg: ManagerBasedRlEnvCfg,
  has_state_estimation: bool,
  tau_mode: TauMode,
  tau_filter: TauFilter,
  play: bool,
  enable_actuator_delay: bool,
  enable_fall_recovery: bool = False,
  recovery_env_ratio: float = 0.5,
  fallen_init_ratio: float = 0.5,
  max_recovery_steps: int = 100,
  bad_tracking_time_threshold_s: float = 2.0,
  recovery_assist_initial_force: float = 250.0,
) -> ManagerBasedRlEnvCfg:
  """Apply Unitree G1 robot-specific overrides to a tracking base config."""
  if tau_mode not in {"off", "reward", "critic", "actor_critic"}:
    raise ValueError(f"Unsupported tau_mode '{tau_mode}'.")
  if tau_filter not in {"mean", "last"}:
    raise ValueError(f"Unsupported tau_filter '{tau_filter}'.")

  robot_cfg = get_g1_robot_cfg()
  if enable_actuator_delay and not play:
    articulation = robot_cfg.articulation
    assert articulation is not None
    robot_cfg.articulation = EntityArticulationInfoCfg(
      actuators=tuple(
        DelayedActuatorCfg(
          base_cfg=act,
          delay_target="position",
          delay_min_lag=0,
          delay_max_lag=3,
          delay_per_feature=True,
        )
        for act in articulation.actuators
      ),
      soft_joint_pos_limit_factor=articulation.soft_joint_pos_limit_factor,
    )
  cfg.scene.entities = {"robot": robot_cfg}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  head_contact_cfg = ContactSensorCfg(
    name="head_contact",
    primary=ContactMatch(mode="geom", pattern="head_collision", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # Hand/wrist ground-contact safety: same measurement as the play-time
  # --print-hand-contact printer (body-mode primaries vs terrain, maxforce).
  wrist_contact_cfg = ContactSensorCfg(
    name="wrist_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"(left|right)_wrist_(pitch|roll|yaw)_link",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
  )
  scene_sensors = [self_collision_cfg, head_contact_cfg, wrist_contact_cfg]
  if tau_mode != "off":
    scene_sensors.append(
      JointTorqueSensorCfg(
        name="joint_torque",
        entity_name="robot",
        window_length=cfg.decimation,
      )
    )
  cfg.scene.sensors = tuple(scene_sensors)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )
  motion_cmd.use_joint_tau = tau_mode != "off"

  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  actor_terms["command"].params["with_joint_tau"] = tau_mode == "actor_critic"
  critic_terms["command"].params["with_joint_tau"] = tau_mode in (
    "critic",
    "actor_critic",
  )

  # Keep full obs/reward terms in tracking_env_cfg, then remove by mode flags.
  if tau_mode != "actor_critic":
    actor_terms.pop("joint_torque", None)
  else:
    actor_terms["joint_torque"].params["mode"] = tau_filter

  if tau_mode not in ("critic", "actor_critic"):
    critic_terms.pop("joint_torque", None)
  else:
    critic_terms["joint_torque"].params["mode"] = tau_filter

  if tau_mode == "off":
    cfg.rewards.pop("motion_joint_torque", None)
  else:
    cfg.rewards["motion_joint_torque"].params["filter_mode"] = tau_filter

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["add_body_mass"].params["asset_cfg"].body_names = ("torso_link",)

  # OmniXtreme power-safety regularization (paper eq. 3 / appendix V-K):
  # penalize regenerative braking power at the knees beyond a 150 W deadband,
  # normalized by 500, weight -10. Knees only — they take the largest
  # braking loads during impacts (e.g. backflip landings) and are the joints
  # where regen events trigger hardware overcurrent/power protection.
  cfg.rewards["neg_power_knee"] = RewardTermCfg(
    func=mdp.negative_knee_power_l2,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=r".*_knee_joint"),
      "power_deadband": 150.0,
      "penalty_scale": 500.0,
    },
  )

  # Wrist (distal arm) contact-force safety penalty: the last three links of
  # each arm (wrist_pitch / wrist_roll / wrist_yaw) are driven by small 4010
  # / 5020 motors and bear hand-ground contact loads during dance /
  # fall-recovery motions. Penalize the wrist-link ground contact force
  # beyond 500 N (linear excess, weight -0.01) via the wrist_contact sensor —
  # the same quantity printed by --print-hand-contact at play time. NOTE: an
  # actuator-force version could never trigger because the trapezoidal
  # envelope clamps wrist torque at 8.6 / 31.9 N·m « 500.
  cfg.rewards["wrist_force_limit"] = RewardTermCfg(
    func=mdp.contact_force_exceedance,
    weight=-0.01,
    params={
      "sensor_name": "wrist_contact",
      "force_threshold": 300.0,
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  if enable_fall_recovery:
    # --- Fall-recovery training (AMP_mjlab + KungFuAthleteBot style) ---
    # 1) Command term: split envs into normal / fall-window / fallen-init
    #    groups on every reset; fallen-init envs start from the .pth pool.
    motion_cmd.recovery_env_ratio = recovery_env_ratio
    motion_cmd.fallen_init_ratio = fallen_init_ratio
    motion_cmd.init_pos_file = _FALLEN_INIT_PTH

    # 2) Termination: replace immediate tracking terminations with a
    #    tolerant version — normal envs terminate instantly; recovery envs
    #    get a continuous-bad-tracking window before the reset is released.
    cfg.terminations["tracking_failure"] = TerminationTermCfg(
      func=mdp.TolerantTermination(
        bad_tracking_time_threshold_s=bad_tracking_time_threshold_s,
        command_name="motion",
        terms=[
          ("anchor_pos_z", mdp.bad_anchor_pos_z_only, {
            "command_name": "motion",
            "threshold": 0.25,
          }),
          ("anchor_ori", mdp.bad_anchor_ori, {
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
          }),
          ("ee_body_pos_z", mdp.bad_motion_body_pos_z_only, {
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": (
              "left_ankle_roll_link",
              "right_ankle_roll_link",
              "left_wrist_yaw_link",
              "right_wrist_yaw_link",
            ),
          }),
        ],
      ),
      params={},
    )
    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("anchor_ori", None)
    cfg.terminations.pop("ee_body_pos", None)

    # 3) Startup: wrap the termination manager with the recovery window.
    cfg.events["install_recovery_termination"] = EventTermCfg(
      func=mdp.install_recovery_termination,
      mode="startup",
      params={
        "recovery_env_ratio": recovery_env_ratio,
        "max_recovery_steps": max_recovery_steps,
      },
    )

    # 4) Interval-every-step: upward assist force on the torso while a
    #    recovery env is down (AMP_mjlab recovery_assist_force), decaying
    #    linearly with the training iteration and fully withdrawn at 0.
    if not play:
      cfg.events["recovery_assist_force"] = EventTermCfg(
        func=mdp.apply_recovery_assist_force,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # == step_dt (decimation 4 @ 200Hz).
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
          # Fire on the first delayed-recovery step (counter==1), i.e. right
          # when TolerantTermination announces failure — not at window midpoint.
          # Waiting until max_recovery_steps//2 leaves too little time to stand.
          "trigger_recovery_steps": 1,
          "apply_prob": 0.8,
          "initial_force": recovery_assist_initial_force,
          "duration_steps": 25,
        },
      )
      # AMP_mjlab-style curriculum term: decays the event's force and logs
      # Curriculum/recovery_assist_force into the per-iteration block.
      cfg.curriculum["recovery_assist_force"] = CurriculumTermCfg(
        func=mdp.recovery_assist_force,
        params={
          "initial_force": recovery_assist_initial_force,
          "force_decay": 20.0,
          "decay_interval_iters": 500,
          "steps_per_iter": 24,
        },
      )

    # 5) Stand-up reward shaping (KungFuAthleteBot penalties):
    #    - penalty_relative_root_orientation (-0.5): pelvis orientation
    #      error w.r.t. the reference — encourages rolling the pelvis
    #      upright first instead of tumbling around.
    #    - penalty_xy_rate_before_stand (-1.0): stand-up-gated horizontal
    #      drift penalty — get up in place instead of scooting away.
    cfg.rewards["penalty_relative_root_orientation"] = RewardTermCfg(
      func=mdp.penalty_relative_root_orientation,
      weight=-0.5,
      params={"command_name": "motion"},
    )
    cfg.rewards["penalty_xy_rate_before_stand"] = RewardTermCfg(
      func=mdp.penalty_xy_rate_before_stand,
      weight=-1.0,
      params={"command_name": "motion", "stand_threshold": 0.1},
    )
    #    - reward_reference_height_before_stand (2.5): stand-up-gated
    #      reward for the pelvis height matching the reference — pulls the
    #      robot up toward the reference posture while it is still down.
    cfg.rewards["reward_reference_height_before_stand"] = RewardTermCfg(
      func=mdp.reward_reference_height_before_stand,
      weight=2.5,
      params={"command_name": "motion", "stand_threshold": 0.1, "std": 0.25},
    )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("push_robot_2", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

    # Play: no fallen-init envs (grading normal tracking from motion start);
    # keep the tolerant termination so manual knock-downs can recover.
    if enable_fall_recovery:
      motion_cmd.fallen_init_ratio = 0.0

  return cfg
