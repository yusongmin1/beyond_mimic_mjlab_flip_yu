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
from mjlab.tasks.tracking.tracking_env_cfg_dance import (
  VELOCITY_RANGE_ADD,
  make_dance_env_cfg,
)

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
  recovery_env_ratio: float = 0.4,
  fallen_init_ratio: float = 1.0,
  max_recovery_steps: int = 100,
  bad_tracking_time_threshold_s: float = 2.0,
  recovery_assist_initial_force: float = 500.0,
  fall_recovery_stage: int = 1,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 dance tracking configuration.

  Same G1 overrides as the tracking task, but built on the dance base
  config (tracking_env_cfg_dance.py) with its relaxed anchor rewards.

  Args:
    enable_fall_recovery: AMP_mjlab-style two-group split — 60% normal /
      40% fall-recovery (``recovery_env_ratio``). Recovery envs get a
      get-up window after falling (delayed termination + tolerant tracking
      termination). By default the entire recovery group
      (``fallen_init_ratio=1.0``) starts lying on the ground from a
      pre-collected pool (KungFuAthleteBot robot_init_states format).
      The split is re-rolled randomly on every reset (dynamic, not fixed
      env ids).
    recovery_env_ratio: Fraction of envs in the fall-recovery group
      (0.4 => 40% recovery / 60% normal, re-sampled each reset).
    fallen_init_ratio: Fraction of the recovery group that is initialized
      fallen (1.0 => all 40% recovery envs start from the fallen pool).
    max_recovery_steps: Delayed get-up window after TolerantTermination
      fires (default 100 ≈ 2 s at step_dt=0.02). Total time down before
      ``penalty_recovery_window_timeout`` (-200) is
      bad_tracking_time_threshold_s + max_recovery_steps * step_dt
      (= 4 s with defaults: 2 s tolerant + 2 s delay ≈ 200 env steps).
    bad_tracking_time_threshold_s: Tolerant-termination window for
      recovery envs (seconds of continuous bad tracking). Together with
      max_recovery_steps the total get-up budget is 4 s by default.
    recovery_assist_initial_force: Initial upward assist force (N) on the
      torso while a recovery env is down; decays to 0 over training.
    fall_recovery_stage: Both stages use anchor z 0.25 m. Stage-I uses
      baseline push/RSI; Stage-II adds KungFu Stage-III-style stronger
      push + larger RSI pose/velocity (``VELOCITY_RANGE_ADD``). Both
      share tolerant + delayed windows.
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
    fall_recovery_stage=fall_recovery_stage,
  )


def unitree_g1_flat_dance_env_cfg_stage_ii(
  has_state_estimation: bool = True,
  tau_mode: TauMode = "off",
  tau_filter: TauFilter = "mean",
  play: bool = False,
  enable_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Dance Stage-II (KungFuAthleteBot-style): tighter tracking + stronger push.

  Relative to Stage-I (plain ``unitree_g1_flat_dance_env_cfg``):
  - terminations: full ``anchor_pos`` (1.0 m) instead of z-only; ``anchor_ori``
    0.8 → 0.6; drop ``ee_body_pos`` z-only (same swaps as KungFu Stage-II);
  - push / RSI velocity: ``VELOCITY_RANGE_ADD`` (same as KungFu Stage-III push);
  - RSI pose noise unchanged from Stage-I (±0.05 m xy / ±0.01 m z).

  Intended to resume from a Stage-I Dance checkpoint.
  """
  cfg = unitree_g1_flat_dance_env_cfg(
    has_state_estimation=has_state_estimation,
    tau_mode=tau_mode,
    tau_filter=tau_filter,
    play=play,
    enable_actuator_delay=enable_actuator_delay,
    enable_fall_recovery=False,
  )
  return _apply_dance_stage_ii_overrides(cfg, play=play)


def _apply_dance_stage_ii_overrides(
  cfg: ManagerBasedRlEnvCfg, *, play: bool
) -> ManagerBasedRlEnvCfg:
  """Apply KungFu-style Stage-II tracking limits + stronger push (config only)."""
  # --- Stricter tracking terminations (KungFu Stage-II) ---
  cfg.terminations["anchor_pos"] = TerminationTermCfg(
    func=mdp.bad_anchor_pos,
    params={"command_name": "motion", "threshold": 1.0},
  )
  cfg.terminations["anchor_ori"].params["threshold"] = 0.6
  cfg.terminations.pop("ee_body_pos", None)

  # Play mode already clears RSI / push in ``_apply_g1_overrides``; keep that.
  if play:
    return cfg

  # --- Stronger push + RSI velocity (KungFu Stage-III push magnitude) ---
  # RSI pose noise stays at Stage-I values (dance base cfg); only the
  # velocity noise is enlarged.
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.velocity_range = VELOCITY_RANGE_ADD

  if "push_robot" in cfg.events:
    cfg.events["push_robot"].params["velocity_range"] = VELOCITY_RANGE_ADD

  return cfg


def _apply_g1_overrides(
  cfg: ManagerBasedRlEnvCfg,
  has_state_estimation: bool,
  tau_mode: TauMode,
  tau_filter: TauFilter,
  play: bool,
  enable_actuator_delay: bool,
  enable_fall_recovery: bool = False,
  recovery_env_ratio: float = 0.5,
  fallen_init_ratio: float = 1.0,
  max_recovery_steps: int = 100,
  bad_tracking_time_threshold_s: float = 2.0,
  recovery_assist_initial_force: float = 250.0,
  fall_recovery_stage: int = 1,
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
    #    Both stages use z 0.25 m (0.5 m is too loose to train well).
    #    Stage-II difference is stronger push / RSI (below), not z.
    z_threshold = 0.25
    cfg.terminations["tracking_failure"] = TerminationTermCfg(
      func=mdp.TolerantTermination(
        bad_tracking_time_threshold_s=bad_tracking_time_threshold_s,
        command_name="motion",
        terms=[
          ("anchor_pos_z", mdp.bad_anchor_pos_z_only, {
            "command_name": "motion",
            "threshold": z_threshold,
          }),
          ("anchor_ori", mdp.bad_anchor_ori, {
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
          }),
        ],
      ),
      params={},
    )
    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("anchor_ori", None)
    cfg.terminations.pop("ee_body_pos", None)

    # 2b) Stage-II: KungFu Stage-III-style stronger disturbance (train only).
    #     Larger RSI pose/velocity + VELOCITY_RANGE_ADD push. Play clears
    #     these later in the play overrides below.
    if fall_recovery_stage >= 2 and not play:
      motion_cmd.pose_range = {
        "x": (-0.15, 0.15),
        "y": (-0.15, 0.15),
        "z": (-0.15, 0.15),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      }
      motion_cmd.velocity_range = VELOCITY_RANGE_ADD
      if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = VELOCITY_RANGE_ADD

    # 3) Startup: wrap the termination manager with the recovery window.
    #    The recovery set is dynamic (re-rolled on every reset via the
    #    command-side mask): 60% normal / 40% fall-recovery (AMP_mjlab
    #    delay_reset_env_ratio=0.4). With fallen_init_ratio=1.0 the whole
    #    40% starts from the fallen pool.
    cfg.events["install_recovery_termination"] = EventTermCfg(
      func=mdp.install_recovery_termination,
      mode="startup",
      params={
        "command_name": "motion",
        "recovery_env_ratio": recovery_env_ratio,
        "max_recovery_steps": max_recovery_steps,
      },
    )

    # 4) Interval-every-step: upward assist force on the torso while a
    #    recovery env is down (AMP_mjlab recovery_assist_force). Training
    #    decays it via curriculum; play freezes force to the checkpoint
    #    iteration on the same schedule (apply_prob=0.8) with force arrows.
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
    # action_rate_l2: curriculum for fall-recovery only (start mild, tighten).
    # Initial -0.025; every 500 PPO iters subtract 0.005; floor at -0.1.
    cfg.rewards["action_rate_l2"].weight = -0.025
    if not play:
      # AMP_mjlab-style curriculum term: decays the event's force and logs
      # Curriculum/recovery_assist_force into the per-iteration block.
      cfg.curriculum["recovery_assist_force"] = CurriculumTermCfg(
        func=mdp.recovery_assist_force,
        params={
          "event_name": "recovery_assist_force",
          "initial_force": recovery_assist_initial_force,
          "force_decay": 20.0,
          "decay_interval_iters": 500,
          "steps_per_iter": 24,
        },
      )
      cfg.curriculum["action_rate_l2"] = CurriculumTermCfg(
        func=mdp.action_rate_l2_weight,
        params={
          "reward_name": "action_rate_l2",
          "initial_weight": -0.025,
          "weight_decay": 0.005,
          "min_weight": -0.1,
          "decay_interval_iters": 500,
          "steps_per_iter": 24,
        },
      )

    # 5) Stand-up reward shaping (AMP_mjlab mask_delay style): only active
    #    on fall-recovery envs that are currently fallen / getting up
    #    (is_recovery_env & (recovery_counters > 0 | bad_tracking > 0)).
    #    Normal envs and upright recovery envs get 0.
    #    - penalty_relative_root_orientation (-0.5): pelvis orientation
    #      error w.r.t. the reference — encourages rolling the pelvis
    #      upright first instead of tumbling around.
    #    - penalty_xy_rate_before_stand (-1.0): horizontal drift penalty —
    #      get up in place instead of scooting away.
    #    - reward_reference_height_before_stand (2.5): pelvis height
    #      matching the reference while down (AMP track_root_height).
    #    - penalty_recovery_window_timeout (-200): single-step penalty when
    #      a recovery env stays down for the full ~4 s get-up budget
    #      (2 s tolerant + 100-step delay); successful get-up / normal
    #      envs get 0.
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
    cfg.rewards["reward_reference_height_before_stand"] = RewardTermCfg(
      func=mdp.reward_reference_height_before_stand,
      weight=2.5,
      params={"command_name": "motion", "stand_threshold": 0.1, "std": 0.25},
    )
    cfg.rewards["penalty_recovery_window_timeout"] = RewardTermCfg(
      func=mdp.penalty_recovery_window_timeout,
      weight=-200.0,
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
    # Fall-recovery play keeps training episode length (10 s) so rounds match
    # train; other tracking play tasks keep the long interactive horizon.
    if not enable_fall_recovery:
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
    # Assist force is frozen in play.py to the checkpoint-iteration value
    # (same curriculum schedule as training); drop live curriculum updates.
    if enable_fall_recovery:
      motion_cmd.fallen_init_ratio = 0.0
      cfg.curriculum.pop("recovery_assist_force", None)
      cfg.curriculum.pop("action_rate_l2", None)

  return cfg
