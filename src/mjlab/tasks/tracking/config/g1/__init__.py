from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_dance_env_cfg,
  unitree_g1_flat_dance_env_cfg_stage_ii,
  unitree_g1_flat_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_dance_ppo_runner_cfg,
  unitree_g1_fall_recovery_ppo_runner_cfg,
  unitree_g1_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation-Tau-Reward",
  env_cfg=unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=False,
    tau_mode="reward",
    tau_filter="mean",
  ),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=False,
    tau_mode="reward",
    tau_filter="mean",
    play=True,
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation-Torque",
  env_cfg=unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=False,
    tau_mode="actor_critic",
    tau_filter="mean",
  ),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=False,
    tau_mode="actor_critic",
    tau_filter="mean",
    play=True,
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Dance",
  env_cfg=unitree_g1_flat_dance_env_cfg(),
  play_env_cfg=unitree_g1_flat_dance_env_cfg(play=True),
  rl_cfg=unitree_g1_dance_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Dance-No-State-Estimation",
  env_cfg=unitree_g1_flat_dance_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_dance_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_dance_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Dance-No-State-Estimation-Stage-II",
  env_cfg=unitree_g1_flat_dance_env_cfg_stage_ii(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_dance_env_cfg_stage_ii(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_dance_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Dance-Fall-Recovery",
  env_cfg=unitree_g1_flat_dance_env_cfg(enable_fall_recovery=True, fall_recovery_stage=1),
  play_env_cfg=unitree_g1_flat_dance_env_cfg(enable_fall_recovery=True, fall_recovery_stage=1, play=True),
  rl_cfg=unitree_g1_fall_recovery_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Dance-Fall-Recovery-No-State-Estimation",
  env_cfg=unitree_g1_flat_dance_env_cfg(
    has_state_estimation=False,
    enable_fall_recovery=True,
    fall_recovery_stage=1,
  ),
  play_env_cfg=unitree_g1_flat_dance_env_cfg(
    has_state_estimation=False,
    enable_fall_recovery=True,
    fall_recovery_stage=1,
    play=True,
  ),
  rl_cfg=unitree_g1_fall_recovery_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Dance-Fall-Recovery-No-State-Estimation-Stage-II",
  env_cfg=unitree_g1_flat_dance_env_cfg(
    has_state_estimation=False,
    enable_fall_recovery=True,
    fall_recovery_stage=2,
  ),
  play_env_cfg=unitree_g1_flat_dance_env_cfg(
    has_state_estimation=False,
    enable_fall_recovery=True,
    fall_recovery_stage=2,
    play=True,
  ),
  rl_cfg=unitree_g1_fall_recovery_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Notes:
# - Plain Dance tasks use tracking_env_cfg_dance.py (relaxed anchors) without
#   fall-recovery machinery; logs go to logs/rsl_rl/g1_dance/.
# - Fall-recovery design: 60% normal / 40% recovery (recovery_env_ratio=0.4);
#   with fallen_init_ratio=1.0 the whole recovery group starts from the
#   fallen-state pool (delayed termination + upward assist, AMP style).
# - Fall-Recovery: both stages use z 0.25 m. Stage-I baseline push/RSI;
#   Stage-II adds KungFu Stage-III-style VELOCITY_RANGE_ADD push and
#   larger RSI pose/velocity. Windows (2s tolerant + 2s delayed) shared.
# - Fall-Recovery logs go to logs/rsl_rl/g1_fall_recovery/.
