from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_flat_dance_env_cfg, unitree_g1_flat_tracking_env_cfg
from .rl_cfg import unitree_g1_fall_recovery_ppo_runner_cfg, unitree_g1_tracking_ppo_runner_cfg

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
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Dance-Fall-Recovery",
  env_cfg=unitree_g1_flat_dance_env_cfg(enable_fall_recovery=True),
  play_env_cfg=unitree_g1_flat_dance_env_cfg(enable_fall_recovery=True, play=True),
  rl_cfg=unitree_g1_fall_recovery_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Dance-Fall-Recovery-No-State-Estimation",
  env_cfg=unitree_g1_flat_dance_env_cfg(
    has_state_estimation=False,
    enable_fall_recovery=True,
  ),
  play_env_cfg=unitree_g1_flat_dance_env_cfg(
    has_state_estimation=False,
    enable_fall_recovery=True,
    play=True,
  ),
  rl_cfg=unitree_g1_fall_recovery_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Notes:
# - The plain Dance tasks (Mjlab-Tracking-Flat-Unitree-G1-Dance and its
#   No-State-Estimation variant) were removed; use the Fall-Recovery tasks.
# - Fall-recovery design: 50% normal envs; of the other 50% recovery envs,
#   half fall mid-episode and get a get-up window (delayed termination +
#   upward assist force, AMP_mjlab style), half initialize lying on the
#   ground from a pre-collected diverse fallen-state pool (KungFuAthleteBot
#   style).
# - Logs go to logs/rsl_rl/g1_fall_recovery/ (separate from g1_tracking).
