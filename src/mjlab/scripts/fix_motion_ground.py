"""Fix root-height calibration drift in a G1 motion npz.

Support phases (lowest foot below 0.15m): shift the whole kinematic tree
down by max(0, lowest_foot_z - STAND_Z) so the support foot rests exactly
on the ground. Airborne phases (flips) are left untouched. The correction
is smoothed (moving average) and vertical velocities are adjusted for
consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

STAND_Z = 0.0361  # ankle_roll origin height when standing (measured from model)
CONTACT_THRESH = 0.15  # lowest foot below this => support phase
SMOOTH_WIN = 21  # frames (~0.4s @ 50Hz)

L_ANKLE_ROLL = 6  # body index in mjlab G1 order
R_ANKLE_ROLL = 12


def ground_motion(src: Path, dst: Path) -> dict[str, float]:
  d = np.load(src)
  pos = d["body_pos_w"].copy()
  vel = d["body_lin_vel_w"].copy()
  fps = float(d["fps"][0])
  dt = 1.0 / fps

  m = np.minimum(pos[:, L_ANKLE_ROLL, 2], pos[:, R_ANKLE_ROLL, 2])

  # Per-frame correction, support phases only.
  support = m < CONTACT_THRESH
  corr = np.where(support, np.maximum(0.0, m - STAND_Z), 0.0)

  # Moving-average smoothing with edge padding (head/tail keep full correction,
  # so the very first frame — the RL reset pose — is already grounded).
  kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
  pad = SMOOTH_WIN // 2
  corr_padded = np.pad(corr, (pad, pad), mode="edge")
  corr_s = np.convolve(corr_padded, kernel, mode="same")[pad : pad + len(corr)]
  # Safety clamp: never sink a frame below ground level. Frames already at or
  # below standing height (landing compression) get zero correction; others
  # are lowered at most to STAND_Z. Prevents smoothing bleed-through from
  # pushing calibrated contact frames into the ground.
  floor = np.minimum(m, STAND_Z)
  corr_s = np.minimum(corr_s, np.maximum(0.0, m - floor))

  # Apply rigid downward translation to all bodies.
  pos[:, :, 2] -= corr_s[:, None]

  # Vertical velocity compensation: v_z -= d(corr)/dt (central difference).
  dcorr = np.zeros_like(corr_s)
  dcorr[1:-1] = (corr_s[2:] - corr_s[:-2]) / (2 * dt)
  dcorr[0] = (corr_s[1] - corr_s[0]) / dt
  dcorr[-1] = (corr_s[-1] - corr_s[-2]) / dt
  vel[:, :, 2] -= dcorr[:, None]

  out = {k: (d[k] if k != "body_pos_w" else pos) for k in d.files}
  out["body_lin_vel_w"] = vel
  dst.parent.mkdir(parents=True, exist_ok=True)
  np.savez(dst, **out)

  m_new = np.minimum(pos[:, L_ANKLE_ROLL, 2], pos[:, R_ANKLE_ROLL, 2])
  stats = {
    "before_head": float(m[:30].mean()),
    "after_head": float(m_new[:30].mean()),
    "before_tail": float(m[-30:].mean()),
    "after_tail": float(m_new[-30:].mean()),
    "max_correction": float(corr_s.max()),
  }
  return stats


if __name__ == "__main__":
  src = Path("motions/chunwan/90_15_stageii_unitree_g1.npz")
  dst = Path("motions/chunwan/90_15_stageii_unitree_g1_fixed.npz")
  stats = ground_motion(src, dst)
  print(f"saved {dst}")
  print(
    f"开头30帧最低脚: {stats['before_head']:.3f} -> {stats['after_head']:.3f} m"
  )
  print(
    f"结尾30帧最低脚: {stats['before_tail']:.3f} -> {stats['after_tail']:.3f} m"
  )
  print(f"最大下沉量: {stats['max_correction']:.3f} m (应出现在腾空段之外)")
