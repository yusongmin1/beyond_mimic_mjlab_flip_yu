"""29-DoF G1 Legged Lab locomotion controller (from legged_lab_yu deploy).

Single-frame obs (96-D, Lab joint order):
  ang_vel*s(3) + projected_gravity(3) + cmd*cmd_scale(3)
  + dof_pos_rel(29) + dof_vel*s(29) + last_action(29)

Policy input = history_length(10) x 96 = 960.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_CFG = _PROJECT_ROOT / "configs" / "g1_legged_lab_rl.yaml"


def project_gravity_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body-frame gravity, matching deploy_mujoco ``math_lab.project_grav``."""
    v = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    q_w = float(quat_wxyz[0])
    q_vec = quat_wxyz[1:4].astype(np.float64)
    a = v * (2.0 * q_w * q_w - 1.0)
    b = 2.0 * q_w * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.dot(q_vec, v)
    return (a - b + c).astype(np.float64)


def _resolve_path(p: str) -> str:
    path = Path(p)
    if path.is_file():
        return str(path.resolve())
    cand = (_PROJECT_ROOT / p).resolve()
    if cand.is_file():
        return str(cand)
    return str(cand)


class LeggedLabG1Controller:
    """TorchScript DWAQ locomotion policy with obs history."""

    def __init__(self, config_path: str | None = None, policy_path: str | None = None):
        import torch

        self.torch = torch
        cfg_path = Path(config_path) if config_path else _DEFAULT_CFG
        if not cfg_path.is_file():
            raise FileNotFoundError(f"RL config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.cfg_path = str(cfg_path.resolve())
        self.joint_names_lab = list(cfg["joint_names_lab"])
        self.joint_names_mujoco = list(cfg["joint_names_mujoco"])
        if len(self.joint_names_lab) != len(self.joint_names_mujoco):
            raise ValueError("joint_names_lab / joint_names_mujoco length mismatch")

        # lab_idx -> mujoco_idx
        self.lab_to_mj = np.array(
            [self.joint_names_mujoco.index(n) for n in self.joint_names_lab], dtype=np.int64
        )

        self.num_actions = int(cfg["num_actions"])
        self.num_obs = int(cfg.get("num_obs_add", cfg["num_obs"]))
        self.history_length = int(cfg["history_length"])
        self.ang_vel_scale = float(cfg["ang_vel_scale"])
        self.dof_pos_scale = float(cfg["dof_pos_scale"])
        self.dof_vel_scale = float(cfg["dof_vel_scale"])
        self.action_scale = float(cfg["action_scale"])
        self.cmd_scale = np.asarray(cfg["cmd_scale"], dtype=np.float64)
        self.clip_actions = float(cfg.get("clip_actions", 100.0))

        self.default_lab = np.asarray(cfg["default_angles_lab"], dtype=np.float64)
        self.default_mj = np.asarray(
            cfg.get("default_angles_mujoco")
            or [
                self.default_lab[self.joint_names_lab.index(n)]
                for n in self.joint_names_mujoco
            ],
            dtype=np.float64,
        )
        # RL default in mujoco order (from lab defaults)
        self.default_lab_as_mj = np.zeros(self.num_actions, dtype=np.float64)
        self.default_lab_as_mj[self.lab_to_mj] = self.default_lab

        self.kp_mj = np.asarray(cfg["kps"], dtype=np.float64)
        self.kd_mj = np.asarray(cfg["kds"], dtype=np.float64)

        path = policy_path or cfg["policy_path"]
        path = _resolve_path(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Legged Lab policy not found: {path}")
        self.path = path
        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()

        self.commands = np.zeros(3, dtype=np.float64)  # vx, vy, yaw
        self.last_action = np.zeros(self.num_actions, dtype=np.float64)  # lab order
        self.history = np.zeros((self.history_length, self.num_obs), dtype=np.float32)
        self._hist_primed = False
        self.enter()

    def enter(self):
        self.commands[:] = 0.0
        self.last_action[:] = 0.0
        self.history[:] = 0.0
        self._hist_primed = False

    def nudge_command(self, dx=0.0, dy=0.0, dyaw=0.0, step=0.2):
        self.commands[0] = float(np.clip(self.commands[0] + dx * step, -1.5, 1.5))
        self.commands[1] = float(np.clip(self.commands[1] + dy * step, -1.0, 1.0))
        self.commands[2] = float(np.clip(self.commands[2] + dyaw * step, -1.5, 1.5))

    def clear_command(self):
        self.commands[:] = 0.0

    def stand_pose_mj(self) -> np.ndarray:
        """Spawn / soft-stand pose in mujoco order."""
        return self.default_mj.copy()

    def rl_gains_mj(self) -> tuple[np.ndarray, np.ndarray]:
        return self.kp_mj.copy(), self.kd_mj.copy()

    def build_obs_row(
        self,
        base_quat_wxyz: np.ndarray,
        ang_vel_body: np.ndarray,
        q_mj: np.ndarray,
        qd_mj: np.ndarray,
    ) -> np.ndarray:
        q_lab = q_mj[self.lab_to_mj]
        qd_lab = qd_mj[self.lab_to_mj]
        omega = ang_vel_body.astype(np.float64) * self.ang_vel_scale
        grav = project_gravity_wxyz(base_quat_wxyz)
        cmd = self.commands * self.cmd_scale
        q_rel = (q_lab - self.default_lab) * self.dof_pos_scale
        qd_s = qd_lab * self.dof_vel_scale
        row = np.concatenate(
            [omega, grav, cmd, q_rel, qd_s, self.last_action]
        ).astype(np.float32)
        assert row.shape[0] == self.num_obs, f"obs dim {row.shape[0]} != {self.num_obs}"
        return row

    def _push_history(self, row: np.ndarray):
        if not self._hist_primed:
            self.history[:] = row.reshape(1, -1)
            self._hist_primed = True
        else:
            self.history = np.concatenate([self.history[1:], row.reshape(1, -1)], axis=0)

    def act(
        self,
        base_quat_wxyz: np.ndarray,
        ang_vel_body: np.ndarray,
        q_mj: np.ndarray,
        qd_mj: np.ndarray,
    ) -> np.ndarray:
        """Return q_des in mujoco / mjlab joint order."""
        row = self.build_obs_row(base_quat_wxyz, ang_vel_body, q_mj, qd_mj)
        self._push_history(row)
        x = self.torch.from_numpy(self.history.reshape(1, -1))
        with self.torch.no_grad():
            action = self.model(x)[0].cpu().numpy().astype(np.float64)
        self.last_action = action
        # target in lab order, then scatter to mujoco
        q_des_lab = self.default_lab + self.action_scale * action
        q_des_lab = np.clip(q_des_lab, -self.clip_actions, self.clip_actions)
        q_des_mj = np.zeros(self.num_actions, dtype=np.float64)
        q_des_mj[self.lab_to_mj] = q_des_lab
        return q_des_mj


def default_config_path() -> str:
    return str(_DEFAULT_CFG)


def default_policy_path() -> str:
    return str((_PROJECT_ROOT / "policies" / "g1_lab" / "policy_dwaq.pt").resolve())
