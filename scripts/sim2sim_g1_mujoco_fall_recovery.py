"""Sim-to-Sim: G1 Fall-Recovery policies in MuJoCo (explicit PD torque).

Same stack as ``sim2sim_g1_mujoco.py``, but policies come from
``logs/rsl_rl/g1_fall_recovery/`` (Stage-I / Stage-II dance fall-recovery
runs). Each run that has a motion-ONNX (``*.onnx`` or ``exported/policy.onnx``)
is listed. The ONNX embeds the FULL reference motion (joint pos/vel +
14 tracking bodies) and takes ``(obs, time_step)``; the reference for the ghost
and reset is read straight out of the graph — no external ``--motion_file``
needed. A plain ``.pt``/``.onnx`` policy path plus explicit ``--motion_file``
is still supported.

PD gains / armature / per-joint action scales match
``mjlab.asset_zoo.robots.unitree_g1.g1_constants`` (NATURAL_FREQ=10 Hz, ζ=2):

    motor       armature      kp         kd        effort   action_scale (0.25*e/kp)
    5020        0.003609725   14.2506    0.9072    25       0.4386   (shoulder/elbow/wrist_roll)
    7520_14     0.010177520   40.1792    2.5579    88       0.5475   (hip_pitch/yaw, waist_yaw)
    7520_22     0.025101925   99.0984    6.3088   139       0.3507   (hip_roll, knee)
    4010        0.004250000   16.7783    1.0681     5       0.0745   (wrist_pitch/yaw)
    2×5020      0.007219450   28.5012    1.8144    50       0.4386   (ankle, waist_pitch/roll)

NOTE: Fall-recovery ONNX metadata stores the TRUE training action_scale
(0.518 / 0.330 / 0.560 / 0.128); it overrides the table above when present.

Anchor body is ``torso_link`` (same as G1 fall-recovery env). Observation default is
Wo-State-Estimation (154-D); pass ``--with_state_estimation`` for 160-D.

Starts in **mimic** on the first fall-recovery policy (override with ``--policy`` /
``--start_rl``). Press ``[`` / ``]`` or ``0`` / ``1`` **any time** (including while
a skill is running) to hot-switch Stage-I ↔ Stage-II; ``L`` returns to RL locomotion.

.. code-block:: bash

    # list Stage-I / Stage-II
    python scripts/sim2sim_g1_mujoco_fall_recovery.py --list

    # start Stage-I; press ] or 1 in the viewer to jump to Stage-II
    python scripts/sim2sim_g1_mujoco_fall_recovery.py --match_isaac

    # start Stage-II directly
    python scripts/sim2sim_g1_mujoco_fall_recovery.py --policy stage2 --match_isaac
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import re
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(
    description="G1 Fall-Recovery sim2sim in MuJoCo (logs/rsl_rl/g1_fall_recovery)."
)
parser.add_argument(
    "--policy",
    type=str,
    default=None,
    help="Initial mimic skill: INDEX (see --list), SUBSTRING (fall/stage2), or path. "
    "Default: first listed policy.",
)
parser.add_argument("--list", action="store_true", help="list available g1_fall_recovery policies and exit.")
parser.add_argument(
    "--start_rl",
    action="store_true",
    help="Start in RL locomotion instead of mimic (default for this script is mimic).",
)
parser.add_argument("--motion_file", type=str, default=None, help="External reference motion npz (only for non-embedded policies).")
parser.add_argument("--xml", type=str, default=None, help="MuJoCo g1 xml (default: unitree_g1/xmls/g1_scene.xml).")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed.")
parser.add_argument("--kp", type=float, default=None, help="Uniform PD stiffness override (mimic mode).")
parser.add_argument("--kd", type=float, default=None, help="Uniform PD damping override (mimic mode).")
parser.add_argument(
    "--action_scale",
    type=float,
    default=None,
    help="Uniform action scale override (default: per-joint mjlab G1_ACTION_SCALE).",
)
parser.add_argument(
    "--with_state_estimation",
    action="store_true",
    help="Include motion_anchor_pos_b + base_lin_vel (160-D). Default is Wo-SE 154-D.",
)
parser.add_argument(
    "--loop_mimic",
    action="store_true",
    default=True,
    help="Loop the mimic skill instead of auto-returning to RL (default: on).",
)
parser.add_argument(
    "--no_loop_mimic",
    action="store_true",
    help="Disable looping; return to RL when the current motion ends.",
)
parser.add_argument("--no_ghost", action="store_true", help="Do not draw reference ghost.")
parser.add_argument(
    "--match_isaac",
    action="store_true",
    help="Set joint armature to mjlab values and zero passive dof damping/frictionloss.",
)
parser.add_argument("--headless", action="store_true", help="No viewer; one motion pass then exit.")
parser.add_argument(
    "--settle",
    type=float,
    default=0.0,
    help="Seconds of PD-hold on frame 0 before the policy starts (default 0).",
)
parser.add_argument(
    "--replay",
    action="store_true",
    help="Bypass policy; PD-track reference joint angles (plant sanity check).",
)
parser.add_argument(
    "--rl_policy",
    type=str,
    default=None,
    help="Path to Legged Lab G1 locomotion TorchScript (default: policies/g1_lab/policy_dwaq.pt).",
)
parser.add_argument(
    "--rl_config",
    type=str,
    default=None,
    help="YAML for Legged Lab RL (default: configs/g1_legged_lab_rl.yaml).",
)
parser.add_argument(
    "--start_mimic",
    action="store_true",
    help="(Deprecated / default) Start in mimic mode. Use --start_rl for RL.",
)
args = parser.parse_args()
if args.no_loop_mimic:
    args.loop_mimic = False

import mujoco
import mujoco.viewer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MJLAB_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_XML = os.path.join(
    _MJLAB_ROOT, "src", "mjlab", "asset_zoo", "robots", "unitree_g1", "xmls", "g1_scene.xml"
)
_TRACKING_DIR = os.path.join(_MJLAB_ROOT, "logs", "rsl_rl", "g1_fall_recovery")

# Joint order used by mjlab csv_to_npz / G1 entity (matches g1.xml qpos order).
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
N = len(JOINT_NAMES)  # 29

# 14 tracking bodies embedded in the fall-recovery motion-ONNX.
TRACKING_BODIES = [
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
]

# mjlab g1_constants: armature from two-stage planetary reflected inertia.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520041
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# (regex, kp, kd, armature, effort_limit)
_ACTUATOR_GROUPS = [
    (r".*_elbow_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_shoulder_yaw_joint|.*_wrist_roll_joint",
     STIFFNESS_5020, DAMPING_5020, ARMATURE_5020, 25.0),
    (r".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint",
     STIFFNESS_7520_14, DAMPING_7520_14, ARMATURE_7520_14, 88.0),
    (r".*_hip_roll_joint|.*_knee_joint",
     STIFFNESS_7520_22, DAMPING_7520_22, ARMATURE_7520_22, 139.0),
    (r".*_wrist_pitch_joint|.*_wrist_yaw_joint",
     STIFFNESS_4010, DAMPING_4010, ARMATURE_4010, 5.0),
    (r"waist_pitch_joint|waist_roll_joint",
     2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 2.0 * ARMATURE_5020, 50.0),
    (r".*_ankle_pitch_joint|.*_ankle_roll_joint",
     2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 2.0 * ARMATURE_5020, 50.0),
]


def _joint_props(name: str):
    for pat, kp, kd, arm, effort in _ACTUATOR_GROUPS:
        if re.fullmatch(pat, name):
            return kp, kd, arm, effort
    raise KeyError(f"No actuator group matched joint '{name}'")


# Default pose = mjlab KNEES_BENT_KEYFRAME / G1 init_state
DEFAULT_JOINT_POS = {n: 0.0 for n in JOINT_NAMES}
for n in JOINT_NAMES:
    if n.endswith("hip_pitch_joint"):
        DEFAULT_JOINT_POS[n] = -0.312
    elif n.endswith("knee_joint"):
        DEFAULT_JOINT_POS[n] = 0.669
    elif n.endswith("ankle_pitch_joint"):
        DEFAULT_JOINT_POS[n] = -0.363
    elif n.endswith("elbow_joint"):
        DEFAULT_JOINT_POS[n] = 0.6
DEFAULT_JOINT_POS["left_shoulder_roll_joint"] = 0.2
DEFAULT_JOINT_POS["left_shoulder_pitch_joint"] = 0.2
DEFAULT_JOINT_POS["right_shoulder_roll_joint"] = -0.2
DEFAULT_JOINT_POS["right_shoulder_pitch_joint"] = 0.2

# Tracking keypoint bodies for ghost (indices into TRACKING_BODIES)
GHOST_EDGES = [
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]

ANCHOR_BODY = "torso_link"

# npz body index (mjcf 30-body tree order) for each of the 14 tracking bodies.
NPZ_BODY_IDX_30 = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]


# ----------------------------------------------------------------- policy discovery
def _short_policy_name(run: str) -> str:
    """Human-readable label for Stage-I / Stage-II fall-recovery runs."""
    low = run.lower()
    if "stage2" in low or "stage_ii" in low or "stage-ii" in low:
        return f"Stage-II  {run}"
    if "fall" in low:
        return f"Stage-I   {run}"
    return run


def list_policies() -> list[tuple[str, str]]:
    """All (display_name, path) pairs from g1_fall_recovery exported ONNX files.

    A run directory can contain BOTH a fresh ``exported/policy.onnx`` and a
    legacy ``<run>.onnx`` dropped by an older exporter. When both exist,
    prefer ``exported/policy.onnx`` and hide the legacy file.
    """
    fresh: dict[str, str] = {}
    legacy: dict[str, str] = {}
    for onnx_file in glob.glob(os.path.join(_TRACKING_DIR, "*", "exported", "policy.onnx")):
        run = os.path.basename(os.path.dirname(os.path.dirname(onnx_file)))
        fresh[run] = onnx_file
    for onnx_file in glob.glob(os.path.join(_TRACKING_DIR, "*", "*.onnx")):
        run = os.path.basename(os.path.dirname(onnx_file))
        if run not in fresh:
            legacy[run] = onnx_file
    out = []

    def _sort_key(run: str) -> tuple[int, str]:
        low = run.lower()
        if "stage2" in low or "stage_ii" in low or "stage-ii" in low:
            return (1, run)
        if "fall" in low:
            return (0, run)
        return (2, run)

    for run in sorted(set(fresh) | set(legacy), key=_sort_key):
        label = _short_policy_name(run)
        if run in fresh:
            out.append((label, fresh[run]))
        else:
            out.append((label + " (legacy onnx)", legacy[run]))
    return out


def resolve_policy(spec: str) -> str:
    polys = list_policies()
    if spec.isdigit():
        i = int(spec)
        if 0 <= i < len(polys):
            return polys[i][1]
        raise SystemExit(f"[ERROR] policy index {i} out of range (0..{len(polys) - 1}); see --list")
    if os.path.isfile(spec):
        return spec
    hits = [p for name, p in polys if spec.lower() in name.lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print("[ERROR] ambiguous policy substring, matches:")
        for name, p in polys:
            if spec.lower() in name.lower():
                print("   ", name)
        raise SystemExit(1)
    raise SystemExit(f"[ERROR] no policy matches '{spec}'; see --list")


if args.list:
    polys = list_policies()
    print(f"{len(polys)} g1_fall_recovery policies in {_TRACKING_DIR}:")
    for i, (name, p) in enumerate(polys):
        print(f"  [{i:2d}] {name}")
        print(f"        {os.path.relpath(p, _MJLAB_ROOT)}")
    raise SystemExit(0)

sys.path.insert(0, _SCRIPT_DIR)
from g1_legged_lab_locomotion import (  # noqa: E402
    LeggedLabG1Controller,
    default_config_path,
)

POLICIES = list_policies()
if not POLICIES and args.policy is None:
    raise SystemExit("[ERROR] no g1_fall_recovery policies found; need at least one for mimic skills")

if args.policy is not None:
    policy_path = resolve_policy(args.policy)
    try:
        cur_policy_idx = next(i for i, (n, p) in enumerate(POLICIES) if os.path.samefile(p, policy_path))
    except (StopIteration, OSError):
        POLICIES.insert(0, ("(adhoc) " + os.path.basename(policy_path), policy_path))
        cur_policy_idx = 0
else:
    cur_policy_idx = 0

# This script defaults to mimic (compare Stage-I / Stage-II). Use --start_rl for RL.
start_mode = "RL" if args.start_rl else "MIMIC"


# ----------------------------------------------------------------- setup
xml_path = args.xml or _DEFAULT_XML
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
model.opt.timestep = 0.005
ctrl_dt = 0.02
decimation = round(ctrl_dt / model.opt.timestep)

# name maps: policy/npz order <-> mujoco qpos/dof (mjlab order == XML order)
mj_qpos_adr, mj_dof_adr = [], []
for name in JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"joint '{name}' not in mujoco model"
    mj_qpos_adr.append(model.jnt_qposadr[jid])
    mj_dof_adr.append(model.jnt_dofadr[jid])
mj_qpos_adr = np.asarray(mj_qpos_adr)
mj_dof_adr = np.asarray(mj_dof_adr)

default_q = np.array([DEFAULT_JOINT_POS[n] for n in JOINT_NAMES], dtype=np.float64)
kp_j = np.zeros(N, dtype=np.float64)
kd_j = np.zeros(N, dtype=np.float64)
arm_j = np.zeros(N, dtype=np.float64)
effort_j = np.zeros(N, dtype=np.float64)
scale_j = np.zeros(N, dtype=np.float64)
for i, name in enumerate(JOINT_NAMES):
    kp, kd, arm, effort = _joint_props(name)
    kp_j[i] = args.kp if args.kp is not None else kp
    kd_j[i] = args.kd if args.kd is not None else kd
    arm_j[i] = arm
    effort_j[i] = effort
    scale_j[i] = 0.25 * effort / kp
scale_default_j = scale_j.copy()

if args.match_isaac:
    for adr, arm in zip(mj_dof_adr, arm_j):
        model.dof_armature[adr] = arm
        model.dof_damping[adr] = 0.0
        model.dof_frictionloss[adr] = 0.0
    print("[INFO]: joint armature/damping/frictionloss matched to mjlab Isaac setup")

free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
free_adr = model.jnt_qposadr[free_jid]
anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
assert anchor_body_id >= 0

obs_dim = 160 if args.with_state_estimation else 154
print(f"obs_dim={obs_dim} ({'with' if args.with_state_estimation else 'wo'} state estimation)")


# ----------------------------------------------------------------- policy + motion
def load_policy(path: str):
    """Returns (run_fn, meta_dict, embedded_motion_or_None)."""
    if path.endswith(".onnx"):
        import onnx
        import onnxruntime as ort

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = dict(session.get_modelmeta().custom_metadata_map)

        # Embedded motion? Extract arrays straight from graph initializers.
        emb = _extract_embedded_motion(path)
        if emb is not None:
            def run(obs: np.ndarray, t: int) -> np.ndarray:
                return session.run(
                    ["actions"],
                    {"obs": obs[None, :].astype(np.float32),
                     "time_step": np.array([[t]], dtype=np.float32)},
                )[0][0]
            return run, meta, emb

        def run_plain(obs: np.ndarray, t: int) -> np.ndarray:
            return session.run(
                ["actions"],
                {"obs": obs[None, :].astype(np.float32),
                 "time_step": np.array([[0]], dtype=np.float32)},
            )[0][0]
        return run_plain, meta, None

    # .pt: TorchScript or rsl_rl checkpoint (legacy external-motion mode)
    import torch

    try:
        module = torch.jit.load(path, map_location="cpu").eval()

        @torch.no_grad()
        def run_ts(obs: np.ndarray, t: int) -> np.ndarray:
            out = module(
                torch.from_numpy(obs.astype(np.float32))[None, :],
                torch.zeros(1, 1, dtype=torch.long),
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out[0].numpy()
        return run_ts, {}, None
    except RuntimeError:
        pass

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["actor_state_dict"]
    norm_key = "obs_norm_state_dict" if "obs_norm_state_dict" in ckpt else None
    if norm_key is not None:
        norm = ckpt[norm_key]
        norm_mean, norm_std = norm["_mean"][0] if norm["_mean"].dim() == 2 else norm["_mean"], \
                              norm["_std"][0] if norm["_std"].dim() == 2 else norm["_std"]
    else:
        norm_mean = sd["obs_normalizer._mean"][0]
        norm_std = sd["obs_normalizer._std"][0]
    import torch.nn as nn
    actor_layers = []
    i = 0
    while f"mlp.{i}.weight" in sd:
        actor_layers.append(nn.Linear(sd[f"mlp.{i}.weight"].shape[1], sd[f"mlp.{i}.weight"].shape[0]))
        actor_layers.append(nn.ELU())
        i += 2
    del actor_layers[-1]
    actor = nn.Sequential(*actor_layers)
    actor.load_state_dict({k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")})
    actor.eval()

    @torch.no_grad()
    def run_ckpt(obs: np.ndarray, t: int) -> np.ndarray:
        x = torch.from_numpy(obs.astype(np.float32))
        x = (x - norm_mean) / (norm_std + 1e-2)
        return actor(x[None, :])[0].numpy()
    return run_ckpt, {}, None


def _extract_embedded_motion(onnx_path: str):
    """Pull (T,29)/(T,14,·) motion arrays out of a motion-ONNX's initializers."""
    import onnx

    m = onnx.load(onnx_path)

    def to_np(init):
        if init.data_type == onnx.TensorProto.FLOAT and init.raw_data:
            return np.frombuffer(init.raw_data, dtype=np.float32).reshape(list(init.dims))
        return np.asarray(onnx.numpy_helper.to_array(init))

    found = {}
    for init in m.graph.initializer:
        d = list(init.dims)
        for key, sig in [
            ("joint_pos", (None, N)),
            ("joint_vel", (None, N)),
            ("body_pos_w", (None, len(TRACKING_BODIES), 3)),
            ("body_quat_w", (None, len(TRACKING_BODIES), 4)),
            ("body_lin_vel_w", (None, len(TRACKING_BODIES), 3)),
            ("body_ang_vel_w", (None, len(TRACKING_BODIES), 3)),
        ]:
            if init.name.startswith(key) and len(d) == len(sig) and (d[1] == sig[1]) and (len(d) < 3 or d[2] == sig[2]):
                found[key] = to_np(init).astype(np.float64)
    if len(found) == 6:
        return found
    return None


if args.action_scale is not None:
    scale_j[:] = args.action_scale

# ----------------------------------------------------------------- runtime
# Hot-switchable runtime: bundles policy fn + embedded motion + per-policy
# action scale so the viewer can swap everything with ONE keystroke.
class Runtime:
    def __init__(self, idx: int):
        name, path = POLICIES[idx]
        self.idx = idx
        self.name = name
        self.path = path
        run, meta, emb = load_policy(path)
        self.run = run                      # obs,t -> action(29)
        self.meta = meta
        self.emb = emb
        # action scale: metadata > global table
        self.scale = scale_j.copy()
        csv = meta.get("action_scale")
        if csv:
            vals = [float(x) for x in csv.split(",")]
            if len(vals) == N:
                self.scale[:] = vals
        if args.action_scale is not None:
            self.scale[:] = args.action_scale

    @property
    def motion_arrays(self):
        emb = self.emb if self.emb is not None else _extract_embedded_motion(self.path)
        return emb

    def switch(self, idx: int) -> "Runtime":
        """Build the next Runtime; returns self if the new one is broken."""
        try:
            nxt = Runtime(idx)
            if nxt.emb is None and args.motion_file is None:
                raise RuntimeError("no embedded motion")
            return nxt
        except Exception as e:
            print(f"[switch] FAILED to load {POLICIES[idx][0]}: {e}")
            return self


# ---- RL locomotion (legged_lab 29dof) + mimic tracking FSM ------------------
locomo = LeggedLabG1Controller(
    config_path=args.rl_config or default_config_path(),
    policy_path=args.rl_policy,
)
rl_kp_j, rl_kd_j = locomo.rl_gains_mj()
rl_stand_q = locomo.stand_pose_mj()
print(f"RL loco: {locomo.path}")
print(f"RL cfg : {locomo.cfg_path}")

rt = None  # mimic Runtime; created on first enter
M = {"jp": None, "jv": None, "bp": None, "bq": None, "bl": None, "ba": None}
T = 1
ref_anchor_pos = None
ref_anchor_quat = None

npz_body_names = TRACKING_BODIES
anchor_npz_idx = npz_body_names.index(ANCHOR_BODY)
pelvis_npz_idx = npz_body_names.index("pelvis")

ghost_edge_idx = []
if not args.no_ghost:
    for a, b in GHOST_EDGES:
        ghost_edge_idx.append((npz_body_names.index(a), npz_body_names.index(b)))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz -> 3x3 (isaaclab / mjlab matrix_from_quat)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Keep only yaw of wxyz quat (mjlab ``yaw_quat``)."""
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by wxyz quat. ``v`` shape (3,) or (..., 3)."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    v = np.asarray(v, dtype=np.float64)
    xyz = q[1:]
    t = np.cross(xyz, v) * 2.0
    return v + q[0] * t + np.cross(xyz, t)


def motion_anchor_ori_b(q_robot: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """mjlab ``motion_anchor_ori_b``: first two columns of R(q_robot^{-1} * q_ref)."""
    q_rel = quat_mul(quat_conj(q_robot), q_ref)
    return quat_to_mat(q_rel)[:, :2].reshape(-1)


def clip_effort(effort: np.ndarray) -> np.ndarray:
    return np.clip(effort, -effort_j, effort_j)


def _bind_motion(emb_new):
    global T, ref_anchor_pos, ref_anchor_quat
    M["jp"], M["jv"] = emb_new["joint_pos"], emb_new["joint_vel"]
    M["bp"], M["bq"] = emb_new["body_pos_w"], emb_new["body_quat_w"]
    M["bl"], M["ba"] = emb_new["body_lin_vel_w"], emb_new["body_ang_vel_w"]
    T = M["jp"].shape[0]
    ref_anchor_pos = M["bp"][:, anchor_npz_idx]
    ref_anchor_quat = M["bq"][:, anchor_npz_idx]


def align_motion_to_robot(emb_raw: dict) -> dict:
    """Yaw+XY align motion to current robot anchor (mjlab ``body_*_relative_w``).

    Training resets the robot onto the motion, so ``motion_anchor_ori_b`` starts
    near identity. Entering mimic from RL without teleport must rotate/translate
    the reference into the robot's current heading/position, otherwise the ori
    obs is dominated by an arbitrary world-yaw offset.
    """
    mujoco.mj_forward(model, data)
    q_robot = data.xquat[anchor_body_id].astype(np.float64).copy()
    p_robot = data.xpos[anchor_body_id].astype(np.float64).copy()

    bp0 = np.asarray(emb_raw["body_pos_w"], dtype=np.float64).copy()
    bq0 = np.asarray(emb_raw["body_quat_w"], dtype=np.float64).copy()
    bl0 = np.asarray(emb_raw["body_lin_vel_w"], dtype=np.float64).copy()
    ba0 = np.asarray(emb_raw["body_ang_vel_w"], dtype=np.float64).copy()

    q_ref0 = bq0[0, anchor_npz_idx]
    p_ref0 = bp0[0, anchor_npz_idx]
    # delta_ori = yaw(q_robot * q_ref^{-1})  — same as MotionCommand._update_command
    delta_q = yaw_quat(quat_mul(q_robot, quat_conj(q_ref0)))

    # positions: robot_xy + ref_z  +  R_yaw * (p_ref - p_ref_anchor)
    rel = bp0 - p_ref0  # (T, B, 3)
    rel_r = quat_apply(delta_q, rel)
    bp = rel_r.copy()
    bp[..., 0] += p_robot[0]
    bp[..., 1] += p_robot[1]
    bp[..., 2] += p_ref0[2]

    # quats / world velocities: rotate by the same yaw
    bq = np.zeros_like(bq0)
    for t_i in range(bq0.shape[0]):
        for b_i in range(bq0.shape[1]):
            bq[t_i, b_i] = quat_mul(delta_q, bq0[t_i, b_i])
            n = np.linalg.norm(bq[t_i, b_i])
            if n > 1e-8:
                bq[t_i, b_i] /= n
    bl = quat_apply(delta_q, bl0)
    ba = quat_apply(delta_q, ba0)

    return {
        "joint_pos": np.asarray(emb_raw["joint_pos"], dtype=np.float64),
        "joint_vel": np.asarray(emb_raw["joint_vel"], dtype=np.float64),
        "body_pos_w": bp,
        "body_quat_w": bq,
        "body_lin_vel_w": bl,
        "body_ang_vel_w": ba,
    }


def load_mimic(idx: int) -> bool:
    """Load / switch mimic Runtime; stash raw emb (unaligned)."""
    global rt
    if rt is None:
        try:
            rt = Runtime(idx)
        except Exception as e:
            print(f"[mimic] FAILED to load {POLICIES[idx][0]}: {e}")
            return False
    else:
        new_rt = rt.switch(idx)
        if new_rt is rt and idx != rt_idx[0]:
            return False
        rt = new_rt
    emb_new = rt.emb if rt.emb is not None else _extract_embedded_motion(rt.path)
    if emb_new is None and args.motion_file is not None:
        motion = np.load(args.motion_file, allow_pickle=True)
        emb_new = {
            "joint_pos": np.asarray(motion["joint_pos"], dtype=np.float64),
            "joint_vel": np.asarray(motion["joint_vel"], dtype=np.float64),
            "body_pos_w": np.asarray(motion["body_pos_w"], dtype=np.float64)[:, NPZ_BODY_IDX_30],
            "body_quat_w": np.asarray(motion["body_quat_w"], dtype=np.float64)[:, NPZ_BODY_IDX_30],
            "body_lin_vel_w": np.asarray(motion["body_lin_vel_w"], dtype=np.float64)[:, NPZ_BODY_IDX_30],
            "body_ang_vel_w": np.asarray(motion["body_ang_vel_w"], dtype=np.float64)[:, NPZ_BODY_IDX_30],
        }
    if emb_new is None:
        print(f"[mimic] {rt.name} has no embedded motion; aborted")
        return False
    rt_idx[0] = idx
    rt.emb_raw = {k: np.asarray(v, dtype=np.float64).copy() for k, v in emb_new.items()}
    _bind_motion(rt.emb_raw)
    print(f"\n[mimic] -> [{idx}] {rt.name}  (T={T}, {T / 50:.1f}s)")
    return True


def reset_mimic(t0: int = 0):
    mujoco.mj_resetData(model, data)
    data.qpos[free_adr : free_adr + 3] = M["bp"][t0, pelvis_npz_idx]
    data.qpos[free_adr + 3 : free_adr + 7] = M["bq"][t0, pelvis_npz_idx]
    data.qpos[mj_qpos_adr] = M["jp"][t0]
    data.qvel[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def reset_rl_stand():
    """Spawn standing pose for locomotion RL (rl_sar defaults)."""
    mujoco.mj_resetData(model, data)
    data.qpos[free_adr : free_adr + 3] = [0.0, 0.0, 0.78]
    data.qpos[free_adr + 3 : free_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[mj_qpos_adr] = rl_stand_q
    data.qvel[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def draw_ghost(scn, t: int):
    if M["bp"] is None:
        scn.ngeom = 0
        return
    scn.ngeom = 0
    for a, b in ghost_edge_idx:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.rgba[:] = [0.2, 1.0, 0.2, 0.6]
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            0.01,
            M["bp"][t, a].astype(np.float64),
            M["bp"][t, b].astype(np.float64),
        )
        scn.ngeom += 1


def apply_pd(q_des: np.ndarray, kp=None, kd=None):
    """Write joint torques into qfrc_applied (g1.xml has no <actuator>)."""
    q = data.qpos[mj_qpos_adr]
    qd = data.qvel[mj_dof_adr]
    kp_use = kp_j if kp is None else kp
    kd_use = kd_j if kd is None else kd
    tau = clip_effort(kp_use * (q_des - q) - kd_use * qd)
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[mj_dof_adr] = tau


# ---- hot-switch state machine ------------------------------------------
obs = np.zeros(obs_dim, dtype=np.float32)
last_action = np.zeros(N, dtype=np.float32)
cur_t = [0]
mode = {"name": start_mode}  # "RL" | "MIMIC"
switch_request = {"idx": None}       # enter / switch mimic
rl_request = {"flag": False}         # return to RL
reset_request = {"flag": False}
paused = {"flag": False}
rt_idx = [cur_policy_idx]

KEY_LBRACKET, KEY_RBRACKET = 91, 93
KEY_SPACE, KEY_R, KEY_L, KEY_C = 32, 82, 76, 67
KEY_0, KEY_9 = 48, 57
KEY_W, KEY_A, KEY_S, KEY_D = 87, 65, 83, 68
KEY_Q, KEY_E = 81, 69


def key_callback(key):
    if key == KEY_LBRACKET:
        if POLICIES:
            switch_request["idx"] = (rt_idx[0] - 1) % len(POLICIES)
    elif key == KEY_RBRACKET:
        if POLICIES:
            switch_request["idx"] = (rt_idx[0] + 1) % len(POLICIES)
    elif KEY_0 <= key <= KEY_9:
        d = key - KEY_0
        if d < len(POLICIES):
            switch_request["idx"] = d
    elif key == KEY_L:
        rl_request["flag"] = True
    elif key == KEY_SPACE:
        paused["flag"] = not paused["flag"]
    elif key == KEY_R:
        reset_request["flag"] = True
    elif key == KEY_C:
        locomo.clear_command()
    elif key == KEY_W:
        locomo.nudge_command(dx=+1)
    elif key == KEY_S:
        locomo.nudge_command(dx=-1)
    elif key == KEY_A:
        locomo.nudge_command(dy=+1)
    elif key == KEY_D:
        locomo.nudge_command(dy=-1)
    elif key == KEY_Q:
        locomo.nudge_command(dyaw=+1)
    elif key == KEY_E:
        locomo.nudge_command(dyaw=-1)


def enter_rl(hard_reset: bool = False):
    mode["name"] = "RL"
    locomo.enter()
    last_action[:] = 0.0
    if hard_reset:
        reset_rl_stand()
    print("\n[mode] -> RL locomotion")


def enter_mimic(idx: int, hard_reset: bool = False) -> bool:
    """Enter a mimic skill.

    ``hard_reset=True`` teleports to motion frame 0 — sim/debug only.
    Real-robot path (default): keep the current pose, yaw+XY-align the
    reference to the robot so ``motion_anchor_ori_b`` starts near identity
    (same as training reset), then track from t=0.
    """
    if not load_mimic(idx):
        return False
    mode["name"] = "MIMIC"
    last_action[:] = 0.0
    if hard_reset:
        _bind_motion(rt.emb_raw)
        reset_mimic(0)
        print("[mode] -> MIMIC skill (teleport to frame 0, sim only)")
    else:
        aligned = align_motion_to_robot(rt.emb_raw)
        _bind_motion(aligned)
        # sanity: ori obs at t=0 should be near identity after yaw align
        q_r = data.xquat[anchor_body_id].astype(np.float64)
        ori0 = motion_anchor_ori_b(q_r, ref_anchor_quat[0])
        print(
            f"[mode] -> MIMIC skill (aligned to robot; "
            f"ori0 L2-to-I={np.linalg.norm(ori0 - np.array([1, 0, 0, 1, 0, 0])):.3f})"
        )
    return True


_HELP_LEFT = "W/S\nA/D\nQ/E\nC\n[/]\n0-9\nL\nSpace\nR"
_HELP_RIGHT = (
    "vx +/- (RL)\n"
    "vy +/- (RL)\n"
    "yaw +/- (RL)\n"
    "clear cmd (RL)\n"
    "mimic (if idle)\n"
    "mimic (if idle)\n"
    "back to RL\n"
    "pause\n"
    "reset mode"
)


def overlay(viewer):
    status = "PAUSED" if paused["flag"] else "RUN"
    if mode["name"] == "RL":
        cmd = locomo.commands
        viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "mode\ncmd\nstatus",
                f"RL locomotion\n"
                f"vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} yaw={cmd[2]:+.2f}\n"
                f"{status}",
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                _HELP_LEFT,
                _HELP_RIGHT,
            ),
        ])
    else:
        name = rt.name if rt and len(rt.name) <= 48 else ((rt.name[:45] + "...") if rt else "?")
        viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "mode\nmimic\nframe\nstatus",
                f"MIMIC -> auto RL\n"
                f"[{rt_idx[0]:2d}/{len(POLICIES) - 1}] {name}\n"
                f"{cur_t[0]:4d} / {T}  ({cur_t[0] / 50.0:5.1f}s)\n"
                f"{status}",
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                _HELP_LEFT,
                _HELP_RIGHT,
            ),
        ])


# ----------------------------------------------------------------- loop
frame_dt = ctrl_dt / max(args.speed, 1e-6)


@contextlib.contextmanager
def maybe_viewer():
    if args.headless:
        yield None
    else:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
            yield v


print(
    "\nControls (viewer window focused):\n"
    "  [ / ]         previous / next fall-recovery policy (anytime, even mid-skill)\n"
    "  0 / 1         jump to Stage-I / Stage-II (anytime)\n"
    "  L             return to RL locomotion\n"
    "  R             reset current skill to frame 0\n"
    "  Space         pause / resume\n"
    "  W/S A/D Q/E   RL velocity commands (only in RL mode)\n"
    "  C             clear velocity command\n"
    f"\nStart mode: {start_mode} | policies: {len(POLICIES)} | loop_mimic={args.loop_mimic}\n"
)
for i, (name, _) in enumerate(POLICIES):
    print(f"  [{i}] {name}")
print()

with maybe_viewer() as viewer:
    if start_mode == "MIMIC":
        if not enter_mimic(cur_policy_idx, hard_reset=True):
            raise SystemExit("[ERROR] failed to start mimic")
    else:
        enter_rl(hard_reset=True)

    if args.settle > 0.0:
        q_hold = M["jp"][0] if mode["name"] == "MIMIC" else rl_stand_q
        kp_hold = kp_j if mode["name"] == "MIMIC" else rl_kp_j
        kd_hold = kd_j if mode["name"] == "MIMIC" else rl_kd_j
        for _ in range(int(args.settle / model.opt.timestep)):
            apply_pd(q_hold, kp_hold, kd_hold)
            mujoco.mj_step(model, data)

    t = 0
    step = 0
    err_pos = []
    while True:
        if viewer is None:
            if mode["name"] == "MIMIC" and step >= T:
                print(f"headless mimic finished. mean anchor pos error: {np.mean(err_pos):.3f} m")
                break
            if mode["name"] == "RL" and step >= 500:
                print("headless RL smoke finished (500 steps)")
                break
        elif not viewer.is_running():
            break

        # ---- mode requests ----------------------------------------------------
        if rl_request["flag"]:
            enter_rl(hard_reset=False)
            t = 0
            err_pos.clear()
            rl_request["flag"] = False

        if switch_request["idx"] is not None:
            idx = switch_request["idx"]
            switch_request["idx"] = None
            # Fall-recovery: allow hot-switch anytime (Stage-I ↔ Stage-II).
            # Hard-reset so each policy starts from a clean motion frame 0.
            if enter_mimic(idx, hard_reset=True):
                t = 0
                err_pos.clear()
                print(f"[switch] active policy [{idx}] {POLICIES[idx][0]}")

        if reset_request["flag"]:
            reset_request["flag"] = False
            if mode["name"] == "RL":
                enter_rl(hard_reset=True)
            elif rt is not None:
                reset_mimic(0)
                last_action[:] = 0.0
            t = 0
            err_pos.clear()

        start = time.time()

        if paused["flag"]:
            if viewer is not None:
                overlay(viewer)
                if mode["name"] == "MIMIC" and not args.no_ghost:
                    draw_ghost(viewer.user_scn, t)
                viewer.sync()
            time.sleep(0.02)
            continue

        base_quat = data.qpos[free_adr + 3 : free_adr + 7].copy()
        ang_vel_b = data.qvel[3:6].copy()
        q_mj = data.qpos[mj_qpos_adr].copy()
        qd_mj = data.qvel[mj_dof_adr].copy()

        if mode["name"] == "RL":
            q_des = locomo.act(base_quat, ang_vel_b, q_mj, qd_mj)
            for _ in range(decimation):
                apply_pd(q_des, rl_kp_j, rl_kd_j)
                mujoco.mj_step(model, data)
            step += 1
            cur_t[0] = step
        else:
            # ---- mimic observation (mjlab PolicyCfg order) -------------------
            # command(58) + motion_anchor_ori_b(6) [+ optional SE]
            # + base_ang_vel(3) + joint_pos_rel(29) + joint_vel(29) + actions(29)
            cmd = np.concatenate([M["jp"][t], M["jv"][t]])
            p_anc = data.xpos[anchor_body_id].copy()
            q_anc = data.xquat[anchor_body_id].copy()  # torso_link wxyz
            # identical to mjlab: mat(quat_mul(quat_inv(q_robot), q_ref))[..., :2].reshape
            anchor_ori_b = motion_anchor_ori_b(q_anc, ref_anchor_quat[t])
            R_anc = quat_to_mat(q_anc)
            R_pelvis = quat_to_mat(base_quat)
            base_ang_vel_b = ang_vel_b  # free-joint qvel[3:6] already body-frame
            joint_pos_rel = q_mj - default_q
            joint_vel_rel = qd_mj

            parts = [cmd]
            if args.with_state_estimation:
                # subtract_frame_transforms translation: R_robot^T @ (p_ref - p_robot)
                anchor_pos_b = R_anc.T @ (ref_anchor_pos[t] - p_anc)
                base_lin_vel_b = R_pelvis.T @ data.qvel[0:3]
                parts.extend([anchor_pos_b, anchor_ori_b, base_lin_vel_b, base_ang_vel_b])
            else:
                parts.extend([anchor_ori_b, base_ang_vel_b])
            parts.extend([joint_pos_rel, joint_vel_rel, last_action])
            obs[:] = np.concatenate(parts).astype(np.float32)

            if args.replay:
                action = ((M["jp"][t] - default_q) / rt.scale).astype(np.float32)
            else:
                action = np.asarray(rt.run(obs, t), dtype=np.float32).reshape(-1)
                assert action.shape[0] == N, f"policy action dim {action.shape[0]} != {N}"
            last_action = action

            q_des = default_q + rt.scale * action
            for _ in range(decimation):
                apply_pd(q_des)
                mujoco.mj_step(model, data)

            err_pos.append(np.linalg.norm(data.xpos[anchor_body_id] - ref_anchor_pos[t]))
            step += 1
            cur_t[0] = t
            t += 1
            if t >= T:
                if args.loop_mimic:
                    t = 0
                else:
                    # rl_sar: skill end -> RequestStateChange(Locomotion)
                    enter_rl(hard_reset=False)
                    t = 0
                    err_pos.clear()

        if viewer is not None:
            overlay(viewer)
            if mode["name"] == "MIMIC" and not args.no_ghost:
                draw_ghost(viewer.user_scn, min(t, T - 1))
            else:
                viewer.user_scn.ngeom = 0
            viewer.sync()

        if step % 10 == 0:
            if mode["name"] == "RL":
                c = locomo.commands
                print(f"step {step:4d}  [RL] cmd=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})")
            else:
                print(
                    f"step {step:4d} t={t:3d}  [MIMIC] anchor err {err_pos[-1]:.3f} m  "
                    f"z_err {data.xpos[anchor_body_id, 2] - ref_anchor_pos[min(t, T - 1), 2]:+.3f}"
                )

        remaining = frame_dt - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)
