![Project banner](docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids 0 1 \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/source/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. mjlab uses WandB to manage motion datasets.
See the [motion preprocessing documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup) for setup instructions.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Go2 Motion Tracking (local npz)

Train / play Unitree Go2 tracking with a local motion file under `motions/`.
Training is headless by default (`MUJOCO_GL=egl`). Logs and TensorBoard events
go to `logs/rsl_rl/go2_tracking/`.

#### Demo Videos

Learned Go2 skills (files under [`videos/`](videos/)).
Two videos per row (50/50) so GitHub's player does not stretch a middle
column.

<table width="100%">
  <tr>
    <td width="50%" align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/800a0da9-42c4-4cb2-a20b-dcb2e0fa96f7" controls width="240" height="427"></video>
      <br />Backflip
    </td>
    <td width="50%" align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/39d54ec3-1454-4242-8abe-e9e5388f7281" controls width="240" height="427"></video>
      <br />Front flip
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/a7419df5-61e3-4dad-9d27-cb5a811a656c" controls width="240" height="427"></video>
      <br />Left flip
    </td>
    <td width="50%" align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/8e613766-5154-401c-acf8-30e16f4333f2" controls width="240" height="427"></video>
      <br />Right flip
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/e52d7050-4e29-4154-b262-4362de74b10c" controls width="240" height="427"></video>
      <br />Jump up
    </td>
    <td width="50%" align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/6671fdd5-c6d3-4ec1-9518-9cd5b92cce30" controls width="240" height="427"></video>
      <br />Jump forward
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/a85a9024-c823-4b3a-ad54-ea8cc9d7476c" controls width="240" height="427"></video>
      <br />Jump back
    </td>
    <td width="50%"></td>
  </tr>
</table>

**Train three long-jump motions sequentially** (4096 envs, 10000 iterations each):

```bash
cd /path/to/mjlab-dev-go2-mimic

for motion in \
  go2_longjump_0p0 \
  go2_longjump_0p7 \
  go2_longjump_m0p5
do
  echo "========== Training $motion =========="
  uv run python -m mjlab.scripts.train \
    Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation \
    --env.commands.motion.motion-file "$PWD/motions/${motion}.npz" \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 \
    --agent.run-name "$motion"
done
```

With state estimation in the actor obs, use task id
`Mjlab-Tracking-Flat-Unitree-Go2` instead.

**Play** (auto-picks name-sorted latest ``model_*.pt``).
``--keep-randomization True`` keeps training-time DR and defaults to **50 envs**:

```bash
uv run python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation \
  --motion-file $PWD/motions/go2_longjump_m0p5.npz \
  --keep-randomization True
```

Override env count if needed: ``--num-envs 16``. Or use the helper:

```bash
uv run play-go2-rand --motion-file $PWD/motions/go2_longjump_m0p5.npz
```

Specify a checkpoint explicitly:

```bash

 python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation \
  --motion-file $PWD/motions/go2_longjump_m0p5.npz \
  --checkpoint-file logs/rsl_rl/go2_tracking/<run>/model_10000.pt \
  --keep-randomization True
```

## G1 tracking:
**Train** (4096 envs, 10000 iterations; loop over any motions): 环境变量必须修改

```bash
conda activate env_isaaclab
cd /home/zju/Downloads/mjlab-dev-go2-mimic
export PYTHONPATH=src
```
```bash
for motion in \
  omnixtreme/g1_motion_long_01\
  omnixtreme/g1_motion_long_06 \
  omnixtreme/g1_motion_long_08 \
  omnixtreme/g1_motion_long_10 \
  omnixtreme/g1_motion_long_13
do
  echo "========== Training $motion =========="
  python -m mjlab.scripts.train \
    Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
    --env.commands.motion.motion-file "$PWD/motions/${motion}.npz" \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 \
    --agent.run-name "$(basename $motion)"
done
```

```bash
python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
  --motion-file $PWD/motions/chunwan/90_15_stageii_unitree_g1.npz  \
```

Specify a checkpoint explicitly: 50个机器人   --keep-randomization True

```bash
python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
  --motion-file $PWD/motions/chunwan/90_15_stageii_unitree_g1.npz \
  --keep-randomization True
```

### 后端可视化，服务器中也可以使用
```bash
python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
  --motion-file $PWD/motions/omnixtreme/g1_motion_long_01.npz \
  --viewer viser
```

### G1 Dance 任务（倒地恢复实验入口）

**该任务用于后续倒地恢复（fall-and-get-up）实验**：基于
`tracking_env_cfg_dance.py` 的宽松锚点奖励配置（root_pos 1.0 / root_ori 0.5），
后续将在此任务上叠加放松终止条件、躺倒惩罚等倒地恢复训练改动。
执行器已启用梯形力矩-速度包络（见下节）。当前动作数据仅一个：
`motions/dance1_subject2.npz`。

```bash
conda activate env_isaaclab
cd /home/zju/Downloads/mjlab-dev-go2-mimic
export PYTHONPATH=src
```

**Train**:

```bash
python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1-Dance-No-State-Estimation \
  --env.commands.motion.motion-file "$PWD/motions/dance1_subject2.npz" \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.run-name "dance1_subject2"
```

带状态估计版本用任务 ID `Mjlab-Tracking-Flat-Unitree-G1-Dance`。
日志位于 `logs/rsl_rl/g1_tracking/dance1_subject2/`。

**Play**:

```bash
python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-G1-Dance-No-State-Estimation \
  --motion-file $PWD/motions/dance1_subject2.npz \
  --keep-randomization True
```


#### Domain Randomization (Go2 vs G1)

Tracking inherits the randomization set from `tracking_env_cfg.py`; the Go2 /
G1 task configs only retarget per-robot names or toggle extras. Both robots
share: interval velocity push, startup COM / body-mass / PD-gain /
encoder-bias / foot-friction randomization, plus motion initial-state (RSI)
pose / velocity / joint-position ranges, actor obs noise, and delayed
actuators where noted.

| Randomization | Mode | Value / Range | Go2 | G1 |
| :--- | :---: | :--- | :---: | :---: |
| Velocity push `push_robot` | interval 1–3 s | lin ±0.5/±0.5/±0.2 m/s, ang ±0.52/±0.52/±0.78 rad/s | ✓ | ✓ |
| Vertical push `push_robot_2` (opt.) | interval 1–2 s | z 0–3 m/s, prob 0.8→0 decay | opt. | opt. |
| COM shift `base_com` | startup | ±0.025/±0.03/±0.03 m on `trunk` / `torso_link` | ✓ | ✓ |
| Body mass `add_body_mass` | startup | add −1.0…+2.0 kg on `trunk` / `torso_link` | ✓ | ✓ |
| PD gains `actuator_gains` | startup | kp, kd ×0.9–1.1 | ✓ | ✓ |
| Encoder bias | startup | ±0.015 rad | ✓ | ✓ |
| Foot friction | startup | ×0.3–1.2, shared across feet (`[FR][LR]_foot` / `^(left|right)_foot[1-7]`) | ✓ | ✓ |
| Actuator delay | startup wrap | position delay 0–3 physics steps | ✓ (flag) | — |
| RSI pose range | reset | x,y ±0.05 m; z ±0.01 m; roll/pitch ±0.1, yaw ±0.2 rad | ✓ | ✓ |
| RSI velocity range | reset | same as push range | ✓ | ✓ |
| RSI joint pos | reset | ±0.1 rad | ✓ | ✓ |
| Obs noise (actor) | per step | anchor pos ±0.25, ori ±0.05, lin vel ±0.5, ang vel ±0.2, joint pos ±0.01, joint vel ±0.5 | ✓ | ✓ |

Play mode removes pushes and RSI ranges; `--keep-randomization True` keeps
training-time DR at play (Go2).

#### Render G1 motions to MP4

Render any G1 motion `.npz` under `motions/` (mimickit / chunwan /
amass_hard) to MP4 with MuJoCo offscreen rendering (chase camera follows
the root; non-29-joint clips are skipped automatically):

```bash
cd /path/to/mjlab-dev-go2-mimic
conda activate env_isaaclab
export PYTHONPATH=src MUJOCO_GL=egl

# All G1 motions → videos/g1_motions/ (mirrors motions/ layout, 1280x720)
python -m mjlab.scripts.render_g1_motions \
  --input-dir motions \
  --output-dir videos/g1_motions \
  --width 1280 --height 720

# Single clip / subset via glob pattern
python -m mjlab.scripts.render_g1_motions \
  --pattern 'chunwan/88_05*.npz' \
  --output-dir videos/g1_motions

# Options: --every-nth 2 (halve fps), --follow False (fixed "track" camera),
# --skip-existing False (re-render)
```

### 4. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```
```bash
conda activate env_isaaclab
cd /home/zju/Downloads/mjlab-dev-go2-mimic
export PYTHONPATH=src MUJOCO_GL=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
for motion in \
    88_06_stageii_unitree_g1 \
    90_08_stageii_unitree_g1 \
    90_15_stageii_unitree_g1
do
  echo "========== Training $motion =========="
  python -m mjlab.scripts.train \
    Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
    --env.commands.motion.motion-file "$PWD/motions/chunwan/${motion}.npz" \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 \
    --agent.run-name "chunwan_${motion}" \
    --gpu-ids '[0]'
done
```
<!-- For development setup: `uvx pre-commit install` -->
## TODO
- [] 手部接触的力进行限制
- [] 塑料手的碰撞体去掉
- [x] 手部接触的动作学不好
- [] 逐渐收紧的终止条件
- [x] 电机力矩转速曲线的梯形包络
- [] 负功率限制(omnixtreme)
- [x] 添加后空翻等极限动作
- [] 查尔斯顿舞的长序列动作(3min) 倒地恢复的力课程
- [] 统一的奖励系数

## G1 执行器建模：梯形力矩-速度包络与非线性摩擦

所有注册的 G1 任务（`Mjlab-Tracking-Flat-Unitree-G1*`、`Mjlab-Tracking-Flat-Unitree-G1-Dance*`）
使用 `UnitreeActuator`（`src/mjlab/actuator/unitree_actuator.py`），
建模来自 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 的
`unitree_actuators.py`，并见 [OmniXtreme](https://arxiv.org/abs/2602.23843) 论文
III-B 节 "Actuation-Aware Torque–Speed Constraints" 与 Table IX。

### 梯形力矩-速度包络（论文式 4–5）

$$
\tau_{\max}(v)=\begin{cases}\tau_{y1}, & |v|\le v_{x1},\ v\cdot\tau>0 \\[2pt]
\tau_{y2}, & |v|\le v_{x1},\ v\cdot\tau\le 0 \\[2pt]
\tau_{\text{peak}}\!\left(1-\dfrac{|v|-v_{x1}}{v_{x2}-v_{x1}}\right), & v_{x1}<|v|\le v_{x2} \\[2pt]
0, & |v|>v_{x2}\end{cases}
$$

其中 $\tau_{\text{peak}}$ 按力矩与速度方向选取：同向（驱动）取 $\tau_{y1}$，
反向（制动）取 $\tau_{y2}$（反电动势使制动峰值更高）。超出 $v_{x1}$（T-N 曲线
拐点）后峰值力矩线性锥削，至空载转速 $v_{x2}$ 降为零；指令力矩最终被裁剪到
$[-\tau_{\max},\ \tau_{\max}]$ 内。

### 非线性摩擦项（论文式 6）

力矩削波之后，再扣除执行器内部摩擦损耗：

$$
\tau_{\text{applied}}=\tau_{\text{clipped}}-\underbrace{\mu_{s}\tanh\!\left(\frac{v}{v_{\text{act}}}\right)}_{\text{平滑库仑摩擦}}-\underbrace{\mu_{d}\,v}_{\text{粘性摩擦}}
$$

- **平滑库仑摩擦项** $\mu_s\tanh(v/v_{\text{act}})$：`tanh` 是符号函数
  $\operatorname{sign}(v)$ 的光滑近似，在 $|v|\lesssim v_{\text{act}}$ 内
  从 0 平滑爬升并很快饱和到 $\pm\mu_s$，捕捉静摩擦→动摩擦的过渡；
- **粘性摩擦项** $\mu_d v$：与速度成正比的耗散，提供附加阻尼。

摩擦始终与速度同号、从指令力矩中扣除，因此总是阻碍运动。实现见
`UnitreeActuator.compute()`（削波后的减摩擦）与 `_clip_effort()`
（包络锥削削波）。

### 电机参数（unitree_rl_lab / OmniXtreme Table IX）

| 电机 | 关节 | $\tau_{y1}$ (N·m) | $\tau_{y2}$ (N·m) | $v_{x1}$ (rad/s) | $v_{x2}$ (rad/s) | $\mu_s$ | $\mu_d$ | $I$ (kg·m²) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5020-16 | 肩/肘/wrist_roll | 24.8 | 31.9 | 30.86 | 40.13 | 0.6 | 0.06 | 3.610e-3 |
| 2×5020-16（并联） | waist_pitch/roll、双踝 | 49.6 | 63.8 | 30.86 | 40.13 | 1.2 | 0.12 | 7.219e-3 |
| 7520-14.3 | hip_pitch / hip_yaw / waist_yaw | 71.0 | 83.3 | 22.63 | 35.52 | 1.6 | 0.16 | 1.018e-2 |
| 7520-22.5 | hip_roll / knee | 111.0 | 131.0 | 14.5 | 22.7 | 2.4 | 0.24 | 2.510e-2 |
| 4010-25 | wrist_pitch / wrist_yaw | 4.8 | 8.6 | 15.3 | 24.76 | 0.6 | 0.06 | 4.250e-3 |

摩擦激活速度 $v_{\text{act}}=0.01$ rad/s。腰部俯仰/侧倾与双踝为两台 5020
电机经四连杆并联驱动（名义 1:1 传动比）：力矩类参数（$\tau_{y1}$、$\tau_{y2}$、
$\mu_s$、$\mu_d$）与折算惯量 $I$、PD 增益均 ×2，速度角点 $v_{x1}$、$v_{x2}$
不缩放（并联电机转速相同）。

## Community Projects

mjlab is used for research and robotics applications around the world. Examples:

<table>
  <tr>
    <td>
      <a href="https://github.com/menloresearch/asimov-mjlab">
        menloresearch/asimov-mjlab
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/menloresearch/asimov-mjlab?style=social"
        />
      </a>
    </td>
    <td>Locomotion fork for the Asimov bipedal robot.</td>
  </tr>
  <tr>
    <td>
      <a href="http://husky-humanoid.github.io/">
        HUSKY
      </a>
      <br />
      <a href="https://github.com/mujocolab/mjlab/discussions/572">#572</a>
      ·
      <a href="https://arxiv.org/abs/2602.03205">Paper</a>
    </td>
    <td>
      Humanoid skateboarding with dynamic balance control.
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/Nagi-ovo/mjlab-homierl">
        Nagi-ovo/mjlab-homierl
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/Nagi-ovo/mjlab-homierl?style=social"
        />
      </a>
    </td>
    <td>Multi-task H1 locomotion (walk/squat/stand) with upper-body disturbance robustness.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/MyoHub/mjlab_myosuite">
        MyoHub/mjlab_myosuite
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/MyoHub/mjlab_myosuite?style=social"
        />
      </a>
    </td>
    <td>Musculoskeletal simulation integration with MyoSuite.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/MarcDcls/mjlab_upkie">
        MarcDcls/mjlab_upkie
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/MarcDcls/mjlab_upkie?style=social"
        />
      </a>
    </td>
    <td>Velocity control for the Upkie wheeled biped.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/unitreerobotics/unitree_rl_mjlab">
        unitreerobotics/unitree_rl_mjlab
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/unitreerobotics/unitree_rl_mjlab?style=social"
        />
      </a>
    </td>
    <td>Official Unitree RL environments for Go2, G1, and H1_2.</td>
  </tr>
</table>

Want to share your project? Post in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell)!

## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Citation

If you use mjlab in your research, please cite:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.

Go2 agile reference motions (flips, jumps, etc.) are generated with
[se3_trajopt](https://github.com/Renkunzhao/se3_trajopt).
Tracking training and demos for this work live on the
[dev/go2-mimic](https://github.com/Renkunzhao/mjlab/tree/dev/go2-mimic)
branch.
