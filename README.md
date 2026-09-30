# Unitree DSC Lab — Stair Climbing on G1

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Replication of *"Explicit Stair Geometry Conditioning for Robust Humanoid
Locomotion"* (arXiv:2605.09944v1) on the Unitree G1.

This repo is structured as a standalone Isaac Lab extension that lives **outside**
the `IsaacLab/` tree so updates to IsaacLab do not clobber the code. Layout and
tooling are modeled on
[`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab).

## What's inside

```
unitree_dsc_lab/
├── source/unitree_dsc_lab/                  # editable-install python package
│   └── unitree_dsc_lab/
│       ├── assets/robots/                   # USD/URDF asset registry (G1)
│       ├── tasks/stair_climb/
│       │   ├── agents/                      # rsl_rl PPO runner configs
│       │   ├── mdp/                         # observations, rewards, terminations, commands
│       │   ├── robots/g1_23dof/             # env cfg + gym.register
│       │   ├── robots/g1_29dof_dex1_1/      # 29-DoF G1 + Dex1-1 stair env
│       │   ├── terrains/                    # procedural staircase generator
│       │   ├── perception/                  # BEV builder, CNN student encoder, teacher
│       │   └── policy/                      # actor-critic, three-stage runner
│       └── utils/                           # cli parsing, deploy cfg export
├── scripts/
│   ├── list_envs.py
│   ├── rsl_rl/{train.py, play.py, cli_args.py, export_onnx.py}
│   └── stages/{train_stage1_policy.py, train_stage2_perception.py, train_stage3_joint.py}
├── deploy/                                  # on-board ROS2 / C++ runtime
├── assets/g1/                               # local URDF / MJCF copies (gitignored)
├── doc/
├── docker/
├── unitree_dsc_lab.sh                       # install / list / train / play helper
├── pyproject.toml
└── README.md
```

## Installation

- Install Isaac Lab 2.3.2 (Isaac Sim 5.1) — see
  [`STAIR_CLIMBING_REPLICATION_GUIDE.md`](../STAIR_CLIMBING_REPLICATION_GUIDE.md)
  sections 3–5.
- Clone this repo outside the `IsaacLab/` directory.
- Install in editable mode against the active Isaac Lab conda env:

  ```bash
  conda activate stairg1
  ./unitree_dsc_lab.sh -i
  ```

- Provide the Unitree G1 robot description. Either:
  - Set `UNITREE_ROS_DIR` in `source/unitree_dsc_lab/unitree_dsc_lab/assets/robots/unitree.py`
    to a checkout of [`unitree_ros`](https://github.com/unitreerobotics/unitree_ros), or
  - Set `UNITREE_MODEL_DIR` to a checkout of the
    [`unitree_model`](https://huggingface.co/datasets/unitreerobotics/unitree_model) USDs.
- The `Unitree-G1-29dof-Dex1-1-StairClimb-v0` task additionally reads
  `source/unitree_dsc_lab/unitree_dsc_lab/assets/robots/g1_description/` directly.
  Keep its mode-15 URDF and referenced meshes available; these large mesh assets
  are external source assets and are not included by the current wheel metadata.

## Tasks

| Task ID                          | Stage  | Notes |
|----------------------------------|--------|-------|
| `Unitree-G1-23dof-StairClimb-v0` | 1/2/3  | PPO with explicit 4-D terrain token `z_t = [s_t, h_step, d_step, theta_yaw]` |
| `Unitree-G1-29dof-Dex1-1-StairClimb-v0` | 1/2/3  | Local mode-15 URDF; 29 body joints are controlled, four Dex1-1 prismatic joints hold their default position |

List, train, play:

```bash
./unitree_dsc_lab.sh -l
./unitree_dsc_lab.sh -t --task Unitree-G1-23dof-StairClimb-v0
./unitree_dsc_lab.sh -p --task Unitree-G1-23dof-StairClimb-v0
```

## Three-stage training

See `scripts/stages/` and §12 of the replication guide.

TensorBoard remains the default logger. To send the same run to Weights & Biases,
authenticate once in the active environment (or provide the API key through the
environment):

```bash
wandb login
# Non-interactive alternative:
export WANDB_API_KEY="your-api-key"

# Optional: select a W&B entity/team.
export WANDB_USERNAME="your-entity"
```

Then enable W&B explicitly for each stage. `--wandb_project` defaults to
`unitree_dsc_lab`; `--run_name` is an optional, path-safe suffix:

```bash
TASK="Unitree-G1-29dof-Dex1-1-StairClimb-v0"
PROJECT="unitree_dsc_lab"
RUN_NAME="g1-stairs"

# Stage 1: privileged-teacher PPO.
python scripts/stages/train_stage1_policy.py \
    --task "$TASK" --max_iterations 6000 --num_envs 4096 --headless \
    --logger wandb --wandb_project "$PROJECT" --run_name "$RUN_NAME"

# Use the exact Stage 1 path printed as "Logging run to".
STAGE1_CKPT="logs/stage1/$TASK/YYYY-MM-DD_HH-MM-SS_stage1_${RUN_NAME}/model_final.pt"

# Stage 2: supervised perception encoder.
python scripts/stages/train_stage2_perception.py \
    --task "$TASK" --policy_ckpt "$STAGE1_CKPT" --epochs 50 \
    --logger wandb --wandb_project "$PROJECT" --run_name "$RUN_NAME" --headless

# Use the exact Stage 2 path printed as "Logging run to".
STAGE2_CKPT="logs/stage2/$TASK/YYYY-MM-DD_HH-MM-SS_stage2_${RUN_NAME}/encoder_best.pt"

# Stage 3: joint policy and encoder fine-tuning.
python scripts/stages/train_stage3_joint.py \
    --task "$TASK" --resume_policy "$STAGE1_CKPT" \
    --resume_encoder "$STAGE2_CKPT" \
    --max_iterations 3000 --num_envs 2048 \
    --logger wandb --wandb_project "$PROJECT" --run_name "$RUN_NAME" --headless
```

For a network-free run, set offline mode before launching a stage. The resulting
run can be uploaded later with `wandb sync`:

```bash
export WANDB_MODE=offline
```

Every launch creates a unique directory (with `_02`, `_03`, and so on added on
same-second collisions):

```text
logs/<stage>/<task>/<YYYY-MM-DD_HH-MM-SS>_<stage>[_<run_name>]/
```

The directory basename is also used as the W&B run name. Local TensorBoard event
files are retained alongside checkpoints. Stage 1 writes periodic `model_<N>.pt`
files and the `model_final.pt` convenience copy; Stage 2 writes
`encoder_best.pt`; Stage 3 writes periodic and final `model_<N>.pt` checkpoints.
Use the exact directory printed by the script when supplying a checkpoint to the
next stage. Running without `--logger wandb` requires no W&B account or network
connection.

## Deploy

Sim-to-real ONNX export and on-board ROS 2 runtime live under
[`scripts/rsl_rl/export_onnx.py`](scripts/rsl_rl/export_onnx.py) and
[`deploy/`](deploy/). See §14–15 of the replication guide.

## Acknowledgements

- [`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab) — repo layout and tooling
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- Paper: arXiv:2605.09944v1
