# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Humanoid-Gym is a reinforcement learning framework based on NVIDIA Isaac Gym for training locomotion policies for humanoid and quadrupedal robots, with focus on zero-shot sim-to-real transfer. Supports XBot-L, Pikachu V01, Pikachu V025, and Pikachu V025 Quad robots.

## Setup

Requires Isaac Gym Preview 4, PyTorch 1.13.1 + CUDA 11.7, NumPy 1.23, and NVIDIA driver >= 515.

```bash
pip install -e .
```

## Common Commands

**Training:**
```bash
python humanoid/scripts/train.py --task=Pikachu_V025 --run_name v1 --headless --num_envs 4096
# Other tasks: humanoid_ppo, Pikachu_V01, Pikachu_V025_Quad
```

**Evaluation / policy export:**
```bash
python humanoid/scripts/play.py --task=Pikachu_V025 --run_name v1
```

**Sim-to-sim transfer (Isaac Gym → MuJoCo):**
```bash
python humanoid/scripts/sim2sim.py --load_model /path/to/policy.pt
```

## Architecture

### Core Flow

```
Task Registry → make_env() → Environment (LeggedRobot subclass)
             → make_alg_runner() → OnPolicyRunner → PPO → ActorCritic
```

The `TaskRegistry` in [humanoid/utils/task_registry.py](humanoid/utils/task_registry.py) maps task names to (env_class, config_class) pairs. All four tasks are registered in [humanoid/envs/\_\_init\_\_.py](humanoid/envs/__init__.py).

### Environment Hierarchy

`BaseTask` → `LeggedRobot` → task-specific env (e.g., `PikachuV025Env`)

- [humanoid/envs/base/legged_robot.py](humanoid/envs/base/legged_robot.py) — core environment (820 LOC): physics setup, reward computation, observation stacking, terrain curriculum
- Each robot variant in `humanoid/envs/<robot>/` overrides only what differs

### Configuration System

All parameters live in nested config classes inheriting from `LeggedRobotCfg` / `LeggedRobotCfgPPO`. Key nested sections: `env`, `asset`, `terrain`, `commands`, `noise`, `init_state`, `reward`, `control`, `sim`, and `runner`/`ppo` for training.

To tune a robot, edit only the config file (e.g., [humanoid/envs/pikachu_v025/pikachu_v025_config.py](humanoid/envs/pikachu_v025/pikachu_v025_config.py)) — no env code changes needed for hyperparameter adjustments.

### Reward System

`_prepare_reward_function()` dynamically discovers reward functions at startup by scanning config for non-zero `reward.scales.*` values and binding the corresponding `_reward_<name>()` methods. To add a new reward: implement `_reward_<name>()` in the env class and add a non-zero scale in the config.

### Key Design Patterns

- **Vectorized simulation**: All training runs thousands of parallel Isaac Gym envs (default 4096)
- **Privileged observations**: Critic receives extra state info not available to the actor policy
- **Observation stacking**: Default 15-frame history stacked into observation
- **Modular rewards**: Composite reward assembled at runtime from config scales
- **Terrain curriculum**: Difficulty auto-progresses during training

### Logs and Checkpoints

Training logs and model checkpoints are saved under `logs/<experiment_name>/`. The `play.py` script loads the latest checkpoint by default; use `--checkpoint` to specify one.
