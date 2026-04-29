# SPDX-License-Identifier: BSD-3-Clause
#
# Self-training entry point for curriculum-style PPO training.

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *  # noqa: F401,F403 - registers tasks
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import launch_tensorboard


DEFAULT_TASK = "Pikachu_V025_No_Yaw"


@dataclass(frozen=True)
class SelfTrainStage:
    name: str
    weight: float
    lin_vel_x: Tuple[float, float]
    lin_vel_y: Tuple[float, float]
    ang_vel_yaw: Tuple[float, float]
    noise_level: float
    add_noise: bool
    push_robots: bool
    max_push_vel_xy: float
    max_push_ang_vel: float
    action_delay: float
    action_noise: float
    friction_range: Tuple[float, float]


SELF_TRAIN_STAGES: List[SelfTrainStage] = [
    SelfTrainStage(
        name="warmup_balance",
        weight=0.15,
        lin_vel_x=(-0.05, 0.20),
        lin_vel_y=(-0.05, 0.05),
        ang_vel_yaw=(-0.05, 0.05),
        noise_level=0.0,
        add_noise=False,
        push_robots=False,
        max_push_vel_xy=0.0,
        max_push_ang_vel=0.0,
        action_delay=0.0,
        action_noise=0.0,
        friction_range=(0.6, 1.2),
    ),
    SelfTrainStage(
        name="easy_walk",
        weight=0.25,
        lin_vel_x=(-0.15, 0.35),
        lin_vel_y=(-0.12, 0.12),
        ang_vel_yaw=(-0.12, 0.12),
        noise_level=0.2,
        add_noise=True,
        push_robots=False,
        max_push_vel_xy=0.0,
        max_push_ang_vel=0.0,
        action_delay=0.15,
        action_noise=0.005,
        friction_range=(0.4, 1.6),
    ),
    SelfTrainStage(
        name="command_tracking",
        weight=0.25,
        lin_vel_x=(-0.25, 0.50),
        lin_vel_y=(-0.20, 0.20),
        ang_vel_yaw=(-0.20, 0.20),
        noise_level=0.4,
        add_noise=True,
        push_robots=True,
        max_push_vel_xy=0.10,
        max_push_ang_vel=0.20,
        action_delay=0.30,
        action_noise=0.01,
        friction_range=(0.2, 1.8),
    ),
    SelfTrainStage(
        name="robust_policy",
        weight=0.35,
        lin_vel_x=(-0.30, 0.60),
        lin_vel_y=(-0.30, 0.30),
        ang_vel_yaw=(-0.30, 0.30),
        noise_level=0.6,
        add_noise=True,
        push_robots=True,
        max_push_vel_xy=0.20,
        max_push_ang_vel=0.40,
        action_delay=0.50,
        action_noise=0.02,
        friction_range=(0.1, 2.0),
    ),
]


def _split_iterations(total_iterations: int) -> List[int]:
    if total_iterations < len(SELF_TRAIN_STAGES):
        return [1] * max(1, total_iterations)

    weights = [stage.weight for stage in SELF_TRAIN_STAGES]
    raw_counts = [
        max(1, int(total_iterations * weight / sum(weights))) for weight in weights
    ]
    diff = total_iterations - sum(raw_counts)
    raw_counts[-1] += diff
    if raw_counts[-1] <= 0:
        raw_counts[-1] = 1
    return raw_counts


def _apply_stage_to_cfg(env_cfg, stage: SelfTrainStage) -> None:
    env_cfg.commands.ranges.lin_vel_x = list(stage.lin_vel_x)
    env_cfg.commands.ranges.lin_vel_y = list(stage.lin_vel_y)
    env_cfg.commands.ranges.ang_vel_yaw = list(stage.ang_vel_yaw)

    env_cfg.noise.add_noise = stage.add_noise
    env_cfg.noise.noise_level = stage.noise_level

    env_cfg.domain_rand.push_robots = stage.push_robots
    env_cfg.domain_rand.max_push_vel_xy = stage.max_push_vel_xy
    env_cfg.domain_rand.max_push_ang_vel = stage.max_push_ang_vel
    env_cfg.domain_rand.action_delay = stage.action_delay
    env_cfg.domain_rand.action_noise = stage.action_noise
    env_cfg.domain_rand.friction_range = list(stage.friction_range)


def _apply_stage(env, env_cfg, stage: SelfTrainStage) -> None:
    _apply_stage_to_cfg(env_cfg, stage)

    env.command_ranges["lin_vel_x"] = list(stage.lin_vel_x)
    env.command_ranges["lin_vel_y"] = list(stage.lin_vel_y)
    env.command_ranges["ang_vel_yaw"] = list(stage.ang_vel_yaw)
    if hasattr(env, "add_noise"):
        env.add_noise = stage.add_noise

    # Resample commands immediately so the new curriculum range is active.
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    env._resample_commands(all_env_ids)


def self_train(args) -> None:
    if args.task == "XBotL_free":
        args.task = DEFAULT_TASK
    if args.run_name is None:
        args.run_name = "self_train"

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _apply_stage_to_cfg(env_cfg, SELF_TRAIN_STAGES[0])
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    if args.launch_tensorboard:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
        launch_tensorboard(log_root)

    stage_iterations = _split_iterations(train_cfg.runner.max_iterations)
    print("Self-training task:", args.task)
    print("Log dir:", ppo_runner.log_dir)

    for stage_id, (stage, num_iterations) in enumerate(
        zip(SELF_TRAIN_STAGES, stage_iterations), start=1
    ):
        print(
            f"\n[SelfTrain] Stage {stage_id}/{len(SELF_TRAIN_STAGES)} "
            f"{stage.name}: {num_iterations} iterations"
        )
        _apply_stage(env, env_cfg, stage)
        ppo_runner.learn(
            num_learning_iterations=num_iterations,
            init_at_random_ep_len=(stage_id == 1),
        )
        if ppo_runner.log_dir is not None:
            ppo_runner.save(
                os.path.join(
                    ppo_runner.log_dir,
                    f"model_stage_{stage_id}_{stage.name}.pt",
                )
            )


if __name__ == "__main__":
    self_train(get_args())
