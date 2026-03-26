# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
import time

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5


    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)
    if not FIX_COMMAND:
        env_cfg.env.get_commands_from_keyboard = True
    else:
        env_cfg.env.get_commands_from_keyboard = False
    
    env_cfg.env.debug = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
    except ValueError as e:
        print(f"WARNING - Resume failed: {e}")
        print("WARNING - Falling back to non-resume mode (random/untrained policy).")
        train_cfg.runner.resume = False
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    plot_steps = 200   # adjustable: plot first N steps
    total_steps = 1000 # adjustable: total play steps
    # Ensure at least one full episode finishes so reward stats are available.
    # total_steps = max(total_steps, int(getattr(env, "max_episode_length", 0)) + 1)
    total_steps =1000
    plot_done = False
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        
        # 相机参数
        camera_distance = 2.0
        camera_height = 1.0
        camera_offset = gymapi.Vec3(camera_distance, 0, camera_height)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(0))
        
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)
        
        # 视频设置
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
        
        # 旋转参数
        rotation_speed = 2 * np.pi / 5.0  # 5秒转一圈
        start_time = time.time()

    # 主循环
    for i in range(total_steps):
        actions = policy(obs.detach())
        
        if FIX_COMMAND:
            env.commands[:, 0] = 0.3
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.0
        
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        
        if RENDER:
            # 更新相机位置
            current_time = time.time()
            elapsed_time = current_time - start_time
            angle = rotation_speed * elapsed_time
            
            # 计算相机位置
            camera_x = camera_distance * np.cos(angle)
            camera_z = camera_distance * np.sin(angle)
            camera_offset = gymapi.Vec3(camera_x, 0, camera_height)
            
            # 计算相机朝向（看向机器人）
            # 方法1：直接设置朝向为指向原点
            # 创建从相机指向机器人的方向向量
            direction = gymapi.Vec3(-camera_x, -camera_height, -camera_z)
            direction = gymapi.normalize(direction) 
            
            # 计算旋转角度
            yaw = np.arctan2(direction.z, direction.x) + np.pi
            pitch = np.arctan2(direction.y, np.sqrt(direction.x**2 + direction.z**2))
            
            # 创建旋转四元数
            camera_rotation = gymapi.Quat.from_euler_zyx(gymapi.Vec3(0, pitch, yaw))
            
            # 更新相机变换
            camera_transform = gymapi.Transform(camera_offset, camera_rotation)
            env.gym.set_camera_transform(h1, env.envs[0], camera_transform)
            
            # 渲染
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            
            # 获取图像
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])


        logger.log_states(
            {
                'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            }
            )
        # ====================== Log states ======================
        episode_info = infos.get("episode", None)
        if episode_info:
            num_episodes = int(torch.sum(dones).item())
            if num_episodes > 0:
                logger.log_rewards(episode_info, num_episodes)

        if (not plot_done) and (i + 1 >= plot_steps):
            logger.plot_states(show=True)
            plot_done = True

    logger.print_rewards()
    
    if RENDER:
        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
