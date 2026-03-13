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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain
import pygame
from humanoid.utils.math import wrap_to_pi


class PikachuQuadEnv(LeggedRobot):
    '''
    XBotLFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._build_joint_index_cache()
        self._validate_observation_dims()
        self._build_rigid_body_index_cache()
        self.last_feet_z = 0.05
        # 键盘模式下的mode状态（0=双足, 1=四足），T键切换
        self._keyboard_mode = 0
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _build_rigid_body_index_cache(self):
        """ 构建刚体索引缓存，用于奖励函数中访问特定刚体 """
        # 从URDF文件可知，手臂末端是elbow_link
        try:
            self.left_elbow_index = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], "left_elbow_link")
            self.right_elbow_index = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], "right_elbow_link")
        except:
            # 如果找不到刚体名称，初始化为-1并在奖励函数中处理
            self.left_elbow_index = -1
            self.right_elbow_index = -1

    def _build_joint_index_cache(self):
        dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}

        def idx(name):
            if name not in dof_name_to_idx:
                raise KeyError(f"Joint '{name}' not found in asset dof names: {self.dof_names}")
            return dof_name_to_idx[name]

        # Reference gait controls hip pitch, knee and ankle for each leg.
        self.left_ref_joint_indices = (
            idx("left_hip_pitch_joint"),
            idx("left_knee_joint"),
            idx("left_ankle_joint"),
        )
        self.right_ref_joint_indices = (
            idx("right_hip_pitch_joint"),
            idx("right_knee_joint"),
            idx("right_ankle_joint"),
        )

        # Arm joint indices for quadrupedal walking - updated for 18 DOF
        self.left_arm_indices = (
            idx("left_arm_pitch_joint"),
            idx("left_arm_roll_joint"),
            idx("left_arm_yaw_joint"),
            idx("left_elbow_joint"),
        )
        self.right_arm_indices = (
            idx("right_arm_pitch_joint"),
            idx("right_arm_roll_joint"),
            idx("right_arm_yaw_joint"),
            idx("right_elbow_joint"),
        )

        # print("Reference joint indices:")
        # print(f"Left leg: hip_pitch={self.left_ref_joint_indices[0]}, knee={self.left_ref_joint_indices[1]}, ankle={self.left_ref_joint_indices[2]}")
        # print(f"Right leg: hip_pitch={self.right_ref_joint_indices[0]}, knee={self.right_ref_joint_indices[1]}, ankle={self.right_ref_joint_indices[2]}")
        # print(f"Left arm: pitch={self.left_arm_indices[0]}, roll={self.left_arm_indices[1]}, yaw={self.left_arm_indices[2]}, elbow={self.left_arm_indices[3]}")
        # print(f"Right arm: pitch={self.right_arm_indices[0]}, roll={self.right_arm_indices[1]}, yaw={self.right_arm_indices[2]}, elbow={self.right_arm_indices[3]}")

        # Default-joint reward explicitly constrains yaw and roll.
        self.left_yaw_roll_indices = (
            idx("left_hip_yaw_joint"),
            idx("left_hip_roll_joint"),
        )
        self.right_yaw_roll_indices = (
            idx("right_hip_yaw_joint"),
            idx("right_hip_roll_joint"),
        )

        # Arm yaw and roll indices for maintaining arm posture
        self.left_arm_yaw_roll_indices = (
            idx("left_arm_roll_joint"),
            idx("left_arm_yaw_joint"),
        )
        self.right_arm_yaw_roll_indices = (
            idx("right_arm_roll_joint"),
            idx("right_arm_yaw_joint"),
        )

    def _validate_observation_dims(self):
        # command_input: 6 (sin, cos, vx, vy, vyaw, mode)
        expected_single_obs = 6 + 3 * self.num_actions + 6
        # privileged: 6(cmd) + 4*n(dof) + 3+3+3+2+3+1+1+2+2=20
        expected_single_priv_obs = 26 + 4 * self.num_actions

        # Validate with new action count (18 for quadrupedal)
        if self.cfg.env.num_single_obs != expected_single_obs:
            raise ValueError(
                f"cfg.env.num_single_obs={self.cfg.env.num_single_obs} does not match "
                f"expected {expected_single_obs} for num_actions={self.num_actions} (should be {6 + 3*18 + 6}=66)."
            )
        if self.cfg.env.single_num_privileged_obs != expected_single_priv_obs:
            raise ValueError(
                f"cfg.env.single_num_privileged_obs={self.cfg.env.single_num_privileged_obs} does not match "
                f"expected {expected_single_priv_obs} for num_actions={self.num_actions} (should be {26 + 4*18}=98)."
            )

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1

        # ---- 腿部参考（双足/四足共用，交替迈步） ----
        sin_pos_l = sin_pos.clone()
        sin_pos_l[sin_pos_l < 0] = 0  # 左腿摆动阶段
        sin_pos_r = sin_pos.clone()
        sin_pos_r[sin_pos_r > 0] = 0  # 右腿摆动阶段

        # ---- 双足模式参考（手臂保持默认，仅腿部运动） ----
        biped_ref = torch.zeros_like(self.dof_pos)
        biped_ref[:, self.left_ref_joint_indices[0]] = sin_pos_l * scale_1
        biped_ref[:, self.left_ref_joint_indices[1]] = sin_pos_l * scale_2
        biped_ref[:, self.left_ref_joint_indices[2]] = sin_pos_l * scale_1
        biped_ref[:, self.right_ref_joint_indices[0]] = -sin_pos_r * scale_1
        biped_ref[:, self.right_ref_joint_indices[1]] = -sin_pos_r * scale_2
        biped_ref[:, self.right_ref_joint_indices[2]] = -sin_pos_r * scale_1
        # 双支撑相清零
        biped_ref[torch.abs(sin_pos) < 0.1] = 0

        # ---- 四足模式参考（对角步态：左腿+右臂同相，右腿+左臂同相） ----
        quad_ref = biped_ref.clone()

        # 左臂与右腿同相（sin_pos < 0 时摆动）
        sin_pos_left_arm = sin_pos.clone()
        sin_pos_left_arm[sin_pos_left_arm > 0] = 0
        quad_ref[:, self.left_arm_indices[0]] = -sin_pos_left_arm * scale_1   # pitch
        quad_ref[:, self.left_arm_indices[1]] =  sin_pos_left_arm * scale_1/2  # roll
        quad_ref[:, self.left_arm_indices[2]] =  sin_pos_left_arm * scale_1/2  # yaw
        quad_ref[:, self.left_arm_indices[3]] =  sin_pos_left_arm * scale_2   # elbow

        # 右臂与左腿同相（sin_pos > 0 时摆动）
        sin_pos_right_arm = sin_pos.clone()
        sin_pos_right_arm[sin_pos_right_arm < 0] = 0
        quad_ref[:, self.right_arm_indices[0]] =  sin_pos_right_arm * scale_1   # pitch
        quad_ref[:, self.right_arm_indices[1]] = -sin_pos_right_arm * scale_1/2  # roll
        quad_ref[:, self.right_arm_indices[2]] = -sin_pos_right_arm * scale_1/2  # yaw
        quad_ref[:, self.right_arm_indices[3]] = -sin_pos_right_arm * scale_2   # elbow

        quad_ref[torch.abs(sin_pos) < 0.1] = 0

        # ---- 按mode混合参考（0=双足, 1=四足） ----
        mode = self.commands[:, 4].unsqueeze(1)  # (num_envs, 1)
        self.ref_dof_pos = biped_ref * (1.0 - mode) + quad_ref * mode
        self.ref_action = 2 * self.ref_dof_pos


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        num_actions = self.num_actions
        action_offset = 6  # command_input now 6 dims: sin,cos,vx,vy,vyaw,mode
        ang_vel_offset = action_offset + 3 * num_actions
        euler_offset = ang_vel_offset + 3
        noise_vec[0: 6] = 0.  # commands + mode (no noise)
        noise_vec[action_offset: action_offset + num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[action_offset + num_actions: action_offset + 2 * num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[action_offset + 2 * num_actions: action_offset + 3 * num_actions] = 0.  # previous actions
        noise_vec[ang_vel_offset: euler_offset] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[euler_offset: euler_offset + 3] = noise_scales.quat * self.obs_scales.quat         # euler
        return noise_vec


    def _resample_commands(self, env_ids):
        """覆写基类方法，增加mode随机采样（0=双足, 1=四足，各50%概率）"""
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        mode_rand = torch.rand(len(env_ids), device=self.device)
        self.commands[env_ids, 4] = (mode_rand > 0.5).float()

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        return super().step(actions)


    def compute_observations(self):

        if self._get_commands_from_keyboard:
            # ---- T键切换双足/四足模式 ----
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    self._keyboard_mode = 1 - self._keyboard_mode
                    mode_name = "四足(QUAD)" if self._keyboard_mode == 1 else "双足(BIPED)"
                    print(f"[Mode切换] → {mode_name}  command[4]={self._keyboard_mode}")

            keys = pygame.key.get_pressed()
            lin_vel_x = 0
            lin_vel_y = 0
            ang_vel = 0
            command_scale=1
            if keys[pygame.K_w]:
                lin_vel_x = torch.tensor(self.command_ranges["lin_vel_x"][1])
            elif keys[pygame.K_s]:
                lin_vel_x = torch.tensor(self.command_ranges["lin_vel_x"][0])
            
            if keys[pygame.K_d]:
                lin_vel_y = torch.tensor(self.command_ranges["lin_vel_y"][0])
            elif keys[pygame.K_a]:
                lin_vel_y = torch.tensor(self.command_ranges["lin_vel_y"][1])
            
            # if keys[pygame.K_q]:
            #     ang_vel = torch.tensor(self.command_ranges["ang_vel_yaw"][1])
            # elif keys[pygame.K_e]:
            #     ang_vel = torch.tensor(self.command_ranges["ang_vel_yaw"][0])

            forward = quat_apply(self.base_quat, self.forward_vec)
            current_heading = torch.atan2(forward[:, 1], forward[:, 0])
            
            if keys[pygame.K_q]:
                self.heading_target += 0.01
            elif keys[pygame.K_e]:
                self.heading_target -= 0.01
            
            self.heading_target = wrap_to_pi(self.heading_target)

            self.commands[:, 0] = lin_vel_x*command_scale
            self.commands[:, 1] = lin_vel_y*command_scale
            self.commands[:, 2] = 0 # ang_vel*command_scale
            self.commands[:, 3] = self.heading_target
            # 每步强制写入当前键盘模式（优先级高于_resample_commands）
            self.commands[:, 4] = float(self._keyboard_mode)

            # print(self.commands[0])

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > self.cfg.env.foot_contact_force

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale,
             self.commands[:, 4:5]), dim=1)  # 6 dims: sin,cos,vx,vy,vyaw,mode
        
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        
        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 6 (sin,cos,vx,vy,vyaw,mode)
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # num_actions
            self.dof_vel * self.obs_scales.dof_vel,  # num_actions
            self.actions,  # num_actions
            diff,  # num_actions
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            stance_mask,   # 2
            contact_mask,  # 2
        ), dim=-1)

        obs_buf = torch.cat((
            self.command_input,  # 6 (sin,cos,vx,vy,vyaw,mode)
            q,    # num_actions
            dq,  # num_actions
            self.actions,   # num_actions
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)


        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
# ================================================ Debugs ================================================== #
        if self._debug:

            measured_heights = torch.sum(
                self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
            base_height = self.root_states[:, 2] - (measured_heights - 0.05)
            # print(base_height)

            foot_pos = self.rigid_state[:, self.feet_indices, :2]
            foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
            # print(foot_dist)

            feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
            delta_z = feet_z - self.last_feet_z
            self.feet_height += delta_z
            # print(self.feet_height)

            foot_pos = self.rigid_state[:, self.knee_indices, :2]
            foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
            # print(foot_dist)

            contact_force = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
            # print(contact_force)

            stance_mask = self._get_gait_phase()
            contact_mask = self.contact_forces[:, self.feet_indices, 2] >  self.cfg.env.foot_contact_force
            # print(stance_mask)
            # print(contact_mask)
            reward = torch.where(contact_mask == stance_mask, 1.0, -0.3)
            # print(torch.mean(reward, dim=1))


            # Compute feet contact mask
            contact = self.contact_forces[:, self.feet_indices, 2] >  self.cfg.env.foot_contact_force
            # Get the z-position of the feet and compute the change in z-position
            feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
            delta_z = feet_z - self.last_feet_z
            self.feet_height += delta_z
            self.last_feet_z = feet_z
            # Compute swing mask
            swing_mask = 1 - self._get_gait_phase()
            # feet height should be closed to target feet height at the peak
            rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
            rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
            self.feet_height *= ~contact
            # return rew_pos
            # print(self.feet_height)

            contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.env.foot_contact_force
            foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
            rew = torch.sqrt(foot_speed_norm)
            rew *= contact
            # print(rew)

            joint_diff = self.dof_pos - self.default_joint_pd_target
            left_yaw_roll = joint_diff[:, list(self.left_yaw_roll_indices)]
            right_yaw_roll = joint_diff[:, list(self.right_yaw_roll_indices)]
            yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
            # yaw_roll = torch.norm(right_yaw_roll, dim=1)

            yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
            # rew = torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
            # print(left_yaw_roll)
            # print(right_yaw_roll)
            # print(yaw_roll)
            # print(list(self.left_yaw_roll_indices), list(self.right_yaw_roll_indices))
            # print(self.default_joint_pd_target)

            left_error = torch.mean(torch.abs(joint_diff[:, self.left_yaw_roll_indices]))
            right_error = torch.mean(torch.abs(joint_diff[:, self.right_yaw_roll_indices]))

            # print("left:", left_error.item())
            # print("right:", right_error.item())

            # print(torch.sum(torch.abs(self.dof_pos - self.default_joint_pd_target), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1))
# ================================================ Debugs ================================================== #

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        Includes both leg and arm joints for quadrupedal walking.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_default_arm_joint_pos(self):
        """
        双足模式：鼓励手臂关节保持默认位置（自然下垂）。
        四足模式下不激活，手臂姿态由joint_pos奖励引导。
        """
        biped_mask = (self.commands[:, 4] < 0.5).float()
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_arm_diff = joint_diff[:, list(self.left_arm_indices)]
        right_arm_diff = joint_diff[:, list(self.right_arm_indices)]
        arm_diff = torch.norm(left_arm_diff, dim=1) + torch.norm(right_arm_diff, dim=1)
        return (torch.exp(-arm_diff * 50) - 0.01 * torch.norm(joint_diff, dim=1)) * biped_mask

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.env.foot_contact_force
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] >  self.cfg.env.foot_contact_force
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)


        # # 真实接触（不再混入期望相位）
        # contact = self.contact_forces[:, self.feet_indices, 2] >  self.cfg.env.foot_contact_force

        # # 简单去抖：当前接触 or 上一帧接触
        # contact_filt = torch.logical_or(contact, self.last_contacts)

        # # 只在“从不接触 -> 接触”瞬间记 first_contact
        # first_contact = torch.logical_and(contact_filt, ~self.last_contacts)

        # self.feet_air_time += self.dt
        # air_time = self.feet_air_time.clamp(0, 0.5) * first_contact

        # # 接触后清零空中计时；离地继续累计
        # self.feet_air_time *= ~contact_filt

        # # 最后再更新上一帧接触
        # self.last_contacts = contact
        # return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        For quadrupedal walking, this includes both feet and arm contacts.
        Rewards or penalizes depending on whether the contact matches the expected gait phase.
        """

        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.env.foot_contact_force
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, list(self.left_yaw_roll_indices)]
        right_yaw_roll = joint_diff[:, list(self.right_yaw_roll_indices)]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_default_joint_pos_left(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, list(self.left_yaw_roll_indices)]
        yaw_roll = torch.norm(left_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100)- 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_default_joint_pos_right(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        right_yaw_roll = joint_diff[:, list(self.right_yaw_roll_indices)]
        yaw_roll = torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100)- 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        双足模式目标高度 ~0.35m（直立），四足模式目标高度 ~0.155m（俯身）。
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)

        mode = self.commands[:, 4]
        biped_mask = (mode < 0.5).float()
        quad_mask  = (mode >= 0.5).float()
        target = self.cfg.rewards.base_height_biped * biped_mask + \
                 self.cfg.rewards.base_height_target * quad_mask
        return torch.exp(-torch.abs(base_height - target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error_y = torch.sum(torch.square(
            self.commands[:, 1] - self.base_lin_vel_lpf[:, 1]), dim=1)
        return torch.exp(-lin_vel_error_y * self.cfg.rewards.tracking_sigma)


    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] >  self.cfg.env.foot_contact_force

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

        # # Compute feet contact mask
        # contact = self.contact_forces[:, self.feet_indices, 2] >  self.cfg.env.foot_contact_force

        # # Get the z-position of the feet and compute the change in z-position
        # feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        # delta_z = feet_z - self.last_feet_z
        # self.feet_height += delta_z
        # self.last_feet_z = feet_z

        # # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()

        # # Continuous reward around target clearance (more stable than hard threshold).
        # height_err = self.feet_height - self.cfg.rewards.target_feet_height
        # rew_pos = torch.exp(-40.0 * torch.square(height_err))
        # rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        # self.feet_height *= ~contact
        # return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        self.feet_state = self.rigid_state[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_stand_still(self):
        """ 惩罚零命令时的运动 """
        # 当命令很小时，惩罚关节位置偏离默认位置
        return torch.sum(torch.abs(self.dof_pos - self.default_joint_pd_target), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_quadrupedal_posture(self):
        """
        仅在四足模式(mode=1)激活：鼓励降低躯干高度并使用手臂。
        """
        quad_mask = (self.commands[:, 4] >= 0.5).float()
        body_height_reward = torch.exp(-torch.abs(self.root_states[:, 2] - 0.1) * 10)
        arm_usage = torch.norm(self.dof_pos[:, list(self.left_arm_indices + self.right_arm_indices)] -
                               self.default_joint_pd_target[:, list(self.left_arm_indices + self.right_arm_indices)], dim=1)
        arm_reward = torch.tanh(arm_usage * 2.0)
        return (body_height_reward * 0.5 + arm_reward * 0.3) * quad_mask

    def _reward_arms_contact(self):
        """
        仅在四足模式(mode=1)激活：鼓励手肘接触地面。
        """
        quad_mask = (self.commands[:, 4] >= 0.5).float()
        if self.left_elbow_index >= 0 and self.right_elbow_index >= 0:
            left_arm_pos = self.rigid_state[:, self.left_elbow_index, 2]
            right_arm_pos = self.rigid_state[:, self.right_elbow_index, 2]
            left_arm_contact = torch.clamp(0.15 - left_arm_pos, 0, 0.15)
            right_arm_contact = torch.clamp(0.15 - right_arm_pos, 0, 0.15)
            left_arm_contact_force = self.contact_forces[:, self.left_elbow_index, 2] > 1.0
            right_arm_contact_force = self.contact_forces[:, self.right_elbow_index, 2] > 1.0
            contact_reward = (left_arm_contact + right_arm_contact) * 2.0 + \
                             left_arm_contact_force.float() + right_arm_contact_force.float()
        else:
            contact_reward = torch.zeros(self.num_envs, device=self.device)
        return contact_reward * quad_mask

    def _reward_body_orientation_for_quadruped(self):
        """
        仅在四足模式(mode=1)激活：鼓励身体保持水平。
        双足模式下由_reward_orientation统一处理。
        """
        quad_mask = (self.commands[:, 4] >= 0.5).float()
        target_orientation = torch.zeros_like(self.projected_gravity[:, :2])
        orientation_error = torch.norm(self.projected_gravity[:, :2] - target_orientation, dim=1)
        return torch.exp(-orientation_error * 10) * quad_mask
