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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class PikachuQuadStandCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 53 # 5 + 3*num_actions + 6 = 5 + 3*14 + 6 = 53
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 81 # 5 + 4*num_actions + 6 = 5 + 4*14 + 20 = 81
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 14
        num_envs = 4096
        episode_length_s = 16
        use_ref_actions = False
        foot_contact_force = 3.0
        hand_contact_force = 3.0
        base_vel_lpf = 0.9

        get_commands_from_keyboard = False
        debug = False
    class safety:
        # safety factors
        pos_limit = 0.9
        vel_limit = 1.0
        torque_limit = 0.2

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Pikachu_V025/urdf/Pikachu_V025_flat_14dof_quad_lite_short.urdf'

        name = "Pikachu_V0025"
        foot_name = "ankle"
        hand_name = "arm_roll"
        knee_name = "knee"

        terminate_after_contacts_on = ['world', 'base_link']  # episode is terminated when contact is detected on these links
        penalize_contacts_on = ["world","base_link"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.4

        class noise_scales:
            dof_pos = 0.03
            dof_vel = 0.2
            ang_vel = 0.08
            lin_vel = 0.05
            quat = 0.02
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.18]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -2.0,         
           'left_knee_joint' : -1.0,       
           'left_ankle_joint' : -0.7,     

           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : 2.0,                                       
           'right_knee_joint' : 1.0,                                             
           'right_ankle_joint' : 0.7,     

           'left_arm_pitch_joint' : -1.77,
           'left_arm_roll_joint' : 0.0,


           'right_arm_pitch_joint' : 1.77,
           'right_arm_roll_joint' : 0.0,

        #    'left_hip_yaw_joint' : 0. ,   
        #    'left_hip_roll_joint' : 0,               
        #    'left_hip_pitch_joint' : 0,         
        #    'left_knee_joint' : 0,       
        #    'left_ankle_joint' : 0,     
        #    'right_hip_yaw_joint' : 0., 
        #    'right_hip_roll_joint' : 0, 
        #    'right_hip_pitch_joint' : 0,                                       
        #    'right_knee_joint' : 0,                                             
        #    'right_ankle_joint' : 0,   
        }

    class random_init:
        enabled = True
        root_pos_xy = 0.05
        root_pos_z_offset = [-0.02, 0.06]
        root_rot_range = {
            "roll": [-0.35, 0.35],
            "pitch": [-0.35, 0.35],
            "yaw": [-0.4, 0.4],
        }
        root_lin_vel_range = {
            "x": [-0.25, 0.25],
            "y": [-0.25, 0.25],
            "z": [-0.1, 0.1],
        }
        root_ang_vel_range = {
            "roll": [-0.8, 0.8],
            "pitch": [-0.8, 0.8],
            "yaw": [-0.8, 0.8],
        }
        joint_pos_range = {
            "hip_pitch": 0.45,
            "hip_roll": 0.12,
            "hip_yaw": 0.12,
            "knee": 0.45,
            "ankle": 0.3,
            "arm_pitch": 0.4,
            "arm_roll": 0.25,
        }
        joint_vel = 0.5

    class recovery:
        orientation_threshold = 0.18
        ang_vel_threshold = 0.6
        lin_vel_threshold = 0.35
        joint_error_threshold = 0.22
        stable_steps = 8

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # stiffness = {'hip_pitch': 80,
        #              'hip_roll': 50,
        #              'hip_yaw': 25,
        #              'knee': 50,
        #              'ankle': 50,
        #              'arm_pitch': 10,
        #              'arm_roll': 10, 
        #              }  # [N*m/rad]
        
        # damping = {  'hip_pitch': 1,
        #              'hip_roll': 0.6,
        #              'hip_yaw': 0.05,
        #              'knee': 0.1,
        #              'ankle': 0.01,
        #              'arm_pitch': 0,
        #              'arm_roll': 0,     
        #              }  # [N*m/rad]  
        

        stiffness = {'hip_pitch': 20,
                     'hip_roll': 9,
                     'hip_yaw': 13.5,
                     'knee': 20,
                     'ankle': 18,
                     'arm_pitch': 2, 
                     'arm_roll': 18,
                     }
        
        damping = {  'hip_pitch': 0.2,
                     'hip_roll': 0,
                     'hip_yaw': 0,
                     'knee': 0.9,
                     'ankle': 0.45,
                     'arm_pitch': 0.45,
                     'arm_roll': 0.09,
                     }
        
        action_scale = 0.25
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.4, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.15, 0.15]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.5
        action_delay = 0.2
        action_noise = 0.01

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 8.
        heading_command = False

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards:
        base_height_target = 0.13
        min_dist = 0.1
        max_dist = 0.3
        target_joint_pos_scale = 0.0
        target_feet_height = 0.0
        cycle_time = 1.0

        only_positive_rewards = True
        tracking_sigma = 5
        max_contact_force = 100

        class scales:
            orientation = 4.0
            base_height = 2.0
            default_joint_pos = 3.0
            feet_distance = 0.3
            knee_distance = 0.2
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            vel_mismatch_exp = 1.0
            stand_still = -0.5
            base_acc = 0.02
            action_smoothness = -0.002
            torques = -2e-4
            dof_vel = -1e-3
            dof_acc = -1e-7
            feet_contact_forces = -0.005
            collision = -1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class PikachuQuadStandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60
        max_iterations = 3001

        save_interval = 100
        experiment_name = 'Pikachu_V025_Quad_Stand'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

# =========刚体顺序=========
# ['world', 'left_arm_pitch_link', 'left_arm_roll_link', 'left_arm_yaw_link', 'left_elbow_ankle_link', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_link', 'right_arm_pitch_link', 'right_arm_roll_link', 'right_arm_yaw_link', 'right_elbow_ankle_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_link']
# =========DOF顺序=========
# DOF数量:18
 
# ['left_arm_pitch_joint', 'left_arm_roll_joint', 'left_arm_yaw_joint', 'left_elbow_ankle_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_joint', 'right_arm_pitch_joint', 'right_arm_roll_joint', 'right_arm_yaw_joint', 'right_elbow_ankle_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_joint']
# DOF属性:
# ('hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature')
# [( True, -3.14, 1.05, 3, 2.,  9. , 0., 0.1 , 0.01, 0.)
#  ( True,  0.  , 1.57, 3, 2.,  9. , 0., 0.1 , 0.01, 0.)
#  ( True, -1.57, 1.57, 3, 2.,  9. , 0., 0.1 , 0.01, 0.)
#  ( True, -2.44, 0.  , 3, 2.,  9. , 0., 0.1 , 0.01, 0.)
#  ( True, -2.44, 0.  , 3, 2., 12.5, 0., 0.4 , 0.01, 0.)
#  ( True, -0.09, 0.26, 3, 1.,  9. , 0., 0.15, 0.01, 0.)
#  ( True, -0.26, 0.09, 3, 1.,  9. , 0., 0.15, 0.01, 0.)
#  ( True, -1.57, 0.  , 3, 3., 12.5, 0., 0.15, 0.01, 0.)
#  ( True, -1.05, 0.52, 3, 2.,  9. , 0., 0.4 , 0.01, 0.)
#  ( True, -1.05, 3.14, 3, 2.,  9. , 0., 0.1 , 0.01, 0.)
#  ( True, -1.57, 0.  , 3, 2.,  9. , 0., 0.1 , 0.01, 0.)
#  ( True, -1.57, 1.57, 3, 2.,  9. , 0., 0.1 , 0.01, 0.)
#  ( True,  0.  , 2.44, 3, 2.,  9. , 0., 0.1 , 0.01, 0.)
#  ( True,  0.  , 2.44, 3, 2., 12.5, 0., 0.4 , 0.01, 0.)
#  ( True, -0.26, 0.09, 3, 1.,  9. , 0., 0.15, 0.01, 0.)
#  ( True, -0.09, 0.26, 3, 1.,  9. , 0., 0.15, 0.01, 0.)
#  ( True,  0.  , 1.57, 3, 3., 12.5, 0., 0.4 , 0.01, 0.)
#  ( True, -0.52, 1.05, 3, 2.,  9. , 0., 0.15, 0.01, 0.)]

# =========================
