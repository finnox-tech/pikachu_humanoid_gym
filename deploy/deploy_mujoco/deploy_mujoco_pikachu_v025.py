import argparse
import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
if LEGGED_GYM_ROOT_DIR not in sys.path:
    sys.path.append(LEGGED_GYM_ROOT_DIR)

TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
joymsg = None


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def quaternion_wxyz_to_euler_xyz(quat_wxyz):
    w, x, y, z = quat_wxyz

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    euler_xyz = np.array([roll_x, pitch_y, yaw_z], dtype=np.float32)
    euler_xyz[euler_xyz > np.pi] -= 2.0 * np.pi
    return euler_xyz


def get_sensor_data(model, data, sensor_name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise KeyError(f"Sensor '{sensor_name}' not found in model.")
    start = model.sensor_adr[sensor_id]
    end = start + model.sensor_dim[sensor_id]
    return data.sensordata[start:end]


def scale_joystick_axis(raw_value, lower, upper):
    return raw_value * upper if raw_value >= 0.0 else raw_value * abs(lower)


def print_model_info(model):
    print("<<------------- Link ------------->>")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            print("link_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Joint ------------->>")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print("joint_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Actuator ------------->>")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            print("actuator_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Sensor ------------->>")
    index = 0
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if name:
            print(
                "sensor_index:",
                index,
                ", name:",
                name,
                ", dim:",
                model.sensor_dim[i],
            )
        index += model.sensor_dim[i]
    print(" ")


def get_actuated_joint_names(model):
    joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "floating_base":
            joint_names.append(name)
    return joint_names


def reset_history(obs_history, frame_stack, num_single_obs):
    obs_history.clear()
    for _ in range(frame_stack):
        obs_history.append(np.zeros(num_single_obs, dtype=np.float32))


def reset_simulation_state(
    model,
    data,
    target_policy,
    action,
    obs_history,
    frame_stack,
    num_single_obs,
    default_angles_mj,
    default_angles_policy,
):
    mujoco.mj_resetData(model, data)
    data.qpos[7: 7 + len(default_angles_mj)] = default_angles_mj
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    target_policy[:] = default_angles_policy
    action[:] = 0.0
    reset_history(obs_history, frame_stack, num_single_obs)


def wireless_controller_handler(msg):
    global joymsg
    joymsg = msg


def init_joystick():
    try:
        from deploy_joystick import DeployJoystick
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
    except ImportError as exc:
        raise ImportError(
            "Joystick mode requires deploy_joystick and unitree_sdk2py dependencies."
        ) from exc

    ChannelFactoryInitialize(1)
    deploy_joy = DeployJoystick()
    deploy_joy.SetupJoystick()

    wirelesscontroller = ChannelSubscriber(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
    wirelesscontroller.Init(wireless_controller_handler, 10)
    return deploy_joy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()

    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy", "deploy_mujoco", "configs", args.config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    cycle_time = config["cycle_time"]
    policy_warmup_steps = config.get("policy_warmup_steps", 0)

    joint_names = config["joint_names"]
    joint_signs = np.array(config["joint_signs_urdf_to_mjcf"], dtype=np.float32)
    default_angles_policy = np.array(config["default_angles"], dtype=np.float32)
    joint_lower = np.array(config["joint_lower"], dtype=np.float32)
    joint_upper = np.array(config["joint_upper"], dtype=np.float32)

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    tau_limit = np.array(config["tau_limit"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    quat_scale = config["quat_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    cmd_min = np.array(config["cmd_min"], dtype=np.float32)
    cmd_max = np.array(config["cmd_max"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_single_obs = config["num_single_obs"]
    frame_stack = config["frame_stack"]
    num_obs = config["num_obs"]
    clip_observations = config["clip_observations"]
    clip_actions = config["clip_actions"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    use_joystick = bool(config["joystick"])

    if num_obs != frame_stack * num_single_obs:
        raise ValueError(
            f"num_obs={num_obs} does not match frame_stack * num_single_obs={frame_stack * num_single_obs}."
        )
    for name, array in {
        "joint_names": joint_names,
        "joint_signs_urdf_to_mjcf": joint_signs,
        "default_angles": default_angles_policy,
        "joint_lower": joint_lower,
        "joint_upper": joint_upper,
        "kps": kps,
        "kds": kds,
        "tau_limit": tau_limit,
    }.items():
        if len(array) != num_actions:
            raise ValueError(f"{name} length {len(array)} does not match num_actions={num_actions}.")

    default_angles_mj = joint_signs * default_angles_policy

    action = np.zeros(num_actions, dtype=np.float32)
    target_policy = default_angles_policy.copy()
    obs_history = deque(maxlen=frame_stack)
    reset_history(obs_history, frame_stack, num_single_obs)

    deploy_joy = init_joystick() if use_joystick else None

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    print_model_info(model)

    model_joint_names = get_actuated_joint_names(model)
    if model_joint_names != joint_names:
        raise ValueError(
            "Configured joint order does not match Mujoco joint order.\n"
            f"config: {joint_names}\n"
            f"model : {model_joint_names}"
        )

    data.qpos[7: 7 + num_actions] = default_angles_mj
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    sim_steps = 0
    control_steps = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            q_mj = data.qpos[7: 7 + num_actions].copy()
            dq_mj = data.qvel[6: 6 + num_actions].copy()
            target_mj = joint_signs * target_policy

            tau = pd_control(target_mj, q_mj, kps, np.zeros_like(kds), dq_mj, kds)
            tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
            tau = np.clip(tau, -tau_limit, tau_limit)

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            sim_steps += 1

            if sim_steps % control_decimation == 0:
                if use_joystick and joymsg is not None:
                    cmd[0] = scale_joystick_axis(joymsg.rx, cmd_min[0], cmd_max[0])
                    cmd[1] = scale_joystick_axis(joymsg.ry, cmd_min[1], cmd_max[1])
                    cmd[2] = scale_joystick_axis(-joymsg.lx, cmd_min[2], cmd_max[2])

                    joykeys_event = deploy_joy.JoykeysPress()
                    if "select" in joykeys_event["press"]:
                        target_policy[:] = default_angles_policy
                        reset_simulation_state(
                            model,
                            data,
                            target_policy,
                            action,
                            obs_history,
                            frame_stack,
                            num_single_obs,
                            default_angles_mj,
                            default_angles_policy,
                        )
                        sim_steps = 0
                        control_steps = 0
                        print("Environment reset!")
                        viewer.sync()
                        continue

                cmd = np.clip(cmd, cmd_min, cmd_max)

                q_mj = data.qpos[7: 7 + num_actions].copy()
                dq_mj = data.qvel[6: 6 + num_actions].copy()
                q_policy = joint_signs * q_mj
                dq_policy = joint_signs * dq_mj

                try:
                    quat_wxyz = get_sensor_data(model, data, "imu_quat").astype(np.float32)
                    omega_body = get_sensor_data(model, data, "imu_gyro").astype(np.float32)
                except KeyError:
                    quat_wxyz = data.qpos[3:7].astype(np.float32)
                    omega_body = data.qvel[3:6].astype(np.float32)

                euler_xyz = quaternion_wxyz_to_euler_xyz(quat_wxyz)

                phase = (sim_steps * simulation_dt) / cycle_time
                single_obs = np.zeros(num_single_obs, dtype=np.float32)
                single_obs[0] = np.sin(2.0 * np.pi * phase)
                single_obs[1] = np.cos(2.0 * np.pi * phase)
                single_obs[2:5] = cmd * cmd_scale
                single_obs[5: 5 + num_actions] = (q_policy - default_angles_policy) * dof_pos_scale
                single_obs[5 + num_actions: 5 + 2 * num_actions] = dq_policy * dof_vel_scale
                single_obs[5 + 2 * num_actions: 5 + 3 * num_actions] = action
                single_obs[5 + 3 * num_actions: 5 + 3 * num_actions + 3] = omega_body * ang_vel_scale
                single_obs[5 + 3 * num_actions + 3:] = euler_xyz * quat_scale

                obs_history.append(single_obs)
                policy_input = np.concatenate(list(obs_history), axis=0).astype(np.float32)
                policy_input = np.clip(policy_input, -clip_observations, clip_observations)

                with torch.no_grad():
                    action = policy(torch.from_numpy(policy_input).unsqueeze(0)).cpu().numpy().squeeze().astype(np.float32)

                action = np.clip(action, -clip_actions, clip_actions)
                action_target_policy = action * action_scale + default_angles_policy
                action_target_policy = np.clip(action_target_policy, joint_lower, joint_upper)

                if control_steps >= policy_warmup_steps:
                    target_policy[:] = action_target_policy
                else:
                    target_policy[:] = default_angles_policy

                control_steps += 1

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
