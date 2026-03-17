import time
import sys
import os

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

from deploy_joystick import DeployJoystick

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

from LPF import LPF

TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
LEGGED_GYM_ROOT_DIR = "/home/finnox/Pikachu/pikachu_gym"


sys.path.append(os.path.join(os.path.dirname(__file__)))

from log.data_logger import DataLogger
from log.plot_data import plot_csv_data

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    # 重力方向
    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_to_mat(quat):
    """将四元数 (w, x, y, z) 转换为旋转矩阵"""
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
    ])
    return R

# PD 控制器  目标位置  当前位置  Kp 目标速度  当前速度  Kd
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def WirelessControllerHandler(msg:WirelessController_):
    global joymsg
    joymsg=msg

if __name__ == "__main__":
    # get config file name from command line
    import argparse
    # 获取传入参数
    parser = argparse.ArgumentParser()
    # 设定参数名称和类型 config_file  str  提示
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()     # 获取参数
    config_file = args.config_file # 传入参数
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        # 加载yaml参数
        config = yaml.load(f, Loader=yaml.FullLoader)
        # 策略路径      replace 用具体参数替代yaml中的位置 
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        # 加载mujoco模型 
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        # 仿真时间
        simulation_duration = config["simulation_duration"]
        # 仿真时间步长
        simulation_dt = config["simulation_dt"]
        # 控制周期倍率  control freq = comtrol_decimation * dt
        control_decimation = config["control_decimation"]

        # 加载PD参数 并设置为np.array格式
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        # 加载初始化默认关节角度
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        zero_angles = np.array(config["zero_angles"], dtype=np.float32)

        # lin_vel_scale = config["lin_vel_scale"]
        # yaw角速度缩放系数?
        ang_vel_scale = config["ang_vel_scale"]
        # 关节角度缩放系数
        dof_pos_scale = config["dof_pos_scale"]
        # 关节速度缩放系数
        dof_vel_scale = config["dof_vel_scale"]
        # 动作缩放系数
        action_scale = config["action_scale"]
        # 控制指令缩放系数
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        # 动作空间
        num_actions = config["num_actions"]
        # 观测器空间
        num_obs = config["num_obs"]
        # 初始化指令
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        # joy
        use_joystick=config["joystick"]

    # define context variables
    # 初始化动作 目标位置 观测器 
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_pos_new= default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    # Joystick初始化
    if use_joystick:
        ChannelFactoryInitialize(1)
        DeployJoy=DeployJoystick()
        DeployJoy.SetupJoystick()

        wirelesscontroller = ChannelSubscriber(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
        wirelesscontroller.Init(WirelessControllerHandler, 10)


    counter = 0

    # Load robot model
    # 加载机器人Mujoco模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt


    # 初始化输出机器人关节信息 
    print("<<------------- Link ------------->> ")
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco._enums.mjtObj.mjOBJ_BODY, i)
        if name:
            print("link_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Joint ------------->> ")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
        if name:
            print("joint_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Actuator ------------->>")
    for i in range(m.nu):
        name = mujoco.mj_id2name(
            m, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
        )
        if name:
            print("actuator_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Sensor ------------->>")
    index = 0
    for i in range(m.nsensor):
        name = mujoco.mj_id2name(
            m, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
        )
        if name:
            print(
                "sensor_index:",
                index,
                ", name:",
                name,
                ", dim:",
                m.sensor_dim[i],
            )
        index = index + m.sensor_dim[i]
    print(" ")



    # load policy
    policy = torch.jit.load(policy_path)


    # LPF
    filter = LPF(alpha=0.5, n_dof=num_actions)
    tau_filter = LPF(alpha=0.9, n_dof=num_actions)

    tau_limit=[12.5,9,9,12.5,9  ,12.5,9,9,12.5,9]

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        # 仿真总时间 duration
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time() # 单步时间记录


            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            tau=np.nan_to_num(tau,nan=0.0,posinf=0.0,neginf=0.0)
          
            d.ctrl[:]=tau

            mujoco.mj_step(m, d) 

            counter += 1
            if counter % control_decimation == 0:
     
                # 周期
                period = 0.8
                count = counter * simulation_dt
                # # 相位
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)


                if np.cos(counter/100)>0:
                    target_dof_pos = default_angles
                else:
                    target_dof_pos = zero_angles


            # 更新GUI 
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)