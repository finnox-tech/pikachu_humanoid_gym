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


import asyncio
import json
from pathlib import Path
import queue
import threading
import math
import xml.etree.ElementTree as ET
from collections import deque

import mujoco
import numpy as np

from humanoid import LEGGED_GYM_ROOT_DIR


def _ws_sender_task(ws_queue, ws_uri='ws://localhost:8765'):
    async def _run():
        try:
            import websockets

            async with websockets.connect(ws_uri) as ws:
                await ws.send(json.dumps({'client_type': 'sim'}))
                while True:
                    payload = await asyncio.get_event_loop().run_in_executor(None, ws_queue.get)
                    if payload is None:
                        break
                    message = json.dumps({'source': 'sim', 'data': payload})
                    await ws.send(message)
        except Exception as e:
            print(f"[WS] sender task error: {e}")

    asyncio.run(_run())


def start_ws_sender(ws_queue, ws_uri='ws://localhost:8765'):
    th = threading.Thread(target=_ws_sender_task, args=(ws_queue, ws_uri), daemon=True)
    th.start()
    return th


def _collect_body_names(root):
    return {elem.get("name") for elem in root.iter("body") if elem.get("name")}


def _prepare_mujoco_model_xml(model_xml_path):
    model_path = Path(model_xml_path).resolve()
    model_tree = ET.parse(model_path)
    model_root = model_tree.getroot()
    body_names = _collect_body_names(model_root)

    generated_paths = []
    patched_model = False

    for include_elem in model_root.iter("include"):
        include_file = include_elem.get("file")
        if not include_file:
            continue

        include_path = (model_path.parent / include_file).resolve()
        if include_path.name != "contact.xml" or not include_path.exists():
            continue

        include_tree = ET.parse(include_path)
        include_root = include_tree.getroot()
        contact_elem = include_root.find("contact")
        if contact_elem is None:
            continue

        removed_pairs = []
        for exclude_elem in list(contact_elem.findall("exclude")):
            body1 = exclude_elem.get("body1")
            body2 = exclude_elem.get("body2")
            if body1 not in body_names or body2 not in body_names:
                removed_pairs.append((body1, body2))
                contact_elem.remove(exclude_elem)

        if not removed_pairs:
            continue

        sanitized_include_path = include_path.with_name(
            f"{include_path.stem}.sim2sim_sanitized{include_path.suffix}"
        )
        include_tree.write(sanitized_include_path, encoding="utf-8")
        include_elem.set("file", sanitized_include_path.name)
        generated_paths.append(sanitized_include_path)
        patched_model = True

        missing_names = sorted(
            {
                name
                for pair in removed_pairs
                for name in pair
                if name and name not in body_names
            }
        )
        print(
            "[sim2sim] Removed"
            f" {len(removed_pairs)} invalid contact excludes from {include_path.name}:"
            f" missing bodies {missing_names}"
        )

    if not patched_model:
        return str(model_path), []

    sanitized_model_path = model_path.with_name(
        f"{model_path.stem}.sim2sim_sanitized{model_path.suffix}"
    )
    model_tree.write(sanitized_model_path, encoding="utf-8")
    generated_paths.append(sanitized_model_path)
    return str(sanitized_model_path), generated_paths


def _cleanup_generated_files(paths):
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def make_ws_entry(**kwargs):
    """接收任意参数，自动转换为 JSON 序列化格式"""
    payload = {}
    for key, val in kwargs.items():
        if isinstance(val, np.ndarray):
            payload[key] = val.tolist() if val.size > 1 else float(val.flat[0])
        elif isinstance(val, (int, np.integer)):
            payload[key] = int(val)
        elif isinstance(val, (float, np.floating)):
            payload[key] = float(val)
        else:
            payload[key] = val
    return payload


class cmd:
    vx = 0.3
    vy = 0.0
    dyaw = 0.0


JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "left_arm_pitch_joint",
    "left_arm_roll_joint",
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
)

DEFAULT_Q = np.array([-2.0, 0.0, 0.0, -1.0, -0.7, 2.0, 0.0, 0.0, 1.0, 0.7, -1.77, 0.0, 1.77, 0.0], dtype=np.double)
# pose_r= [ 2.9, 0.0, 0.0, 1.57, 0.8]
# pose_l= [-2.9, 0.0, 0.0,-1.57,-0.8]
# pose_a= [0.0,-1.57,1.57,0.0]
Q_stand_ready = np.array([-2.44, 0.0, 0.0, -1.57, -0.8, 2.44, 0.0, 0.0, 1.57, 0.8, -1.57, 0.0, 1.57, 0.0], dtype=np.double)
# 
# pose_r= [ 0.8, 0.0, 0.0, 1.12,0.6]
# pose_l= [-0.8, 0.0, 0.0, -1.12,-0.6]
Q_stand_up = np.array([-0.8, 0.0, 0.0, -1.12, -0.6, 0.8, 0.0, 0.0, 1.12, 0.6, -0, 0.0, 0, 0.0], dtype=np.double)

class Sim2simCfg:
    class env:
        frame_stack = 15
        num_single_obs = 53
        num_observations = frame_stack * num_single_obs
        num_actions = 14

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0

        clip_observations = 18.0
        clip_actions = 18.0

    class control:
        action_scale = 0.25

    class rewards:
        cycle_time = 0.56

    class sim_config:
        mujoco_model_path = (
            f"{LEGGED_GYM_ROOT_DIR}/resources/robots/Pikachu_V025/mjcf_14dof_lite/Pikachu_V025_quad_14dof_lite.xml"
        )
        sim_duration = 60.0
        dt = 0.001
        decimation = 10

    class plot:
        enabled = False
        max_points = 1500
        redraw_interval = 5

    class robot_config:
        default_q = DEFAULT_Q.copy()
        kps = np.array([20, 9, 13.5, 20, 18, 20, 9, 13.5, 20, 18, 2, 18, 2, 18], dtype=np.double)
        kds = np.array([0.2, 0.0, 0.0, 0.9, 0.45, 0.2, 0.0, 0.0, 0.9, 0.45, 0.45, 0.09, 0.45, 0.09], dtype=np.double)
        tau_limit = 0.2 * np.array([12.5, 9.0, 9.0, 12.5, 9.0, 12.5, 9.0, 9.0, 12.5, 9.0, 9.0, 9.0, 9.0, 9.0], dtype=np.double)


        # <<------------- Actuator ------------->>
        # actuator_index: 0 , name: act_Lhip_pitch
        # actuator_index: 1 , name: act_Lhip_roll
        # actuator_index: 2 , name: act_Lhip_yaw
        # actuator_index: 3 , name: act_Lknee
        # actuator_index: 4 , name: act_Lankle
        # actuator_index: 5 , name: act_Rhip_pitch
        # actuator_index: 6 , name: act_Rhip_roll
        # actuator_index: 7 , name: act_Rhip_yaw
        # actuator_index: 8 , name: act_Rknee
        # actuator_index: 9 , name: act_Rankle
        # actuator_index: 10 , name: act_Larm_pitch
        # actuator_index: 11 , name: act_Larm_roll
        # actuator_index: 12 , name: act_Rarm_pitch
        # actuator_index: 13 , name: act_Rarm_roll
 
class ObsPlotter:
    def __init__(self, max_points, redraw_interval):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.max_points = max_points
        self.redraw_interval = max(1, redraw_interval)
        self.steps = deque(maxlen=max_points)
        self.sin_values = deque(maxlen=max_points)
        self.cos_values = deque(maxlen=max_points)
        self.omega_values = [deque(maxlen=max_points) for _ in range(3)]
        self.eu_ang_values = [deque(maxlen=max_points) for _ in range(3)]
        self.update_count = 0

        self.plt.ion()
        self.fig, self.axes = self.plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Pikachu V025 Quad Obs Plot")

        self.sin_line, = self.axes[0].plot([], [], label="sin")
        self.cos_line, = self.axes[0].plot([], [], label="cos")

        omega_labels = ("omega_x", "omega_y", "omega_z")
        self.omega_lines = [
            self.axes[1].plot([], [], label=label)[0] for label in omega_labels
        ]

        eu_ang_labels = ("roll", "pitch", "yaw")
        self.eu_ang_lines = [
            self.axes[2].plot([], [], label=label)[0] for label in eu_ang_labels
        ]

        self.axes[0].set_title("obs_sin_cos")
        self.axes[1].set_title("obs_omega")
        self.axes[2].set_title("obs_eu_ang")
        self.axes[2].set_xlabel("record step")

        for ax in self.axes:
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(self, step, sin_cos, omega, eu_ang):
        self.steps.append(step)
        self.sin_values.append(float(sin_cos[0]))
        self.cos_values.append(float(sin_cos[1]))

        for idx in range(3):
            self.omega_values[idx].append(float(omega[idx]))
            self.eu_ang_values[idx].append(float(eu_ang[idx]))

        self.update_count += 1
        if self.update_count % self.redraw_interval != 0:
            return

        x = np.asarray(self.steps, dtype=np.int32)
        self.sin_line.set_data(x, np.asarray(self.sin_values, dtype=np.float32))
        self.cos_line.set_data(x, np.asarray(self.cos_values, dtype=np.float32))

        for idx, line in enumerate(self.omega_lines):
            line.set_data(x, np.asarray(self.omega_values[idx], dtype=np.float32))

        for idx, line in enumerate(self.eu_ang_lines):
            line.set_data(x, np.asarray(self.eu_ang_values[idx], dtype=np.float32))

        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self):
        if self.fig.canvas.manager is not None:
            self.plt.ioff()
            self.fig.canvas.draw_idle()
            self.plt.show()


def quaternion_to_euler_array(quat):
    x, y, z, w = quat

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])


def quaternion_to_rotation_axes(quat):
    # quat is [x, y, z, w] from IMU sensor order output
    x, y, z, w = quat
    # world-frame axis directions of robot base local axes
    x_axis = np.array([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)])
    y_axis = np.array([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)])
    z_axis = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    return x_axis, y_axis, z_axis


def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("imu_quat").data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor("imu_gyro").data.astype(np.double)
    return q, dq, quat, omega


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg, ws_queue=None):
    plotter = None
    import mujoco.viewer
    import torch
    from tqdm import tqdm

    model_xml_path, generated_files = _prepare_mujoco_model_xml(cfg.sim_config.mujoco_model_path)
    try:
        model = mujoco.MjModel.from_xml_path(model_xml_path)
    finally:
        _cleanup_generated_files(generated_files)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    # 初始化输出机器人关节信息
    print("<<------------- Link ------------->> ")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            print("link_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Joint ------------->> ")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print("joint_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Actuator ------------->>")
    for i in range(model.nu):
        name = mujoco.mj_id2name(
            model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
        )
        if name:
            print("actuator_index:", i, ", name:", name)
    print(" ")

    data.qpos[7:] = cfg.robot_config.default_q
    mujoco.mj_forward(model, data)

    if cfg.plot.enabled:
        plotter = ObsPlotter(
            max_points=cfg.plot.max_points,
            redraw_interval=cfg.plot.redraw_interval,
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_q = cfg.robot_config.default_q.copy()
        action = np.zeros(cfg.env.num_actions, dtype=np.double)
        last_action = action.copy()

        hist_obs = deque()
        for _ in range(cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

        count_lowlevel = 0
        max_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
        obs_plot = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)

        with tqdm(total=max_steps, desc="Simulating...") as pbar:
            while viewer.is_running() and count_lowlevel < max_steps:
                # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY # mjFRAME_SITE
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE                
                q, dq, quat, omega = get_obs(data)
                q = q[-cfg.env.num_actions:]
                dq = dq[-cfg.env.num_actions:]

                if count_lowlevel >= 250:
                    if count_lowlevel % cfg.sim_config.decimation == 0:
                        obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                        eu_ang = quaternion_to_euler_array(quat)
                        eu_ang[eu_ang > math.pi] -= 2 * math.pi

                        obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
                        obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
                        obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                        obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                        obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel

                        obs[0, 5:19] = (q - cfg.robot_config.default_q) * cfg.normalization.obs_scales.dof_pos
                        obs[0, 19:33] = dq * cfg.normalization.obs_scales.dof_vel
                        obs[0, 33:47] = action

                        obs[0, 47:50] = omega * cfg.normalization.obs_scales.ang_vel
                        obs[0, 50:53] = eu_ang * cfg.normalization.obs_scales.quat

                        obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                        # print(f"obs_sin_cos: {obs[0, 0:2]}")
                        # print(f"obs_cmd: {obs[0, 2:5]}")
                        # print(f"obs_dof_pos: {obs[0, 5:19]}")
                        # print(f"obs_dof_vel: {obs[0, 19:33]}")
                        # print(f"obs_action: {obs[0, 33:47]}")
                        # print(f"obs_omega: {obs[0, 47:50]}")
                        # print(f"obs_eu_ang: {obs[0, 50:53]}")
                        obs_plot=obs.copy()

                        if plotter is not None:
                            plotter.update(
                                step=count_lowlevel,
                                sin_cos=obs_plot[0, 0:2],
                                omega=obs_plot[0, 47:50],
                                eu_ang=obs_plot[0, 50:53],
                            )

                        hist_obs.append(obs)
                        hist_obs.popleft()

                        policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                        for i in range(cfg.env.frame_stack):
                            start = i * cfg.env.num_single_obs
                            end = (i + 1) * cfg.env.num_single_obs
                            policy_input[0, start:end] = hist_obs[i][0, :]

                        action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                        action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                        
                        delay = np.random.uniform(0, 1)
                        action = (1 - delay) * action + delay * last_action
                        last_action = action.copy()

                        # 4000 步内按三段插值：默认 -> 站立准备 -> 站立抬起
                        # if count_lowlevel < 500:
                        #     t = float(count_lowlevel) / 500.0
                        #     target_q = (1 - t) * cfg.robot_config.default_q + t * Q_stand_ready
                        # elif count_lowlevel < 2000:
                        #     t = float(count_lowlevel - 500) / 1500.0
                        #     target_q = (1 - t) * Q_stand_ready + t * Q_stand_up
                            
                        # elif count_lowlevel < 1500:
                        #     t = float(count_lowlevel - 1000) / 500.0
                        #     target_q = (1 - t) * Q_stand_ready + t * Q_stand_up
                        # else:
                        target_q = action * cfg.control.action_scale + cfg.robot_config.default_q
                else:
                    # 250 步前也做移动插值过渡, 不直接用 default_q 固定体态
                    if count_lowlevel < 2000:
                        t = float(count_lowlevel) / 2000.0
                        target_q = (1 - t) * cfg.robot_config.default_q + t * Q_stand_ready
                    else:
                        t = float(max(0, count_lowlevel - 2000)) / 2000.0
                        t = min(t, 1.0)
                        target_q = (1 - t) * Q_stand_ready + t * Q_stand_up


                target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)
                tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
                tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
                # print(tau)
                data.ctrl = tau
                mujoco.mj_step(model, data)
                viewer.sync()
                count_lowlevel += 1
                pbar.update(1)

    if plotter is not None:
        plotter.close()

if __name__ == "__main__":
    import argparse
    import torch

    default_load_model = (
        f"{LEGGED_GYM_ROOT_DIR}/pre_train/Pikachu_V025_Quad/jit/policy_stand_still_quad_lite.pt"
    )
    parser = argparse.ArgumentParser(description="Pikachu V025 sim2sim deployment script.")
    parser.add_argument(
        "--load_model",
        type=str,
        default=default_load_model,
        help="Run to load from.",
    )
    parser.add_argument(
        "--ws_uri",
        type=str,
        default=None,
        help="WebSocket server URI. If omitted, websocket streaming is disabled.",
    )
    args = parser.parse_args()

    policy = torch.jit.load(args.load_model)

    ws_queue = None
    if args.ws_uri:
        ws_queue = queue.Queue(maxsize=1000)
        start_ws_sender(ws_queue, args.ws_uri)

    try:
        run_mujoco(policy, Sim2simCfg(), ws_queue)
    finally:
        if ws_queue is not None:
            ws_queue.put(None)
