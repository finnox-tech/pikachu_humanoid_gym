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

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from multiprocessing import Process

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self, save_path=None, show=True):
        # Use a separate process for interactive plotting so simulation thread is never blocked.
        if self.plot_process is not None and self.plot_process.is_alive():
            return
        state_snapshot = dict(self.state_log)
        self.plot_process = Process(
            target=self._plot_worker,
            args=(state_snapshot, self.dt, save_path, show),
        )
        self.plot_process.start()

    @staticmethod
    def _plot_worker(state_log, dt, save_path, show):
        fig = Logger._plot_from_state_log(state_log, dt)
        if fig is None:
            return
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved state plot to: {save_path}")
        if show:
            # In child process we can block safely; this keeps window alive.
            plt.show()
        else:
            plt.close(fig)

    def _plot(self):
        return self._plot_from_state_log(self.state_log, self.dt)

    @staticmethod
    def _plot_from_state_log(log, dt):
        try:
            def series(key):
                return log.get(key, [])

            if len(log) == 0:
                print("No states logged. Skip plotting.")
                return None

            nb_rows = 3
            nb_cols = 3
            fig, axs = plt.subplots(nb_rows, nb_cols)
            time = None
            for _, value in log.items():
                time = np.linspace(0, len(value) * dt, len(value))
                break
            if time is None:
                print("No valid state series found. Skip plotting.")
                plt.close(fig)
                return None
            # plot joint targets and measured positions
            a = axs[1, 0]
            if series("dof_pos"): a.plot(time, series("dof_pos"), label='measured')
            if series("dof_pos_target"): a.plot(time, series("dof_pos_target"), label='target')
            a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
            a.legend()
            # plot joint velocity
            a = axs[1, 1]
            if series("dof_vel"): a.plot(time, series("dof_vel"), label='measured')
            if series("dof_vel_target"): a.plot(time, series("dof_vel_target"), label='target')
            a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
            a.legend()
            # plot base vel x
            a = axs[0, 0]
            if series("base_vel_x"): a.plot(time, series("base_vel_x"), label='measured')
            if series("command_x"): a.plot(time, series("command_x"), label='commanded')
            a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
            a.legend()
            # plot base vel y
            a = axs[0, 1]
            if series("base_vel_y"): a.plot(time, series("base_vel_y"), label='measured')
            if series("command_y"): a.plot(time, series("command_y"), label='commanded')
            a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
            a.legend()
            # plot base vel yaw
            a = axs[0, 2]
            if series("base_vel_yaw"): a.plot(time, series("base_vel_yaw"), label='measured')
            if series("command_yaw"): a.plot(time, series("command_yaw"), label='commanded')
            a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
            a.legend()
            # plot base vel z
            a = axs[1, 2]
            if series("base_vel_z"): a.plot(time, series("base_vel_z"), label='measured')
            a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
            a.legend()
            # plot contact forces
            a = axs[2, 0]
            if series("contact_forces_z"):
                forces = np.array(series("contact_forces_z"))
                for i in range(forces.shape[1]):
                    a.plot(time, forces[:, i], label=f'force {i}')
            a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
            a.legend()
            # plot torque/vel curves
            a = axs[2, 1]
            if series("dof_vel") != [] and series("dof_torque") != []:
                a.plot(series("dof_vel"), series("dof_torque"), 'x', label='measured')
            a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
            a.legend()
            # plot torques
            a = axs[2, 2]
            if series("dof_torque") != []: a.plot(time, series("dof_torque"), label='measured')
            a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
            a.legend()
            return fig

        except Exception as e:
            print(f"Error creating plot: {e}")
            return None
        finally:
            plt.close('all')  # 确保清理
    def print_rewards(self):
        print("Average rewards per second:")
        if self.num_episodes == 0:
            print(" - No completed episodes yet; rewards were not aggregated.")
            print(f"Total number of episodes: {self.num_episodes}")
            return
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
