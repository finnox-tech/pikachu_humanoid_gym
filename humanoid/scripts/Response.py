#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import copy
import csv
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import isaacgym  # noqa: F401
from isaacgym import gymutil, gymtorch
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
import humanoid.envs  # noqa: F401
from humanoid.utils.task_registry import task_registry


def parse_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "Pikachu_V025_Quad",
            "help": "Registered task name.",
        },
        {
            "name": "--show_viewer",
            "action": "store_true",
            "default": False,
            "help": "Open Isaac Gym viewer during the response test.",
        },
        {
            "name": "--output_dir",
            "type": str,
            "default": "",
            "help": "Optional output directory. Defaults to logs/response/<task>/<timestamp>.",
        },
        {
            "name": "--step_fraction",
            "type": float,
            "default": 0.35,
            "help": "Step size as a fraction of available room from default pose to the selected joint limit.",
        },
        {
            "name": "--position_margin",
            "type": float,
            "default": 0.03,
            "help": "Safety margin from the effective joint limit [rad].",
        },
        {
            "name": "--step_direction",
            "type": str,
            "default": "auto",
            "help": "One of: auto, positive, negative, both.",
        },
        {
            "name": "--pre_steps",
            "type": int,
            "default": 25,
            "help": "Number of policy steps to hold the default target before the step.",
        },
        {
            "name": "--response_steps",
            "type": int,
            "default": 220,
            "help": "Number of policy steps recorded after the step target is applied.",
        },
        {
            "name": "--settle_tol_ratio",
            "type": float,
            "default": 0.02,
            "help": "Settling band as a fraction of step amplitude.",
        },
        {
            "name": "--settle_tol_abs",
            "type": float,
            "default": 0.01,
            "help": "Minimum absolute settling band [rad].",
        },
        {
            "name": "--joint_regex",
            "type": str,
            "default": "",
            "help": "Only test joints whose names match this regex.",
        },
        {
            "name": "--free_base",
            "action": "store_true",
            "default": False,
            "help": "Use the task's free-base setup instead of fixed-base isolation.",
        },
        {
            "name": "--no_detail_csv",
            "action": "store_true",
            "default": False,
            "help": "Do not save the full time series CSV.",
        },
        {
            "name": "--kp_scale",
            "type": float,
            "default": 1.0,
            "help": "Scale factor for proportional gain (Kp). Increase for overshoot, decrease for slower response.",
        },
        {
            "name": "--kd_scale",
            "type": float,
            "default": 1.0,
            "help": "Scale factor for derivative gain (Kd). Decrease for overshoot, increase for more damping.",
        },
    ]

    args = gymutil.parse_arguments(
        description="Automated joint step-response evaluation",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    args.headless = not args.show_viewer
    args.num_envs = 1
    if args.step_direction not in {"auto", "positive", "negative", "both"}:
        raise ValueError("--step_direction must be one of: auto, positive, negative, both")
    if args.step_fraction <= 0.0:
        raise ValueError("--step_fraction must be > 0")
    if args.position_margin < 0.0:
        raise ValueError("--position_margin must be >= 0")
    return args


def resolve_output_dir(task_name, output_dir):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "response", task_name, timestamp)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_task_cfg(task_name):
    if task_name not in task_registry.env_cfgs:
        known = ", ".join(sorted(task_registry.env_cfgs.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Known tasks: {known}")
    env_cfg, _ = task_registry.get_cfgs(task_name)
    return copy.deepcopy(env_cfg)


def resolve_asset_path(asset_file_template):
    return asset_file_template.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)


def parse_urdf_joint_limits(urdf_path):
    if not urdf_path.endswith(".urdf"):
        return {}
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    info = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        joint_type = joint.attrib.get("type", "")
        if joint_type == "fixed" or not name:
            continue
        limit = joint.find("limit")
        lower = None
        upper = None
        effort = None
        velocity = None
        if limit is not None:
            if "lower" in limit.attrib:
                lower = float(limit.attrib["lower"])
            if "upper" in limit.attrib:
                upper = float(limit.attrib["upper"])
            if "effort" in limit.attrib:
                effort = float(limit.attrib["effort"])
            if "velocity" in limit.attrib:
                velocity = float(limit.attrib["velocity"])
        info[name] = {
            "type": joint_type,
            "lower": lower,
            "upper": upper,
            "effort": effort,
            "velocity": velocity,
        }
    return info


def configure_env_for_response(env_cfg, args):
    env_cfg.env.num_envs = 1
    env_cfg.env.debug = False
    if hasattr(env_cfg.env, "plot_debug"):
        env_cfg.env.plot_debug = False
    if hasattr(env_cfg.env, "get_commands_from_keyboard"):
        env_cfg.env.get_commands_from_keyboard = False
    if hasattr(env_cfg.env, "use_ref_actions"):
        env_cfg.env.use_ref_actions = False

    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.measure_heights = False

    env_cfg.noise.add_noise = False

    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg.domain_rand, "action_delay"):
        env_cfg.domain_rand.action_delay = 0.0
    if hasattr(env_cfg.domain_rand, "action_noise"):
        env_cfg.domain_rand.action_noise = 0.0
    if hasattr(env_cfg.domain_rand, "joint_angle_noise"):
        env_cfg.domain_rand.joint_angle_noise = 0.0

    env_cfg.asset.fix_base_link = not args.free_base


def make_env(task_name, args, env_cfg):
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    return env


def reset_env_to_nominal_state(env):
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    env_ids_int32 = env_ids.to(dtype=torch.int32)

    if hasattr(env, "action_buffer"):
        delattr(env, "action_buffer")

    env.dof_pos[0] = env.default_dof_pos[0]
    env.dof_vel[0] = 0.0
    env.actions[0] = 0.0
    env.last_actions[0] = 0.0
    env.last_last_actions[0] = 0.0
    env.last_dof_vel[0] = 0.0
    env.torques[0] = 0.0

    env.root_states[0] = env.base_init_state
    env.root_states[0, :3] += env.env_origins[0]
    if env.cfg.asset.fix_base_link:
        env.root_states[0, 2] += 1.8
        env.root_states[0, 7:13] = 0.0

    env.episode_length_buf[0] = 0
    env.reset_buf[0] = 0
    env.time_out_buf[0] = False
    env.commands[0] = 0.0

    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.compute_observations()


def build_joint_metadata(env, urdf_limits):
    metadata = []
    default_pos = env.default_dof_pos[0].detach().cpu().numpy()
    effective_limits = env.dof_pos_limits.detach().cpu().numpy()
    torque_limits = env.torque_limits.detach().cpu().numpy()
    p_gains = env.p_gains[0].detach().cpu().numpy()
    d_gains = env.d_gains[0].detach().cpu().numpy()
    vel_limits = env.dof_vel_limits.detach().cpu().numpy()

    for idx, joint_name in enumerate(env.dof_names):
        urdf_info = urdf_limits.get(joint_name, {})
        metadata.append({
            "joint_index": idx,
            "joint_name": joint_name,
            "default_pos": float(default_pos[idx]),
            "effective_lower": float(effective_limits[idx, 0]),
            "effective_upper": float(effective_limits[idx, 1]),
            "effective_torque_limit": float(torque_limits[idx]),
            "effective_vel_limit": float(vel_limits[idx]),
            "kp": float(p_gains[idx]),
            "kd": float(d_gains[idx]),
            "urdf_lower": urdf_info.get("lower"),
            "urdf_upper": urdf_info.get("upper"),
            "urdf_effort": urdf_info.get("effort"),
            "urdf_velocity": urdf_info.get("velocity"),
            "joint_type": urdf_info.get("type"),
        })
    return metadata


def select_joints(joint_metadata, joint_regex):
    if not joint_regex:
        return joint_metadata
    pattern = re.compile(joint_regex)
    return [item for item in joint_metadata if pattern.search(item["joint_name"])]


def choose_pretty_step_amplitude(raw_amplitude, max_room):
    raw_amplitude = float(raw_amplitude)
    max_room = float(max_room)
    if max_room <= 0.0:
        return 0.0

    # Prefer clean 0.x values first, then fall back to finer steps for small-range joints.
    pretty_candidates = [
        1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
        0.05, 0.02, 0.01,
    ]
    limit = min(raw_amplitude, max_room)
    for candidate in pretty_candidates:
        if candidate <= limit + 1e-9:
            return candidate

    # If the available range is tiny, keep a conservative rounded value.
    if limit >= 0.005:
        return math.floor(limit / 0.005) * 0.005
    return 0.0


def choose_joint_targets(joint_info, action_scale, clip_actions, step_fraction, position_margin, direction_mode):
    lower = joint_info["effective_lower"]
    upper = joint_info["effective_upper"]
    default = float(np.clip(joint_info["default_pos"], lower, upper))
    lower_safe = lower + position_margin
    upper_safe = upper - position_margin

    if lower_safe >= upper_safe:
        return []

    max_target_from_action = default + action_scale * clip_actions
    min_target_from_action = default - action_scale * clip_actions
    lower_safe = max(lower_safe, min_target_from_action)
    upper_safe = min(upper_safe, max_target_from_action)

    pos_room = upper_safe - default
    neg_room = default - lower_safe

    candidates = []
    if direction_mode in ("auto", "positive", "both") and pos_room > 1e-5:
        candidates.append(("positive", pos_room))
    if direction_mode in ("auto", "negative", "both") and neg_room > 1e-5:
        candidates.append(("negative", neg_room))

    if direction_mode == "auto" and candidates:
        candidates = [max(candidates, key=lambda item: item[1])]

    targets = []
    for direction, room in candidates:
        raw_amplitude = room * step_fraction
        amplitude = choose_pretty_step_amplitude(raw_amplitude, room)
        if amplitude < 0.01:
            continue
        if direction == "positive":
            target = default + amplitude
        else:
            target = default - amplitude
        target = float(np.clip(target, lower_safe, upper_safe))
        action = (target - default) / action_scale if action_scale != 0 else 0.0
        action = float(np.clip(action, -clip_actions, clip_actions))
        applied_target = default + action * action_scale
        targets.append({
            "direction": direction,
            "target_pos": float(applied_target),
            "action_value": action,
            "step_amplitude_cmd": float(applied_target - default),
        })
    return targets


def compute_step_metrics(time_s, target_series, actual_series, torque_series, settle_tol_ratio, settle_tol_abs):
    target_series = np.asarray(target_series, dtype=np.float64)
    actual_series = np.asarray(actual_series, dtype=np.float64)
    torque_series = np.asarray(torque_series, dtype=np.float64)

    if len(time_s) == 0:
        return {
            "step_index": None,
            "step_time_s": None,
            "initial_pos_rad": None,
            "target_final_rad": None,
            "final_pos_rad": None,
            "step_amplitude_rad": None,
            "rise_time_s": None,
            "peak_time_s": None,
            "settling_time_s": None,
            "max_overshoot_rad": None,
            "max_overshoot_pct": None,
            "steady_state_error_rad": None,
            "peak_abs_torque_nm": None,
            "settle_band_rad": None,
            "peak_pos_rad": None,
            "peak_index": None,
            "rise_10_time_s": None,
            "rise_90_time_s": None,
            "settling_index": None,
        }

    initial_target = target_series[0]
    diff = np.abs(target_series - initial_target)
    changed = np.where(diff > 1e-7)[0]
    step_index = int(changed[0]) if len(changed) > 0 else 0
    step_time = float(time_s[step_index])
    initial_pos = float(np.mean(actual_series[: max(1, step_index + 1)]))
    target_final = float(np.mean(target_series[-10:]))
    final_pos = float(np.mean(actual_series[-10:]))
    step_amplitude = float(target_final - initial_pos)
    sign = 1.0 if step_amplitude >= 0.0 else -1.0
    response = sign * (actual_series[step_index:] - initial_pos)
    target_mag = abs(step_amplitude)

    rise_time = None
    peak_time = None
    settling_time = None
    overshoot_rad = 0.0
    overshoot_pct = 0.0
    peak_abs_torque = float(np.max(np.abs(torque_series[step_index:]))) if len(torque_series[step_index:]) else 0.0
    steady_state_error = float(target_final - final_pos)
    settle_band = max(settle_tol_abs, target_mag * settle_tol_ratio)
    peak_pos = float(np.max(actual_series[step_index:])) if len(actual_series[step_index:]) else float(actual_series[-1])
    peak_index = None
    rise_10_time = None
    rise_90_time = None
    settling_index = None

    if target_mag > 1e-8:
        t_post = time_s[step_index:] - step_time

        above_10 = np.where(response >= 0.1 * target_mag)[0]
        above_90 = np.where(response >= 0.9 * target_mag)[0]
        if len(above_10) > 0:
            rise_10_time = float(t_post[above_10[0]])
        if len(above_90) > 0:
            rise_90_time = float(t_post[above_90[0]])
        if len(above_10) > 0 and len(above_90) > 0:
            rise_time = float(t_post[above_90[0]] - t_post[above_10[0]])

        peak_idx = int(np.argmax(response))
        peak_index = step_index + peak_idx
        peak_time = float(t_post[peak_idx])
        peak_response = float(response[peak_idx])
        peak_pos = float(actual_series[peak_index])
        overshoot_rad = max(0.0, peak_response - target_mag)
        overshoot_pct = 100.0 * overshoot_rad / target_mag

        abs_error = np.abs(actual_series[step_index:] - target_final)
        within_band = abs_error <= settle_band
        for idx in range(len(within_band)):
            if np.all(within_band[idx:]):
                settling_index = step_index + idx
                settling_time = float(t_post[idx])
                break

    return {
        "step_index": step_index,
        "step_time_s": step_time,
        "initial_pos_rad": initial_pos,
        "target_final_rad": target_final,
        "final_pos_rad": final_pos,
        "step_amplitude_rad": step_amplitude,
        "rise_time_s": rise_time,
        "peak_time_s": peak_time,
        "settling_time_s": settling_time,
        "max_overshoot_rad": overshoot_rad,
        "max_overshoot_pct": overshoot_pct,
        "steady_state_error_rad": steady_state_error,
        "peak_abs_torque_nm": peak_abs_torque,
        "settle_band_rad": settle_band,
        "peak_pos_rad": peak_pos,
        "peak_index": peak_index,
        "rise_10_time_s": rise_10_time,
        "rise_90_time_s": rise_90_time,
        "settling_index": settling_index,
    }


def run_joint_step_test(env, joint_info, target_info, args):
    reset_env_to_nominal_state(env)

    # Apply Kp/Kd scaling to enable overshoot testing
    # Reduce Kd relative to Kp to achieve underdamped response (zeta < 1)
    if hasattr(args, 'kp_scale') and hasattr(args, 'kd_scale'):
        env.p_gains[0, :] *= args.kp_scale
        env.d_gains[0, :] *= args.kd_scale

    zero_actions = torch.zeros((1, env.num_actions), device=env.device, dtype=torch.float)
    step_actions = zero_actions.clone()
    step_actions[0, joint_info["joint_index"]] = float(target_info["action_value"])

    total_steps = args.pre_steps + args.response_steps
    joint_idx = joint_info["joint_index"]

    time_s = []
    target_series = []
    actual_series = []
    torque_series = []
    env.commands.zero_()

    terminated = False
    for step in range(total_steps):
        env.commands.zero_()
        actions = zero_actions if step < args.pre_steps else step_actions
        _, _, _, dones, _ = env.step(actions)

        applied_target = (
            env.actions[0] * env.cfg.control.action_scale + env.default_dof_pos[0]
        ).detach().cpu().numpy()
        actual_pos = env.dof_pos[0].detach().cpu().numpy()
        torques = env.torques[0].detach().cpu().numpy()

        time_s.append(step * env.dt)
        target_series.append(float(applied_target[joint_idx]))
        actual_series.append(float(actual_pos[joint_idx]))
        torque_series.append(float(torques[joint_idx]))

        if bool(dones[0].item()):
            terminated = True
            break

    time_s = np.asarray(time_s, dtype=np.float64)
    target_series = np.asarray(target_series, dtype=np.float64)
    actual_series = np.asarray(actual_series, dtype=np.float64)
    torque_series = np.asarray(torque_series, dtype=np.float64)

    metrics = compute_step_metrics(
        time_s,
        target_series,
        actual_series,
        torque_series,
        settle_tol_ratio=args.settle_tol_ratio,
        settle_tol_abs=args.settle_tol_abs,
    )
    return {
        "joint_name": joint_info["joint_name"],
        "joint_index": joint_idx,
        "direction": target_info["direction"],
        "command_action": float(target_info["action_value"]),
        "command_target_rad": float(target_info["target_pos"]),
        "time_s": time_s,
        "target_rad": target_series,
        "actual_rad": actual_series,
        "torque_nm": torque_series,
        "terminated_early": terminated,
        "joint_info": joint_info,
        "metrics": metrics,
    }


def format_metric(value, fmt="{:.3f}", empty="n/a"):
    if value is None:
        return empty
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return empty
    return fmt.format(value)


def plot_grouped_results(results, output_dir, max_per_figure=5):
    if not results:
        return []

    saved_paths = []
    for start in range(0, len(results), max_per_figure):
        chunk = results[start:start + max_per_figure]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(14, 2.9 * len(chunk)), sharex=False)
        axes = np.atleast_1d(axes)

        for ax, result in zip(axes, chunk):
            metrics = result["metrics"]
            time_rel = result["time_s"] - (metrics["step_time_s"] or 0.0)
            ax.plot(time_rel, result["target_rad"], "--", linewidth=1.6, label="target")
            ax.plot(time_rel, result["actual_rad"], "-", linewidth=1.6, label="actual")
            ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)

            target_final = metrics["target_final_rad"]
            settle_band = metrics["settle_band_rad"]
            if target_final is not None and settle_band is not None:
                ax.axhspan(
                    target_final - settle_band,
                    target_final + settle_band,
                    color="tab:green",
                    alpha=0.10,
                    label="settling band",
                )

            if metrics["rise_90_time_s"] is not None:
                ax.axvline(metrics["rise_90_time_s"], color="tab:orange", linestyle="--", linewidth=1.0)
            if metrics["peak_time_s"] is not None:
                ax.axvline(metrics["peak_time_s"], color="tab:red", linestyle="--", linewidth=1.0)
            if metrics["settling_time_s"] is not None:
                ax.axvline(metrics["settling_time_s"], color="tab:green", linestyle="--", linewidth=1.0)

            peak_index = metrics.get("peak_index")
            if peak_index is not None and 0 <= peak_index < len(result["actual_rad"]):
                peak_time_rel = time_rel[peak_index]
                peak_pos = result["actual_rad"][peak_index]
                ax.plot(peak_time_rel, peak_pos, "o", color="tab:red", markersize=4)
                ax.annotate(
                    "peak",
                    xy=(peak_time_rel, peak_pos),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="tab:red",
                )

            settling_index = metrics.get("settling_index")
            if settling_index is not None and 0 <= settling_index < len(result["actual_rad"]):
                settling_time_rel = time_rel[settling_index]
                settling_pos = result["actual_rad"][settling_index]
                ax.plot(settling_time_rel, settling_pos, "o", color="tab:green", markersize=4)
                ax.annotate(
                    "settled",
                    xy=(settling_time_rel, settling_pos),
                    xytext=(6, -12),
                    textcoords="offset points",
                    fontsize=8,
                    color="tab:green",
                )

            title = f"{result['joint_name']} [{result['direction']}]"
            if result["terminated_early"]:
                title += "  (terminated early)"
            ax.set_title(title)
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

            textbox = "\n".join([
                f"Ts: {format_metric(metrics['settling_time_s'], '{:.3f}s')}",
                f"Tr: {format_metric(metrics['rise_time_s'], '{:.3f}s')}",
                f"Mp: {format_metric(metrics['max_overshoot_pct'], '{:.1f}%')}",
                f"Ess: {format_metric(metrics['steady_state_error_rad'], '{:.3f}rad')}",
                f"|tau|max: {format_metric(metrics['peak_abs_torque_nm'], '{:.2f}Nm')}",
            ])
            ax.text(
                0.01,
                0.97,
                textbox,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )

            if metrics["peak_time_s"] is not None and metrics["peak_pos_rad"] is not None:
                peak_x = metrics["peak_time_s"]
                peak_y = metrics["peak_pos_rad"]
                text_y = peak_y + 0.08 * max(0.1, np.ptp(result["actual_rad"]))
                ax.annotate(
                    f"Tp={metrics['peak_time_s']:.3f}s\nMp={metrics['max_overshoot_pct']:.1f}%",
                    xy=(peak_x, peak_y),
                    xytext=(peak_x, text_y),
                    arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.0},
                    fontsize=8,
                    color="tab:red",
                    ha="left",
                )

            if metrics["settling_time_s"] is not None and target_final is not None:
                ts = metrics["settling_time_s"]
                ax.annotate(
                    f"Ts={ts:.3f}s",
                    xy=(ts, target_final),
                    xytext=(ts, target_final - 0.12 * max(0.1, np.ptp(result["actual_rad"]))),
                    arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 1.0},
                    fontsize=8,
                    color="tab:green",
                    ha="left",
                )

            if metrics["rise_90_time_s"] is not None:
                tr90 = metrics["rise_90_time_s"]
                y_min, y_max = ax.get_ylim()
                y_text = y_min + 0.15 * (y_max - y_min)
                ax.annotate(
                    f"t90={tr90:.3f}s",
                    xy=(tr90, y_text),
                    xytext=(tr90 + 0.02 * max(1.0, time_rel[-1] - time_rel[0]), y_text),
                    arrowprops={"arrowstyle": "->", "color": "tab:orange", "lw": 1.0},
                    fontsize=8,
                    color="tab:orange",
                    ha="left",
                )

        axes[-1].set_xlabel("time relative to applied step [s]")
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f"response_group_{start // max_per_figure + 1:02d}.png")
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(fig_path)
    return saved_paths


def save_summary_csv(results, output_dir):
    path = os.path.join(output_dir, "response_summary.csv")
    fieldnames = [
        "joint_name",
        "joint_index",
        "direction",
        "command_action",
        "command_target_rad",
        "initial_pos_rad",
        "target_final_rad",
        "final_pos_rad",
        "step_amplitude_rad",
        "rise_time_s",
        "peak_time_s",
        "settling_time_s",
        "max_overshoot_rad",
        "max_overshoot_pct",
        "steady_state_error_rad",
        "peak_abs_torque_nm",
        "torque_limit_nm",
        "kp",
        "kd",
        "urdf_lower_rad",
        "urdf_upper_rad",
        "effective_lower_rad",
        "effective_upper_rad",
        "urdf_effort_nm",
        "terminated_early",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            joint_info = result["joint_info"]
            writer.writerow({
                "joint_name": result["joint_name"],
                "joint_index": result["joint_index"],
                "direction": result["direction"],
                "command_action": result["command_action"],
                "command_target_rad": result["command_target_rad"],
                "initial_pos_rad": metrics["initial_pos_rad"],
                "target_final_rad": metrics["target_final_rad"],
                "final_pos_rad": metrics["final_pos_rad"],
                "step_amplitude_rad": metrics["step_amplitude_rad"],
                "rise_time_s": metrics["rise_time_s"],
                "peak_time_s": metrics["peak_time_s"],
                "settling_time_s": metrics["settling_time_s"],
                "max_overshoot_rad": metrics["max_overshoot_rad"],
                "max_overshoot_pct": metrics["max_overshoot_pct"],
                "steady_state_error_rad": metrics["steady_state_error_rad"],
                "peak_abs_torque_nm": metrics["peak_abs_torque_nm"],
                "torque_limit_nm": joint_info["effective_torque_limit"],
                "kp": joint_info["kp"],
                "kd": joint_info["kd"],
                "urdf_lower_rad": joint_info["urdf_lower"],
                "urdf_upper_rad": joint_info["urdf_upper"],
                "effective_lower_rad": joint_info["effective_lower"],
                "effective_upper_rad": joint_info["effective_upper"],
                "urdf_effort_nm": joint_info["urdf_effort"],
                "terminated_early": result["terminated_early"],
            })
    return path


def save_detail_csv(results, output_dir):
    path = os.path.join(output_dir, "response_detail.csv")
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "joint_name",
            "joint_index",
            "direction",
            "time_s",
            "target_rad",
            "actual_rad",
            "torque_nm",
        ])
        for result in results:
            for time_s, target, actual, torque in zip(
                result["time_s"],
                result["target_rad"],
                result["actual_rad"],
                result["torque_nm"],
            ):
                writer.writerow([
                    result["joint_name"],
                    result["joint_index"],
                    result["direction"],
                    time_s,
                    target,
                    actual,
                    torque,
                ])
    return path


def save_report_json(results, output_dir, config_info):
    path = os.path.join(output_dir, "response_report.json")
    payload = {
        "config": config_info,
        "results": [],
    }
    for result in results:
        item = {
            "joint_name": result["joint_name"],
            "joint_index": result["joint_index"],
            "direction": result["direction"],
            "command_action": result["command_action"],
            "command_target_rad": result["command_target_rad"],
            "terminated_early": result["terminated_early"],
            "joint_info": result["joint_info"],
            "metrics": result["metrics"],
        }
        payload["results"].append(item)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return path


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.task, args.output_dir)
    env_cfg = resolve_task_cfg(args.task)
    configure_env_for_response(env_cfg, args)

    asset_path = resolve_asset_path(env_cfg.asset.file)
    urdf_limits = parse_urdf_joint_limits(asset_path)

    env = None
    try:
        env = make_env(args.task, args, env_cfg)
        joint_metadata = build_joint_metadata(env, urdf_limits)
        joint_metadata = select_joints(joint_metadata, args.joint_regex)
        if not joint_metadata:
            raise ValueError("No joints selected for response evaluation.")

        results = []
        for joint_info in joint_metadata:
            targets = choose_joint_targets(
                joint_info=joint_info,
                action_scale=float(env.cfg.control.action_scale),
                clip_actions=float(env.cfg.normalization.clip_actions),
                step_fraction=float(args.step_fraction),
                position_margin=float(args.position_margin),
                direction_mode=args.step_direction,
            )
            if not targets:
                print(f"[skip] {joint_info['joint_name']}: not enough available joint range for the requested step.")
                continue

            for target_info in targets:
                print(
                    f"[test] joint={joint_info['joint_name']} direction={target_info['direction']} "
                    f"target={target_info['target_pos']:.4f} rad action={target_info['action_value']:.4f}"
                )
                result = run_joint_step_test(env, joint_info, target_info, args)
                results.append(result)

        if not results:
            raise RuntimeError("No valid joint response tests were produced.")

        summary_csv = save_summary_csv(results, output_dir)
        detail_csv = save_detail_csv(results, output_dir) if not args.no_detail_csv else None
        figure_paths = plot_grouped_results(results, output_dir, max_per_figure=5)
        report_json = save_report_json(
            results,
            output_dir,
            config_info={
                "task": args.task,
                "asset_path": asset_path,
                "fixed_base": bool(env.cfg.asset.fix_base_link),
                "dt": float(env.dt),
                "sim_dt": float(env.cfg.sim.dt),
                "decimation": int(env.cfg.control.decimation),
                "control_frequency_hz": 1.0 / float(env.dt),
                "action_scale": float(env.cfg.control.action_scale),
                "torque_limit_scale": float(env.cfg.safety.torque_limit),
                "pos_limit_scale": float(env.cfg.safety.pos_limit),
                "step_fraction": float(args.step_fraction),
                "position_margin": float(args.position_margin),
                "step_direction": args.step_direction,
                "pre_steps": int(args.pre_steps),
                "response_steps": int(args.response_steps),
                "kp_scale": float(args.kp_scale),
                "kd_scale": float(args.kd_scale),
            },
        )

        print(f"\nResponse test completed. Output directory: {output_dir}")
        print(f"Summary CSV : {summary_csv}")
        if detail_csv:
            print(f"Detail CSV  : {detail_csv}")
        print(f"Report JSON : {report_json}")
        print("Figures     :")
        for fig_path in figure_paths:
            print(f"  - {fig_path}")

    finally:
        if env is not None and getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        if env is not None and getattr(env, "sim", None) is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
