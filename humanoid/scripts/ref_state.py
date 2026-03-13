# SPDX-License-Identifier: BSD-3-Clause
#
# Visualize reference gait trajectories from task config.
# Supports:
# 1) simulation playback in Isaac Gym (URDF motion visualization)
# 2) optional trajectory plotting

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from humanoid import LEGGED_GYM_ROOT_DIR


def _get_env_cfg_by_task(task_name: str):
    # Import isaacgym first to avoid "torch imported before isaacgym" issue.
    import isaacgym  # noqa: F401
    import humanoid.envs  # noqa: F401  # register tasks
    from humanoid.utils.task_registry import task_registry

    if task_name not in task_registry.env_cfgs:
        known = ", ".join(sorted(task_registry.env_cfgs.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Known tasks: {known}")
    env_cfg, _ = task_registry.get_cfgs(task_name)
    return env_cfg


def _resolve_urdf_path(asset_file_template: str) -> str:
    return asset_file_template.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)


def _parse_urdf_dof_names(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    # IsaacGym DOF order follows non-fixed joints in URDF declaration order.
    return [
        j.attrib["name"]
        for j in root.findall("joint")
        if j.attrib.get("type", "") != "fixed"
    ]


def _select_ref_joint_names(default_joint_angles):
    # Prefer Pikachu naming.
    pikachu = (
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_joint",
    )
    if all(name in default_joint_angles for name in pikachu):
        return pikachu

    # Fallback for humanoid/XBot style naming.
    humanoid = (
        "left_leg_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_leg_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
    )
    if all(name in default_joint_angles for name in humanoid):
        return humanoid

    raise ValueError(
        "Cannot infer reference gait joints from default_joint_angles. "
        "Expected either Pikachu or humanoid naming."
    )


def _build_ref_traj(cfg, dof_names, seconds, dt, cycle_time_override, scale_override):
    joint_to_idx = {name: i for i, name in enumerate(dof_names)}
    default_joint_angles = cfg.init_state.default_joint_angles
    ref_names = _select_ref_joint_names(default_joint_angles)

    for name in ref_names:
        if name not in joint_to_idx:
            raise KeyError(f"Joint '{name}' not found in URDF dof names.")

    lh, lk, la, rh, rk, ra = (joint_to_idx[n] for n in ref_names)

    cycle_time = (
        float(cycle_time_override)
        if cycle_time_override is not None
        else float(cfg.rewards.cycle_time)
    )
    scale_1 = (
        float(scale_override)
        if scale_override is not None
        else float(cfg.rewards.target_joint_pos_scale)
    )
    scale_2 = 2.0 * scale_1
    left_sign = float(getattr(cfg.rewards, "left_ref_sign", 1.0))
    right_sign = float(getattr(cfg.rewards, "right_ref_sign", 1.0))

    num_steps = int(seconds / dt)
    t = np.arange(num_steps, dtype=np.float32) * dt
    phase = t / cycle_time
    sin_pos = np.sin(2.0 * np.pi * phase)
    sin_pos_l = sin_pos.copy()
    sin_pos_r = sin_pos.copy()

    ref_dof_pos = np.zeros((num_steps, len(dof_names)), dtype=np.float32)

    sin_pos_l[sin_pos_l > 0.0] = 0.0
    ref_dof_pos[:, lh] = left_sign * sin_pos_l * scale_1
    ref_dof_pos[:, lk] = left_sign * sin_pos_l * scale_2
    ref_dof_pos[:, la] = left_sign * sin_pos_l * scale_1

    sin_pos_r[sin_pos_r < 0.0] = 0.0
    ref_dof_pos[:, rh] = right_sign * sin_pos_r * scale_1
    ref_dof_pos[:, rk] = right_sign * sin_pos_r * scale_2
    ref_dof_pos[:, ra] = right_sign * sin_pos_r * scale_1

    ds_mask = np.abs(sin_pos) < 0.1
    ref_dof_pos[ds_mask, :] = 0.0

    ref_action = 2.0 * ref_dof_pos
    action_scale = float(cfg.control.action_scale)
    ref_pd_offset = action_scale * ref_action

    return {
        "t": t,
        "sin_pos": sin_pos,
        "ds_mask": ds_mask.astype(np.float32),
        "ref_dof_pos": ref_dof_pos,
        "ref_action": ref_action,
        "ref_pd_offset": ref_pd_offset,
        "dof_names": list(dof_names),
        "ref_joint_indices": (lh, lk, la, rh, rk, ra),
        "ref_joint_names": ref_names,
        "cycle_time": cycle_time,
        "scale_1": scale_1,
        "action_scale": action_scale,
        "left_ref_sign": left_sign,
        "right_ref_sign": right_sign,
    }


def _plot_ref(data, actual_data=None):
    import matplotlib.pyplot as plt

    t = data["t"]
    ref_dof_pos = data["ref_dof_pos"]
    sin_pos = data["sin_pos"]
    ref_names = data["ref_joint_names"]
    lh, lk, la, rh, rk, ra = data["ref_joint_indices"]
    actual_t = None
    actual_pos = None
    if actual_data is not None:
        actual_t = actual_data.get("t", None)
        actual_pos = actual_data.get("dof_pos", None)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
    ax_phase = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharex=ax_left)
    cmap = plt.get_cmap("tab10")

    ax_phase.plot(t, sin_pos, label="sin(2πphase)", color="black", linewidth=1.2)
    ax_phase.axhline(0.0, color="gray", linewidth=0.8)
    ax_phase.set_ylabel("Phase")
    ax_phase.set_title("Reference Gait Phase Signal")
    ax_phase.legend()
    ax_phase.grid(alpha=0.3)

    # Left leg: same color per joint, dashed=ref, solid=actual
    left_specs = [
        (lh, ref_names[0], cmap(0)),
        (lk, ref_names[1], cmap(1)),
        (la, ref_names[2], cmap(2)),
    ]
    for idx, name, color in left_specs:
        ax_left.plot(t, ref_dof_pos[:, idx], linestyle="--", color=color, label=f"{name} ref")
        if actual_t is not None and actual_pos is not None and len(actual_t) > 0:
            ax_left.plot(actual_t, actual_pos[:, idx], linestyle="-", color=color, label=f"{name} actual")

    # Right leg: same color per joint, dashed=ref, solid=actual
    right_specs = [
        (rh, ref_names[3], cmap(0)),
        (rk, ref_names[4], cmap(1)),
        (ra, ref_names[5], cmap(2)),
    ]
    for idx, name, color in right_specs:
        ax_right.plot(t, ref_dof_pos[:, idx], linestyle="--", color=color, label=f"{name} ref")
        if actual_t is not None and actual_pos is not None and len(actual_t) > 0:
            ax_right.plot(actual_t, actual_pos[:, idx], linestyle="-", color=color, label=f"{name} actual")

    ax_left.set_title("Left Leg: Ref (dashed) vs Actual (solid)")
    ax_left.set_xlabel("time [s]")
    ax_left.set_ylabel("joint offset [rad]")
    ax_left.grid(alpha=0.3)
    ax_left.legend(ncol=2, fontsize=8)

    ax_right.set_title("Right Leg: Ref (dashed) vs Actual (solid)")
    ax_right.set_xlabel("time [s]")
    ax_right.set_ylabel("joint offset [rad]")
    ax_right.grid(alpha=0.3)
    ax_right.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()


def _joint_type(joint_name: str) -> str:
    """Strip side prefix (left_/right_) and '_joint' suffix to get the joint type key.

    E.g. 'left_hip_pitch_joint' -> 'hip_pitch',
         'left_elbow_ankle_joint' -> 'elbow_ankle'.
    Using exact type matching avoids substring false-positives such as
    'ankle' accidentally matching 'elbow_ankle_joint'.
    """
    name = joint_name
    if name.endswith("_joint"):
        name = name[:-6]
    for prefix in ("left_", "right_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def _get_pd_gain_for_joint(joint_name, stiffness_cfg, damping_cfg):
    """Return (k, d) PD gains for *joint_name* by exact type-key matching.

    Falls back to small non-zero gains for joints not listed in the config
    (e.g. arm joints on the quad URDF) so they stay near their default
    position during fixed-base visualization instead of drooping under gravity.
    """
    jtype = _joint_type(joint_name)
    # Small fallback so uncontrolled joints (e.g. quad arm joints) hold pose.
    k = 5.0
    d = 0.5
    for key, value in stiffness_cfg.items():
        if key == jtype:
            k = float(value)
            break
    for key, value in damping_cfg.items():
        if key == jtype:
            d = float(value)
            break
    return k, d


def _simulate_ref_motion(
    cfg,
    dof_names,
    data,
    headless=False,
    fix_base=True,
    base_lift=0.5,
    loop=True,
    collect_actual=False,
):
    from isaacgym import gymapi

    def _resolve_contact_collection(value):
        iv = int(value)
        # Newer bindings: enum constructor from int.
        try:
            return gymapi.ContactCollection(iv)
        except Exception:
            pass
        # Fallback for bindings exposing enum constants as attributes.
        name_map = {
            0: ("CC_NEVER",),
            1: ("CC_LAST_SUBSTEP", "CC_LAST_SUBSTEP_ONLY"),
            2: ("CC_ALL_SUBSTEPS",),
        }
        for attr in name_map.get(iv, ()):
            if hasattr(gymapi, attr):
                return getattr(gymapi, attr)
            if hasattr(gymapi.ContactCollection, attr):
                return getattr(gymapi.ContactCollection, attr)
        raise ValueError(f"Unsupported contact_collection value: {iv}")

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = float(cfg.sim.dt)
    sim_params.substeps = int(cfg.sim.substeps)
    sim_params.up_axis = gymapi.UP_AXIS_Z if int(cfg.sim.up_axis) == 1 else gymapi.UP_AXIS_Y
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.num_threads = int(cfg.sim.physx.num_threads)
    sim_params.physx.solver_type = int(cfg.sim.physx.solver_type)
    sim_params.physx.num_position_iterations = int(cfg.sim.physx.num_position_iterations)
    sim_params.physx.num_velocity_iterations = int(cfg.sim.physx.num_velocity_iterations)
    sim_params.physx.contact_offset = float(cfg.sim.physx.contact_offset)
    sim_params.physx.rest_offset = float(cfg.sim.physx.rest_offset)
    sim_params.physx.bounce_threshold_velocity = float(cfg.sim.physx.bounce_threshold_velocity)
    sim_params.physx.max_depenetration_velocity = float(cfg.sim.physx.max_depenetration_velocity)
    sim_params.physx.default_buffer_size_multiplier = float(cfg.sim.physx.default_buffer_size_multiplier)
    sim_params.physx.contact_collection = _resolve_contact_collection(cfg.sim.physx.contact_collection)

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation.")

    viewer = None
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            gym.destroy_sim(sim)
            raise RuntimeError("Failed to create viewer.")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = float(cfg.terrain.static_friction)
    plane_params.dynamic_friction = float(cfg.terrain.dynamic_friction)
    plane_params.restitution = float(cfg.terrain.restitution)
    gym.add_ground(sim, plane_params)

    asset_path = _resolve_urdf_path(cfg.asset.file)
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.collapse_fixed_joints = bool(cfg.asset.collapse_fixed_joints)
    asset_options.replace_cylinder_with_capsule = bool(cfg.asset.replace_cylinder_with_capsule)
    asset_options.flip_visual_attachments = bool(cfg.asset.flip_visual_attachments)
    # For reference-trajectory inspection, default to fixed-base visualization.
    asset_options.fix_base_link = bool(fix_base)
    asset_options.density = float(cfg.asset.density)
    asset_options.angular_damping = float(cfg.asset.angular_damping)
    asset_options.linear_damping = float(cfg.asset.linear_damping)
    asset_options.max_angular_velocity = float(cfg.asset.max_angular_velocity)
    asset_options.max_linear_velocity = float(cfg.asset.max_linear_velocity)
    asset_options.armature = float(cfg.asset.armature)
    asset_options.thickness = float(cfg.asset.thickness)
    asset_options.disable_gravity = bool(cfg.asset.disable_gravity)

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        raise RuntimeError(f"Failed to load asset: {asset_path}")

    env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
    env_upper = gymapi.Vec3(1.0, 1.0, 1.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    start_pose = gymapi.Transform()
    base_z = float(cfg.init_state.pos[2]) + float(base_lift)
    start_pose.p = gymapi.Vec3(float(cfg.init_state.pos[0]), float(cfg.init_state.pos[1]), base_z)
    actor = gym.create_actor(env, asset, start_pose, cfg.asset.name, 0, int(cfg.asset.self_collisions), 0)

    asset_dof_names = gym.get_asset_dof_names(asset)
    asset_dof_names = list(asset_dof_names)
    src_dof_names = list(data.get("dof_names", dof_names))

    # Ensure reference action columns follow IsaacGym DOF order.
    ref_action = data["ref_action"]
    if asset_dof_names != src_dof_names:
        src_name_to_idx = {n: i for i, n in enumerate(src_dof_names)}
        remapped = np.zeros((ref_action.shape[0], len(asset_dof_names)), dtype=np.float32)
        missing = []
        for dst_idx, joint_name in enumerate(asset_dof_names):
            src_idx = src_name_to_idx.get(joint_name, None)
            if src_idx is None:
                missing.append(joint_name)
                continue
            remapped[:, dst_idx] = ref_action[:, src_idx]
        if missing:
            print(
                "WARNING - These IsaacGym DOFs were not found in source DOF list; "
                f"their ref_action remains zero: {missing}"
            )
        ref_action = remapped
        dof_names = asset_dof_names
    else:
        dof_names = src_dof_names

    dof_props = gym.get_asset_dof_properties(asset)
    stiffness_cfg = cfg.control.stiffness
    damping_cfg = cfg.control.damping
    for i, joint_name in enumerate(dof_names):
        k, d = _get_pd_gain_for_joint(joint_name, stiffness_cfg, damping_cfg)
        dof_props["driveMode"][i] = int(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][i] = k
        dof_props["damping"][i] = d
    gym.set_actor_dof_properties(env, actor, dof_props)

    default_joint_angles = cfg.init_state.default_joint_angles
    default_pos = np.zeros(len(dof_names), dtype=np.float32)
    for i, joint_name in enumerate(dof_names):
        default_pos[i] = float(default_joint_angles.get(joint_name, 0.0))

    dof_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_state["pos"][:] = default_pos
    dof_state["vel"][:] = 0.0
    gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)

    if viewer is not None:
        cam_pos = gymapi.Vec3(2.0, 2.0, 1.2)
        cam_target = gymapi.Vec3(0.0, 0.0, float(cfg.init_state.pos[2]))
        gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

    t = data["t"]
    policy_dt = float(t[1] - t[0]) if len(t) > 1 else float(cfg.control.decimation * cfg.sim.dt)
    sim_dt = float(cfg.sim.dt)
    substeps = max(1, int(round(policy_dt / sim_dt)))
    action_scale = float(data["action_scale"])

    i = 0
    traj_len = ref_action.shape[0]
    actual_t = []
    actual_pos = []
    while True:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        if (not loop) and (i >= traj_len):
            break

        idx = i % traj_len
        target_pos = default_pos + action_scale * ref_action[idx]
        gym.set_actor_dof_position_targets(env, actor, target_pos)

        for _ in range(substeps):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
        if collect_actual:
            states = gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
            actual_pos.append(np.array(states["pos"], dtype=np.float32))
            actual_t.append(i * policy_dt)
        i += 1

    if viewer is not None:
        while not gym.query_viewer_has_closed(viewer):
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    if collect_actual and len(actual_t) > 0:
        return {
            "t": np.array(actual_t, dtype=np.float32),
            "dof_pos": np.stack(actual_pos, axis=0),
        }
    return None


def main():
    parser = argparse.ArgumentParser("Visualize config-based reference gait trajectories.")
    parser.add_argument(
        "--task",
        type=str,
        default="Pikachu_V025",
        help="task name, e.g. Pikachu_V025, Pikachu_V025_Quad, Pikachu_V01, humanoid_ppo",
    )
    parser.add_argument("--seconds", type=float, default=4.0, help="visualization duration")
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="policy dt for reference calculation; default uses cfg.control.decimation * cfg.sim.dt",
    )
    parser.add_argument("--cycle_time", type=float, default=None, help="override cfg.rewards.cycle_time")
    parser.add_argument(
        "--target_joint_pos_scale",
        type=float,
        default=None,
        help="override cfg.rewards.target_joint_pos_scale",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sim",
        choices=["sim", "plot", "both"],
        help="sim: URDF motion in IsaacGym, plot: curves only, both: run both",
    )
    parser.add_argument("--headless", action="store_true", help="run sim mode without viewer")
    parser.add_argument(
        "--base_lift",
        type=float,
        default=0.5,
        help="extra base height (m) applied in sim visualization",
    )
    parser.add_argument(
        "--free_base",
        action="store_true",
        help="disable fixed-base (default is fixed base for ref visualization)",
    )
    parser.add_argument(
        "--no_loop",
        action="store_true",
        help="play trajectory once (default is loop forever until viewer is closed)",
    )
    args = parser.parse_args()

    env_cfg = _get_env_cfg_by_task(args.task)

    urdf_path = _resolve_urdf_path(env_cfg.asset.file)
    dof_names = _parse_urdf_dof_names(urdf_path)
    policy_dt = float(args.dt) if args.dt is not None else float(env_cfg.control.decimation * env_cfg.sim.dt)

    print(f"Task: {args.task}")
    print(f"URDF: {urdf_path}")
    print(f"DOF count (non-fixed joints): {len(dof_names)}")
    print(f"Policy dt: {policy_dt:.6f}s")
    print(f"cycle_time: {args.cycle_time if args.cycle_time is not None else env_cfg.rewards.cycle_time}")
    print(
        "target_joint_pos_scale: "
        f"{args.target_joint_pos_scale if args.target_joint_pos_scale is not None else env_cfg.rewards.target_joint_pos_scale}"
    )
    print(f"sim fixed base: {not args.free_base}")
    print(f"sim base lift: {args.base_lift:.3f} m")
    print(f"sim loop: {not args.no_loop}")

    data = _build_ref_traj(
        cfg=env_cfg,
        dof_names=dof_names,
        seconds=args.seconds,
        dt=policy_dt,
        cycle_time_override=args.cycle_time,
        scale_override=args.target_joint_pos_scale,
    )

    actual_data = None
    if args.mode in ("sim", "both"):
        actual_data = _simulate_ref_motion(
            env_cfg,
            dof_names,
            data,
            headless=args.headless,
            fix_base=(not args.free_base),
            base_lift=args.base_lift,
            loop=(not args.no_loop),
            collect_actual=(args.mode == "both"),
        )
    if args.mode in ("plot", "both"):
        _plot_ref(data, actual_data=actual_data)


if __name__ == "__main__":
    main()
