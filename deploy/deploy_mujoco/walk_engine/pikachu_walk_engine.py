"""
Pikachu Walk Engine - Specialized walk engine for Pikachu_V025 robot
Based on PlacoWalkEngine with Pikachu_V025 specific configurations
"""

import time
import warnings
import json

import numpy as np
import placo
import os

warnings.filterwarnings("ignore")

DT = 0.01
REFINE = 10


class PikachuWalkEngine:
    """
    Walk engine for the Pikachu_V025 quadruped robot using Placo walk planning.
    """
    
    def __init__(
        self,
        asset_path: str = "",
        model_filename: str = "Pikachu_V025_flat.urdf",
        init_params: dict = {},
        ignore_feet_contact: bool = False,
        knee_limits: list = None,
    ) -> None:
        """
        Initialize the Pikachu walk engine.
        
        Args:
            asset_path: Path to the robot assets directory
            model_filename: Name of the URDF model file
            init_params: Initial parameters dictionary
            ignore_feet_contact: Whether to ignore foot contact sensors
            knee_limits: Knee joint limits
        """
        model_filename = os.path.join(asset_path, model_filename)
        self.asset_path = asset_path
        self.model_filename = model_filename
        self.ignore_feet_contact = ignore_feet_contact

        # Pikachu_V025 has positive knee limits (unlike open_duck)
        knee_limits = knee_limits or [0.0, 1.57]

        # Loading the robot
        self.robot = placo.HumanoidRobot(model_filename)

        self.parameters = placo.HumanoidParameters()
        if init_params is not None:
            self.load_parameters(init_params)
        else:
            defaults_filename = os.path.join(asset_path, "placo_defaults.json")
            self.load_defaults(defaults_filename)

        # Creating the kinematics solver
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.enable_velocity_limits(True)
        self.robot.set_velocity_limits(12.0)
        self.solver.enable_joint_limits(False)
        self.solver.dt = DT / REFINE

        # Set knee joint limits for both legs
        self.robot.set_joint_limits("left_knee_joint", *knee_limits)
        self.robot.set_joint_limits("right_knee_joint", *knee_limits)

        # Creating the walk QP tasks
        self.tasks = placo.WalkTasks()
        if hasattr(self.parameters, 'trunk_mode'):
            self.tasks.trunk_mode = self.parameters.trunk_mode
        self.tasks.com_x = 0.0
        self.tasks.initialize_tasks(self.solver, self.robot)
        self.tasks.left_foot_task.orientation().mask.set_axises("yz", "local")
        self.tasks.right_foot_task.orientation().mask.set_axises("yz", "local")

        # Creating a joint task to assign DoF values for upper body
        self.joints = self.parameters.joints
        # `joint_angles` 可以是字典（joint->degrees），也可能是列表或未设置。
        # 为了兼容不同来源的参数，先检查类型以避免 AttributeError。
        joint_degrees = self.parameters.joint_angles
        if isinstance(joint_degrees, dict):
            joint_radians = {joint: np.deg2rad(degrees) for joint, degrees in joint_degrees.items()}
        else:
            joint_radians = {}
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints(joint_radians)
        self.joints_task.configure("joints", "soft", 1.0)

        # Placing the robot in the initial position
        print("Placing the robot in the initial position...")
        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )
        print("Initial position reached")

        print(self.get_angles())

        # Creating the FootstepsPlanner
        self.repetitive_footsteps_planner = placo.FootstepsPlannerRepetitive(
            self.parameters
        )
        self.d_x = 0.0
        self.d_y = 0.0
        self.d_theta = 0.0
        self.nb_steps = 5
        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

        # Planning footsteps
        self.T_world_left = placo.flatten_on_floor(self.robot.get_T_world_left())
        self.T_world_right = placo.flatten_on_floor(self.robot.get_T_world_right())
        self.footsteps = self.repetitive_footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self.T_world_left, self.T_world_right
        )

        self.supports = placo.FootstepsPlanner.make_supports(
            self.footsteps, True, self.parameters.has_double_support(), True
        )

        # Creating the pattern generator and making an initial plan
        self.walk = placo.WalkPatternGenerator(self.robot, self.parameters)
        self.trajectory = self.walk.plan(self.supports, self.robot.com_world(), 0.0)

        self.time_since_last_right_contact = 0.0
        self.time_since_last_left_contact = 0.0
        self.start = None
        self.initial_delay = -1.0
        self.t = self.initial_delay
        self.last_replan = 0

        # Calculate the period of one walking cycle
        self.period = (
            2 * self.parameters.single_support_duration
            + 2 * self.parameters.double_support_duration()
        )
        print("## period:", self.period)

        # Live tuning UI state
        self._pending_params = None
        self._pending_reset = False
        self._tuning_started = False
        self._qt_app = None
        self._qt_window = None
        self._qt_param_inputs = {}
        self._qt_joints_edit = None
        self._qt_joint_angles_edit = None
        self._qt_trunk_mode_edit = None
        self._qt_status_label = None
        self._manual_traj_override = False

    def _collect_current_params(self):
        params = self.parameters
        return {
            "double_support_ratio": params.double_support_ratio,
            "startend_double_support_ratio": params.startend_double_support_ratio,
            "planned_timesteps": params.planned_timesteps,
            "replan_timesteps": params.replan_timesteps,
            "walk_com_height": params.walk_com_height,
            "walk_foot_height": params.walk_foot_height,
            "walk_trunk_pitch": np.rad2deg(params.walk_trunk_pitch),
            "walk_foot_rise_ratio": params.walk_foot_rise_ratio,
            "single_support_duration": params.single_support_duration,
            "single_support_timesteps": params.single_support_timesteps,
            "foot_length": params.foot_length,
            "feet_spacing": params.feet_spacing,
            "zmp_margin": params.zmp_margin,
            "foot_zmp_target_x": params.foot_zmp_target_x,
            "foot_zmp_target_y": params.foot_zmp_target_y,
            "walk_max_dtheta": params.walk_max_dtheta,
            "walk_max_dy": params.walk_max_dy,
            "walk_max_dx_forward": params.walk_max_dx_forward,
            "walk_max_dx_backward": params.walk_max_dx_backward,
            "joints": list(params.joints) if hasattr(params, "joints") else [],
            "joint_angles": params.joint_angles if hasattr(params, "joint_angles") else {},
            "trunk_mode": getattr(params, "trunk_mode", None),
            "d_x": self.d_x,
            "d_y": self.d_y,
            "dtheta": self.d_theta,
        }

    def start_tuning_ui(self, title="Pikachu Walk Tuner"):
        """Launch a live parameter tuning window."""
        if self._tuning_started:
            return
        self._tuning_started = True

        print("[walk_engine] opening tuning UI (PyQt5)...")
        try:
            from PyQt5.QtWidgets import (
                QApplication,
                QWidget,
                QLabel,
                QVBoxLayout,
                QHBoxLayout,
                QDoubleSpinBox,
                QSpinBox,
                QSlider,
                QScrollArea,
                QPushButton,
                QLineEdit,
            )
            from PyQt5.QtCore import Qt
        except Exception as e:
            print(f"[walk_engine] tuning UI unavailable: {e}")
            self._tuning_started = False
            return

        specs = {
            "double_support_ratio": (0.0, 1.0, 0.01),
            "startend_double_support_ratio": (0.0, 1.0, 0.01),
            "planned_timesteps": (10, 500, 1),
            "replan_timesteps": (1, 300, 1),
            "walk_com_height": (0.05, 0.5, 0.001),
            "walk_foot_height": (0.0, 0.2, 0.001),
            "walk_trunk_pitch": (-30.0, 30.0, 0.1),
            "walk_foot_rise_ratio": (0.0, 1.0, 0.01),
            "single_support_duration": (0.1, 1.5, 0.01),
            "single_support_timesteps": (1, 400, 1),
            "foot_length": (0.01, 0.3, 0.001),
            "feet_spacing": (0.05, 0.5, 0.001),
            "zmp_margin": (-0.05, 0.1, 0.001),
            "foot_zmp_target_x": (-0.1, 0.1, 0.001),
            "foot_zmp_target_y": (-0.1, 0.1, 0.001),
            "walk_max_dtheta": (-1, 1, 0.01),
            "walk_max_dy": (-1, 1, 0.001),
            "walk_max_dx_forward": (-1, 1, 0.001),
            "walk_max_dx_backward": (-1, 1, 0.001),
            "d_x": (-1, 1, 0.01),
            "d_y": (-1, 1, 0.01),
            "dtheta": (-1, 1, 0.01),
        }
        integer_keys = {"planned_timesteps", "replan_timesteps", "single_support_timesteps"}
        current = self._collect_current_params()
        self._qt_app = QApplication.instance() or QApplication([])

        window = QWidget()
        window.setWindowTitle(title)
        window.resize(760, 920)
        self._qt_window = window
        self._qt_param_inputs = {}

        root_layout = QVBoxLayout(window)
        scroll_area = QScrollArea(window)
        scroll_area.setWidgetResizable(True)
        root_layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)

        for key, (vmin, vmax, step) in specs.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(key))
            if key in integer_keys:
                editor = QSpinBox()
                editor.setRange(int(vmin), int(vmax))
                editor.setSingleStep(int(step))
                editor.setValue(int(current[key]))
                slider = QSlider(Qt.Horizontal)
                slider.setRange(int(vmin), int(vmax))
                slider.setSingleStep(int(step))
                slider.setValue(int(current[key]))
                slider.valueChanged.connect(editor.setValue)
                editor.valueChanged.connect(slider.setValue)
            else:
                editor = QDoubleSpinBox()
                editor.setDecimals(4)
                editor.setRange(float(vmin), float(vmax))
                editor.setSingleStep(float(step))
                editor.setValue(float(current[key]))
                slider = QSlider(Qt.Horizontal)
                scale = max(1, int(round(1.0 / float(step))))
                slider.setRange(int(round(float(vmin) * scale)), int(round(float(vmax) * scale)))
                slider.setSingleStep(1)
                slider.setValue(int(round(float(current[key]) * scale)))
                slider.valueChanged.connect(lambda v, e=editor, s=scale: e.setValue(v / s))
                editor.valueChanged.connect(lambda v, sl=slider, s=scale: sl.setValue(int(round(v * s))))
            slider.setFixedWidth(300)
            row.addWidget(slider)
            editor.setFixedWidth(200)
            row.addWidget(editor)
            row.addStretch(1)
            scroll_layout.addLayout(row)
            self._qt_param_inputs[key] = editor

        row_joints = QHBoxLayout()
        row_joints.addWidget(QLabel("joints (JSON list)"))
        self._qt_joints_edit = QLineEdit(json.dumps(current["joints"]))
        row_joints.addWidget(self._qt_joints_edit)
        scroll_layout.addLayout(row_joints)

        row_joint_angles = QHBoxLayout()
        row_joint_angles.addWidget(QLabel("joint_angles (JSON dict, deg)"))
        self._qt_joint_angles_edit = QLineEdit(json.dumps(current["joint_angles"]))
        row_joint_angles.addWidget(self._qt_joint_angles_edit)
        scroll_layout.addLayout(row_joint_angles)

        row_trunk_mode = QHBoxLayout()
        row_trunk_mode.addWidget(QLabel("trunk_mode (optional)"))
        trunk_text = "" if current["trunk_mode"] is None else str(current["trunk_mode"])
        self._qt_trunk_mode_edit = QLineEdit(trunk_text)
        row_trunk_mode.addWidget(self._qt_trunk_mode_edit)
        scroll_layout.addLayout(row_trunk_mode)

        self._qt_status_label = QLabel("Ready")
        scroll_layout.addWidget(self._qt_status_label)

        def _push_params(reset_pose):
            data = {}
            for key, editor in self._qt_param_inputs.items():
                value = editor.value()
                if key in integer_keys:
                    value = int(value)
                data[key] = value
            try:
                joints_text = self._qt_joints_edit.text().strip()
                data["joints"] = json.loads(joints_text) if joints_text else []
                angles_text = self._qt_joint_angles_edit.text().strip()
                data["joint_angles"] = json.loads(angles_text) if angles_text else {}
            except Exception as e:
                self._qt_status_label.setText(f"JSON parse error: {e}")
                return
            trunk_mode_text = self._qt_trunk_mode_edit.text().strip()
            if trunk_mode_text:
                data["trunk_mode"] = trunk_mode_text
            self._pending_params = data
            self._pending_reset = reset_pose
            self._qt_status_label.setText("Pending apply on next control tick")

        button_row = QHBoxLayout()
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(lambda: _push_params(False))
        button_row.addWidget(btn_apply)
        btn_apply_reset = QPushButton("Apply + Reset Pose")
        btn_apply_reset.clicked.connect(lambda: _push_params(True))
        button_row.addWidget(btn_apply_reset)
        button_row.addStretch(1)
        scroll_layout.addLayout(button_row)

        def _on_close(_event):
            self._qt_window = None
            self._tuning_started = False

        window.closeEvent = _on_close
        window.show()

    def poll_tuning_ui(self):
        if self._qt_app is None or self._qt_window is None:
            return
        try:
            self._qt_app.processEvents()
        except Exception:
            self._qt_window = None
            self._tuning_started = False

    def apply_pending_tuning(self):
        """Apply pending tuning values from UI in simulation thread."""
        params = self._pending_params
        reset_pose = self._pending_reset
        self._pending_params = None
        self._pending_reset = False
        if params is None:
            return False

        self.load_parameters(params)
        if hasattr(self.parameters, "trunk_mode"):
            self.tasks.trunk_mode = self.parameters.trunk_mode

        joint_degrees = self.parameters.joint_angles
        if isinstance(joint_degrees, dict):
            joint_radians = {joint: np.deg2rad(degrees) for joint, degrees in joint_degrees.items()}
            self.joints_task.set_joints(joint_radians)

        if "d_x" in params or "d_y" in params or "dtheta" in params:
            d_x = float(params.get("d_x", self.d_x))
            d_y = float(params.get("d_y", self.d_y))
            d_theta = float(params.get("dtheta", self.d_theta))
            self.set_traj(d_x, d_y, d_theta)
            self._manual_traj_override = True

        self.repetitive_footsteps_planner.configure(self.d_x, self.d_y, self.d_theta, self.nb_steps)
        if reset_pose:
            self.reset()
        print("[walk_engine] applied live tuning parameters")
        return True

    def get_traj_command(self, cmd_scaled):
        """Return trajectory command, preferring manual UI values when set."""
        if self._manual_traj_override:
            return np.array([self.d_x, self.d_y, self.d_theta], dtype=np.float32)
        return np.array(cmd_scaled, dtype=np.float32)

    def load_defaults(self, filename):
        """Load default parameters from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        params = self.parameters
        self.load_parameters(data)

    def load_parameters(self, data):
        """Load walk parameters from a dictionary."""
        params = self.parameters
        params.double_support_ratio = data.get('double_support_ratio', params.double_support_ratio)
        params.startend_double_support_ratio = data.get('startend_double_support_ratio', params.startend_double_support_ratio)
        params.planned_timesteps = data.get('planned_timesteps', params.planned_timesteps)
        params.replan_timesteps = data.get('replan_timesteps', params.replan_timesteps)
        params.walk_com_height = data.get('walk_com_height', params.walk_com_height)
        params.walk_foot_height = data.get('walk_foot_height', params.walk_foot_height)
        params.walk_trunk_pitch = np.deg2rad(data.get('walk_trunk_pitch', np.rad2deg(params.walk_trunk_pitch)))
        params.walk_foot_rise_ratio = data.get('walk_foot_rise_ratio', params.walk_foot_rise_ratio)
        params.single_support_duration = data.get('single_support_duration', params.single_support_duration)
        params.single_support_timesteps = data.get('single_support_timesteps', params.single_support_timesteps)
        params.foot_length = data.get('foot_length', params.foot_length)
        params.feet_spacing = data.get('feet_spacing', params.feet_spacing)
        params.zmp_margin = data.get('zmp_margin', params.zmp_margin)
        params.foot_zmp_target_x = data.get('foot_zmp_target_x', params.foot_zmp_target_x)
        params.foot_zmp_target_y = data.get('foot_zmp_target_y', params.foot_zmp_target_y)
        params.walk_max_dtheta = data.get('walk_max_dtheta', params.walk_max_dtheta)
        params.walk_max_dy = data.get('walk_max_dy', params.walk_max_dy)
        params.walk_max_dx_forward = data.get('walk_max_dx_forward', params.walk_max_dx_forward)
        params.walk_max_dx_backward = data.get('walk_max_dx_backward', params.walk_max_dx_backward)
        params.joints = data.get('joints', [])
        params.joint_angles = data.get('joint_angles', [])
        if 'trunk_mode' in data:
            params.trunk_mode = data.get('trunk_mode')

    def get_angles(self):
        """Get current joint angles for all tracked joints."""
        angles = {joint: self.robot.get_joint(joint) for joint in self.joints}
        return angles

    def reset(self):
        """Reset the walk engine to initial state."""
        self.t = 0
        self.start = None
        self.last_replan = 0
        self.time_since_last_right_contact = 0.0
        self.time_since_last_left_contact = 0.0

        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )

        # Planning footsteps
        self.T_world_left = placo.flatten_on_floor(self.robot.get_T_world_left())
        self.T_world_right = placo.flatten_on_floor(self.robot.get_T_world_right())
        self.footsteps = self.repetitive_footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self.T_world_left, self.T_world_right
        )

        self.supports = placo.FootstepsPlanner.make_supports(
            self.footsteps, True, self.parameters.has_double_support(), True
        )
        self.trajectory = self.walk.plan(self.supports, self.robot.com_world(), 0.0)

    def set_traj(self, d_x, d_y, d_theta):
        """
        Set the desired trajectory parameters.
        
        Args:
            d_x: Forward step distance
            d_y: Lateral step distance
            d_theta: Rotational step angle
        """
        self.d_x = d_x
        self.d_y = d_y
        self.d_theta = d_theta
        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

    def get_footsteps_in_world(self):
        """Get planned footsteps in world frame."""
        footsteps = self.trajectory.get_supports()
        footsteps_in_world = []
        for footstep in footsteps:
            if not footstep.is_both():
                footsteps_in_world.append(footstep.frame())

        for i in range(len(footsteps_in_world)):
            footsteps_in_world[i][:3, 3][1] += self.parameters.feet_spacing / 2

        return footsteps_in_world

    def get_footsteps_in_robot_frame(self):
        """Get planned footsteps in robot frame."""
        T_world_fbase = self.robot.get_T_world_fbase()

        footsteps = self.trajectory.get_supports()
        footsteps_in_robot_frame = []
        for footstep in footsteps:
            if not footstep.is_both():
                T_world_footstepFrame = footstep.frame().copy()
                T_fbase_footstepFrame = (
                    np.linalg.inv(T_world_fbase) @ T_world_footstepFrame
                )
                T_fbase_footstepFrame = placo.flatten_on_floor(T_fbase_footstepFrame)
                T_fbase_footstepFrame[:3, 3][2] = -T_world_fbase[:3, 3][2]

                footsteps_in_robot_frame.append(T_fbase_footstepFrame)

        return footsteps_in_robot_frame

    def get_current_support_phase(self):
        """Get current support phase (which feet are in contact)."""
        if self.trajectory.support_is_both(self.t):
            return [1, 1]
        elif str(self.trajectory.support_side(self.t)) == "left":
            return [1, 0]
        elif str(self.trajectory.support_side(self.t)) == "right":
            return [0, 1]
        else:
            raise AssertionError(f"Invalid phase: {self.trajectory.support_side(self.t)}")

    def tick(self, dt, left_contact=True, right_contact=True):
        """
        Update the walk engine for one timestep.
        
        Args:
            dt: Timestep duration
            left_contact: Whether left foot is in contact
            right_contact: Whether right foot is in contact
        """
        if self.start is None:
            self.start = time.time()

        if not self.ignore_feet_contact:
            if left_contact:
                self.time_since_last_left_contact = 0.0
            if right_contact:
                self.time_since_last_right_contact = 0.0

        # Check if falling based on contact duration
        falling = not self.ignore_feet_contact and (
            self.time_since_last_left_contact > self.parameters.single_support_duration
            or self.time_since_last_right_contact
            > self.parameters.single_support_duration
        )

        # Solve kinematics with refinement
        for k in range(REFINE):
            # Updating the QP tasks from planned trajectory
            if not falling:
                self.tasks.update_tasks_from_trajectory(
                    self.trajectory, self.t - dt + k * dt / REFINE
                )

            self.robot.update_kinematics()
            _ = self.solver.solve(True)

        # If enough time elapsed and we can replan, do the replanning
        if (
            self.t - self.last_replan
            > self.parameters.replan_timesteps * self.parameters.dt()
            and self.walk.can_replan_supports(self.trajectory, self.t)
        ):
            self.last_replan = self.t

            # Replanning footsteps from current trajectory
            self.supports = self.walk.replan_supports(
                self.repetitive_footsteps_planner, self.trajectory, self.t
            )

            # Replanning CoM trajectory, yielding a new trajectory we can switch to
            self.trajectory = self.walk.replan(self.supports, self.trajectory, self.t)

        self.time_since_last_left_contact += dt
        self.time_since_last_right_contact += dt
        self.t += dt
