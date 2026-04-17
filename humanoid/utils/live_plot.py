from collections import deque
from queue import Empty, Full
import multiprocessing as mp

import numpy as np


def _plot_worker_main(queue, joint_names, max_points, redraw_interval, title):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[JointResponsePlotter] matplotlib unavailable in plot worker: {exc}")
        return

    max_subplots_per_figure = 5
    joint_names = list(joint_names)
    num_joints = len(joint_names)
    steps = deque(maxlen=max_points)
    target_history = [deque(maxlen=max_points) for _ in range(num_joints)]
    actual_history = [deque(maxlen=max_points) for _ in range(num_joints)]

    plt.ion()
    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = []
    if color_cycle is not None:
        colors = color_cycle.by_key().get("color", [])
    if not colors:
        colors = [f"C{i}" for i in range(max(1, num_joints))]

    figures = []
    target_lines = []
    actual_lines = []

    for idx, joint_name in enumerate(joint_names):
        figure_slot = idx % max_subplots_per_figure
        if figure_slot == 0:
            joints_in_figure = min(max_subplots_per_figure, num_joints - idx)
            fig, axes = plt.subplots(joints_in_figure, 1, figsize=(13, 2.8 * joints_in_figure), sharex=True)
            axes = np.atleast_1d(axes)
            figure_index = len(figures) + 1
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(f"{title} [{figure_index}]")
            figures.append({"fig": fig, "axes": axes})

        figure_info = figures[-1]
        axis = figure_info["axes"][figure_slot]
        color = colors[idx % len(colors)]
        target_line = axis.plot(
            [], [], linestyle="--", color=color, linewidth=1.5, label="target"
        )[0]
        actual_line = axis.plot(
            [], [], linestyle="-", color=color, linewidth=1.5, alpha=0.85, label="actual"
        )[0]
        axis.set_title(joint_name)
        axis.set_ylabel("rad")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
        target_lines.append(target_line)
        actual_lines.append(actual_line)

    for figure_info in figures:
        figure_info["axes"][-1].set_xlabel("Policy Step")
        figure_info["fig"].tight_layout()
        figure_info["fig"].canvas.draw_idle()
        figure_info["fig"].canvas.flush_events()

    update_count = 0
    running = True
    while running:
        try:
            message = queue.get(timeout=0.05)
        except Empty:
            if figures:
                plt.pause(0.001)
            continue

        msg_type = message[0]
        if msg_type == "close":
            running = False
            continue
        if msg_type != "update":
            continue

        _, step, target, actual = message
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        actual = np.asarray(actual, dtype=np.float32).reshape(-1)
        if target.shape[0] != num_joints or actual.shape[0] != num_joints:
            continue

        steps.append(int(step))
        for idx in range(num_joints):
            target_history[idx].append(float(target[idx]))
            actual_history[idx].append(float(actual[idx]))

        update_count += 1
        if update_count % redraw_interval != 0:
            continue

        x = np.asarray(steps, dtype=np.int32)
        for idx in range(num_joints):
            target_lines[idx].set_data(x, np.asarray(target_history[idx], dtype=np.float32))
            actual_lines[idx].set_data(x, np.asarray(actual_history[idx], dtype=np.float32))

        for figure_info in figures:
            for ax in figure_info["axes"]:
                ax.relim()
                ax.autoscale_view()
            figure_info["fig"].canvas.draw_idle()
            figure_info["fig"].canvas.flush_events()

        plt.pause(0.001)

    plt.ioff()
    for figure_info in figures:
        plt.close(figure_info["fig"])


class JointResponsePlotter:
    """Live plotter for actuator target/actual position tracking."""

    def __init__(self, joint_names, max_points=600, redraw_interval=2, title="Joint Response"):
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names)
        self.max_points = max(10, int(max_points))
        self.redraw_interval = max(1, int(redraw_interval))
        self.title = title
        self.enabled = False
        self._closed = False
        self._ctx = None
        self._queue = None
        self._process = None

        if self.num_joints == 0:
            print("[JointResponsePlotter] No joint selected, live plot disabled.")
            return

        try:
            self._ctx = mp.get_context("spawn")
            self._queue = self._ctx.Queue(maxsize=8)
            self._process = self._ctx.Process(
                target=_plot_worker_main,
                args=(
                    self._queue,
                    self.joint_names,
                    self.max_points,
                    self.redraw_interval,
                    self.title,
                ),
                daemon=True,
            )
            self._process.start()
        except Exception as exc:
            print(f"[JointResponsePlotter] Failed to start plot worker, live plot disabled: {exc}")
            self._queue = None
            self._process = None
            return

        self.enabled = True

    def update(self, step, target, actual):
        if not self.enabled or self._closed or self._queue is None:
            return

        target = np.asarray(target, dtype=np.float32).reshape(-1)
        actual = np.asarray(actual, dtype=np.float32).reshape(-1)
        if target.shape[0] != self.num_joints or actual.shape[0] != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints, got target={target.shape[0]}, actual={actual.shape[0]}"
            )

        payload = ("update", int(step), target.tolist(), actual.tolist())
        try:
            self._queue.put_nowait(payload)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except Full:
                pass

    def close(self):
        if not self.enabled or self._closed:
            return

        self._closed = True
        if self._queue is not None:
            try:
                self._queue.put_nowait(("close",))
            except Full:
                pass

        if self._process is not None:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
