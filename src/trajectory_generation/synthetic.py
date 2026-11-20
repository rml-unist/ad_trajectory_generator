"""Deterministic, dataset-free examples. No planner predictions or GT are embedded."""
import time

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


CASES = ("straight", "curve", "stop", "accelerate", "stop_start")
STATE_COLUMNS = ("time_us", "x_m", "y_m", "heading_rad", "vx_mps", "vy_mps",
                 "ax_mps2", "ay_mps2", "steering_rad", "steering_rate_rps",
                 "yaw_rate_rps", "yaw_accel_rps2")


def make_trajectory(kind="straight", *, jitter=False):
    if kind not in CASES:
        raise ValueError("Unknown synthetic case: " + str(kind))
    timestamps = np.arange(81, dtype=np.int64) * 100000
    if jitter:
        timestamps[1::2] += 500
    seconds = timestamps * 1e-6
    speeds = np.full(81, 2.0)
    if kind == "stop":
        speeds[:] = 0.0
    elif kind == "accelerate":
        speeds = 2.0 + 0.8 * seconds
    elif kind == "stop_start":
        speeds = np.where(seconds < 3.0, np.maximum(0.0, 1.0 - 0.5 * seconds),
                          np.minimum(2.0, 0.4 * (seconds - 3.0)))
    distance = np.r_[0.0, np.cumsum(speeds[:-1] * np.diff(seconds))]
    accel = np.r_[0.0, np.diff(speeds) / np.diff(seconds)]
    states = []
    for i, progress in enumerate(distance):
        angle = progress / 40.0 if kind == "curve" else 0.0
        x, y = ((40.0 * np.sin(angle), 40.0 * (1.0 - np.cos(angle)))
                if kind == "curve" else (progress, 0.0))
        states.append(EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(float(x), float(y), float(angle)),
            rear_axle_velocity_2d=StateVector2D(float(speeds[i]), 0.0),
            rear_axle_acceleration_2d=StateVector2D(float(accel[i]), 0.0),
            tire_steering_angle=0.0, time_point=TimePoint(int(timestamps[i])),
            vehicle_parameters=get_pacifica_parameters()))
    return InterpolatedTrajectory(states), states[0]


def state_values(state):
    dynamics = state.dynamic_car_state
    return [int(state.time_point.time_us), float(state.rear_axle.x), float(state.rear_axle.y),
            float(state.rear_axle.heading), float(dynamics.rear_axle_velocity_2d.x),
            float(dynamics.rear_axle_velocity_2d.y), float(dynamics.rear_axle_acceleration_2d.x),
            float(dynamics.rear_axle_acceleration_2d.y), float(state.tire_steering_angle),
            float(dynamics.tire_steering_rate), float(dynamics.angular_velocity),
            float(dynamics.angular_acceleration)]


def run_case(config, kind, *, jitter=False):
    from trajectory_generation import TrajectoryGenerationPipeline
    from .window_validation import compact_window

    raw, initial = make_trajectory(kind, jitter=jitter)
    started = time.perf_counter()
    times = [state.time_point.time_us for state in raw.get_sampled_trajectory()]
    session = TrajectoryGenerationPipeline(config).start_session(
        raw, initial, feedback_times_us=times if jitter else None)
    preparation_s = time.perf_counter() - started
    states, windows, previews, tracking = [state_values(initial)], [], [], []
    refresh_steps = []
    for index, timestamp in enumerate(times[:-1]):
        started = time.perf_counter()
        result = session.generate_reference(TimePoint(timestamp))
        elapsed = time.perf_counter() - started
        if result.used_fallback:
            raise RuntimeError("Synthetic example unexpectedly used fallback")
        windows.append(compact_window(result, elapsed,
                       "logged_origin" if index == 0 else "kmpc_model_state"))
        previews.append([state_values(s) for s in result.trajectory.get_sampled_trajectory()])
        tracking.append([state_values(s) for s in
                         result.trajectory.tracking_reference_trajectory.get_sampled_trajectory()])
        states.append(state_values(session.reference_state))
        if result.diagnostics["qp_refreshed"]:
            refresh_steps.append(index)
    numeric = dict(case=kind, jitter=jitter, raw=[state_values(s) for s in raw.get_sampled_trajectory()],
                   retimed=[state_values(s) for s in session.target.get_sampled_trajectory()],
                   reference=states, previews=previews, tracking_previews=tracking,
                   refresh_steps=refresh_steps, mode=result.applied_mode, fallback=0)
    diagnostics = dict(preparation_s=preparation_s, retiming=session.retiming,
                       timestamps=session.timestamp_info, windows=windows)
    return numeric, diagnostics
