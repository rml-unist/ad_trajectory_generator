from __future__ import annotations

from typing import Any

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from ._utils import build_state, endpoint_preserving_times, finite_difference, speed_mps
from .config import KinematicGeneratorConfig, KMPCQPConfig
from .physics import acceleration_envelope


def anchor_current_state(trajectory: AbstractTrajectory, initial: EgoState):
    states = list(trajectory.get_sampled_trajectory())
    times = np.asarray([s.time_point.time_us for s in states], dtype=np.int64)
    poses = np.asarray([[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading] for s in states])
    if len(states) < 2 or not np.isfinite(poses).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Retiming requires finite poses and strictly increasing timestamps")
    origin = initial.time_point.time_us
    if times[0] < origin or times[-1] <= origin:
        raise ValueError("Retiming cannot shift planner timestamps or reuse a past origin")
    future = [s for s in states if s.time_point.time_us > origin]
    anchored = InterpolatedTrajectory([initial, *future])
    return anchored, {
        "current_state_source": "actual_ego",
        "replaced_start_sample": bool(times[0] == origin),
        "input_start_error_m": float(np.linalg.norm(poses[0, :2] - [initial.rear_axle.x, initial.rear_axle.y])),
        "future_timestamps_unchanged": True,
    }


def _speed_profile(desired, dt, initial_accel, limits, config):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    count = len(desired)
    rows, columns, coefficients, rhs = [], [], [], []

    def add(values, upper):
        row = len(rhs)
        for column, value in values.items():
            rows.append(row)
            columns.append(column)
            coefficients.append(value)
        rhs.append(upper)

    for i, target in enumerate(desired):
        add({i: 1.0, count + i: -1.0}, target)
        add({i: -1.0, count + i: -1.0}, -target)
    accel_lower, accel_upper = acceleration_envelope(initial_accel, dt, limits, config)
    for i, interval in enumerate(dt):
        add({i: -1.0, i + 1: 1.0}, accel_upper[i] * interval)
        add({i: 1.0, i + 1: -1.0}, -accel_lower[i] * interval)
    matrix = coo_matrix((coefficients, (rows, columns)), shape=(len(rhs), 2 * count)).tocsr()
    bounds = [(float(desired[0]), float(desired[0]))] + [(0.0, None)] * (count - 1)
    result = linprog(np.r_[np.zeros(count), np.ones(count)], A_ub=matrix,
                     b_ub=np.asarray(rhs), bounds=bounds + [(0.0, None)] * count, method="highs")
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Infeasible path retiming: {result.message}")
    return np.asarray(result.x[:count])


class RetimedPathGenerator:
    """Build optimization targets p(s(t)); the final MPC rollout supplies model feasibility."""

    def __init__(self, limits: KinematicGeneratorConfig, config: KMPCQPConfig):
        self.limits = limits
        self.config = config

    def generate(self, trajectory: AbstractTrajectory, initial: EgoState) -> tuple[AbstractTrajectory, dict[str, Any]]:
        times = endpoint_preserving_times(trajectory, self.limits.dt_s)
        templates = list(trajectory.get_state_at_times(times))
        times_s = np.asarray([t.time_us - times[0].time_us for t in times], dtype=float) * 1e-6
        dt = np.diff(times_s)
        points = np.asarray([[s.rear_axle.x, s.rear_axle.y] for s in templates])
        headings = np.unwrap([s.rear_axle.heading for s in templates])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        path_s = np.r_[0.0, np.cumsum(distances)]
        keep = np.r_[True, distances > 1e-8]
        wheelbase = float(initial.car_footprint.vehicle_parameters.wheel_base)
        if np.count_nonzero(keep) > 1:
            unique_s = path_s[keep]
            curvature = np.gradient(headings[keep], unique_s)
            geometric_speed = np.r_[distances / dt, distances[-1] / dt[-1]]
            curve_limit = np.sqrt(self.limits.max_lateral_accel_mps2 / np.maximum(np.abs(np.interp(path_s, unique_s, curvature)), 1e-6))
            if self.limits.speed_profile_scale != 1.0:
                geometric_speed = geometric_speed * self.limits.speed_profile_scale
            desired = np.minimum(geometric_speed, curve_limit)
        else:
            unique_s, curvature = np.asarray([0.0]), np.asarray([0.0])
            desired = np.zeros(len(templates))
        desired[0] = speed_mps(initial)
        initial_accel = float(initial.dynamic_car_state.rear_axle_acceleration_2d.x)
        speeds = _speed_profile(desired, dt, initial_accel, self.limits, self.config)
        retimed_s = np.r_[0.0, np.cumsum(speeds[:-1] * dt)]
        # Hold the target at the path end; the downstream physical solve handles unavoidable overrun.
        requested_progress = retimed_s.copy()
        retimed_s = np.minimum(retimed_s, path_s[-1])
        target_speeds = speeds.copy()
        overrun = requested_progress > path_s[-1] + 1e-8
        terminal_path_speed = float(desired[-1])
        # A finite horizon is not a stop command; respect the speed of its last path segment.
        at_end = requested_progress >= path_s[-1]
        target_speeds[at_end] = np.minimum(speeds[at_end], terminal_path_speed)
        target_speeds[0] = speed_mps(initial)
        x = np.interp(retimed_s, unique_s, points[keep, 0])
        y = np.interp(retimed_s, unique_s, points[keep, 1])
        heading = np.interp(retimed_s, unique_s, headings[keep])
        retimed_curvature = np.interp(retimed_s, unique_s, curvature)
        steering = np.arctan(wheelbase * retimed_curvature)
        steering[0] = initial.tire_steering_angle
        acceleration = np.r_[initial_accel, np.diff(target_speeds) / dt]
        yaw_rate = target_speeds * retimed_curvature
        yaw_accel = finite_difference(yaw_rate, times_s)
        rate = np.r_[initial.dynamic_car_state.tire_steering_rate, np.diff(steering) / dt]
        states = [build_state(template, StateSE2(x[i], y[i], heading[i]), target_speeds[i],
                              acceleration[i], steering[i], rate[i], yaw_rate[i], yaw_accel[i])
                  for i, template in enumerate(templates)]
        states[0] = initial
        generated = InterpolatedTrajectory(states)
        return generated, {
            "method": "arc_length_retiming_with_highs_speed_projection",
            "speed_profile_scale": self.limits.speed_profile_scale,
            "jerk_hard_constraint": False,
            "comfort_acceleration_recovery_jerk_mps3": self.limits.max_jerk_mps3,
            "reference_is_optimization_target_not_feasibility_certificate": True,
            "input_path_length_m": float(path_s[-1]),
            "retimed_progress_m": float(retimed_s[-1]),
            "projected_speed_progress_m": float(requested_progress[-1]),
            "target_path_overrun_m": max(float(requested_progress[-1] - path_s[-1]), 0.0),
            "target_endpoint_policy": "hold_without_spatial_extrapolation",
            "terminal_speed_policy": "path_terminal_speed",
            "terminal_path_speed_mps": terminal_path_speed,
            "terminal_target_speed_mps": float(target_speeds[-1]),
            "moving_endpoint_forced_stop_count": int(np.count_nonzero(
                overrun & (speeds > self.limits.stop_speed_threshold_mps)
                & (target_speeds <= self.limits.stop_speed_threshold_mps)
            )) if terminal_path_speed > self.limits.stop_speed_threshold_mps else 0,
            "path_length_is_hard_speed_constraint": False,
            "endpoint_preserved": False,
            "max_position_deviation_m": float(np.max(np.linalg.norm(np.column_stack((x, y)) - points, axis=1))),
            "max_arc_integration_residual_m": float(np.max(np.abs(np.diff(retimed_s) - speeds[:-1] * dt))),
            "input_samples": len(templates), "output_samples": len(states),
        }
