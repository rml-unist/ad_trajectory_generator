from __future__ import annotations

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


def speed_mps(state: EgoState) -> float:
    velocity = state.dynamic_car_state.rear_axle_velocity_2d
    return float(np.hypot(velocity.x, velocity.y))


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def endpoint_preserving_times(trajectory: AbstractTrajectory, dt_s: float) -> list[TimePoint]:
    """Use nominal interior steps and one actual endpoint, not a tiny extra interval."""
    start, end = trajectory.start_time.time_us, trajectory.end_time.time_us
    step = int(round(dt_s * 1e6))
    if step <= 0 or end <= start:
        raise ValueError("Invalid endpoint-preserving time grid")
    count = max(1, (end - start + step // 2) // step)
    values = [start + i * step for i in range(count)] + [end]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("Non-increasing endpoint-preserving time grid")
    return [TimePoint(int(t)) for t in values]


def build_state(
    template: EgoState,
    pose: StateSE2,
    speed: float,
    acceleration: float,
    steering_angle: float,
    steering_rate: float,
    yaw_rate: float,
    yaw_acceleration: float,
) -> EgoState:
    return EgoState.build_from_rear_axle(
        rear_axle_pose=pose,
        rear_axle_velocity_2d=StateVector2D(float(max(speed, 0.0)), 0.0),
        rear_axle_acceleration_2d=StateVector2D(float(acceleration), 0.0),
        tire_steering_angle=float(steering_angle),
        time_point=template.time_point,
        vehicle_parameters=template.car_footprint.vehicle_parameters,
        is_in_auto_mode=template.is_in_auto_mode,
        angular_vel=float(yaw_rate),
        angular_accel=float(yaw_acceleration),
        tire_steering_rate=float(steering_rate),
    )


def finite_difference(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if len(values) < 2:
        return result
    dt = np.maximum(np.diff(times_s), 1e-3)
    result[0] = (values[1] - values[0]) / dt[0]
    result[1:] = np.diff(values) / dt
    return result
