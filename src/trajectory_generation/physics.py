"""Bounds and independent nonlinear checks on the returned vehicle states."""
import numpy as np

from ._utils import speed_mps, wrap_angle


class InitialPhysicsError(ValueError):
    """The initial state of a QP window is outside its physical envelope."""


def acceleration_envelope(initial_accel, dt, limits, config):
    """Recover an already-outside comfort acceleration at max jerk, within fixed actuator caps."""
    elapsed = np.cumsum(dt)
    upper = np.maximum(limits.max_accel_mps2, initial_accel - limits.max_jerk_mps3 * elapsed)
    lower = np.minimum(-limits.max_decel_mps2, initial_accel + limits.max_jerk_mps3 * elapsed)
    return np.maximum(lower, -config.actuator_max_decel_mps2), np.minimum(upper, config.actuator_max_accel_mps2)


def local_states(states, initial):
    xy = np.asarray([[s.rear_axle.x, s.rear_axle.y] for s in states]) - [initial.rear_axle.x, initial.rear_axle.y]
    angle = initial.rear_axle.heading
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    headings = np.unwrap([float(wrap_angle(s.rear_axle.heading - angle)) for s in states])
    return np.column_stack((xy @ rotation, headings, [speed_mps(s) for s in states],
                            [s.tire_steering_angle for s in states]))


def _model_residual(states, controls, dt, wheelbase):
    expected = states[:-1].copy()
    expected[:, 0] += states[:-1, 3] * np.cos(states[:-1, 2]) * dt
    expected[:, 1] += states[:-1, 3] * np.sin(states[:-1, 2]) * dt
    expected[:, 2] += states[:-1, 3] / wheelbase * np.tan(states[:-1, 4]) * dt
    expected[:, 3] += controls[:, 0] * dt
    expected[:, 4] += controls[:, 1] * dt
    residual = states[1:] - expected
    residual[:, 2] = wrap_angle(residual[:, 2])
    return float(np.max(np.abs(residual)))


def validate_rollout(predicted, controls, reference, dt, wheelbase, initial_accel, cfg, limits):
    if not np.isfinite(predicted).all() or not np.isfinite(controls).all():
        raise RuntimeError('QP rollout contains nonfinite states or controls')
    position = np.linalg.norm(predicted[:, :2] - reference[:, :2], axis=1)
    heading = np.abs(wrap_angle(predicted[:, 2] - reference[:, 2]))
    lateral = predicted[:, 3] ** 2 * np.tan(predicted[:, 4]) / wheelbase
    jerk = np.diff(np.r_[initial_accel, controls[:, 0]]) / dt
    stop = reference[:, 3] <= limits.stop_speed_threshold_mps
    max_stop = float(np.max(predicted[stop, 3])) if np.any(stop) else 0.0
    residual = _model_residual(predicted, controls, dt, wheelbase)
    accel_lower, accel_upper = acceleration_envelope(initial_accel, dt, limits, cfg)
    bounds = {'acceleration_recovery': (float(np.max(controls[:, 0] - accel_upper)), 0.0),
        'deceleration_recovery': (float(np.max(accel_lower - controls[:, 0])), 0.0),
        'acceleration': (float(np.max(controls[:, 0])), cfg.actuator_max_accel_mps2),
        'deceleration': (float(np.max(-controls[:, 0])), cfg.actuator_max_decel_mps2),
        'steering_rate': (float(np.max(np.abs(controls[:, 1]))), cfg.max_steering_rate_rps),
        'steering_angle': (float(np.max(np.abs(predicted[:, 4]))), min(cfg.max_steering_angle_rad, limits.max_steering_angle_rad)),
        'lateral_acceleration': (float(np.max(np.abs(lateral))), limits.max_lateral_accel_mps2)}
    max_jerk = float(np.max(np.abs(jerk)))
    reasons = [k for k, (value, bound) in bounds.items() if value > bound + cfg.validation_tolerance]
    if float(np.min(predicted[:, 3])) < -cfg.model_tolerance:
        reasons.append('negative_model_speed')
    if residual > cfg.model_tolerance:
        reasons.append('model_residual')
    position_slack = max(float(np.max(position)) - cfg.lateral_tube_m, 0.0)
    heading_slack = max(float(np.max(heading)) - cfg.heading_tube_rad, 0.0)
    stop_slack = max(max_stop - limits.stop_speed_threshold_mps, 0.0)
    return {'accepted': not reasons, 'rejection_reason': ','.join(reasons) or None,
        'hard_physics_validated': not reasons, 'tracking_constraint_mode': 'soft',
        'max_position_error_m': float(np.max(position)), 'max_heading_error_rad': float(np.max(heading)),
        'max_position_tracking_slack_m': position_slack, 'max_heading_tracking_slack_rad': heading_slack,
        'max_stop_tracking_slack_mps': stop_slack,
        'tracking_tubes_satisfied': position_slack <= cfg.validation_tolerance and heading_slack <= cfg.validation_tolerance,
        'max_abs_lateral_accel_mps2': bounds['lateral_acceleration'][0],
        'max_abs_jerk_mps3': max_jerk, 'max_abs_steering_rate_rps': bounds['steering_rate'][0],
        'jerk_hard_constraint': False, 'jerk_penalty_weight': cfg.jerk_weight,
        'jerk_samples_mps3': jerk.tolist(),
        'comfort_jerk_excess_mps3': max(max_jerk - limits.max_jerk_mps3, 0.0),
        'hard_constraint_names': sorted([*bounds, 'nonnegative_speed', 'model_residual']),
        'initial_comfort_bound_recovery_steps': int(np.sum((accel_upper > limits.max_accel_mps2)
                                                          | (accel_lower < -limits.max_decel_mps2))),
        'max_accel_mps2': float(np.max(controls[:, 0])), 'max_decel_mps2': float(np.max(-controls[:, 0])),
        'max_model_residual': residual, 'min_speed_mps': float(np.min(predicted[:, 3])), 'max_stop_speed_mps': max_stop}
