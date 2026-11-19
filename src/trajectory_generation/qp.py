"""Bounded sequential LTV QPs, followed by an independent nonlinear rollout check."""

from dataclasses import asdict
from functools import lru_cache
import time

import numpy as np
import osqp
from scipy import sparse

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from trajectory_generation._utils import build_state, finite_difference, speed_mps, wrap_angle
from .config import GENERATOR_MODE, PREVIEW_STEPS
from .physics import InitialPhysicsError, acceleration_envelope, local_states, validate_rollout


NUMERICAL_REVISION = 'same_qp_unscaled_retry_v1'


def numerical_revision(cfg):
    return 'same_qp_unscaled_retry_inaccurate_v2' if cfg.retry_unscaled_on_inaccurate else NUMERICAL_REVISION


def solve_qp(p, q, matrix, lower, upper, warm_start, cfg, started):
    """Optionally retry numerical nonconvergence with the same QP and warm start."""
    attempts = []
    retry_statuses = {7} if cfg.retry_unscaled_on_max_iterations else set()
    if cfg.retry_unscaled_on_inaccurate:
        retry_statuses.add(2)
    for scaling in (10, 0) if retry_statuses else (10,):
        remaining = cfg.solver_timeout_s - (time.perf_counter() - started)
        if cfg.enforce_wall_time_limits and remaining <= 0:
            raise TimeoutError(f'QP window exceeded its deadline before solve; attempts={attempts}')
        timing = {'time_limit': max(1e-6, remaining)} if cfg.enforce_wall_time_limits else {}
        attempt_started = time.perf_counter()
        solver = osqp.OSQP()
        solver.setup(P=p, q=q, A=matrix, l=lower, u=upper, verbose=False,
            eps_abs=cfg.solver_tolerance, eps_rel=cfg.solver_relative_tolerance,
            max_iter=cfg.max_iterations, polishing=True, warm_starting=True,
            adaptive_rho=True, adaptive_rho_interval=50, scaling=scaling, **timing)
        solver.warm_start(x=warm_start.copy())
        solution = solver.solve()
        info = solution.info
        attempts.append(dict(status=info.status, status_val=int(info.status_val),
            scaling=scaling, iterations=int(info.iter), primal_residual=float(info.prim_res),
            dual_residual=float(info.dual_res), polishing_status=int(info.status_polish),
            rho_updates=int(info.rho_updates), runtime_s=float(info.run_time),
            elapsed_s=time.perf_counter() - attempt_started,
            eps_abs=cfg.solver_tolerance, eps_rel=cfg.solver_relative_tolerance,
            max_iterations=cfg.max_iterations, warm_start='original_nominal'))
        if info.status_val not in retry_statuses:
            break
    return solution, attempts


def step(state, control, dt, wheelbase):
    x, y, heading, speed, steering = state
    return np.array([x + dt * speed * np.cos(heading),
        y + dt * speed * np.sin(heading),
        heading + dt * speed * np.tan(steering) / wheelbase,
        speed + dt * control[0], steering + dt * control[1]])


def linearize(state, control, dt, wheelbase):
    """Affine first-order expansion of the same Euler model used for acceptance."""
    _, _, heading, speed, steering = state
    a, b = np.eye(5), np.zeros((5, 2))
    a[0, 2], a[0, 3] = -dt * speed * np.sin(heading), dt * np.cos(heading)
    a[1, 2], a[1, 3] = dt * speed * np.cos(heading), dt * np.sin(heading)
    a[2, 3] = dt * np.tan(steering) / wheelbase
    a[2, 4] = dt * speed / (wheelbase * np.cos(steering) ** 2)
    b[3, 0], b[4, 1] = dt, dt
    return a, b, step(state, control, dt, wheelbase) - a @ state - b @ control


def rollout(initial, controls, dt, wheelbase):
    states = [np.asarray(initial, dtype=float)]
    for control, interval in zip(controls, dt):
        states.append(step(states[-1], control, interval, wheelbase))
    return np.asarray(states)


@lru_cache(maxsize=128)
def _quadratic_template(intervals, cfg):
    """Cache only reference-independent costs, keyed by exact intervals and config."""
    dt = np.asarray(intervals, dtype=float)
    horizon = len(dt)
    offset = 5 * (horizon + 1)
    size = offset + 2 * horizon
    p = sparse.lil_matrix((size, size))

    def square(terms, weight):
        for i, ai in terms.items():
            for j, aj in terms.items():
                p[i, j] += 2 * weight * ai * aj

    weights = np.tile([cfg.position_weight, cfg.position_weight, cfg.heading_weight,
                       cfg.speed_weight, cfg.steering_weight], (horizon + 1, 1))
    weights[-1] += cfg.terminal_weight * np.array([1., 1., 1., .25, 0.])
    for i, w in enumerate(weights):
        for j in range(5):
            square({5 * i + j: 1.}, w[j])
    for i, interval in enumerate(dt):
        j = offset + 2 * i
        square({j: 1.}, cfg.acceleration_weight)
        square({j + 1: 1.}, cfg.steering_rate_weight)
        square({j: 1.} if i == 0 else {j: 1., j - 2: -1.},
               cfg.jerk_weight / interval ** 2)
        if i:
            square({j + 1: 1., j - 1: -1.}, cfg.steering_rate_delta_weight / interval ** 2)
    matrix = sparse.triu(p.tocsc(), format='csc')
    for array in (matrix.data, matrix.indices, matrix.indptr, weights):
        array.flags.writeable = False
    return matrix, weights


def formulate(reference, nominal, controls, dt, wheelbase, initial_accel, cfg, limits):
    """Return the same QP, reusing its immutable quadratic cost template."""
    horizon = len(dt)
    offset = 5 * (horizon + 1)
    size = offset + 2 * horizon
    p, weights = _quadratic_template(tuple(dt), cfg)
    q = np.zeros(size)
    q[:offset] -= (2 * weights * reference).reshape(-1)
    q[offset] -= 2 * (cfg.jerk_weight / dt[0] ** 2) * initial_accel

    rows, columns, values, lower, upper = [], [], [], [], []

    def constraint(terms, lo, hi):
        row = len(lower)
        for j, value in terms.items():
            if value:
                rows.append(row)
                columns.append(j)
                values.append(value)
        lower.append(lo)
        upper.append(hi)

    for j in range(5):
        constraint({j: 1.}, nominal[0, j], nominal[0, j])
    accel_lo, accel_hi = acceleration_envelope(initial_accel, dt, limits, cfg)
    steering_limit = min(limits.max_steering_angle_rad, cfg.max_steering_angle_rad)
    lateral_limit = limits.max_lateral_accel_mps2 - cfg.lateral_margin_mps2
    for i, interval in enumerate(dt):
        a, b, c = linearize(nominal[i], controls[i], interval, wheelbase)
        for j in range(5):
            terms = {5 * (i + 1) + j: 1.}
            terms.update({5 * i + k: -a[j, k] for k in range(5)})
            terms.update({offset + 2 * i + k: -b[j, k] for k in range(2)})
            constraint(terms, c[j], c[j])
        constraint({offset + 2 * i: 1.}, accel_lo[i], accel_hi[i])
        constraint({offset + 2 * i + 1: 1.}, -cfg.max_steering_rate_rps, cfg.max_steering_rate_rps)
        j = 5 * (i + 1)
        center = nominal[i + 1]
        constraint({j + 2: 1.}, center[2] - cfg.heading_trust_rad, center[2] + cfg.heading_trust_rad)
        constraint({j + 3: 1.}, max(0., center[3] - cfg.speed_trust_mps), center[3] + cfg.speed_trust_mps)
        constraint({j + 4: 1.}, max(-steering_limit, center[4] - cfg.steering_trust_rad),
                   min(steering_limit, center[4] + cfg.steering_trust_rad))
        v, delta = center[3:5]
        g = v * v * np.tan(delta) / wheelbase
        gv, gd = 2 * v * np.tan(delta) / wheelbase, v * v / (wheelbase * np.cos(delta) ** 2)
        constant = g - gv * v - gd * delta
        constraint({j + 3: gv, j + 4: gd}, -lateral_limit - constant, lateral_limit - constant)
    matrix = sparse.csc_matrix((values, (rows, columns)), shape=(len(lower), size))
    # Keep caller ownership: OSQP or tests must not mutate the cached template.
    return p.copy(), q, matrix, np.asarray(lower), np.asarray(upper)


class WindowQPSolver:
    def __init__(self, config, limits):
        self.config, self.limits = config, limits
        self._config_dict = asdict(config)

    def solve_window(self, target, initial, warm_controls=None):
        started = time.perf_counter()
        cfg = self.config
        templates = list(target) if isinstance(target, (list, tuple)) else list(target.get_sampled_trajectory())
        horizon = len(templates) - 1
        if not 1 <= horizon <= cfg.horizon_steps or templates[0].time_point != initial.time_point:
            raise ValueError('Invalid QP window size or timestamp')
        if not all(isinstance(state, initial.__class__) for state in templates):
            raise TypeError('QP reference states must have the initial state type')
        times = np.array([s.time_point.time_us - initial.time_point.time_us for s in templates]) * 1e-6
        dt, reference = np.diff(times), local_states(templates, initial)
        wheelbase = float(initial.car_footprint.vehicle_parameters.wheel_base)
        z0 = np.array([0., 0., 0., speed_mps(initial), initial.tire_steering_angle])
        a0 = float(initial.dynamic_car_state.rear_axle_acceleration_2d.x)
        if (not np.isfinite(reference).all() or not np.isfinite(z0).all()
                or not np.all(dt > 0) or not np.isfinite(a0) or not np.isfinite(wheelbase) or wheelbase <= 0):
            raise ValueError('Nonfinite QP input or invalid time/vehicle geometry')
        if (z0[3] < 0 or abs(z0[4]) > min(cfg.max_steering_angle_rad, self.limits.max_steering_angle_rad)
                + cfg.validation_tolerance
                or abs(z0[3] ** 2 * np.tan(z0[4]) / wheelbase) > self.limits.max_lateral_accel_mps2 + cfg.validation_tolerance):
            raise InitialPhysicsError('Initial measured state is outside the physical envelope')
        lo, hi = acceleration_envelope(a0, dt, self.limits, cfg)
        if warm_controls is not None:
            controls = np.asarray(warm_controls, dtype=float).copy()
            if controls.shape != (horizon, 2) or not np.isfinite(controls).all():
                raise ValueError('Invalid per-origin shifted controls')
        else:
            controls = np.zeros((horizon, 2))
        speed, steering = z0[3:5]
        steering_limit = min(cfg.max_steering_angle_rad, self.limits.max_steering_angle_rad)
        for i, interval in enumerate(dt):
            if warm_controls is None:
                # A saturated first difference loses its unexecuted remainder.
                # Track each absolute reference value from the realized seed state.
                controls[i] = [(reference[i + 1, 3] - speed) / interval,
                               (reference[i + 1, 4] - steering) / interval]
            controls[i, 0] = np.clip(controls[i, 0], max(lo[i], -speed / interval), hi[i])
            controls[i, 1] = np.clip(controls[i, 1],
                max(-cfg.max_steering_rate_rps, (-steering_limit - steering) / interval),
                min(cfg.max_steering_rate_rps, (steering_limit - steering) / interval))
            speed += controls[i, 0] * interval
            steering += controls[i, 1] * interval
        nominal = rollout(z0, controls, dt, wheelbase)
        solves, corrections = [], 0
        max_correction = 0.
        for outer in range(cfg.outer_iterations):
            remaining = cfg.solver_timeout_s - (time.perf_counter() - started)
            if cfg.enforce_wall_time_limits and remaining <= 0:
                raise TimeoutError('QP window exceeded its complete generation deadline')
            p, q, matrix, lower, upper = formulate(reference, nominal, controls, dt, wheelbase, a0, cfg, self.limits)
            solution, attempts = solve_qp(p, q, matrix, lower, upper,
                np.r_[nominal.reshape(-1), controls.reshape(-1)], cfg, started)
            if solution.info.status_val != 1 or solution.x is None or not np.isfinite(solution.x).all():
                raise RuntimeError(f'QP solve failed: {solution.info.status}; outer={outer + 1}; '
                    f'primal_residual={solution.info.prim_res}; dual_residual={solution.info.dual_res}; '
                    f'polishing_status={solution.info.status_polish}; iterations={solution.info.iter}; '
                    f'attempts={attempts}')
            lhs = matrix @ solution.x
            violation = float(max(0., np.max(lower - lhs), np.max(lhs - upper)))
            if violation > cfg.affine_tolerance:
                raise RuntimeError(f'QP affine constraints failed independent check: {violation}')
            linear = solution.x[:5 * (horizon + 1)].reshape(-1, 5)
            controls = solution.x[5 * (horizon + 1):].reshape(-1, 2).copy()
            speed = z0[3]
            for i, interval in enumerate(dt):
                next_speed = speed + controls[i, 0] * interval
                if next_speed < 0:
                    if next_speed < -cfg.affine_tolerance:
                        raise RuntimeError('Negative QP speed beyond solver roundoff')
                    change = -speed / interval - controls[i, 0]
                    controls[i, 0] += change
                    corrections += 1
                    max_correction = max(max_correction, abs(change))
                speed = max(0., speed + controls[i, 0] * interval)
            nonlinear = rollout(z0, controls, dt, wheelbase)
            validation = validate_rollout(nonlinear, controls, reference, dt, wheelbase, a0, cfg, self.limits)
            mismatch = float(np.max(np.abs(nonlinear[:PREVIEW_STEPS + 1, :3] - linear[:PREVIEW_STEPS + 1, :3])))
            if not np.isfinite(mismatch):
                raise RuntimeError('Nonfinite nonlinear/affine model disagreement')
            solves.append(dict(status=solution.info.status,
                iterations=sum(a['iterations'] for a in attempts), accepted_iterations=int(solution.info.iter),
                attempts=attempts, numerical_retry_count=len(attempts) - 1,
                numerical_revision=numerical_revision(cfg), retry_uses_same_qp_and_original_warm_start=True,
                affine_constraint_violation=violation, linearization_error=mismatch,
                full_prediction_linearization_error=float(np.max(np.abs(nonlinear[:, :3] - linear[:, :3]))),
                nonlinear_physics_accepted=validation['accepted'],
                primal_residual=float(solution.info.prim_res), dual_residual=float(solution.info.dual_res),
                polishing_status=int(solution.info.status_polish),
                objective=float(solution.info.obj_val), runtime_s=sum(a['runtime_s'] for a in attempts),
                attempts_elapsed_s=sum(a['elapsed_s'] for a in attempts)))
            nominal = nonlinear
            if validation['accepted'] and mismatch <= cfg.linearization_tolerance:
                break
        else:
            if not validation['accepted'] or cfg.enforce_linearization_tolerance:
                raise RuntimeError(f'QP outer budget exhausted: physics={validation["rejection_reason"]}, model_error={mismatch}; solves={solves}')
        angle = initial.rear_axle.heading
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        xy = nonlinear[:, :2] @ rotation.T + [initial.rear_axle.x, initial.rear_axle.y]
        heading = wrap_angle(nonlinear[:, 2] + angle)
        yaw = nonlinear[:, 3] * np.tan(nonlinear[:, 4]) / wheelbase
        yaw_accel = finite_difference(yaw, times)
        accel = np.r_[a0, controls[:, 0]]
        rate = np.r_[initial.dynamic_car_state.tire_steering_rate, controls[:, 1]]
        returned = [initial] + [build_state(templates[i], StateSE2(*xy[i], heading[i]), nonlinear[i, 3],
            accel[i], nonlinear[i, 4], rate[i], yaw[i], yaw_accel[i]) for i in range(1, len(templates))]
        validation = validate_rollout(local_states(returned, initial), controls, reference, dt, wheelbase, a0, cfg, self.limits)
        if not validation['accepted']:
            raise RuntimeError('Returned QP rollout failed physics: ' + str(validation))
        elapsed = time.perf_counter() - started
        if cfg.enforce_wall_time_limits and elapsed > cfg.solver_timeout_s:
            raise TimeoutError('QP window exceeded its deadline including validation')
        info = dict(validation, generator_version=GENERATOR_MODE, config=self._config_dict.copy(),
            solver='osqp_bounded_sequential_ltv_qp', solver_success=True, solver_status='solved',
            solver_iterations=sum(s['iterations'] for s in solves), outer_iterations=len(solves), solves=solves,
            solver_elapsed_s=elapsed, horizon_steps=horizon, full_horizon_validated=True,
            wall_time_limits_enforced=cfg.enforce_wall_time_limits,
            linearization_limit_enforced=cfg.enforce_linearization_tolerance,
            linearization_target=cfg.linearization_tolerance,
            linearization_target_met=mismatch <= cfg.linearization_tolerance,
            linearization_acceptance=('converged' if mismatch <= cfg.linearization_tolerance
                                      else 'outer_budget_exhausted_physics_valid'),
            solver_time_reference_s=cfg.solver_timeout_s,
            solver_time_reference_exceeded=elapsed > cfg.solver_timeout_s,
            suffix_appended=False, endpoint_overwritten=False, coordinate_frame='ego_local',
            warm_start_source='shifted_within_origin' if warm_controls is not None else 'bounded_reference_tracking_rollout',
            tracking_constraint_mode='quadratic_cost_only_no_hinge_or_hard_tubes',
            numerical_control_bound_corrections=corrections, max_numerical_accel_correction_mps2=max_correction)
        return InterpolatedTrajectory(returned), controls, info
