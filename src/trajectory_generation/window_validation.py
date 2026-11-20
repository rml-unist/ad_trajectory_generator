"""Compact and verify solver, physics and reference-refresh evidence."""
import numpy as np

from .config import PREVIEW_STEPS
from .qp import numerical_revision

CYCLE_DEADLINE_S = 0.1

def check_solver_attempts(solve, cfg):
    attempts = solve['attempts']
    count = len(attempts)
    retry_statuses = ({7} if cfg.retry_unscaled_on_max_iterations else set()) | ({2} if cfg.retry_unscaled_on_inaccurate else set())
    if (count not in (1, 2) or (count == 2 and not retry_statuses)
            or [a['scaling'] for a in attempts] != ([10] if count == 1 else [10, 0])
            or (count == 2 and attempts[0]['status_val'] not in retry_statuses)
            or attempts[-1]['status_val'] != 1 or attempts[-1]['status'] != 'solved'
            or solve['numerical_revision'] != numerical_revision(cfg)
            or solve['retry_uses_same_qp_and_original_warm_start'] is not True
            or solve['numerical_retry_count'] != count - 1
            or solve['iterations'] != sum(a['iterations'] for a in attempts)
            or solve['accepted_iterations'] != attempts[-1]['iterations']
            or abs(solve['runtime_s'] - sum(a['runtime_s'] for a in attempts)) > 1e-12
            or abs(solve['attempts_elapsed_s'] - sum(a['elapsed_s'] for a in attempts)) > 1e-12):
        raise ValueError('Invalid numerical retry evidence or incomplete iteration/runtime accounting')
    for a in attempts:
        if (a['warm_start'] != 'original_nominal' or a['eps_abs'] != cfg.solver_tolerance
                or a['eps_rel'] != cfg.solver_relative_tolerance or a['max_iterations'] != cfg.max_iterations
                or not 0 <= a['iterations'] <= cfg.max_iterations
                or any(not np.isfinite(a[k]) for k in ('primal_residual', 'dual_residual', 'runtime_s', 'elapsed_s'))
                or a['runtime_s'] < 0 or a['elapsed_s'] < 0):
            raise ValueError('Retry changed solver tolerances/budget or has invalid diagnostics')


def check_windows(windows, cfg, origin_us, *, window_count=80):
    q = cfg.kmpc_qp
    period = q.reference_update_steps
    if window_count not in (1, 80) or [w['step'] for w in windows] != list(range(window_count)):
        raise ValueError('Incomplete reference window evidence')
    for i, item in enumerate(windows):
        source = 'logged_origin' if i == 0 else 'kmpc_model_state'
        source_index = i - i % period
        age = i - source_index
        refreshed = age == 0
        count = min(q.horizon_steps, 80 - source_index) - age
        if (item.get('qp_refreshed', True) is not refreshed
                or item.get('reference_update_steps', 1) != period
                or item.get('source_window_step', i) != source_index
                or item.get('reference_age_steps', 0) != age
                or item.get('extra_preview_repeats', 0) != 0
                or count < min(PREVIEW_STEPS, 80 - i)):
            raise ValueError('Invalid reference refresh/suffix ownership')
        if (item['initial_time_us'] != origin_us + i * 100000 or item['initial_state_source'] != source
                or item['horizon_steps'] != count or not item['accepted'] or not item['solver_success']
                or item['solver_status'] != 'solved' or item['generator_version'] != cfg.mode
                or item['reference_only'] is not q.reference_only
                or not item['hard_physics_validated'] or not item['full_horizon_validated']
                or item['suffix_appended'] or item['endpoint_overwritten']
                or item['max_model_residual'] > q.model_tolerance
                or item['max_abs_lateral_accel_mps2'] > cfg.kinematic.max_lateral_accel_mps2 + q.validation_tolerance
                or item['max_abs_steering_rate_rps'] > q.max_steering_rate_rps + q.validation_tolerance
                or item['linearization_checked_steps'] != max(0, min(PREVIEW_STEPS, count + age) - age)
                or item['terminal_preview_repeats'] != max(0, PREVIEW_STEPS - count)
                or not np.isfinite(item['cycle_elapsed_s'])
                or item['cycle_elapsed_s'] < 0
                or not np.isfinite(item['solver_elapsed_s']) or item['solver_elapsed_s'] < 0
                or item['wall_time_limits_enforced'] is not q.enforce_wall_time_limits
                or item['cycle_time_reference_s'] != CYCLE_DEADLINE_S
                or item['solver_time_reference_s'] != q.solver_timeout_s
                or item['cycle_time_reference_exceeded'] != (item['cycle_elapsed_s'] > CYCLE_DEADLINE_S)
                or item['solver_time_reference_exceeded'] != (item['solver_elapsed_s'] > q.solver_timeout_s)
                or (q.enforce_wall_time_limits and
                    (item['cycle_time_reference_exceeded'] or item['solver_time_reference_exceeded']))
                or any(not np.isfinite(item[key]) for key in
                       ('max_model_residual', 'max_abs_lateral_accel_mps2', 'max_abs_steering_rate_rps'))):
            raise ValueError(f'Invalid feedback/physical/time evidence in cycle {i}')
        if not refreshed:
            original = windows[source_index]
            inherited = ('max_model_residual', 'max_position_error_m', 'max_heading_error_rad',
                         'max_abs_lateral_accel_mps2', 'max_abs_steering_rate_rps',
                         'linearization_limit_enforced', 'linearization_target',
                         'linearization_target_met', 'linearization_acceptance')
            if (item['solves'] or item['solver_iterations'] or item['outer_iterations']
                    or item['solver_elapsed_s'] != 0
                    or item.get('validation_source') != 'unchanged_suffix_of_source_solve'
                    or any(item[key] != original[key] for key in inherited)):
                raise ValueError('Held reference must retain its source validation without a new solve')
            continue
        if (item.get('validation_source', 'current_solve') != 'current_solve'
                or not 1 <= item['outer_iterations'] == len(item['solves']) <= q.outer_iterations
                or any(s['status'] != 'solved' or s['affine_constraint_violation'] > q.affine_tolerance for s in item['solves'])
                or (q.enforce_linearization_tolerance and item['solves'][-1]['linearization_error'] > q.linearization_tolerance)
                or not item['solves'][-1]['nonlinear_physics_accepted']):
            raise ValueError(f'Invalid feedback/physical/solver/time evidence in cycle {i}')
        met = item['solves'][-1]['linearization_error'] <= q.linearization_tolerance
        if (item['linearization_limit_enforced'] is not q.enforce_linearization_tolerance
                or item['linearization_target'] != q.linearization_tolerance
                or item['linearization_target_met'] is not met
                or item['linearization_acceptance'] != ('converged' if met else 'outer_budget_exhausted_physics_valid')
                or (not met and item['outer_iterations'] != q.outer_iterations)
                or any(not np.isfinite(s['linearization_error']) or s['linearization_error'] < 0
                       or not np.isfinite(s['full_prediction_linearization_error']) for s in item['solves'])):
            raise ValueError('Invalid or hidden linearization quality warning')
        for solve in item['solves']:
            check_solver_attempts(solve, q)
        if item['solver_iterations'] != sum(s['iterations'] for s in item['solves']):
            raise ValueError('QP window omitted failed-attempt iterations')


def compact_window(result, cycle_s, state_source):
    info = result.diagnostics['kmpc']
    solves = info['solves']
    return dict(step=result.diagnostics['step'], initial_time_us=result.trajectory.start_time.time_us,
        qp_refreshed=result.diagnostics.get('qp_refreshed', True),
        reference_update_steps=result.diagnostics.get('reference_update_steps', 1),
        source_window_step=result.diagnostics.get('source_window_step', result.diagnostics['step']),
        reference_age_steps=result.diagnostics.get('reference_age_steps', 0),
        extra_preview_repeats=result.diagnostics.get('extra_preview_repeats', 0),
        validation_source=result.diagnostics.get('validation_source', 'current_solve'),
        reference_only=result.diagnostics['reference_only'],
        initial_state_source=state_source, horizon_steps=info['horizon_steps'],
        cycle_elapsed_s=float(cycle_s), generator_elapsed_s=result.diagnostics['total_elapsed_s'],
        wall_time_limits_enforced=info['wall_time_limits_enforced'],
        solver_elapsed_s=info['solver_elapsed_s'], solver_time_reference_s=info['solver_time_reference_s'],
        solver_time_reference_exceeded=info['solver_time_reference_exceeded'],
        cycle_time_reference_s=CYCLE_DEADLINE_S, cycle_time_reference_exceeded=bool(cycle_s > CYCLE_DEADLINE_S),
        accepted=info['accepted'], solver_success=info['solver_success'],
        hard_physics_validated=info['hard_physics_validated'], full_horizon_validated=info['full_horizon_validated'],
        solver_status=info['solver_status'], generator_version=info['generator_version'],
        max_model_residual=info['max_model_residual'], max_position_error_m=info['max_position_error_m'],
        max_heading_error_rad=info['max_heading_error_rad'],
        max_abs_lateral_accel_mps2=info['max_abs_lateral_accel_mps2'],
        max_abs_steering_rate_rps=info['max_abs_steering_rate_rps'],
        outer_iterations=len(solves), solver_iterations=info['solver_iterations'],
        linearization_checked_steps=info['linearization_checked_steps'],
        linearization_limit_enforced=info['linearization_limit_enforced'],
        linearization_target=info['linearization_target'],
        linearization_target_met=info['linearization_target_met'],
        linearization_acceptance=info['linearization_acceptance'],
        terminal_preview_repeats=result.diagnostics['terminal_preview_repeats'],
        solves=solves, suffix_appended=info['suffix_appended'], endpoint_overwritten=info['endpoint_overwritten'])
