"""Generate time-indexed references and reuse the unmodified suffix between solves."""

from dataclasses import replace
import time

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from .config import GENERATOR_MODE, PREVIEW_STEPS
from .retiming import anchor_current_state
from .result import TrajectoryGenerationResult


RAW_TIMESTAMP_ROUNDOFF_US = 2
GT_INTERVAL_JITTER_US = 2000


def normalize_raw_times(raw, initial):
    states = list(raw.get_sampled_trajectory())
    expected = initial.time_point.time_us + np.arange(len(states), dtype=np.int64) * 100000
    errors = np.asarray([s.time_point.time_us for s in states], dtype=np.int64) - expected
    if len(states) < 2 or np.max(np.abs(errors)) > RAW_TIMESTAMP_ROUNDOFF_US:
        raise ValueError('Raw prediction is not a 0.1s grid within the 2us serialization tolerance')
    corrected = [s if s.time_point.time_us == int(t) else EgoState(
        car_footprint=s.car_footprint, dynamic_car_state=s.dynamic_car_state,
        tire_steering_angle=s.tire_steering_angle, is_in_auto_mode=s.is_in_auto_mode,
        time_point=TimePoint(int(t))) for s, t in zip(states, expected)]
    return InterpolatedTrajectory(corrected), dict(policy='integer_100000us_grid; state_values_unchanged',
        max_timestamp_correction_us=int(np.max(np.abs(errors))), corrected_samples=int(np.count_nonzero(errors)),
        raw_end_time_us=raw.end_time.time_us, normalized_end_time_us=int(expected[-1]))


class TrackingPreview(AbstractTrajectory):
    def __init__(self, trajectory):
        self.tracking_reference_trajectory = trajectory

    @property
    def start_time(self):
        return self.tracking_reference_trajectory.start_time

    @property
    def end_time(self):
        return self.tracking_reference_trajectory.end_time

    def get_state_at_time(self, time_point):
        return self.tracking_reference_trajectory.get_state_at_time(time_point)

    def get_state_at_times(self, time_points):
        return self.tracking_reference_trajectory.get_state_at_times(time_points)

    def get_sampled_trajectory(self):
        return self.tracking_reference_trajectory.get_sampled_trajectory()


class HeldReferenceView(AbstractTrajectory):
    """Expose an unchanged suffix of a solved window, without extrapolation."""

    def __init__(self, trajectory, time_point):
        self._trajectory = trajectory
        self._states = [s for s in trajectory.get_sampled_trajectory() if s.time_point >= time_point]
        if len(self._states) < 2 or self._states[0].time_point != time_point:
            raise ValueError('Held reference does not cover the current control step')

    @property
    def start_time(self):
        return self._states[0].time_point

    @property
    def end_time(self):
        return self._states[-1].time_point

    def get_state_at_time(self, time_point):
        if not self.start_time <= time_point <= self.end_time:
            raise ValueError('Held reference interpolation must remain within its solved window')
        return self._trajectory.get_state_at_time(time_point)

    def get_state_at_times(self, time_points):
        if any(not self.start_time <= t <= self.end_time for t in time_points):
            raise ValueError('Held reference interpolation must remain within its solved window')
        return self._trajectory.get_state_at_times(time_points)

    def get_sampled_trajectory(self):
        return list(self._states)


class ReferenceSession:
    def __init__(self, solver, retimer, raw, initial, feedback_times_us=None):
        started = time.perf_counter()
        self.solver, self.raw = solver, raw
        if feedback_times_us is None:
            raw, self.timestamp_info = normalize_raw_times(raw, initial)
        else:
            times = np.asarray(feedback_times_us, dtype=np.int64)
            if (len(times) < 2 or times[0] != initial.time_point.time_us or times[-1] != raw.end_time.time_us
                    or np.max(np.abs(np.diff(times) - 100000)) > GT_INTERVAL_JITTER_US):
                raise ValueError('GT feedback timeline must retain its measured nominal-10Hz timestamps')
            self.timestamp_info = dict(policy='measured_GT_timestamps; canonical_target_interpolated',
                max_interval_jitter_us=int(np.max(np.abs(np.diff(times) - 100000))),
                max_timestamp_correction_us=0, corrected_samples=0,
                raw_end_time_us=raw.end_time.time_us, normalized_end_time_us=raw.end_time.time_us)
        self.anchored, self.anchor = anchor_current_state(raw, initial)
        self.target, self.retiming = retimer.generate(self.anchored, initial)
        self.targets = (list(self.target.get_sampled_trajectory()) if feedback_times_us is None
            else list(self.target.get_state_at_times([TimePoint(int(t)) for t in feedback_times_us])))
        self.origin_us = initial.time_point.time_us
        self.times = np.array([s.time_point.time_us for s in self.targets], dtype=np.int64)
        if (self.times[0] != self.origin_us or len(self.times) < 2
                or self.times[-1] != raw.end_time.time_us
                or np.max(np.abs(np.diff(self.times) - 100000)) > (2 if feedback_times_us is None else GT_INTERVAL_JITTER_US)):
            raise ValueError('Short QP requires an endpoint-preserving 0.1s target grid')
        self.previous, self.last_index = None, None
        self._held_result = None
        self._reference_state = initial
        self.preparation_s = time.perf_counter() - started

    @property
    def reference_state(self):
        return self._reference_state

    def generate_reference(self, time_point):
        if not isinstance(time_point, TimePoint):
            raise TypeError('Reference-only KMPC requires a TimePoint, never an EgoState')
        if time_point != self._reference_state.time_point:
            raise ValueError('Reference time must advance exactly one model step from the origin')
        if time_point.time_us >= self.times[-1]:
            raise ValueError('Reference time is outside this origin')
        started = time.perf_counter()
        index = int(np.argmin(np.abs(self.times - time_point.time_us)))
        period = self.solver.config.reference_update_steps
        refreshed = self.last_index is None or index - self.last_index >= period
        if refreshed:
            result = self._generate(self._reference_state)
            self._held_result = result
        else:
            source = self._held_result
            view = HeldReferenceView(source.trajectory, time_point)
            info = dict(source.diagnostics['kmpc'])
            age = index - self.last_index
            # Reuse the source solve's validation, but never count it as a new solve.
            info.update(horizon_steps=len(view.get_sampled_trajectory()) - 1,
                solver_iterations=0, outer_iterations=0, solves=[], solver_elapsed_s=0.,
                solver_time_reference_exceeded=False,
                linearization_checked_steps=max(0, source.diagnostics['kmpc']['linearization_checked_steps'] - age))
            diagnostics = dict(source.diagnostics, kmpc=info, step=index)
            result = replace(source, trajectory=TrackingPreview(view), diagnostics=diagnostics)
        available = len(result.trajectory.get_sampled_trajectory()) - 1
        original_remaining = len(self.targets) - 1 - index
        repeats = max(0, PREVIEW_STEPS - available)
        result.diagnostics.update(reference_update_steps=period,
            qp_refreshed=refreshed, source_window_step=self.last_index,
            source_window_time_us=int(self.times[self.last_index]), reference_age_steps=index - self.last_index,
            available_future_steps=available, terminal_preview_repeats=repeats,
            extra_preview_repeats=max(0, repeats - max(0, PREVIEW_STEPS - original_remaining)),
            total_elapsed_s=time.perf_counter() - started,
            validation_source='current_solve' if refreshed else 'unchanged_suffix_of_source_solve')
        self._reference_state = result.trajectory.get_sampled_trajectory()[1]
        return result

    def _generate(self, state):
        started = time.perf_counter()
        index = int(np.argmin(np.abs(self.times - state.time_point.time_us)))
        if abs(int(self.times[index]) - state.time_point.time_us) > 2 or index >= len(self.targets) - 1:
            raise ValueError('Reference time is outside this origin; no padding or extrapolation')
        shift = self.solver.config.reference_update_steps
        if self.last_index is not None and index != self.last_index + shift:
            raise ValueError('Short QP refresh must follow its declared update period')
        count = min(self.solver.config.horizon_steps, len(self.targets) - index - 1)
        target_states = self.targets[index:index + count + 1]
        if target_states[0].time_point != state.time_point:
            raise ValueError('Target and reference timestamps differ')
        warm = None
        if self.previous is not None:
            remaining = self.previous[shift:]
            if len(remaining):
                warm = np.vstack((remaining, np.repeat(remaining[-1:], max(0, count - len(remaining)), axis=0)))[:count]
        trajectory, controls, info = self.solver.solve_window(target_states, state, warm)
        self.previous, self.last_index = controls, index
        info.update(generator_version=GENERATOR_MODE, linearization_checked_steps=min(count, PREVIEW_STEPS),
            linearization_scope=('AR-PPO preview, max of separate x/y metres and heading radians; '
                                 f'target={self.solver.config.linearization_tolerance}; '
                                 f'enforced={self.solver.config.enforce_linearization_tolerance}'))
        preview = TrackingPreview(trajectory)
        return TrajectoryGenerationResult(trajectory=preview, raw_trajectory=self.raw,
            estimated_trajectory=self.anchored, kinematic_trajectory=self.target,
            requested_mode=GENERATOR_MODE, applied_mode=GENERATOR_MODE,
            diagnostics=dict(kmpc=info, step=index, canonical_reference_preparations=1,
                estimate_calls=0, estimate_bypassed=True,
                timestamps=self.timestamp_info,
                reference_only=True,
                current_error_reference='kmpc_reference_state',
                future_reference='nonlinear_QP_rollout',
                input_preview_steps=PREVIEW_STEPS, available_future_steps=count,
                terminal_preview_repeats=max(0, PREVIEW_STEPS - count),
                total_elapsed_s=time.perf_counter() - started))
