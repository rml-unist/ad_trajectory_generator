"""Configuration of the reference-only short-horizon generator."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, get_type_hints


GENERATOR_MODE = "kmpc_qp_short_v1"
PREVIEW_STEPS = 10


@dataclass(frozen=True)
class KinematicGeneratorConfig:
    dt_s: float = 0.1
    max_accel_mps2: float = 1.5
    max_decel_mps2: float = 3.0
    max_lateral_accel_mps2: float = 3.0
    max_jerk_mps3: float = 4.0
    max_steering_angle_rad: float = 0.7
    stop_speed_threshold_mps: float = 0.2
    speed_profile_scale: float = 1.0


@dataclass(frozen=True)
class KMPCQPConfig:
    reference_only: bool = True
    estimate_enabled: bool = False
    horizon_steps: int = 15
    execution_steps: int = 1
    reference_update_steps: int = 5
    lateral_tube_m: float = 0.25
    heading_tube_rad: float = 0.1
    max_steering_angle_rad: float = 0.7
    max_steering_rate_rps: float = 0.5
    position_weight: float = 20.0
    heading_weight: float = 6.0
    speed_weight: float = 2.0
    terminal_weight: float = 30.0
    acceleration_weight: float = 0.02
    steering_rate_weight: float = 0.05
    steering_rate_delta_weight: float = 0.02
    jerk_weight: float = 0.08
    steering_weight: float = 0.005
    solver_timeout_s: float = 0.08
    max_iterations: int = 20000
    outer_iterations: int = 12
    enforce_wall_time_limits: bool = False
    retry_unscaled_on_max_iterations: bool = True
    retry_unscaled_on_inaccurate: bool = True
    enforce_linearization_tolerance: bool = False
    solver_tolerance: float = 1e-5
    solver_relative_tolerance: float = 1e-7
    affine_tolerance: float = 2e-5
    validation_tolerance: float = 1e-3
    model_tolerance: float = 1e-6
    linearization_tolerance: float = 0.03
    speed_trust_mps: float = 3.0
    steering_trust_rad: float = 0.1
    heading_trust_rad: float = 0.5
    lateral_margin_mps2: float = 0.02
    actuator_max_accel_mps2: float = 3.0
    actuator_max_decel_mps2: float = 6.0


def _mapping(value, path, names):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object/mapping")
    unknown = set(value) - set(names)
    if unknown:
        raise ValueError(f"Unknown {path} field(s): {', '.join(sorted(map(str, unknown)))}")
    return dict(value)


def _settings(cls, value, name):
    return _mapping(value, name, (field.name for field in fields(cls)))


def _check_component(component, cls, name):
    if not isinstance(component, cls):
        raise ValueError(f"{name} must be {cls.__name__}")
    for key, expected in get_type_hints(cls).items():
        value = getattr(component, key)
        valid = type(value) in (int, float) if expected is float else type(value) is expected
        if not valid:
            raise ValueError(f"{name}.{key} must be {expected.__name__}")
        if expected is not bool and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name}.{key} must be finite and nonnegative")


_EXAM_ESTIMATOR = dict(mode="lqr_pose_fit", identity_if_valid=True, preserve_endpoint=True,
    speed_mismatch_tolerance_mps=0.75, lqr_discretization_time_s=0.1,
    max_accel_mps2=1.5, max_decel_mps2=3.0, lqr_jerk_penalty=1e-4, lqr_curvature_rate_penalty=1e-2)
_EXAM_KMPC = dict(horizon_steps=30, lateral_tube_m=0.25, heading_tube_rad=0.1,
    max_steering_angle_rad=0.7, max_steering_rate_rps=0.5, solver_timeout_s=1.0, max_iterations=50,
    position_weight=20.0, heading_weight=6.0, speed_weight=2.0, terminal_weight=30.0,
    acceleration_weight=0.02, steering_rate_weight=0.05, steering_rate_delta_weight=0.02, jerk_weight=0.08)


def _archived_qp_settings(payload, qp):
    for name, recorded in (("estimator", _EXAM_ESTIMATOR), ("kmpc", _EXAM_KMPC)):
        section = _mapping(payload.get(name, {}), name, recorded)
        for key, value in section.items():
            if type(value) is not type(recorded[key]) or value != recorded[key]:
                raise ValueError(f"Unsupported archived {name}.{key}")
    if payload.get("estimator", {}).get("preserve_endpoint") is not True:
        raise ValueError("Archived short QP requires endpoint preservation")
    # Only the QP defaults differed in the archived reader.
    return {"reference_update_steps": 1, "max_iterations": 10000, "outer_iterations": 4,
            "solver_timeout_s": 1.0, "steering_trust_rad": 0.2, "enforce_wall_time_limits": True,
            "enforce_linearization_tolerance": True, "retry_unscaled_on_inaccurate": False,
            "retry_unscaled_on_max_iterations": False, **qp}


@dataclass(frozen=True)
class TrajectoryGenerationConfig:
    mode: str = GENERATOR_MODE
    fail_on_fallback: bool = True
    kinematic: KinematicGeneratorConfig = KinematicGeneratorConfig()
    kmpc_qp: KMPCQPConfig = KMPCQPConfig()

    def __post_init__(self):
        if self.mode != GENERATOR_MODE or self.fail_on_fallback is not True:
            raise ValueError(f"Only {GENERATOR_MODE} with fail_on_fallback=true is supported")
        q, k = self.kmpc_qp, self.kinematic
        _check_component(q, KMPCQPConfig, "kmpc_qp")
        _check_component(k, KinematicGeneratorConfig, "kinematic")
        if q.estimate_enabled or not q.reference_only:
            raise ValueError("Only reference_only=true and estimate_enabled=false are supported")
        if (q.horizon_steps not in (10, 15, 20) or q.execution_steps != 1
                or not 1 <= q.reference_update_steps <= q.horizon_steps or abs(k.dt_s - 0.1) > 1e-9):
            raise ValueError("Invalid short QP horizon, refresh period or 0.1s interval")
        if not 0.9 <= k.speed_profile_scale <= 1.0:
            raise ValueError("speed_profile_scale must be in [0.9, 1.0]")
        if (q.outer_iterations < 1 or q.max_iterations < 1 or q.solver_tolerance <= 0
                or q.solver_timeout_s <= 0 or q.model_tolerance <= 0 or q.affine_tolerance <= 0
                or min(q.speed_trust_mps, q.steering_trust_rad, q.heading_trust_rad) <= 0
                or q.lateral_margin_mps2 >= k.max_lateral_accel_mps2
                or q.actuator_max_accel_mps2 < k.max_accel_mps2
                or q.actuator_max_decel_mps2 < k.max_decel_mps2):
            raise ValueError("Invalid QP solver budget, trust region or physical bounds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrajectoryGenerationConfig:
        payload = _mapping(value, "config", ("mode", "fail_on_fallback", "kinematic", "kmpc_qp", "estimator", "kmpc"))
        if payload.get("mode") != GENERATOR_MODE:
            raise ValueError(f"Unsupported trajectory generator mode: {payload.get('mode')!r}")
        kin = _settings(KinematicGeneratorConfig, payload.get("kinematic", {}), "kinematic")
        qp = _settings(KMPCQPConfig, payload.get("kmpc_qp", {}), "kmpc_qp")
        if (qp.get("reference_only") is not True or qp.get("estimate_enabled") is not False
                or "horizon_steps" not in qp or "execution_steps" not in qp):
            raise ValueError("Explicit short-QP horizon, execution_steps, reference_only=true and estimate_enabled=false required")
        if "estimator" in payload or "kmpc" in payload:
            qp = _archived_qp_settings(payload, qp)
        return cls(mode=payload["mode"], fail_on_fallback=payload.get("fail_on_fallback", True),
                   kinematic=KinematicGeneratorConfig(**kin), kmpc_qp=KMPCQPConfig(**qp))
