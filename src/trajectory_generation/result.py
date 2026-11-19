from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


@dataclass(frozen=True)
class TrajectoryGenerationResult:
    trajectory: AbstractTrajectory
    requested_mode: str
    applied_mode: str
    fallback_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    raw_trajectory: AbstractTrajectory | None = None
    estimated_trajectory: AbstractTrajectory | None = None
    kinematic_trajectory: AbstractTrajectory | None = None

    @property
    def used_fallback(self) -> bool:
        return self.applied_mode != self.requested_mode

    def summary(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "applied_mode": self.applied_mode,
            "fallback_reason": self.fallback_reason,
            "used_fallback": self.used_fallback,
            "diagnostics": self.diagnostics,
        }
