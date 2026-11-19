"""One reference-only trajectory generator; each prediction origin owns a session."""
from __future__ import annotations

from typing import Sequence

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from .config import TrajectoryGenerationConfig
from .qp import WindowQPSolver
from .retiming import RetimedPathGenerator
from .session import ReferenceSession


class TrajectoryGenerationPipeline:
    def __init__(self, config: TrajectoryGenerationConfig | None = None):
        self.config = TrajectoryGenerationConfig() if config is None else config
        if not isinstance(self.config, TrajectoryGenerationConfig):
            raise TypeError("config must be a TrajectoryGenerationConfig; use from_dict() for a saved config")
        self._solver = WindowQPSolver(self.config.kmpc_qp, self.config.kinematic)
        self._retimer = RetimedPathGenerator(self.config.kinematic, self.config.kmpc_qp)

    def start_session(
        self,
        trajectory: AbstractTrajectory,
        initial_ego_state: EgoState,
        *,
        feedback_times_us: Sequence[int] | None = None,
    ) -> ReferenceSession:
        """Start at the logged state; measured GT timestamps may be supplied."""
        return ReferenceSession(self._solver, self._retimer, trajectory, initial_ego_state, feedback_times_us)
