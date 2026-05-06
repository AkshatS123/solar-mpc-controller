"""Receding-horizon MPC controller built on cvxpy.

The controller solves a horizon-`H` optimization at each control step,
applies the first action, then re-solves on the next step with new
measurements. See `docs/algorithm.md` for the full formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    """Static configuration for an MPC instance.

    All fields are immutable so a single config can be reused across rolling steps.
    """

    horizon_steps: int
    step_minutes: int
    amp_max: float
    amp_step: float
    delta_amp_max: float
    soc_target: float
    smoothness_weight: float
    soc_terminal_weight: float


@dataclass(frozen=True)
class ControllerInputs:
    """Per-step exogenous inputs to the controller."""

    solar_kw: np.ndarray
    load_kw: np.ndarray
    grid_price: np.ndarray
    soc_now: float
    deadline_step: int


@dataclass(frozen=True)
class ControlAction:
    """The action returned at each step."""

    amperage: float
    horizon_plan: np.ndarray
    objective_value: float


class MPCController:
    """Receding-horizon controller. See module docstring for the formulation."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def step(self, inputs: ControllerInputs) -> ControlAction:
        raise NotImplementedError("Phase 1: cvxpy formulation goes here.")
