"""Receding-horizon MPC controller built on cvxpy.

The controller solves a horizon-`H` optimization at each control step,
applies the first action, then re-solves on the next step with new
measurements. See `docs/algorithm.md` for the full formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
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
    voltage: float
    battery_capacity_kwh: float
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
        cfg = self.config
        horizon = cfg.horizon_steps
        step_hours = cfg.step_minutes / 60.0

        if len(inputs.solar_kw) != horizon:
            raise ValueError(
                f"solar_kw length {len(inputs.solar_kw)} != horizon_steps {horizon}"
            )

        charge_amp = cp.Variable(horizon, nonneg=True)
        charge_kw = charge_amp * cfg.voltage / 1000.0
        soc = cp.cumsum(charge_kw * step_hours / cfg.battery_capacity_kwh) + inputs.soc_now

        net_load = inputs.load_kw + charge_kw - inputs.solar_kw
        grid_import = cp.pos(net_load)
        energy_cost = cp.sum(cp.multiply(inputs.grid_price, grid_import) * step_hours)

        smoothness = cp.sum_squares(cp.diff(charge_amp))
        terminal_bonus = -cfg.soc_terminal_weight * soc[horizon - 1]

        objective = cp.Minimize(
            energy_cost + cfg.smoothness_weight * smoothness + terminal_bonus
        )

        constraints = [
            charge_amp <= cfg.amp_max,
            soc <= 1.0,
            cp.abs(cp.diff(charge_amp)) <= cfg.delta_amp_max,
        ]

        if 0 < inputs.deadline_step <= horizon:
            constraints.append(soc[inputs.deadline_step - 1] >= cfg.soc_target)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        if problem.status not in ("optimal", "optimal_inaccurate") or charge_amp.value is None:
            fallback = np.full(horizon, cfg.amp_max)
            return ControlAction(
                amperage=cfg.amp_max,
                horizon_plan=fallback,
                objective_value=float("inf"),
            )

        plan = np.asarray(charge_amp.value, dtype=float).flatten()
        plan = np.clip(plan, 0.0, cfg.amp_max)
        return ControlAction(
            amperage=float(plan[0]),
            horizon_plan=plan,
            objective_value=float(problem.value),
        )
