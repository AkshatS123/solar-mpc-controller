"""Receding-horizon MPC controllers built on cvxpy.

Three controllers live here:

- `MPCController` — nominal, point-forecast input.
- `RobustMPCController` — scenario-based, takes a `ForecastBundle`.
- `MPCControllerInteger` — opt-in MIP variant of the nominal controller
  for discrete amperage. Falls back to round(continuous) if no MIP solver
  is available.

The controller solves a horizon-`H` optimization at each control step,
applies the first action, then re-solves on the next step with new
measurements. See `docs/algorithm.md` for the full formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .forecast import ForecastBundle


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

    def step_bundle(
        self, bundle: ForecastBundle, soc_now: float, deadline_step: int
    ) -> ControlAction:
        """Bundle-aware entry point — collapses scenarios to point forecast."""
        solar_pt, load_pt, price = bundle.point()
        return self.step(
            ControllerInputs(
                solar_kw=solar_pt,
                load_kw=load_pt,
                grid_price=price,
                soc_now=soc_now,
                deadline_step=deadline_step,
            )
        )


class RobustMPCController:
    """Scenario-based robust MPC with optional CVaR risk-aversion.

    Decision variable `a[H]` is shared across all K scenarios
    (non-anticipative first-stage control). The objective is a convex
    combination of expected cost and conditional value-at-risk:

        (1 - λ) · E[cost(a, s_k)]  +  λ · CVaR_α(cost(a, s_k))

    With `cvar_weight = 0` (default) this reduces to expected-cost MPC.
    Set `cvar_weight ∈ (0, 1]` to trade mean cost for tail-cost reduction
    via the standard Rockafellar–Uryasev linear-programming form of CVaR.

    SoC dynamics are deterministic given `a`, so the deadline constraint
    is nominal. CVaR is the lever that makes this controller behave
    differently from nominal MPC under unbiased noise.
    """

    def __init__(
        self,
        config: ControllerConfig,
        *,
        cvar_alpha: float = 0.90,
        cvar_weight: float = 0.0,
    ) -> None:
        if not 0.0 <= cvar_weight <= 1.0:
            raise ValueError(f"cvar_weight must be in [0, 1], got {cvar_weight}")
        if not 0.0 < cvar_alpha < 1.0:
            raise ValueError(f"cvar_alpha must be in (0, 1), got {cvar_alpha}")
        self.config = config
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight

    def step(self, inputs: ControllerInputs) -> ControlAction:
        """Deterministic-trace entry point: degenerate K=1 bundle.

        Lets `simulate()` run RobustMPCController on the same code path as
        nominal MPC — useful for the perfect-forecast comparison and for
        the Tesphase replay.
        """
        bundle = ForecastBundle.degenerate(inputs.solar_kw, inputs.load_kw, inputs.grid_price)
        return self.step_bundle(bundle, soc_now=inputs.soc_now, deadline_step=inputs.deadline_step)

    def step_bundle(
        self, bundle: ForecastBundle, soc_now: float, deadline_step: int
    ) -> ControlAction:
        cfg = self.config
        horizon = cfg.horizon_steps
        step_hours = cfg.step_minutes / 60.0
        n_scenarios = bundle.n_scenarios

        if bundle.horizon != horizon:
            raise ValueError(
                f"Bundle horizon {bundle.horizon} != config.horizon_steps {horizon}"
            )

        charge_amp = cp.Variable(horizon, nonneg=True)
        charge_kw = charge_amp * cfg.voltage / 1000.0
        soc = cp.cumsum(charge_kw * step_hours / cfg.battery_capacity_kwh) + soc_now

        # Per-scenario energy cost expressions
        scenario_costs = []
        for k in range(n_scenarios):
            net_load_k = bundle.load_kw_scenarios[k] + charge_kw - bundle.solar_kw_scenarios[k]
            grid_import_k = cp.pos(net_load_k)
            scenario_costs.append(
                cp.sum(cp.multiply(bundle.grid_price, grid_import_k) * step_hours)
            )
        expected_cost = cp.sum(scenario_costs) / n_scenarios

        # Rockafellar–Uryasev CVaR_α(L) = min_t  t + (1/(1-α)) · E[(L−t)_+]
        if self.cvar_weight > 0.0:
            t_var = cp.Variable()
            tail_excess = cp.sum(
                [cp.pos(scenario_costs[k] - t_var) for k in range(n_scenarios)]
            ) / n_scenarios
            cvar = t_var + (1.0 / (1.0 - self.cvar_alpha)) * tail_excess
            cost_term = (1.0 - self.cvar_weight) * expected_cost + self.cvar_weight * cvar
        else:
            cost_term = expected_cost

        smoothness = cp.sum_squares(cp.diff(charge_amp))
        terminal_bonus = -cfg.soc_terminal_weight * soc[horizon - 1]

        objective = cp.Minimize(
            cost_term + cfg.smoothness_weight * smoothness + terminal_bonus
        )

        constraints = [
            charge_amp <= cfg.amp_max,
            soc <= 1.0,
            cp.abs(cp.diff(charge_amp)) <= cfg.delta_amp_max,
        ]
        if 0 < deadline_step <= horizon:
            constraints.append(soc[deadline_step - 1] >= cfg.soc_target)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        if problem.status not in ("optimal", "optimal_inaccurate") or charge_amp.value is None:
            fallback = np.full(horizon, cfg.amp_max)
            return ControlAction(
                amperage=cfg.amp_max,
                horizon_plan=fallback,
                objective_value=float("inf"),
            )

        plan = np.clip(np.asarray(charge_amp.value, dtype=float).flatten(), 0.0, cfg.amp_max)
        return ControlAction(
            amperage=float(plan[0]),
            horizon_plan=plan,
            objective_value=float(problem.value),
        )


class MPCControllerInteger(MPCController):
    """Opt-in MIP variant: amperage in 1 A integer steps.

    Tries available MIP solvers in order; if none is installed, falls
    back to solving the continuous LP and rounding. Round-fallback is
    intentionally lossy and clearly documented in `docs/algorithm.md`.
    """

    _MIP_SOLVERS: tuple[str, ...] = ("SCIP", "CBC", "GLPK_MI", "GUROBI", "MOSEK")

    def __init__(self, config: ControllerConfig) -> None:
        super().__init__(config)
        installed = set(cp.installed_solvers())
        self.mip_solver: str | None = next(
            (s for s in self._MIP_SOLVERS if s in installed), None
        )

    def step(self, inputs: ControllerInputs) -> ControlAction:  # noqa: D401
        if self.mip_solver is None:
            continuous = super().step(inputs)
            rounded = np.clip(np.round(continuous.horizon_plan), 0.0, self.config.amp_max)
            return ControlAction(
                amperage=float(rounded[0]),
                horizon_plan=rounded,
                objective_value=continuous.objective_value,
            )

        cfg = self.config
        horizon = cfg.horizon_steps
        step_hours = cfg.step_minutes / 60.0

        charge_amp = cp.Variable(horizon, integer=True)
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
            charge_amp >= 0,
            charge_amp <= cfg.amp_max,
            soc <= 1.0,
            cp.abs(cp.diff(charge_amp)) <= cfg.delta_amp_max,
        ]
        if 0 < inputs.deadline_step <= horizon:
            constraints.append(soc[inputs.deadline_step - 1] >= cfg.soc_target)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.mip_solver)

        if problem.status not in ("optimal", "optimal_inaccurate") or charge_amp.value is None:
            return ControlAction(
                amperage=cfg.amp_max,
                horizon_plan=np.full(horizon, cfg.amp_max),
                objective_value=float("inf"),
            )

        plan = np.clip(np.asarray(charge_amp.value, dtype=float).flatten(), 0.0, cfg.amp_max)
        return ControlAction(
            amperage=float(plan[0]),
            horizon_plan=plan,
            objective_value=float(problem.value),
        )
