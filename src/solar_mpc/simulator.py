"""Rolling-horizon simulators.

Two flavors:

- `simulate` — deterministic trace; controller sees and gets the same
  values. Used in Phases 1–3.
- `simulate_under_noise` — independent realization vs forecast. The
  controller sees a `ForecastBundle` (point or scenario set). The state
  advances using a *separately sampled* realization of solar/load. This
  is the test of robustness to forecast error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .controller import ControllerConfig, ControllerInputs
from .noise import NoiseSpec, generate_scenarios, realize_trace


@dataclass(frozen=True)
class SimulationResult:
    """Per-step trace produced by `simulate()`."""

    history: pd.DataFrame
    total_cost: float
    final_soc: float
    deadline_met: bool


def simulate(
    controller,
    trace: pd.DataFrame,
    *,
    soc_initial: float,
    deadline_index: int | None = None,
) -> SimulationResult:
    """Run `controller` over `trace`, returning per-step history.

    `controller` must expose `.config: ControllerConfig` and
    `step(inputs) -> ControlAction`.
    """
    cfg: ControllerConfig = controller.config
    horizon = cfg.horizon_steps
    step_hours = cfg.step_minutes / 60.0
    n_steps = len(trace) - horizon
    if n_steps < 1:
        raise ValueError(
            f"Trace length {len(trace)} too short for horizon {horizon}"
        )

    rows: list[dict] = []
    soc = soc_initial
    total_cost = 0.0

    for i in range(n_steps):
        window = trace.iloc[i : i + horizon]
        deadline_step = (
            min(deadline_index - i, horizon)
            if deadline_index is not None and deadline_index > i
            else horizon
        )

        inputs = ControllerInputs(
            solar_kw=window["solar_kw"].to_numpy(),
            load_kw=window["load_kw"].to_numpy(),
            grid_price=window["grid_price"].to_numpy(),
            soc_now=soc,
            deadline_step=deadline_step,
        )
        action = controller.step(inputs)

        amp = action.amperage
        commanded_charge_kw = amp * cfg.voltage / 1000.0
        # The EV's BMS refuses current at full SoC: the actual flow is
        # capped by remaining capacity for this step.
        remaining_kwh = max(0.0, (1.0 - soc) * cfg.battery_capacity_kwh)
        realized_charge_kwh = min(commanded_charge_kw * step_hours, remaining_kwh)
        charge_kw = realized_charge_kwh / step_hours if step_hours > 0 else 0.0

        net_load = float(trace.iloc[i]["load_kw"] + charge_kw - trace.iloc[i]["solar_kw"])
        grid_import = max(0.0, net_load)
        cost = grid_import * float(trace.iloc[i]["grid_price"]) * step_hours
        total_cost += cost

        new_soc = float(np.clip(soc + charge_kw * step_hours / cfg.battery_capacity_kwh, 0.0, 1.0))

        rows.append(
            {
                "timestamp": trace.iloc[i]["timestamp"],
                "solar_kw": float(trace.iloc[i]["solar_kw"]),
                "load_kw": float(trace.iloc[i]["load_kw"]),
                "grid_price": float(trace.iloc[i]["grid_price"]),
                "amperage": amp,
                "charge_kw": charge_kw,
                "grid_import_kw": grid_import,
                "cost": cost,
                "soc_pre": soc,
                "soc_post": new_soc,
            }
        )
        soc = new_soc

    history = pd.DataFrame(rows)
    return SimulationResult(
        history=history,
        total_cost=total_cost,
        final_soc=soc,
        deadline_met=bool(soc >= cfg.soc_target),
    )


def simulate_under_noise(
    controller,
    point_trace: pd.DataFrame,
    *,
    soc_initial: float,
    n_realizations: int,
    seed: int,
    noise_spec: NoiseSpec | None = None,
    n_scenarios: int = 10,
    deadline_index: int | None = None,
) -> list[SimulationResult]:
    """Run `controller` over `n_realizations` noisy realizations.

    Per realization:
    - One sample of (solar, load) from `noise_spec` defines what *actually*
      unfolds at each step.
    - At each control step, a fresh batch of `n_scenarios` is drawn around
      the point forecast `point_trace`. This is what the controller sees
      via `controller.step_bundle(...)`.
    - Realization seeds and scenario seeds come from independent rng
      streams so the controller never sees the realization.

    Returns one `SimulationResult` per realization.
    """
    spec = noise_spec or NoiseSpec()
    cfg: ControllerConfig = controller.config
    horizon = cfg.horizon_steps
    step_hours = cfg.step_minutes / 60.0
    n_steps = len(point_trace) - horizon
    if n_steps < 1:
        raise ValueError(f"Trace length {len(point_trace)} too short for horizon {horizon}")

    rng = np.random.default_rng(seed)
    realization_seeds = rng.integers(0, 2**31 - 1, size=n_realizations).tolist()
    scenario_seeds = rng.integers(0, 2**31 - 1, size=n_realizations).tolist()

    results: list[SimulationResult] = []
    point_solar = point_trace["solar_kw"].to_numpy()
    point_load = point_trace["load_kw"].to_numpy()
    grid_price = point_trace["grid_price"].to_numpy()

    for r_seed, s_seed in zip(realization_seeds, scenario_seeds, strict=True):
        realized_solar, realized_load = realize_trace(
            point_solar, point_load, seed=int(r_seed), noise_spec=spec
        )
        scenario_rng = np.random.default_rng(int(s_seed))

        rows: list[dict] = []
        soc = soc_initial
        total_cost = 0.0

        for i in range(n_steps):
            sub_seed = int(scenario_rng.integers(0, 2**31 - 1))
            bundle = generate_scenarios(
                point_solar[i : i + horizon],
                point_load[i : i + horizon],
                grid_price[i : i + horizon],
                n_scenarios=n_scenarios,
                seed=sub_seed,
                noise_spec=spec,
            )
            deadline_step = (
                min(deadline_index - i, horizon)
                if deadline_index is not None and deadline_index > i
                else horizon
            )

            action = controller.step_bundle(bundle, soc_now=soc, deadline_step=deadline_step)

            amp = action.amperage
            commanded_kw = amp * cfg.voltage / 1000.0
            remaining_kwh = max(0.0, (1.0 - soc) * cfg.battery_capacity_kwh)
            realized_charge_kwh = min(commanded_kw * step_hours, remaining_kwh)
            charge_kw = realized_charge_kwh / step_hours if step_hours > 0 else 0.0

            net_load = float(realized_load[i] + charge_kw - realized_solar[i])
            grid_import = max(0.0, net_load)
            cost = grid_import * float(grid_price[i]) * step_hours
            total_cost += cost

            new_soc = float(
                np.clip(soc + charge_kw * step_hours / cfg.battery_capacity_kwh, 0.0, 1.0)
            )

            rows.append(
                {
                    "timestamp": point_trace.iloc[i]["timestamp"],
                    "solar_kw_realized": float(realized_solar[i]),
                    "load_kw_realized": float(realized_load[i]),
                    "grid_price": float(grid_price[i]),
                    "amperage": amp,
                    "charge_kw": charge_kw,
                    "grid_import_kw": grid_import,
                    "cost": cost,
                    "soc_pre": soc,
                    "soc_post": new_soc,
                }
            )
            soc = new_soc

        results.append(
            SimulationResult(
                history=pd.DataFrame(rows),
                total_cost=total_cost,
                final_soc=soc,
                deadline_met=bool(soc >= cfg.soc_target),
            )
        )

    return results
