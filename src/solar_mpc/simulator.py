"""Rolling-horizon simulator.

Drives any controller (`MPCController`, `GreedyController`, `TouRuleController`)
across a recorded or synthetic trace, advancing real SoC by the realized
charge and recording per-step state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .controller import ControllerConfig, ControllerInputs


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
        charge_kw = amp * cfg.voltage / 1000.0
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
