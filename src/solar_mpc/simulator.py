"""Rolling-horizon simulator.

Drives any controller (`MPCController`, `GreedyController`, `TouRuleController`)
across a recorded or synthetic trace and records cost, grid import, SoC, and
amperage at each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SimulationResult:
    """Per-step trace produced by `simulate()`."""

    history: pd.DataFrame
    total_cost: float
    final_soc: float
    deadline_met: bool


def simulate(controller, trace: pd.DataFrame, *, soc_initial: float) -> SimulationResult:
    """Run `controller` over `trace`, returning per-step history.

    `controller` must expose `step(inputs) -> ControlAction`.
    `trace` must have columns: timestamp, solar_kw, load_kw, grid_price.
    """
    raise NotImplementedError("Phase 2: rolling-horizon driver.")
