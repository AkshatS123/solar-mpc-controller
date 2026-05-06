"""Baseline controllers for honest comparison against MPC.

Both baselines expose the same `step(inputs) -> ControlAction` interface
as `MPCController`, so the simulator can swap them without branching.
"""

from __future__ import annotations

import numpy as np

from .controller import ControlAction, ControllerConfig, ControllerInputs


class GreedyController:
    """Charge at maximum amperage whenever solar export exceeds a threshold.

    No look-ahead. The "solar excess only" baseline most DIY home energy
    automations land on.
    """

    def __init__(self, config: ControllerConfig, *, export_threshold_kw: float = 0.5) -> None:
        self.config = config
        self.export_threshold_kw = export_threshold_kw

    def step(self, inputs: ControllerInputs) -> ControlAction:
        excess = float(inputs.solar_kw[0] - inputs.load_kw[0])
        amp = self.config.amp_max if excess > self.export_threshold_kw else 0.0
        plan = np.full(len(inputs.solar_kw), amp)
        return ControlAction(amperage=amp, horizon_plan=plan, objective_value=float("nan"))


class TouRuleController:
    """Price-threshold controller — charges full whenever current price is cheap.

    Despite the name, this generalizes time-of-use: if the trace encodes a
    TOU schedule via the price series, this reduces to "charge during
    off-peak windows."
    """

    def __init__(self, config: ControllerConfig, *, cheap_price_threshold: float) -> None:
        self.config = config
        self.cheap_price_threshold = cheap_price_threshold

    def step(self, inputs: ControllerInputs) -> ControlAction:
        amp = self.config.amp_max if inputs.grid_price[0] <= self.cheap_price_threshold else 0.0
        plan = np.full(len(inputs.solar_kw), amp)
        return ControlAction(amperage=amp, horizon_plan=plan, objective_value=float("nan"))
