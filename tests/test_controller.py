"""Placeholder tests for the MPC step. Marked xfail until Phase 1 lands."""

import numpy as np
import pytest

from solar_mpc.controller import ControllerConfig, ControllerInputs, MPCController


@pytest.mark.xfail(reason="Phase 1 — controller.step() not implemented yet", strict=True)
def test_step_returns_action_within_amp_bounds() -> None:
    config = ControllerConfig(
        horizon_steps=4,
        step_minutes=15,
        amp_max=32.0,
        amp_step=1.0,
        delta_amp_max=8.0,
        soc_target=0.8,
        smoothness_weight=0.01,
        soc_terminal_weight=1.0,
    )
    inputs = ControllerInputs(
        solar_kw=np.array([3.0, 3.5, 4.0, 4.0]),
        load_kw=np.array([1.0, 1.0, 1.0, 1.0]),
        grid_price=np.array([0.30, 0.30, 0.30, 0.30]),
        soc_now=0.4,
        deadline_step=4,
    )
    controller = MPCController(config)
    action = controller.step(inputs)
    assert 0.0 <= action.amperage <= config.amp_max
