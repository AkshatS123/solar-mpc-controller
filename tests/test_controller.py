"""Tests for the MPC controller's solve step."""

import numpy as np

from solar_mpc.controller import ControllerConfig, ControllerInputs, MPCController


def _config(horizon: int = 4) -> ControllerConfig:
    return ControllerConfig(
        horizon_steps=horizon,
        step_minutes=15,
        amp_max=32.0,
        amp_step=1.0,
        delta_amp_max=8.0,
        voltage=240.0,
        battery_capacity_kwh=60.0,
        soc_target=0.8,
        smoothness_weight=0.01,
        soc_terminal_weight=0.5,
    )


def test_step_returns_action_within_amp_bounds() -> None:
    config = _config(horizon=4)
    inputs = ControllerInputs(
        solar_kw=np.array([3.0, 3.5, 4.0, 4.0]),
        load_kw=np.array([1.0, 1.0, 1.0, 1.0]),
        grid_price=np.array([0.30, 0.30, 0.30, 0.30]),
        soc_now=0.4,
        deadline_step=4,
    )
    action = MPCController(config).step(inputs)

    assert 0.0 <= action.amperage <= config.amp_max
    assert action.horizon_plan.shape == (config.horizon_steps,)
    assert np.all(action.horizon_plan >= -1e-6)
    assert np.all(action.horizon_plan <= config.amp_max + 1e-6)


def test_step_prefers_charging_during_solar_excess() -> None:
    """With ample solar and zero load, the optimal first action should be > 0."""
    config = _config(horizon=4)
    inputs = ControllerInputs(
        solar_kw=np.array([5.0, 5.0, 5.0, 5.0]),
        load_kw=np.array([0.5, 0.5, 0.5, 0.5]),
        grid_price=np.array([0.40, 0.40, 0.40, 0.40]),
        soc_now=0.4,
        deadline_step=4,
    )
    action = MPCController(config).step(inputs)
    assert action.amperage > 0.0


def test_step_respects_slew_constraint() -> None:
    config = _config(horizon=4)
    inputs = ControllerInputs(
        solar_kw=np.array([0.0, 5.0, 5.0, 5.0]),
        load_kw=np.array([0.5, 0.5, 0.5, 0.5]),
        grid_price=np.array([0.40, 0.40, 0.40, 0.40]),
        soc_now=0.4,
        deadline_step=4,
    )
    action = MPCController(config).step(inputs)
    diffs = np.abs(np.diff(action.horizon_plan))
    assert np.all(diffs <= config.delta_amp_max + 1e-6)
