"""Tests for `RobustMPCController`."""

import numpy as np

from solar_mpc.controller import ControllerConfig, RobustMPCController
from solar_mpc.forecast import ForecastBundle
from solar_mpc.noise import NoiseSpec, generate_scenarios


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


def test_step_bundle_returns_action_within_amp_bounds() -> None:
    config = _config(horizon=4)
    bundle = ForecastBundle(
        solar_kw_scenarios=np.tile([3.0, 3.5, 4.0, 4.0], (5, 1)),
        load_kw_scenarios=np.tile([1.0, 1.0, 1.0, 1.0], (5, 1)),
        grid_price=np.full(4, 0.30),
    )
    action = RobustMPCController(config).step_bundle(bundle, soc_now=0.4, deadline_step=4)
    assert 0.0 <= action.amperage <= config.amp_max
    assert action.horizon_plan.shape == (4,)


def test_robust_charges_more_under_high_solar_uncertainty_with_tight_deadline() -> None:
    """With a tight deadline, the worst-case scenario forces earlier charging."""
    config = _config(horizon=8)
    point_solar = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    point_load = np.full(8, 1.0)
    price = np.full(8, 0.30)

    low_noise = generate_scenarios(
        point_solar, point_load, price,
        n_scenarios=10, seed=0, noise_spec=NoiseSpec(0.05, 0.05),
    )
    high_noise = generate_scenarios(
        point_solar, point_load, price,
        n_scenarios=10, seed=0, noise_spec=NoiseSpec(0.5, 0.30),
    )

    ctrl = RobustMPCController(config)
    low = ctrl.step_bundle(low_noise, soc_now=0.4, deadline_step=8)
    high = ctrl.step_bundle(high_noise, soc_now=0.4, deadline_step=8)

    assert low.horizon_plan.sum() <= high.horizon_plan.sum() + 1e-3


def test_step_bundle_respects_slew_constraint() -> None:
    config = _config(horizon=6)
    rng = np.random.default_rng(0)
    bundle = ForecastBundle(
        solar_kw_scenarios=rng.uniform(0, 6, (8, 6)),
        load_kw_scenarios=rng.uniform(0, 2, (8, 6)),
        grid_price=np.full(6, 0.30),
    )
    action = RobustMPCController(config).step_bundle(bundle, soc_now=0.5, deadline_step=6)
    assert (np.abs(np.diff(action.horizon_plan)) <= config.delta_amp_max + 1e-6).all()
