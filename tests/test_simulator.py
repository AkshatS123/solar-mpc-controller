"""Tests for the rolling-horizon simulator and the synthetic trace."""

import pytest

from solar_mpc.baselines import GreedyController
from solar_mpc.controller import ControllerConfig, RobustMPCController
from solar_mpc.noise import NoiseSpec
from solar_mpc.simulator import simulate, simulate_under_noise
from solar_mpc.traces import synthetic_trace


def _config() -> ControllerConfig:
    return ControllerConfig(
        horizon_steps=8,
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


def test_synthetic_trace_shape_and_columns() -> None:
    trace = synthetic_trace(days=1, step_minutes=15)
    assert len(trace) == 24 * 4
    assert set(trace.columns) == {"timestamp", "solar_kw", "load_kw", "grid_price"}
    assert (trace["solar_kw"] >= 0).all()
    assert (trace["load_kw"] >= 0).all()
    assert (trace["grid_price"] > 0).all()


def test_simulator_runs_with_greedy_controller() -> None:
    config = _config()
    trace = synthetic_trace(days=1, step_minutes=config.step_minutes)
    controller = GreedyController(config, export_threshold_kw=0.5)

    result = simulate(controller, trace, soc_initial=0.4)

    assert len(result.history) == len(trace) - config.horizon_steps
    assert result.total_cost >= 0.0
    assert 0.0 <= result.final_soc <= 1.0


def test_simulator_rejects_too_short_trace() -> None:
    config = _config()
    trace = synthetic_trace(days=1, step_minutes=config.step_minutes).iloc[: config.horizon_steps]
    controller = GreedyController(config)
    with pytest.raises(ValueError, match="too short"):
        simulate(controller, trace, soc_initial=0.4)


def test_simulate_under_noise_returns_n_realizations() -> None:
    config = _config()
    trace = synthetic_trace(days=1, step_minutes=config.step_minutes)
    controller = GreedyController(config, export_threshold_kw=0.5)

    results = simulate_under_noise(
        controller,
        trace,
        soc_initial=0.4,
        n_realizations=4,
        seed=0,
        n_scenarios=5,
        noise_spec=NoiseSpec(0.2, 0.1),
    )
    assert len(results) == 4
    for r in results:
        assert 0.0 <= r.final_soc <= 1.0
        assert (r.history["soc_post"] <= 1.0 + 1e-9).all()  # BMS phantom-charge regression


def test_simulate_under_noise_is_deterministic_with_seed() -> None:
    config = _config()
    trace = synthetic_trace(days=1, step_minutes=config.step_minutes)
    controller = GreedyController(config)

    a = simulate_under_noise(
        controller, trace, soc_initial=0.4, n_realizations=2, seed=99, n_scenarios=3
    )
    b = simulate_under_noise(
        controller, trace, soc_initial=0.4, n_realizations=2, seed=99, n_scenarios=3
    )
    assert a[0].total_cost == b[0].total_cost
    assert a[1].total_cost == b[1].total_cost


def test_simulate_under_noise_runs_with_robust_mpc() -> None:
    config = _config()
    trace = synthetic_trace(days=1, step_minutes=config.step_minutes)
    controller = RobustMPCController(config)
    results = simulate_under_noise(
        controller, trace, soc_initial=0.4, n_realizations=2, seed=1, n_scenarios=4
    )
    assert len(results) == 2
