"""Smoke tests that run today so CI is green from the start."""

import solar_mpc
from solar_mpc.controller import ControllerConfig, MPCController


def test_package_version_present() -> None:
    assert isinstance(solar_mpc.__version__, str)
    assert solar_mpc.__version__


def test_controller_constructs() -> None:
    config = ControllerConfig(
        horizon_steps=12,
        step_minutes=5,
        amp_max=32.0,
        amp_step=1.0,
        delta_amp_max=8.0,
        soc_target=0.8,
        smoothness_weight=0.01,
        soc_terminal_weight=1.0,
    )
    controller = MPCController(config)
    assert controller.config.horizon_steps == 12
