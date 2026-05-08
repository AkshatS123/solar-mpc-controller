"""Tests for `ForecastBundle`."""

import numpy as np
import pytest

from solar_mpc.forecast import ForecastBundle


def _bundle(k: int = 5, h: int = 8) -> ForecastBundle:
    rng = np.random.default_rng(0)
    return ForecastBundle(
        solar_kw_scenarios=rng.uniform(0, 6, (k, h)),
        load_kw_scenarios=rng.uniform(0, 2, (k, h)),
        grid_price=np.full(h, 0.30),
    )


def test_shapes_and_accessors() -> None:
    b = _bundle(k=7, h=12)
    assert b.n_scenarios == 7
    assert b.horizon == 12


def test_point_returns_means_with_correct_shape() -> None:
    b = _bundle(k=5, h=8)
    s, load, p = b.point()
    assert s.shape == (8,) and load.shape == (8,) and p.shape == (8,)
    np.testing.assert_allclose(s, b.solar_kw_scenarios.mean(axis=0))


def test_degenerate_wraps_point_as_single_scenario() -> None:
    s = np.array([1.0, 2.0, 3.0, 4.0])
    load = np.array([0.5, 0.5, 0.5, 0.5])
    p = np.full(4, 0.30)
    b = ForecastBundle.degenerate(s, load, p)
    assert b.n_scenarios == 1
    np.testing.assert_array_equal(b.solar_kw_scenarios[0], s)


def test_validation_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        ForecastBundle(
            solar_kw_scenarios=np.zeros((3, 4)),
            load_kw_scenarios=np.zeros((3, 5)),
            grid_price=np.zeros(4),
        )
    with pytest.raises(ValueError):
        ForecastBundle(
            solar_kw_scenarios=np.zeros((3, 4)),
            load_kw_scenarios=np.zeros((3, 4)),
            grid_price=np.zeros(5),
        )
