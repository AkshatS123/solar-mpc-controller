"""Tests for the `solar-tsfm-bench` adapter shim.

We never import tsfm-bench. Instead we feed duck-typed objects with the
expected `point` and `quantiles` attributes.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from solar_mpc.forecast_adapter import from_tsfm_result


@dataclass
class _Result:
    point: np.ndarray
    quantiles: dict | None


def test_quantile_path_produces_bundle_with_shape_K_H() -> None:
    horizon = 6
    qs = {
        0.1: np.full(horizon, 1.0),
        0.5: np.full(horizon, 3.0),
        0.9: np.full(horizon, 6.0),
    }
    solar = _Result(point=np.full(horizon, 3.0), quantiles=qs)
    load = _Result(
        point=np.full(horizon, 1.0),
        quantiles={0.1: np.full(horizon, 0.5), 0.5: np.full(horizon, 1.0), 0.9: np.full(horizon, 1.5)},
    )
    bundle = from_tsfm_result(
        solar, load, grid_price=np.full(horizon, 0.3), n_scenarios=20, seed=42
    )
    assert bundle.solar_kw_scenarios.shape == (20, horizon)
    assert bundle.solar_kw_scenarios.min() >= 0.0
    # Sampled values should sit between p10 and p90 (with small tolerance for clipping)
    assert bundle.solar_kw_scenarios.min() >= 1.0 - 1e-6
    assert bundle.solar_kw_scenarios.max() <= 6.0 + 1e-6


def test_point_only_with_residuals_falls_back_to_bootstrap() -> None:
    horizon = 4
    solar = _Result(point=np.full(horizon, 3.0), quantiles=None)
    load = _Result(point=np.full(horizon, 1.0), quantiles=None)
    residuals = np.array([-0.5, 0.0, 0.5, 1.0])

    bundle = from_tsfm_result(
        solar,
        load,
        grid_price=np.full(horizon, 0.3),
        n_scenarios=8,
        seed=7,
        solar_residuals=residuals,
        load_residuals=residuals,
    )
    assert bundle.solar_kw_scenarios.shape == (8, horizon)
    # Each scenario value is point + one of the residuals (then clipped to ≥0)
    expected_values = {3.0 + r for r in residuals}
    expected_values.add(max(0.0, 3.0 + residuals[0]))
    seen = set(np.round(bundle.solar_kw_scenarios.flatten(), 6).tolist())
    assert seen.issubset({round(v, 6) for v in expected_values})


def test_point_only_without_residuals_warns_and_returns_degenerate() -> None:
    horizon = 4
    solar = _Result(point=np.full(horizon, 3.0), quantiles=None)
    load = _Result(point=np.full(horizon, 1.0), quantiles=None)
    with pytest.warns(UserWarning):
        bundle = from_tsfm_result(
            solar, load, grid_price=np.full(horizon, 0.3), n_scenarios=5, seed=0
        )
    # All scenarios are identical to point
    assert (bundle.solar_kw_scenarios == 3.0).all()
    assert (bundle.load_kw_scenarios == 1.0).all()
