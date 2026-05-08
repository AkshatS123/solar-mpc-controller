"""Adapter from `solar-tsfm-bench` ForecastResult shape to ForecastBundle.

This module imports nothing from `solar-tsfm-bench` — the adapter takes
a duck-typed object with the expected attributes. When tsfm-bench ships
quantile-capable models, the wiring is one line at the call site.

Two paths:

- **Quantile path:** `result.quantiles` is a dict like
  `{0.1: arr, 0.5: arr, 0.9: arr}`. We sample K scenarios via inverse-CDF
  using piecewise-linear interpolation across the sorted quantiles.
- **Point fallback:** only `result.point` is available. We fall back to
  bootstrap residuals if the caller passes a residual sample, otherwise
  return a degenerate (K=1) bundle and emit a warning.
"""

from __future__ import annotations

import warnings
from typing import Any, Protocol

import numpy as np

from .forecast import ForecastBundle


class _ForecastResultLike(Protocol):
    """Shape we expect from `solar_tsfm_bench.ForecastResult`."""

    point: np.ndarray
    quantiles: dict[float, np.ndarray] | None


def from_tsfm_result(
    solar_result: Any,
    load_result: Any,
    grid_price: np.ndarray,
    *,
    n_scenarios: int,
    seed: int,
    solar_residuals: np.ndarray | None = None,
    load_residuals: np.ndarray | None = None,
) -> ForecastBundle:
    """Build a `ForecastBundle` from two duck-typed forecast results."""
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be ≥ 1, got {n_scenarios}")

    rng = np.random.default_rng(seed)
    solar_scenarios = _scenarios_from_result(
        solar_result, rng, n_scenarios, residuals=solar_residuals, name="solar"
    )
    load_scenarios = _scenarios_from_result(
        load_result, rng, n_scenarios, residuals=load_residuals, name="load"
    )
    if solar_scenarios.shape[1] != len(grid_price):
        raise ValueError(
            f"solar horizon {solar_scenarios.shape[1]} != grid_price {len(grid_price)}"
        )
    if load_scenarios.shape != solar_scenarios.shape:
        raise ValueError(
            f"load shape {load_scenarios.shape} != solar shape {solar_scenarios.shape}"
        )
    return ForecastBundle(
        solar_kw_scenarios=np.clip(solar_scenarios, 0.0, None),
        load_kw_scenarios=np.clip(load_scenarios, 0.0, None),
        grid_price=grid_price,
    )


def _scenarios_from_result(
    result: Any, rng: np.random.Generator, n_scenarios: int, *, residuals, name: str
) -> np.ndarray:
    quantiles = getattr(result, "quantiles", None)
    point = np.asarray(result.point, dtype=float)
    horizon = len(point)

    if quantiles:
        return _sample_from_quantiles(quantiles, rng, n_scenarios, horizon)

    if residuals is not None:
        residuals = np.asarray(residuals, dtype=float)
        if residuals.ndim != 1 or len(residuals) < 1:
            raise ValueError(f"{name} residuals must be a non-empty 1D array")
        idx = rng.integers(0, len(residuals), size=(n_scenarios, horizon))
        return point[None, :] + residuals[idx]

    warnings.warn(
        f"{name} result has no quantiles and no residuals — returning K=1 bundle",
        stacklevel=2,
    )
    return np.tile(point[None, :], (n_scenarios, 1))


def _sample_from_quantiles(
    quantiles: dict[float, np.ndarray],
    rng: np.random.Generator,
    n_scenarios: int,
    horizon: int,
) -> np.ndarray:
    qs = sorted(quantiles.keys())
    if not qs:
        raise ValueError("quantiles dict is empty")
    q_arr = np.array(qs, dtype=float)
    values = np.stack([np.asarray(quantiles[q], dtype=float) for q in qs], axis=0)
    if values.shape[1] != horizon:
        raise ValueError(
            f"quantile values horizon {values.shape[1]} != expected {horizon}"
        )

    u = rng.uniform(size=(n_scenarios, horizon))
    out = np.empty((n_scenarios, horizon), dtype=float)
    for h in range(horizon):
        out[:, h] = np.interp(u[:, h], q_arr, values[:, h])
    return out
