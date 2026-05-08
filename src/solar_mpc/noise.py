"""Synthetic forecast-noise generator.

Solar: log-normal multiplicative — captures the "how off can the
forecast be?" question with a heavier-on-the-low-side tail. Clipped to
[0, 3·point] to prevent extreme samples.

Load: additive Gaussian, scaled as a fraction of the point load.
Clipped to ≥ 0.

Price: passthrough — TOU is published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecast import ForecastBundle


@dataclass(frozen=True)
class NoiseSpec:
    """Forecast-error model parameters."""

    solar_log_sigma: float = 0.25
    load_sigma_frac: float = 0.15


def generate_scenarios(
    point_solar: np.ndarray,
    point_load: np.ndarray,
    grid_price: np.ndarray,
    *,
    n_scenarios: int,
    seed: int,
    noise_spec: NoiseSpec | None = None,
) -> ForecastBundle:
    """Sample K scenarios from `noise_spec` around the point forecasts."""
    spec = noise_spec or NoiseSpec()
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be ≥ 1, got {n_scenarios}")

    rng = np.random.default_rng(seed)
    horizon = len(point_solar)
    if len(point_load) != horizon or len(grid_price) != horizon:
        raise ValueError("point_solar, point_load, grid_price must share length H")

    # Solar: log-normal with mean ≈ point (mean correction shifts μ)
    sigma = spec.solar_log_sigma
    mu = -0.5 * sigma**2
    solar_mult = rng.lognormal(mean=mu, sigma=sigma, size=(n_scenarios, horizon))
    point_solar_b = point_solar[None, :]
    solar_scenarios = np.clip(point_solar_b * solar_mult, 0.0, 3.0 * point_solar_b)

    # Load: additive Gaussian, std proportional to base
    load_std = spec.load_sigma_frac * np.maximum(point_load, 1e-6)
    load_noise = rng.normal(loc=0.0, scale=load_std[None, :], size=(n_scenarios, horizon))
    load_scenarios = np.clip(point_load[None, :] + load_noise, 0.0, None)

    return ForecastBundle(
        solar_kw_scenarios=solar_scenarios,
        load_kw_scenarios=load_scenarios,
        grid_price=grid_price,
    )


def realize_trace(
    point_solar: np.ndarray,
    point_load: np.ndarray,
    *,
    seed: int,
    noise_spec: NoiseSpec | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one realization of (solar, load) from `noise_spec`.

    Used to construct what *actually* unfolds in the simulator, distinct
    from the scenario set the controller sees.
    """
    spec = noise_spec or NoiseSpec()
    rng = np.random.default_rng(seed)
    horizon = len(point_solar)

    sigma = spec.solar_log_sigma
    mu = -0.5 * sigma**2
    solar_mult = rng.lognormal(mean=mu, sigma=sigma, size=horizon)
    realized_solar = np.clip(point_solar * solar_mult, 0.0, 3.0 * point_solar)

    load_std = spec.load_sigma_frac * np.maximum(point_load, 1e-6)
    load_noise = rng.normal(loc=0.0, scale=load_std, size=horizon)
    realized_load = np.clip(point_load + load_noise, 0.0, None)

    return realized_solar, realized_load
