"""Forecast bundle: K scenarios for solar/load + deterministic price.

Decouples the controller from how scenarios were produced. Three sources
today: synthetic noise (`noise.generate_scenarios`), bootstrap residuals
from a point forecaster, and the `solar-tsfm-bench` adapter shim
(`forecast_adapter.from_tsfm_result`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastBundle:
    """K scenarios over horizon H.

    `solar_kw_scenarios` and `load_kw_scenarios` are (K, H).
    `grid_price` is (H,) — treated as deterministic since utility TOU
    schedules are published.
    """

    solar_kw_scenarios: np.ndarray
    load_kw_scenarios: np.ndarray
    grid_price: np.ndarray

    def __post_init__(self) -> None:
        if self.solar_kw_scenarios.ndim != 2:
            raise ValueError(f"solar_kw_scenarios must be 2D, got {self.solar_kw_scenarios.shape}")
        if self.load_kw_scenarios.shape != self.solar_kw_scenarios.shape:
            raise ValueError(
                f"load shape {self.load_kw_scenarios.shape} != "
                f"solar shape {self.solar_kw_scenarios.shape}"
            )
        if self.grid_price.shape != (self.solar_kw_scenarios.shape[1],):
            raise ValueError(
                f"grid_price shape {self.grid_price.shape} must be (H,) where H="
                f"{self.solar_kw_scenarios.shape[1]}"
            )

    @property
    def n_scenarios(self) -> int:
        return self.solar_kw_scenarios.shape[0]

    @property
    def horizon(self) -> int:
        return self.solar_kw_scenarios.shape[1]

    def point(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mean-over-scenarios point forecast for back-compat use."""
        return (
            self.solar_kw_scenarios.mean(axis=0),
            self.load_kw_scenarios.mean(axis=0),
            self.grid_price,
        )

    @classmethod
    def degenerate(
        cls, solar: np.ndarray, load: np.ndarray, grid_price: np.ndarray
    ) -> ForecastBundle:
        """Wrap a point forecast as a single-scenario bundle (K=1)."""
        return cls(
            solar_kw_scenarios=solar.reshape(1, -1),
            load_kw_scenarios=load.reshape(1, -1),
            grid_price=grid_price,
        )
