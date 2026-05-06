"""Trace loaders.

Two sources today:
- `synthetic_trace` — analytic solar bell curve + bimodal load + TOU price.
- `load_recorded_trace` — read a parquet from `data/raw/` (gitignored).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_trace(
    days: int = 1,
    step_minutes: int = 5,
    *,
    peak_solar_kw: float = 6.0,
    base_load_kw: float = 0.6,
    peak_price: float = 0.45,
    off_peak_price: float = 0.18,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    """Return a synthetic (timestamp, solar_kw, load_kw, grid_price) trace.

    Solar: Gaussian bell centered at noon, zero before 6am or after 6pm.
    Load: small base + morning ramp + evening cooking spike.
    Price: TOU schedule — peak 4pm-9pm, off-peak otherwise.
    """
    n = int(days * 24 * 60 / step_minutes)
    timestamps = pd.date_range(start=start, periods=n, freq=f"{step_minutes}min")
    hours = np.array([t.hour + t.minute / 60 for t in timestamps])

    solar = peak_solar_kw * np.exp(-((hours - 12) ** 2) / 8)
    solar[hours < 6] = 0.0
    solar[hours > 18] = 0.0

    morning = 0.4 * np.exp(-((hours - 7) ** 2) / 1.5)
    evening = 0.9 * np.exp(-((hours - 19) ** 2) / 1.5)
    load = base_load_kw + morning + evening

    price = np.where((hours >= 16) & (hours < 21), peak_price, off_peak_price)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "solar_kw": solar,
            "load_kw": load,
            "grid_price": price,
        }
    )


def load_recorded_trace(path: str | Path) -> pd.DataFrame:
    """Load a recorded trace from parquet. Validates required columns."""
    df = pd.read_parquet(path)
    required = {"timestamp", "solar_kw", "load_kw", "grid_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Recorded trace missing required columns: {sorted(missing)}")
    return df
