"""Trace loaders.

Two sources today:
- `synthetic_trace` — analytic solar curve + sinusoidal load, for tests
  and the demo script.
- `load_recorded_trace` — read a parquet from `data/raw/` (gitignored).

Real Tesphase replay traces will live in a separate dataset repo so this
package stays import-cheap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def synthetic_trace(days: int = 1, step_minutes: int = 5) -> pd.DataFrame:
    """Return a synthetic (timestamp, solar_kw, load_kw, grid_price) trace."""
    raise NotImplementedError("Phase 1: analytic solar/load curves.")


def load_recorded_trace(path: str | Path) -> pd.DataFrame:
    """Load a recorded trace from parquet."""
    raise NotImplementedError("Phase 3: parquet loader.")
