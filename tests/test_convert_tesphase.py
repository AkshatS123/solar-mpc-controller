"""Smoke test for the Tesphase-export converter using a synthetic input."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "convert_tesphase_export.py"

# Load the script as a module so we can call `convert()` directly.
spec = importlib.util.spec_from_file_location("convert_tesphase_export", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["convert_tesphase_export"] = mod
spec.loader.exec_module(mod)


def _synthetic_export(hours: int = 6) -> pd.DataFrame:
    """Mimic a Tesphasev2 export — required columns only."""
    n = hours * 12  # 5-min steps
    ts = pd.date_range("2026-04-15 12:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "y": [3.5] * n,
            "consumption_w": [1500] * n,
            "house_load_w": [800] * n,
            "net_w": [-2000] * n,
            "blackout": [False] * n,
            "tou_period": ["neutral"] * n,
            "decision_action": ["maintain"] * n,
        }
    )


def test_convert_produces_required_schema() -> None:
    src = _synthetic_export(hours=4)
    out = mod.convert(src, mod.PGE_EV2A_SUMMER, step_minutes=15)
    assert set(out.columns) == {"timestamp", "solar_kw", "load_kw", "grid_price"}
    assert len(out) == 4 * 4  # 4 hours × 4 fifteen-minute steps
    assert (out["solar_kw"] >= 0).all()
    assert (out["load_kw"] >= 0).all()
    assert (out["grid_price"] > 0).all()


def test_anonymizes_first_timestamp_to_2026_01_01() -> None:
    src = _synthetic_export(hours=2)
    out = mod.convert(src, mod.PGE_EV2A_SUMMER, step_minutes=15)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 00:00", tz="UTC")


def test_pge_summer_charges_peak_in_4pm_to_9pm_window() -> None:
    assert mod.price_for_hour(2.0, mod.PGE_EV2A_SUMMER) == 0.31
    assert mod.price_for_hour(15.5, mod.PGE_EV2A_SUMMER) == 0.51
    assert mod.price_for_hour(18.0, mod.PGE_EV2A_SUMMER) == 0.62
    assert mod.price_for_hour(22.0, mod.PGE_EV2A_SUMMER) == 0.51


def test_rejects_input_with_no_house_load() -> None:
    src = _synthetic_export(hours=2)
    src["house_load_w"] = None
    with pytest.raises(ValueError, match="estimatedHouseLoadW"):
        mod.convert(src, mod.PGE_EV2A_SUMMER, step_minutes=15)
