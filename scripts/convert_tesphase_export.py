"""Convert a Tesphasev2 cycle-telemetry export to solar-mpc-controller schema.

Inputs the parquet produced by `Tesphasev2/scripts/export_cycle_telemetry.py`.

Three transformations:
1. **Schema reshape** — pick `solar_kw`, `load_kw`, `timestamp`. Drop everything
   else. `load_kw` uses `estimatedHouseLoadW` (house load excluding EV
   charging) so the MPC simulator's own charging decisions don't double-count.
2. **Resample** to 15-minute means. Tesphase polls every 5 min; the MPC
   controller's default step is 15 min.
3. **Tariff overlay** — synthesize `grid_price` from a published utility
   TOU schedule based on hour-of-day. Two tariffs built in (PG&E EV2-A
   summer, SCE TOU-D-PRIME summer); pick one with `--tariff` or supply
   your own JSON.
4. **Anonymize timestamps** — shift so the first row is 2026-01-01 00:00 UTC.

Usage:
    uv run python scripts/convert_tesphase_export.py \\
        --input ~/Documents/Dev/Tesphasev2/exports/own_house_2026.parquet \\
        --tariff pge-ev2a-summer \\
        --output data/raw/tesphase_anonymized.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# PG&E EV2-A residential summer rates (June-September), $/kWh.
# Source: pge.com/tariffs/electric (publicly published, customer-side).
PGE_EV2A_SUMMER = {
    "name": "PG&E EV2-A summer",
    "schedule": [
        {"start_hour": 0, "end_hour": 15, "price": 0.31},   # off-peak
        {"start_hour": 15, "end_hour": 16, "price": 0.51},  # partial peak
        {"start_hour": 16, "end_hour": 21, "price": 0.62},  # peak
        {"start_hour": 21, "end_hour": 24, "price": 0.51},  # partial peak
    ],
}

# SCE TOU-D-PRIME residential summer rates (June-September), $/kWh.
SCE_TOUD_PRIME_SUMMER = {
    "name": "SCE TOU-D-PRIME summer",
    "schedule": [
        {"start_hour": 0, "end_hour": 16, "price": 0.26},
        {"start_hour": 16, "end_hour": 21, "price": 0.59},
        {"start_hour": 21, "end_hour": 24, "price": 0.36},
    ],
}

BUILT_IN_TARIFFS = {
    "pge-ev2a-summer": PGE_EV2A_SUMMER,
    "sce-toud-prime-summer": SCE_TOUD_PRIME_SUMMER,
}


def price_for_hour(hour: float, tariff: dict) -> float:
    """Look up the $/kWh price for a fractional hour-of-day."""
    for window in tariff["schedule"]:
        if window["start_hour"] <= hour < window["end_hour"]:
            return float(window["price"])
    return float(tariff["schedule"][-1]["price"])


def convert(
    src: pd.DataFrame, tariff: dict, *, step_minutes: int = 15
) -> pd.DataFrame:
    df = src.copy()
    df = df.dropna(subset=["house_load_w"])
    if df.empty:
        raise ValueError(
            "no rows with non-null estimatedHouseLoadW — needed for honest "
            "load_kw (Tesphase's own EV charging is in `consumption_w` and "
            "would double-count)"
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    df["solar_kw"] = df["y"].astype(float)
    df["load_kw"] = df["house_load_w"].astype(float) / 1000.0

    resampled = df[["solar_kw", "load_kw"]].resample(f"{step_minutes}min").mean()
    resampled = resampled.dropna(how="all").interpolate(limit=2)
    resampled = resampled.dropna()

    if resampled.empty:
        raise ValueError("trace empty after resampling — input may be too sparse")

    epoch = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    shift = epoch - resampled.index[0]
    out = resampled.reset_index()
    out["timestamp"] = out["ts"] + shift
    out = out.drop(columns=["ts"])

    hours = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    out["grid_price"] = np.array([price_for_hour(h, tariff) for h in hours])

    out["solar_kw"] = out["solar_kw"].clip(lower=0.0)
    out["load_kw"] = out["load_kw"].clip(lower=0.0)

    return out[["timestamp", "solar_kw", "load_kw", "grid_price"]]


def load_tariff(name_or_path: str) -> dict:
    if name_or_path in BUILT_IN_TARIFFS:
        return BUILT_IN_TARIFFS[name_or_path]
    path = Path(name_or_path).expanduser()
    if not path.exists():
        raise SystemExit(
            f"tariff '{name_or_path}' is not built-in and is not a JSON file. "
            f"Built-in options: {sorted(BUILT_IN_TARIFFS)}"
        )
    with open(path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Parquet from Tesphasev2 export_cycle_telemetry.py",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/tesphase_anonymized.parquet"),
        help="Where to write the solar-mpc-controller-shaped trace",
    )
    p.add_argument(
        "--tariff",
        default="pge-ev2a-summer",
        help=f"Built-in tariff name {sorted(BUILT_IN_TARIFFS)} or path to JSON",
    )
    p.add_argument("--step-minutes", type=int, default=15)
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        print(
            "\nFirst run the Tesphasev2 export to produce a parquet:\n"
            "  cd ~/Documents/Dev/Tesphasev2 && \\\n"
            "  export DATABASE_URL='postgres://...neon...' && \\\n"
            "  export OWNER_USER_ID='<your-user-id>' && \\\n"
            "  ./scripts/export_cycle_telemetry.py --out exports/own_house_2026.parquet",
            file=sys.stderr,
        )
        return 2

    tariff = load_tariff(args.tariff)
    src = pd.read_parquet(args.input)
    out = convert(src, tariff, step_minutes=args.step_minutes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    print(f"wrote {len(out):,} rows → {args.output}")
    print(f"tariff: {tariff['name']}")
    print(f"timestamp range: {out['timestamp'].min()} → {out['timestamp'].max()}")
    print("preview:")
    print(out.head().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
