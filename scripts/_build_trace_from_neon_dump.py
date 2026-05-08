"""One-shot script: parse the Neon MCP JSON dump, anonymize, tariff-overlay,
write `data/raw/tesphase_anonymized.parquet`. Lives under `scripts/` rather
than the repo root because it's reusable if we re-pull data later, but it's
not intended to be a public CLI — keep it short.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import convert_tesphase_export as cv  # type: ignore  # noqa: E402

DUMP = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = REPO / "data" / "raw" / "tesphase_anonymized.parquet"


def _extract_json_array(path: Path) -> list[dict]:
    """The MCP file may have leading prose before the JSON array. Find it."""
    text = path.read_text()
    match = re.search(r"\[\s*\{", text)
    if not match:
        raise RuntimeError(f"no JSON array found in {path}")
    json_text = text[match.start() :]
    end = json_text.rfind("]")
    if end == -1:
        raise RuntimeError(f"no closing ] in {path}")
    return json.loads(json_text[: end + 1])


def main() -> int:
    if DUMP is None:
        print("usage: _build_trace_from_neon_dump.py <dump-file>", file=sys.stderr)
        return 2

    rows = _extract_json_array(DUMP)
    print(f"parsed {len(rows):,} rows from {DUMP.name}")

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["prod_w_avg"] = df["prod_w_avg"].astype(float)
    df["load_w_avg"] = df["load_w_avg"].astype(float)

    df = df.sort_values("ts").reset_index(drop=True)
    df["solar_kw"] = (df["prod_w_avg"] / 1000.0).clip(lower=0.0)
    df["load_kw"] = (df["load_w_avg"] / 1000.0).clip(lower=0.0)

    epoch = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    shift = epoch - df["ts"].iloc[0].floor("15min")
    df["timestamp"] = (df["ts"] + shift).dt.floor("15min")

    hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    tariff = cv.PGE_EV2A_SUMMER
    df["grid_price"] = np.array([cv.price_for_hour(h, tariff) for h in hours])

    out = df[["timestamp", "solar_kw", "load_kw", "grid_price"]].copy()

    # Sanity: drop duplicate buckets (shouldn't exist post-Postgres GROUP BY,
    # but the floor() above can collide on partial-minute boundaries).
    out = out.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print(f"wrote {len(out):,} rows → {OUT}")
    print(f"tariff:        {tariff['name']}")
    print(f"timestamp:     {out['timestamp'].min()}  →  {out['timestamp'].max()}")
    print(f"solar_kw mean: {out['solar_kw'].mean():.2f}  peak: {out['solar_kw'].max():.2f}")
    print(f"load_kw mean:  {out['load_kw'].mean():.2f}  peak: {out['load_kw'].max():.2f}")
    print("preview:")
    print(out.head().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
