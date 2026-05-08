"""Replay an anonymized Tesphase trace through all four controllers.

Usage:
    uv run python scripts/replay_tesphase.py \
        --trace data/raw/tesphase_anonymized.parquet

The trace must be a parquet with columns:
    timestamp     (datetime64)
    solar_kw      (float, ≥ 0)
    load_kw       (float, ≥ 0)
    grid_price    (float, $/kWh)

The trace file is gitignored. The script writes
`notebooks/figures/04_tesphase_replay.png` and prints a comparison table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solar_mpc.baselines import GreedyController, TouRuleController
from solar_mpc.controller import ControllerConfig, MPCController, RobustMPCController
from solar_mpc.simulator import simulate
from solar_mpc.traces import load_recorded_trace

COLORS = {
    "MPC": "#1f77b4",
    "RobustMPC": "#2ca02c",
    "Greedy": "#ff7f0e",
    "TOU": "#9467bd",
}


def _config() -> ControllerConfig:
    return ControllerConfig(
        horizon_steps=24,
        step_minutes=15,
        amp_max=32.0,
        amp_step=1.0,
        delta_amp_max=8.0,
        voltage=240.0,
        battery_capacity_kwh=60.0,
        soc_target=0.8,
        smoothness_weight=0.01,
        soc_terminal_weight=0.5,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Path to anonymized trace parquet (gitignored).",
    )
    parser.add_argument(
        "--soc-initial",
        type=float,
        default=0.4,
        help="Starting SoC (fraction of capacity).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "figures" / "04_tesphase_replay.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    if not args.trace.exists():
        print(f"error: trace file not found: {args.trace}", file=sys.stderr)
        print(
            "\nTo create one, export your Tesphase dry-run telemetry to a parquet "
            "with columns {timestamp, solar_kw, load_kw, grid_price}, anonymize "
            "timestamps (shift to a generic 2026-01-01 epoch), and place it under "
            "data/raw/. The directory is gitignored.",
            file=sys.stderr,
        )
        return 2

    trace = load_recorded_trace(args.trace)
    config = _config()
    if len(trace) <= config.horizon_steps:
        print(
            f"error: trace length {len(trace)} too short for horizon "
            f"{config.horizon_steps}",
            file=sys.stderr,
        )
        return 2

    # Auto-tune TOU threshold from the trace so it works with any utility
    # tariff: midpoint of min and max prices catches the off-peak band.
    price_threshold = float((trace["grid_price"].min() + trace["grid_price"].max()) / 2)

    controllers = {
        "MPC": MPCController(config),
        "RobustMPC": RobustMPCController(config),
        "Greedy": GreedyController(config, export_threshold_kw=0.5),
        "TOU": TouRuleController(config, cheap_price_threshold=price_threshold),
    }
    print(f"TOU threshold (auto-tuned to trace): ${price_threshold:.3f}/kWh")

    results = {}
    print(f"\nReplay over {len(trace)}-step trace from {args.trace.name}:")
    for name, ctrl in controllers.items():
        result = simulate(ctrl, trace, soc_initial=args.soc_initial)
        results[name] = result
        print(
            f"  {name:<9} cost ${result.total_cost:6.2f}   "
            f"final_SoC {result.final_soc:.3f}   deadline_met {result.deadline_met}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, result in results.items():
        h = np.arange(len(result.history)) * (config.step_minutes / 60.0)
        axes[0].plot(h, result.history["soc_post"], color=COLORS[name], label=name, lw=2)
        axes[1].plot(
            h, result.history["cost"].cumsum(), color=COLORS[name], label=name, lw=2
        )
    axes[0].axhline(config.soc_target, color="gray", linestyle="--", lw=1)
    axes[0].set_ylabel("SoC")
    axes[0].set_title(f"Tesphase replay: {args.trace.stem}")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("cumulative cost ($)")
    axes[1].set_xlabel("hour")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
