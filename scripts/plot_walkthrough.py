"""Generate the walkthrough figures embedded in README + notebooks/01_walkthrough.md.

Regenerate with:  uv run python scripts/plot_walkthrough.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solar_mpc.baselines import GreedyController, TouRuleController
from solar_mpc.controller import ControllerConfig, MPCController
from solar_mpc.simulator import simulate
from solar_mpc.traces import synthetic_trace

COLORS = {"MPC": "#1f77b4", "Greedy": "#ff7f0e", "TOU": "#9467bd"}


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


def _hour_axis(history) -> np.ndarray:
    return np.arange(len(history)) * 0.25  # 15-minute step


def main() -> None:
    config = _config()
    trace = synthetic_trace(days=2, step_minutes=config.step_minutes)
    soc_initial = 0.4

    controllers = {
        "MPC": MPCController(config),
        "Greedy": GreedyController(config, export_threshold_kw=0.5),
        "TOU": TouRuleController(config, cheap_price_threshold=0.25),
    }
    results = {n: simulate(c, trace, soc_initial=soc_initial) for n, c in controllers.items()}

    out_dir = Path(__file__).resolve().parents[1] / "notebooks" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Figure 1: input trace (1 day) ----------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    one_day = trace.iloc[: 24 * 4]
    hours = np.arange(len(one_day)) * 0.25

    axes[0].fill_between(hours, one_day["solar_kw"], alpha=0.4, color="gold", label="solar")
    axes[0].plot(hours, one_day["load_kw"], color="darkgreen", label="house load", lw=1.8)
    axes[0].set_ylabel("kW")
    axes[0].set_title("Synthetic input trace (day 1)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].step(hours, one_day["grid_price"], where="post", color="crimson", lw=1.8)
    axes[1].set_ylabel("$/kWh")
    axes[1].set_xlabel("hour of day")
    axes[1].set_title("Grid price (TOU)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "01_inputs.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure 2: 3-controller comparison ----------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for name, result in results.items():
        h = _hour_axis(result.history)
        axes[0].plot(h, result.history["soc_post"], color=COLORS[name], label=name, lw=2)
        axes[1].plot(h, result.history["cost"].cumsum(), color=COLORS[name], label=name, lw=2)
        axes[2].step(
            h, result.history["amperage"], where="post", color=COLORS[name], label=name, lw=1.5
        )

    axes[0].axhline(
        config.soc_target, color="gray", linestyle="--", lw=1, label=f"target = {config.soc_target}"
    )
    axes[0].set_ylabel("SoC")
    axes[0].set_title("State of charge")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("cumulative cost ($)")
    axes[1].set_title("Cumulative grid-import cost")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    axes[2].set_ylabel("amperage (A)")
    axes[2].set_xlabel("simulation hour")
    axes[2].set_title("Charging amperage chosen by each controller")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "02_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"figures saved to {out_dir}")
    for name, result in results.items():
        print(
            f"  {name:<7} cost=${result.total_cost:6.2f}  "
            f"final_SoC={result.final_soc:.3f}  deadline_met={result.deadline_met}"
        )


if __name__ == "__main__":
    main()
