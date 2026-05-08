"""Generate the walkthrough figures embedded in README + notebooks/01_walkthrough.md.

Regenerate with:  uv run python scripts/plot_walkthrough.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solar_mpc.baselines import GreedyController, TouRuleController
from solar_mpc.controller import ControllerConfig, MPCController, RobustMPCController
from solar_mpc.noise import NoiseSpec
from solar_mpc.simulator import simulate, simulate_under_noise
from solar_mpc.traces import synthetic_trace

COLORS = {
    "MPC": "#1f77b4",
    "RobustMPC": "#2ca02c",
    "RobustCVaR": "#17becf",
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
            f"  {name:<9} cost=${result.total_cost:6.2f}  "
            f"final_SoC={result.final_soc:.3f}  deadline_met={result.deadline_met}"
        )

    # ---------------- Figure 3: robustness under forecast noise ----------------
    _make_robustness_figure(config, trace, soc_initial, out_dir)


def _make_robustness_figure(config, trace, _soc_initial_unused, out_dir) -> None:
    """Run all four controllers under N noisy realizations.

    Tight setup so the deadline actually binds in tail scenarios:
      - SoC 0.25 → 0.90 target (needs 39 kWh added, ≈5h of full charge)
      - Deadline at step 48 (=12:00 in the trace), tight relative to solar window
      - σ_solar=0.40, σ_load=0.20 — substantial forecast error

    Two-panel figure:
      (top) cost distribution over 50 realizations (boxplot)
      (bottom) deadline-miss rate (bar)
    """
    # Stress test config
    stress_config = ControllerConfig(
        horizon_steps=config.horizon_steps,
        step_minutes=config.step_minutes,
        amp_max=config.amp_max,
        amp_step=config.amp_step,
        delta_amp_max=config.delta_amp_max,
        voltage=config.voltage,
        battery_capacity_kwh=config.battery_capacity_kwh,
        soc_target=0.90,
        smoothness_weight=config.smoothness_weight,
        soc_terminal_weight=config.soc_terminal_weight,
    )
    soc_initial = 0.25
    deadline_index = 48
    n_realizations = 50
    noise_spec = NoiseSpec(solar_log_sigma=0.40, load_sigma_frac=0.20)

    controllers = {
        "MPC": MPCController(stress_config),
        "RobustMPC": RobustMPCController(stress_config),
        "Greedy": GreedyController(stress_config, export_threshold_kw=0.5),
        "TOU": TouRuleController(stress_config, cheap_price_threshold=0.25),
    }

    summary = {}
    print(
        "\nRobustness sweep (N=50, SoC 0.25→0.90 by step 48, "
        "σ_solar=0.40, σ_load=0.20):"
    )
    for name, ctrl in controllers.items():
        runs = simulate_under_noise(
            ctrl,
            trace,
            soc_initial=soc_initial,
            n_realizations=n_realizations,
            seed=2026,
            n_scenarios=10,
            noise_spec=noise_spec,
            deadline_index=deadline_index,
        )
        costs = np.array([r.total_cost for r in runs])
        # "deadline met" = SoC ≥ target at the deadline timestep (not at trace end)
        deadline_socs = np.array(
            [
                r.history.iloc[deadline_index - 1]["soc_post"]
                if len(r.history) > deadline_index - 1
                else r.final_soc
                for r in runs
            ]
        )
        miss_rate = float(np.mean(deadline_socs < stress_config.soc_target - 1e-3))
        summary[name] = {
            "cost_mean": float(costs.mean()),
            "cost_std": float(costs.std()),
            "miss_rate": miss_rate,
            "costs": costs,
        }
        print(
            f"  {name:<9} cost ${costs.mean():6.2f} ± ${costs.std():4.2f}   "
            f"deadline-miss-rate {miss_rate:.0%}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})
    names = list(summary.keys())
    box_data = [summary[n]["costs"] for n in names]
    box_colors = [COLORS[n] for n in names]

    bp = axes[0].boxplot(
        box_data,
        labels=names,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("cost over 50 realizations ($)")
    axes[0].set_title(
        "Reliability under noisy forecasts: cost distribution over 50 realizations"
    )
    axes[0].grid(alpha=0.3, axis="y")

    miss_rates = [summary[n]["miss_rate"] * 100 for n in names]
    bars = axes[1].bar(names, miss_rates, color=box_colors, alpha=0.8)
    axes[1].set_ylabel("deadline-miss rate (%)")
    axes[1].set_title(
        "Deadline-miss rate (SoC < 0.90 by step 48; σ_solar=0.40, σ_load=0.20·base)"
    )
    axes[1].set_ylim(0, max(100.0, max(miss_rates) * 1.2))
    axes[1].grid(alpha=0.3, axis="y")
    for bar, rate in zip(bars, miss_rates, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_dir / "03_robustness.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nrobustness figure: {out_dir / '03_robustness.png'}")


if __name__ == "__main__":
    main()
