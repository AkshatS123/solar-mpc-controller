"""End-to-end demo: run all three controllers over the same synthetic trace
and print a side-by-side cost comparison.
"""

from __future__ import annotations

from solar_mpc.baselines import GreedyController, TouRuleController
from solar_mpc.controller import ControllerConfig, MPCController
from solar_mpc.simulator import simulate
from solar_mpc.traces import synthetic_trace


def main() -> None:
    config = ControllerConfig(
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
    trace = synthetic_trace(days=1, step_minutes=config.step_minutes)
    soc_initial = 0.4

    controllers = {
        "MPC":    MPCController(config),
        "Greedy": GreedyController(config, export_threshold_kw=0.5),
        "TOU":    TouRuleController(config, cheap_price_threshold=0.25),
    }

    print(f"{'controller':<10} {'cost ($)':>10} {'final SoC':>11} {'deadline':>10}")
    for name, controller in controllers.items():
        result = simulate(controller, trace, soc_initial=soc_initial)
        print(
            f"{name:<10} {result.total_cost:>10.3f} {result.final_soc:>11.3f}"
            f" {str(result.deadline_met):>10}"
        )


if __name__ == "__main__":
    main()
