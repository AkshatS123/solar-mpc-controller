"""Command-line entry point: `solar-mpc …`."""

from __future__ import annotations

import click

from . import __version__
from .baselines import GreedyController, TouRuleController
from .controller import ControllerConfig, MPCController
from .simulator import simulate
from .traces import synthetic_trace


def _default_config() -> ControllerConfig:
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


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Solar MPC controller."""


@main.command()
@click.option("--days", default=1, show_default=True, help="Days of synthetic trace.")
@click.option(
    "--controller",
    "controller_kind",
    type=click.Choice(["mpc", "greedy", "tou"]),
    default="mpc",
    show_default=True,
)
@click.option("--soc-initial", default=0.4, show_default=True, type=float)
def run(days: int, controller_kind: str, soc_initial: float) -> None:
    """Run a controller over a synthetic trace and print summary stats."""
    config = _default_config()
    trace = synthetic_trace(days=days, step_minutes=config.step_minutes)

    if controller_kind == "mpc":
        controller = MPCController(config)
    elif controller_kind == "greedy":
        controller = GreedyController(config, export_threshold_kw=0.5)
    else:
        controller = TouRuleController(config, cheap_price_threshold=0.25)

    result = simulate(controller, trace, soc_initial=soc_initial)
    click.echo(f"controller       : {controller_kind}")
    click.echo(f"steps simulated  : {len(result.history)}")
    click.echo(f"total cost ($)   : {result.total_cost:.3f}")
    click.echo(f"final SoC        : {result.final_soc:.3f}")
    click.echo(f"deadline met     : {result.deadline_met}")


if __name__ == "__main__":
    main()
