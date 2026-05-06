"""Command-line entry point: `solar-mpc …`."""

from __future__ import annotations

import click

from . import __version__


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Solar MPC controller."""


@main.command()
@click.option("--days", default=1, show_default=True, help="Days of synthetic trace.")
@click.option(
    "--controller",
    type=click.Choice(["mpc", "greedy", "tou"]),
    default="mpc",
    show_default=True,
)
def run(days: int, controller: str) -> None:
    """Run a controller over a synthetic trace and print summary stats."""
    raise NotImplementedError("Phase 2: wire simulator + controller.")


if __name__ == "__main__":
    main()
