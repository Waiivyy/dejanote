"""Command line interface."""

from __future__ import annotations

import typer

from dejanote import __version__

app = typer.Typer(
    help="Search your notes by meaning, entirely offline.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"dejanote {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_print_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """Search your notes by meaning, entirely offline."""
