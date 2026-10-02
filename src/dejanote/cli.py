"""Command line interface."""

from __future__ import annotations

import typer
from rich.console import Console

from dejanote import __version__
from dejanote.embedding import (
    DEFAULT_MODEL,
    ModelIntegrityError,
    default_model_dir,
    download_model,
    is_model_downloaded,
    verify_model_files,
)

app = typer.Typer(
    help="Search your notes by meaning, entirely offline.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(highlight=False, soft_wrap=True)


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


@app.command()
def setup(
    yes: bool = typer.Option(False, "--yes", "-y", help="Download without asking for confirmation."),
) -> None:
    """Download the embedding model once. This is the only command that uses the network."""
    spec = DEFAULT_MODEL
    target = default_model_dir(spec)
    if is_model_downloaded(target, spec):
        try:
            verify_model_files(target, spec)
        except ModelIntegrityError as err:
            console.print(f"[yellow]The model at {target} failed verification:[/] {err}")
        else:
            console.print(f"Model already downloaded and verified: {target}")
            console.print("Nothing to do. Every other command runs fully offline.")
            return

    console.print(
        "dejanote needs its embedding model on this machine. This one-time download\n"
        "is the only time dejanote uses the network. Your notes are not read or sent.\n"
    )
    console.print(f"  model     {spec.repo_id}")
    console.print(f"  revision  {spec.revision} (pinned)")
    console.print("  from      https://huggingface.co")
    console.print(f"  size      about {spec.download_mb} MB, {len(spec.files)} files, each checked against a pinned sha256")
    console.print(f"  to        {target}\n")
    if not yes:
        typer.confirm("Download it now?", abort=True)

    with console.status("Downloading and verifying..."):
        try:
            download_model(spec, target)
        except Exception as err:  # network, proxy or integrity problems: explain, don't dump a traceback
            console.print(f"[red]Download failed:[/] {err}")
            console.print("Nothing was saved. Check your connection and run `dejanote setup` again.")
            raise typer.Exit(1) from None
    console.print(f"[green]Done.[/] Model saved to {target}")
    console.print("From here on, dejanote works fully offline.")
