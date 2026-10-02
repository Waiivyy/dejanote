"""Command line interface."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from dejanote import __version__, config
from dejanote.embedding import (
    DEFAULT_MODEL,
    Embedder,
    ModelIntegrityError,
    ModelNotFoundError,
    default_model_dir,
    download_model,
    is_model_downloaded,
    verify_model_files,
)
from dejanote.indexer import index_folder
from dejanote.privacy import NetworkGuard, block_network
from dejanote.search import SearchResult, search, snippet
from dejanote.store import IndexMismatchError, Store

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
    console.print(
        f"  size      about {spec.download_mb} MB, {len(spec.files)} files, each checked against a pinned sha256"
    )
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


@app.command()
def index(
    folder: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Folder of notes to index. Subfolders are included.",
    ),
) -> None:
    """Index a folder of Markdown and text notes. Runs fully offline.

    Only new and changed notes are embedded, so running it again is cheap.
    """
    started = time.perf_counter()
    with block_network() as guard:
        embedder = _load_embedder()
        with _open_index(embedder, rebuild_if_incompatible=True) as store:
            if store.rebuilt:
                console.print(
                    "[yellow]The index was made with a different model or chunking version, "
                    "so it is being rebuilt from scratch.[/]"
                )
            with _progress() as progress:
                task = progress.add_task("Indexing", total=None)
                report = index_folder(
                    folder,
                    store,
                    embedder,
                    on_progress=lambda done, total, _path: progress.update(task, completed=done, total=total),
                )
    elapsed = time.perf_counter() - started

    changes = ", ".join(
        f"{count} {label}"
        for count, label in ((report.added, "new"), (report.updated, "changed"), (report.unchanged, "unchanged"))
        if count
    )
    summary = f"Indexed {_plural(report.notes, 'note')} from {escape(_display(folder))}"
    summary += f" ({changes})." if changes else "."
    if report.chunks:
        summary += f" Embedded {_plural(report.chunks, 'chunk')} in {elapsed:.1f}s."
    else:
        summary += f" Nothing new to embed ({elapsed:.1f}s)."
    console.print(summary)
    if report.removed:
        verb = "exists" if report.removed == 1 else "exist"
        console.print(f"Removed {_plural(report.removed, 'note')} that no longer {verb}.")
    if report.skipped:
        console.print(f"[yellow]Skipped {_plural(len(report.skipped), 'file')}:[/]")
        for path, reason in report.skipped:
            console.print(f"  {escape(_display(Path(path)))} ({escape(reason)})")
    console.print(f"Index: {escape(_display(config.index_path()))}")
    _report_network(guard)


@app.command("search")
def search_notes(
    query: str = typer.Argument(..., help="What you are looking for, in your own words."),
    limit: int = typer.Option(5, "--limit", "-n", min=1, help="How many results to show."),
) -> None:
    """Search your indexed notes by meaning. Runs fully offline."""
    if not query.strip():
        console.print("[red]Give something to search for.[/]")
        raise typer.Exit(1)
    with block_network() as guard:
        embedder = _load_embedder()
        if not config.index_path().exists():
            console.print("There is no index yet. Run `dejanote index <folder>` first.", markup=False)
            raise typer.Exit(1)
        with _open_index(embedder) as store:
            results = search(store, embedder, query, limit=limit)
    if results:
        _print_results(results)
    else:
        console.print("The index is empty. Run `dejanote index <folder>` to add notes.", markup=False)
    _report_network(guard)


def _print_results(results: list[SearchResult]) -> None:
    indent = " " * 6
    width = max(40, console.width)
    for result in results:
        location = f"{_display(Path(result.path))}:{result.chunk.start_line}"
        console.print(f"[bold]{result.score:.2f}[/]  [cyan]{escape(location)}[/]")
        console.print(f"{indent}[dim]{escape(' > '.join(result.chunk.headings))}[/]")
        for line in snippet(result.chunk.text).split("\n"):
            wrapped = textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
            console.print(wrapped, markup=False)
        if result.also:
            # A note's untitled opening only carries the note's own title, so call it the intro.
            others = ", ".join(
                f"{s.chunk.headings[-1] if len(s.chunk.headings) > 1 else 'intro'} (line {s.chunk.start_line})"
                for s in result.also
            )
            wrapped = textwrap.fill(f"also: {others}", width=width, initial_indent=indent, subsequent_indent=indent)
            console.print(wrapped, style="dim", markup=False)
        console.print()


def _load_embedder() -> Embedder:
    try:
        return Embedder()
    except ModelNotFoundError as err:
        console.print(str(err), style="red", markup=False)
        raise typer.Exit(1) from None


def _open_index(embedder: Embedder, rebuild_if_incompatible: bool = False) -> Store:
    try:
        return Store(
            config.index_path(),
            embedder.model_id,
            embedder.dimension,
            rebuild_if_incompatible=rebuild_if_incompatible,
        )
    except IndexMismatchError as err:
        console.print(str(err), style="red", markup=False)
        raise typer.Exit(1) from None


def _progress() -> Progress:
    """A progress bar that is only drawn on a terminal, so piped output stays clean."""
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
        disable=not console.is_terminal,
    )


def _report_network(guard: NetworkGuard) -> None:
    """Every offline command ends with a receipt: nothing attempted, or exactly what was blocked."""
    if not guard.attempts:
        console.print("[dim]No network connections were attempted.[/]")
        return
    console.print(
        f"[yellow]Warning: blocked {_plural(len(guard.attempts), 'network attempt')} during this run. "
        "Nothing was sent.[/]"
    )
    for attempt in guard.attempts:
        console.print(f"  {escape(attempt)}")


def _display(path: Path) -> str:
    """A path as short as possible: relative to the current folder, else to the home folder."""
    if path.is_relative_to(Path.cwd()):
        return str(path.relative_to(Path.cwd()))
    if path.is_relative_to(Path.home()):
        return str(Path("~") / path.relative_to(Path.home()))
    return str(path)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
