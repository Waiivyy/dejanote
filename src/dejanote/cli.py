"""Command line interface."""

from __future__ import annotations

import os
import tempfile
import textwrap
import time
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.text import Text

from dejanote import __version__, config
from dejanote.display import plural, short_path
from dejanote.editor import configured_editor
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
from dejanote.indexer import IndexReport, index_folder
from dejanote.privacy import NetworkGuard, block_network
from dejanote.search import Passage, SearchResult, best_passages, passage_line, search, snippet_lines
from dejanote.store import IndexMismatchError, Store

app = typer.Typer(
    help="Search your notes by meaning, entirely offline.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(highlight=False, soft_wrap=True)
HIGHLIGHT = "bold"  # the passage that best matches the query; plain bold reads on light and dark themes


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
            console.print(
                f"[yellow]The model at {escape(short_path(target))} failed verification:[/] {escape(str(err))}"
            )
        else:
            console.print(f"Model already downloaded and verified: {escape(short_path(target))}")
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
    console.print(f"  to        {escape(short_path(target))}\n")
    if not yes:
        typer.confirm("Download it now?", abort=True)

    with console.status("Downloading and verifying..."):
        try:
            download_model(spec, target)
        except Exception as err:  # network, proxy or integrity problems: explain, don't dump a traceback
            console.print(f"[red]Download failed:[/] {err}")
            console.print("Nothing was saved. Check your connection and run `dejanote setup` again.")
            raise typer.Exit(1) from None
    console.print(f"[green]Done.[/] Model saved to {escape(short_path(target))}")
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
    summary = f"Indexed {plural(report.notes, 'note')} from {escape(short_path(folder))}"
    summary += f" ({changes})." if changes else "."
    if report.chunks:
        summary += f" Embedded {plural(report.chunks, 'chunk')} in {elapsed:.1f}s."
    else:
        summary += f" Nothing new to embed ({elapsed:.1f}s)."
    console.print(summary)
    if report.removed:
        verb = "exists" if report.removed == 1 else "exist"
        console.print(f"Removed {plural(report.removed, 'note')} that no longer {verb}.")
    if report.skipped:
        console.print(f"[yellow]Skipped {plural(len(report.skipped), 'file')}:[/]")
        for path, reason in report.skipped:
            console.print(f"  {escape(short_path(Path(path)))} ({escape(reason)})")
    console.print(f"Index: {escape(short_path(config.index_path()))}")
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
        highlights = best_passages(embedder, query, [result.chunk.text for result in results])
    if results:
        _print_results(results, highlights)
    else:
        console.print("The index is empty. Run `dejanote index <folder>` to add notes.", markup=False)
    _report_network(guard)


@app.command()
def browse(query: str = typer.Argument("", help="Start with this search.")) -> None:
    """Search interactively: results update as you type. Runs fully offline.

    Arrow keys move through the results and Enter opens the note in $VISUAL or
    $EDITOR at the matching line. Without an editor, Enter quits and prints the
    note's path:line. Escape quits.
    """
    os.environ.pop("TEXTUAL", None)  # keeps Textual's devtools client off, even if configured
    with block_network() as guard:
        embedder = _load_embedder()
        if not config.index_path().exists():
            console.print("There is no index yet. Run `dejanote index <folder>` first.", markup=False)
            raise typer.Exit(1)
        with _open_index(embedder) as store:
            index = store.snapshot()
        from dejanote.tui import BrowseApp  # Textual is only imported for this command

        chosen = BrowseApp(index, embedder, query=query, guard=guard, editor=configured_editor()).run()
    if guard.attempts:
        _report_network(guard, Console(stderr=True, highlight=False))
    if chosen:
        typer.echo(chosen)


@app.command()
def watch(
    folder: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Folder of notes to keep indexed. Subfolders are included.",
    ),
) -> None:
    """Keep the index current while you edit: notes are reindexed as soon as they change.

    Runs fully offline until you press Ctrl+C. The model stays loaded, so each
    change is searchable within a second or two of saving.
    """
    from dejanote.watcher import watch_folder  # watchfiles is only imported for this command

    with block_network() as guard:
        embedder = _load_embedder()
        with _open_index(embedder, rebuild_if_incompatible=True) as store:
            if store.rebuilt:
                console.print(
                    "[yellow]The index was made with a different model or chunking version, "
                    "so it is being rebuilt from scratch.[/]"
                )
            with console.status("Loading the model so changes are indexed as soon as you save..."):
                embedder.embed_query("warm up")
            console.print(f"Watching {escape(short_path(folder))} for changes. Press Ctrl+C to stop.")
            updates = 0
            attempts_shown = 0

            def on_update(report: IndexReport, seconds: float) -> None:
                nonlocal updates, attempts_shown
                _print_watch_update(report, seconds, folder, first=updates == 0)
                updates += 1
                for attempt in guard.attempts[attempts_shown:]:  # report blocked attempts as they happen
                    console.print(f"[yellow]Blocked a network attempt, nothing was sent:[/] {escape(attempt)}")
                attempts_shown = len(guard.attempts)

            try:
                watch_folder(folder, store, embedder, on_update)
            except KeyboardInterrupt:
                pass
    console.print("Stopped watching.")
    _report_network(guard)


def _print_watch_update(report: IndexReport, seconds: float, root: Path, first: bool) -> None:
    """One line per update: what changed, or for the first update, where the index stands."""
    stamp = time.strftime("%H:%M:%S")
    if first and not report.changes:
        console.print(f"[dim]{stamp}[/]  up to date: {plural(report.notes, 'note')}")
    elif report.changes:
        changes = _describe_changes(report.changes, root)
        embedded = f" ({plural(report.chunks, 'chunk')} embedded, {seconds:.2f}s)" if report.chunks else ""
        console.print(f"[dim]{stamp}[/]  {changes}{embedded}")
    for path, reason in report.skipped:
        console.print(f"[dim]{stamp}[/]  [yellow]skipped {escape(short_path(Path(path)))} ({escape(reason)})[/]")


def _describe_changes(changes: list[tuple[str, str]], root: Path, listed: int = 3) -> str:
    """Name the notes in a small batch; count them in a large one, such as a first run."""
    if len(changes) <= listed:
        return ", ".join(
            f"{kind} {escape(Path(path).relative_to(root).as_posix())}" for kind, path in sorted(changes)
        )
    counts = Counter(kind for kind, _ in changes)
    kinds = [kind for kind in ("added", "updated", "removed") if counts[kind]]
    return ", ".join(f"{kind} {plural(counts[kind], 'note')}" for kind in kinds)


_SAMPLE_NOTES = {
    "sourdough.md": "# Sourdough starter\n\nDiscard half, then feed it flour and water twice a day.\n"
    "Grey liquid on top means it is hungry, not dead.\n",
    "boiler.md": "# Boiler\n\nIf the gauge drops below one bar, top it up with the filling loop.\n",
    "japan.md": "# Japan trip\n\nPut a Suica card in the phone wallet for trains and buses.\n",
}
_SAMPLE_QUERY, _SAMPLE_ANSWER = "keeping my bread yeast culture alive", "sourdough.md"


@app.command()
def verify() -> None:
    """Check the model files, then index and search sample notes with the network blocked."""
    console.print("Checking dejanote with every network connection blocked.")
    with block_network() as guard:
        embedder = _load_embedder()
        try:
            verify_model_files(embedder.model_dir, embedder.spec)
        except ModelIntegrityError as err:
            console.print(f"[red]The model files do not match their pinned hashes:[/] {escape(str(err))}")
            console.print("Run `dejanote setup` to download a fresh copy.", markup=False)
            raise typer.Exit(1) from None
        console.print("  Model files match their pinned sha256 hashes.")
        with tempfile.TemporaryDirectory() as scratch:
            notes = Path(scratch) / "notes"
            notes.mkdir()
            for name, text in _SAMPLE_NOTES.items():
                (notes / name).write_text(text)
            with Store(Path(scratch) / "index.db", embedder.model_id, embedder.dimension) as store:
                report = index_folder(notes, store, embedder)
                results = search(store, embedder, _SAMPLE_QUERY, limit=1)
    found = Path(results[0].path).name if results else "nothing"
    console.print(f"  Indexed {plural(report.notes, 'sample note')} into a throwaway index.")
    console.print(f'  Searched for "{_SAMPLE_QUERY}": best match {found}.')

    if guard.attempts:
        console.print(f"[red]Network connections attempted: {len(guard.attempts)}. All were blocked:[/]")
        for attempt in guard.attempts:
            console.print(f"  {escape(attempt)}")
        raise typer.Exit(1)
    if found != _SAMPLE_ANSWER:
        console.print(f"[red]Search went wrong: expected {_SAMPLE_ANSWER} first.[/]")
        raise typer.Exit(1)
    console.print("[green]Network connections attempted: 0.[/] Indexing and search work fully offline.")
    console.print(
        "For a check that does not rely on dejanote's own guard, see \"Verify it yourself\" in the README."
    )


def _print_results(results: list[SearchResult], highlights: list[Passage | None]) -> None:
    indent = " " * 6
    width = max(40, console.width)
    for result, highlight in zip(results, highlights):
        location = f"{short_path(Path(result.path))}:{passage_line(result.chunk, highlight)}"
        console.print(f"[bold]{result.score:.2f}[/]  [cyan]{escape(location)}[/]")
        console.print(f"{indent}[dim]{escape(' > '.join(result.chunk.headings))}[/]")
        for line in snippet_lines(result.chunk.text, focus=highlight):
            text = Text.assemble(*((part, HIGHLIGHT if is_focus else "") for part, is_focus in line))
            for wrapped in text.wrap(console, width - len(indent)):
                wrapped.rstrip()  # in place: no trailing spaces when output is piped
                console.print(Text(indent) + wrapped)
        if result.also:
            # A note's untitled opening only carries the note's own title, so call it the intro.
            others = ", ".join(
                f"{s.chunk.headings[-1] if len(s.chunk.headings) > 1 else 'intro'} (line {s.chunk.start_line})"
                for s in result.also
            )
            wrapped = textwrap.fill(
                f"also: {others}", width=width, initial_indent=indent, subsequent_indent=indent
            )
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


def _report_network(guard: NetworkGuard, out: Console | None = None) -> None:
    """Every offline command ends with a receipt: nothing attempted, or exactly what was blocked."""
    out = out or console
    if not guard.attempts:
        out.print("[dim]No network connections were attempted.[/]")
        return
    out.print(
        f"[yellow]Warning: blocked {plural(len(guard.attempts), 'network attempt')} during this run. "
        "Nothing was sent.[/]"
    )
    for attempt in guard.attempts:
        out.print(f"  {escape(attempt)}")
