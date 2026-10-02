"""Interactive search: results update as you type, arrow keys move through them, Enter opens one.

The model loads once in the background and the index is held in memory, so
after the first few seconds every search takes milliseconds instead of the
seconds a one-off `dejanote search` spends starting up.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.timer import Timer
from textual.widgets import Footer, Input, OptionList, Static
from textual.widgets.option_list import Option
from textual.worker import get_current_worker

from dejanote.chunking import MARKDOWN_SUFFIXES
from dejanote.display import plural, short_path
from dejanote.editor import editor_command
from dejanote.privacy import NetworkGuard
from dejanote.search import QueryEmbedder, SearchResult, search
from dejanote.store import IndexSnapshot

RESULTS = 20


class BrowseApp(App[str | None]):
    """Browse the index interactively.

    With an editor, Enter opens the selected note at the matching line and
    browsing continues. Without one, Enter quits and the app returns the
    note's "path:line", which the command prints for use in scripts.
    """

    TITLE = "dejanote"
    ENABLE_COMMAND_PALETTE = False  # its extras, such as saving screenshots, are not needed here
    CSS = """
    #query { margin-bottom: 1; }
    #body { height: 1fr; }
    #results { width: 2fr; border: none; text-wrap: nowrap; text-overflow: ellipsis; }
    #preview-pane { width: 3fr; padding: 0 1; border-left: solid $panel-lighten-2; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("down", "move(1)", "next", priority=True),
        Binding("up", "move(-1)", "previous", priority=True),
        Binding("enter", "open", "open", priority=True),
        Binding("escape", "quit", "quit", priority=True),
        Binding("ctrl+c", "quit", "quit", show=False, priority=True),
    ]

    def __init__(
        self,
        index: IndexSnapshot,
        embedder: QueryEmbedder,
        *,
        query: str = "",
        guard: NetworkGuard | None = None,
        editor: str | None = None,
        launch: Callable[[list[str]], object] | None = None,
        debounce: float = 0.15,
    ):
        super().__init__()
        self.index = index
        self.root = _common_folder(index.paths())  # paths are shown relative to it
        self.embedder = embedder
        self.initial_query = query
        self.guard = guard or NetworkGuard()
        self.editor = editor
        self.launch = launch or self._run_in_terminal
        self.debounce = debounce
        self.results: list[SearchResult] = []
        self.selected: SearchResult | None = None
        self._model_ready = threading.Event()
        self._model_state = "loading the model"
        self._timer: Timer | None = None
        self._scheduled: str | None = None

    def compose(self) -> ComposeResult:
        yield Input(value=self.initial_query, placeholder="Search your notes by meaning", id="query")
        with Horizontal(id="body"):
            yield OptionList(id="results")
            with VerticalScroll(id="preview-pane"):
                yield Static(id="preview")
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self._refresh_status()
        self.warm_up()
        self._schedule(self.initial_query, delay=0)

    @work(thread=True, group="model")
    def warm_up(self) -> None:
        self.embedder.embed_query("warm up")  # the model loads on first use
        self._model_ready.set()
        self.call_from_thread(self._model_loaded)

    def _model_loaded(self) -> None:
        self._model_state = "ready"
        self._refresh_status()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._schedule(event.value, delay=self.debounce)

    def _schedule(self, query: str, delay: float) -> None:
        """Search for query once typing pauses; repeated requests for the same query are dropped."""
        if query == self._scheduled:
            return
        self._scheduled = query
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if delay:
            self._timer = self.set_timer(delay, lambda: self.run_search(query))
        else:  # Textual timers cannot have a zero delay
            self.run_search(query)

    @work(thread=True, exclusive=True, group="search")
    def run_search(self, query: str) -> None:
        results: list[SearchResult] = []
        if query.strip():
            self._model_ready.wait()
            results = search(self.index, self.embedder, query, limit=RESULTS)
        if not get_current_worker().is_cancelled:
            self.call_from_thread(self._show_results, query, results)

    def _show_results(self, query: str, results: list[SearchResult]) -> None:
        if query != self.query_one(Input).value:
            return  # typing went on; a newer search is on its way
        self.results = results
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options([Option(_result_line(result, self.root)) for result in results])
        if results:
            options.highlighted = 0
        self._preview(results[0] if results else None)
        self._refresh_status()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index < len(self.results):
            self._preview(self.results[event.option_index])

    def action_move(self, step: int) -> None:
        options = self.query_one(OptionList)
        if step > 0:
            options.action_cursor_down()
        else:
            options.action_cursor_up()

    def action_open(self) -> None:
        if self.selected is None:
            return
        path, line = self.selected.path, self.selected.chunk.start_line
        if self.editor is None:
            self.exit(f"{short_path(Path(path))}:{line}")
        else:
            self.launch(editor_command(self.editor, path, line))

    def _run_in_terminal(self, command: list[str]) -> None:
        """Hand the terminal to the editor, and take it back when the editor closes."""
        with self.suspend():
            subprocess.run(command, check=False)

    def _preview(self, result: SearchResult | None) -> None:
        self.selected = result
        self.query_one("#preview", Static).update(_preview(result, self.root) if result else "")
        self.query_one("#preview-pane", VerticalScroll).scroll_home(animate=False)

    def _refresh_status(self) -> None:
        notes, chunks = self.index.counts()
        status = Text(f"{plural(notes, 'note')}, {plural(chunks, 'chunk')} · {self._model_state} · ")
        blocked = len(self.guard.attempts)
        if blocked:
            status.append(f"{plural(blocked, 'network attempt')} blocked", style="bold red")
        else:
            status.append("no network connections attempted", style="green")
        self.query_one("#status", Static).update(status)


def _common_folder(paths: list[str]) -> Path | None:
    """The deepest folder that holds every note, or None for an empty index."""
    if not paths:
        return None
    return Path(os.path.commonpath([str(Path(path).parent) for path in paths]))


def _relative(path: str, root: Path | None) -> str:
    return Path(path).relative_to(root).as_posix() if root else short_path(Path(path))


def _result_line(result: SearchResult, root: Path | None) -> Text:
    """Two lines per result, cut with an ellipsis rather than wrapped, so the list stays scannable."""
    line = Text(no_wrap=True, overflow="ellipsis")
    line.append(f"{result.score:.2f}  ", style="bold")
    line.append(f"{_relative(result.path, root)}:{result.chunk.start_line}", style="cyan")
    line.append("\n      " + " > ".join(result.chunk.headings), style="dim")
    return line


def _preview(result: SearchResult, root: Path | None) -> RenderableType:
    chunk = result.chunk
    if chunk.start_line == chunk.end_line:
        lines = f"line {chunk.start_line}"
    else:
        lines = f"lines {chunk.start_line}-{chunk.end_line}"
    parts: list[RenderableType] = [
        Text(f"{_relative(result.path, root)}, {lines}", style="bold cyan"),
        Text(" > ".join(chunk.headings), style="dim"),
        Text(""),
    ]
    if Path(result.path).suffix.lower() in MARKDOWN_SUFFIXES:
        parts.append(Markdown(chunk.text, hyperlinks=False))  # nothing clickable that could open a browser
    else:
        parts.append(Text(chunk.text))
    if result.also:
        parts.append(Text("\nAlso matches in this note:", style="dim"))
        for other in result.also:
            name = other.chunk.headings[-1] if len(other.chunk.headings) > 1 else "intro"
            parts.append(Text(f"  {name}, line {other.chunk.start_line}", style="dim"))
    return Group(*parts)
