"""Interactive search: results update as you type, arrow keys move through them, Enter opens one.

The model loads once in the background and the index is held in memory, so
after the first few seconds every search takes milliseconds instead of the
seconds a one-off `dejanote search` spends starting up. The passage to
highlight is found only for the selected result, a few tens of milliseconds
after it is selected, so moving through results stays instant.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from rich.console import Group, RenderableType
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.timer import Timer
from textual.widgets import Footer, Input, OptionList, Static
from textual.widgets.option_list import Option
from textual.worker import get_current_worker

from dejanote.chunking import Chunk
from dejanote.display import plural, short_path
from dejanote.editor import editor_command
from dejanote.privacy import NetworkGuard
from dejanote.search import Passage, PassageEmbedder, SearchResult, best_passages, passage_line, search
from dejanote.store import IndexSnapshot

RESULTS = 20
HIGHLIGHT = Style(bold=True, bgcolor="#4a4220")  # a highlighter pen that keeps text readable on the dark theme


class BrowseApp(App[str | None]):
    """Browse the index interactively.

    With an editor, Enter opens the selected note at the highlighted line and
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
        embedder: PassageEmbedder,
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
        self.results_query = ""  # the query self.results answer
        self.selected: SearchResult | None = None
        self.highlight: Passage | None = None  # in the selected result, once found
        self._highlights: dict[tuple[str, Chunk], Passage | None] = {}  # found for self.results
        self._requested: set[tuple[str, Chunk]] = set()
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
        self.results_query = query
        self._highlights.clear()
        self._requested.clear()
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options([Option(_result_line(result, self.root)) for result in results])
        if results:
            options.highlighted = 0
        self._select(results[0] if results else None)
        self._refresh_status()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index < len(self.results):
            self._select(self.results[event.option_index])

    def _select(self, result: SearchResult | None) -> None:
        self.selected = result
        self.highlight = None
        if result is not None:
            key = (result.path, result.chunk)
            if key in self._highlights:
                self.highlight = self._highlights[key]
            elif key not in self._requested:
                self._requested.add(key)
                self.find_highlight(self.results_query, result)
        self._show_preview()

    @work(thread=True, group="highlight")
    def find_highlight(self, query: str, result: SearchResult) -> None:
        [passage] = best_passages(self.embedder, query, [result.chunk.text])
        self.call_from_thread(self._highlight_found, query, result, passage)

    def _highlight_found(self, query: str, result: SearchResult, passage: Passage | None) -> None:
        if query != self.results_query:
            return  # the results have changed since
        self._highlights[(result.path, result.chunk)] = passage
        if result is self.selected:  # not if the selection has moved on meanwhile
            self.highlight = passage
            self._show_preview()

    def action_move(self, step: int) -> None:
        options = self.query_one(OptionList)
        if step > 0:
            options.action_cursor_down()
        else:
            options.action_cursor_up()

    def action_open(self) -> None:
        if self.selected is None:
            return
        path, line = self.selected.path, passage_line(self.selected.chunk, self.highlight)
        if self.editor is None:
            self.exit(f"{short_path(Path(path))}:{line}")
        else:
            self.launch(editor_command(self.editor, path, line))

    def _run_in_terminal(self, command: list[str]) -> None:
        """Hand the terminal to the editor, and take it back when the editor closes."""
        with self.suspend():
            subprocess.run(command, check=False)

    def _show_preview(self) -> None:
        result = self.selected
        self.query_one("#preview", Static).update(_preview(result, self.root, self.highlight) if result else "")
        self.query_one("#preview-pane", VerticalScroll).scroll_home(animate=False)
        if self.highlight is not None:
            self.call_after_refresh(self._scroll_to_highlight)

    def _scroll_to_highlight(self) -> None:
        """Bring the highlight into view when the section is too long to show whole."""
        preview = self.query_one("#preview", Static)
        pane = self.query_one("#preview-pane", VerticalScroll)
        options = self.console.options.update_width(preview.size.width)
        lines = self.console.render_lines(preview.content, options, pad=False)
        rows = [y for y, line in enumerate(lines) if _shows_highlight(line)]
        if rows and rows[-1] >= pane.scrollable_content_region.height:
            pane.scroll_to(y=max(0, rows[0] - 2), animate=False)

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


def _preview(result: SearchResult, root: Path | None, highlight: Passage | None) -> RenderableType:
    chunk = result.chunk
    if chunk.start_line == chunk.end_line:
        lines = f"line {chunk.start_line}"
    else:
        lines = f"lines {chunk.start_line}-{chunk.end_line}"
    parts: list[RenderableType] = [
        Text(f"{_relative(result.path, root)}, {lines}", style="bold cyan"),
        Text(" > ".join(chunk.headings), style="dim"),
        Text(""),
        _numbered(chunk, highlight),
    ]
    if result.also:
        parts.append(Text("\nAlso matches in this note:", style="dim"))
        for other in result.also:
            name = other.chunk.headings[-1] if len(other.chunk.headings) > 1 else "intro"
            parts.append(Text(f"  {name}, line {other.chunk.start_line}", style="dim"))
    return Group(*parts)


def _numbered(chunk: Chunk, highlight: Passage | None) -> Table:
    """The section as it is in the note, each line numbered as in the note, the highlight marked.

    Plain text rather than rendered Markdown: the line numbers then match what
    the editor shows, and nothing in the preview is a link that could be clicked.
    """
    text = Text(chunk.text)
    highlighted_lines = range(0)
    if highlight is not None:
        text.stylize(HIGHLIGHT, highlight.start, highlight.end)
        first = passage_line(chunk, highlight)
        highlighted_lines = range(first, first + chunk.text.count("\n", highlight.start, highlight.end) + 1)
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", no_wrap=True)
    table.add_column()
    for number, line in enumerate(text.split("\n", allow_blank=True), start=chunk.start_line):
        table.add_row(Text(str(number), style="bold" if number in highlighted_lines else "dim"), line)
    return table


def _shows_highlight(line: list[Segment]) -> bool:
    return any(segment.style is not None and segment.style.bgcolor == HIGHLIGHT.bgcolor for segment in line)
