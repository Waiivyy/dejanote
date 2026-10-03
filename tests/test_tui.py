"""The interactive browser, driven headlessly: type to search, move with the arrow keys, open."""

from __future__ import annotations

import asyncio
import html
import threading
from pathlib import Path

import pytest
from rich.console import Console
from textual.widgets import OptionList, Static

from conftest import FakeEmbedder
from dejanote.indexer import index_folder
from dejanote.privacy import NetworkGuard
from dejanote.store import Store
from dejanote.tui import BrowseApp

EXAMPLES = Path(__file__).parent.parent / "examples" / "notes"

# With the fake embedder, similarity is word overlap, so these rank predictably.
NOTES = {
    "home/boiler.md": "# Boiler\n\n## Pressure\n\nboiler pressure gauge low, top up with the filling loop\n\n"
    "## Radiators\n\nbleed the radiators with the key in the drawer\n",
    "cooking/sourdough.md": "# Sourdough\n\nfeed the sourdough starter flour and water\n",
    "travel/japan.md": "# Japan\n\nput a suica card in the phone for trains in tokyo\n",
}


# Several sentences each, so there is a passage to pick out. With the fake embedder,
# "hurts to walk see a physio" ranks the shins note first and the knee note second.
SHINS = "# Shins\n\nRest for a week.\nIce the sore spot twice a day.\nIf it hurts to walk, see a physio.\n"
KNEE = "# Knee\n\nThe knee is fine.\nIt hurts only on stairs.\n"
HURTS = "hurts to walk see a physio"


def index_of(tmp_path, notes: dict[str, str]):
    folder = tmp_path / "notes"
    for name, text in notes.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)
    embedder = FakeEmbedder()
    with Store(tmp_path / "index.db", embedder.model_id, embedder.dimension) as store:
        index_folder(folder, store, embedder)
        return store.snapshot()


@pytest.fixture
def snapshot(tmp_path):
    return index_of(tmp_path, NOTES)


def run(app, script, size=(110, 30)):
    """Run the app headless, let script(pilot) drive it, and return what the app exited with."""

    async def main():
        async with app.run_test(size=size) as pilot:
            await script(pilot)
        return app.return_value

    return asyncio.run(main())


async def settle(pilot):
    """Wait for the debounce timer, then for every search and highlight it leads to, and the screen."""
    await pilot.pause(0.3)
    while pilot.app.workers:
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()


async def until(pilot, condition, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if condition():
            return
        await pilot.pause(0.02)
    raise AssertionError("the app never got there")


def on_screen(app) -> str:
    return html.unescape(app.export_screenshot()).replace("\xa0", " ")


def preview_lines(app) -> list[list]:
    """The preview as rendered at its width on screen: a list of segments per line."""
    preview = app.query_one("#preview", Static)
    console = Console(width=preview.size.width, force_terminal=True, color_system="truecolor")
    return console.render_lines(preview.content, console.options, pad=False)


def preview_text(app) -> list[str]:
    return ["".join(segment.text for segment in line).rstrip() for line in preview_lines(app)]


def highlighted(app) -> str:
    """The text the preview marks with a background colour, wrapping undone."""
    marked = [s.text for line in preview_lines(app) for s in line if s.style and s.style.bgcolor]
    return " ".join("".join(marked).split())


class HeldBack(FakeEmbedder):
    """Holds back scoring the passages of any section that mentions `word`, until released."""

    def __init__(self, word: str):
        super().__init__()
        self.word = word
        self.release = threading.Event()

    def embed_documents(self, texts):
        if len(texts) > 1 and any(self.word in text for text in texts):
            self.release.wait(timeout=10)
        return super().embed_documents(texts)


def test_typing_lists_the_best_matching_note_first_and_previews_it(snapshot):
    app = BrowseApp(snapshot, FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await pilot.press(*"boiler pressure")
        await settle(pilot)
        assert Path(app.results[0].path).name == "boiler.md"
        # The entry's own text: what wraps where on screen depends on the machine's paths.
        entry = app.query_one(OptionList).get_option_at_index(0).prompt.plain
        assert "boiler.md:5" in entry  # the results list shows where the match is
        assert "filling loop" in on_screen(app)  # the preview shows the matching section

    run(app, script)


def test_the_arrow_keys_move_through_results_and_the_preview_follows(snapshot):
    app = BrowseApp(snapshot, FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await pilot.press(*"boiler pressure")
        await settle(pilot)
        await pilot.press("down")
        await pilot.pause()
        assert app.selected == app.results[1]
        assert app.results[1].chunk.text.split()[0] in on_screen(app)
        await pilot.press("up")
        await pilot.pause()
        assert app.selected == app.results[0]

    run(app, script)


def test_a_starting_query_is_searched_right_away(snapshot):
    app = BrowseApp(snapshot, FakeEmbedder(), query="sourdough starter", debounce=0.05)

    async def script(pilot):
        await settle(pilot)
        assert Path(app.results[0].path).name == "sourdough.md"

    run(app, script)


def test_enter_opens_the_note_in_the_editor_at_the_matching_line_and_keeps_browsing(snapshot):
    launched = []
    app = BrowseApp(snapshot, FakeEmbedder(), editor="vim", launch=launched.append, debounce=0.05)

    async def script(pilot):
        await pilot.press(*"sourdough starter")
        await settle(pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert launched == [["vim", "+3", app.results[0].path]]
        assert app.is_running

    run(app, script)


def test_without_an_editor_enter_quits_with_the_location(snapshot):
    app = BrowseApp(snapshot, FakeEmbedder(), editor=None, debounce=0.05)

    async def script(pilot):
        await pilot.press(*"sourdough starter")
        await settle(pilot)
        await pilot.press("enter")

    assert run(app, script).endswith("sourdough.md:3")


def test_the_preview_highlights_the_passage_closest_to_the_query(tmp_path):
    app = BrowseApp(index_of(tmp_path, {"health/shins.md": SHINS}), FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await pilot.press(*HURTS)
        await settle(pilot)
        assert highlighted(app) == "If it hurts to walk, see a physio."

    run(app, script)


def test_the_preview_numbers_each_line_as_in_the_note(tmp_path):
    app = BrowseApp(index_of(tmp_path, {"health/shins.md": SHINS}), FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await pilot.press(*HURTS)
        await settle(pilot)
        shown = [line.strip() for line in preview_text(app)]
        assert shown[-3:] == [
            "3 Rest for a week.",
            "4 Ice the sore spot twice a day.",
            "5 If it hurts to walk, see a physio.",
        ]

    run(app, script)


def test_enter_opens_the_editor_at_the_highlighted_line(tmp_path):
    launched = []
    index = index_of(tmp_path, {"health/shins.md": SHINS})
    app = BrowseApp(index, FakeEmbedder(), editor="vim", launch=launched.append, debounce=0.05)

    async def script(pilot):
        await pilot.press(*HURTS)
        await settle(pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert launched == [["vim", "+5", app.results[0].path]]

    run(app, script)


def test_without_an_editor_enter_quits_with_the_highlighted_line(tmp_path):
    app = BrowseApp(index_of(tmp_path, {"health/shins.md": SHINS}), FakeEmbedder(), editor=None, debounce=0.05)

    async def script(pilot):
        await pilot.press(*HURTS)
        await settle(pilot)
        await pilot.press("enter")

    assert run(app, script).endswith("health/shins.md:5")


def test_a_highlight_that_arrives_after_moving_on_is_not_shown(tmp_path):
    embedder = HeldBack("physio")  # the shins note's highlight is held back
    app = BrowseApp(index_of(tmp_path, {"shins.md": SHINS, "knee.md": KNEE}), embedder, debounce=0.05)

    async def script(pilot):
        await pilot.press(*HURTS)
        await until(pilot, lambda: len(app.results) == 2)
        assert Path(app.results[0].path).name == "shins.md"
        await pilot.press("down")
        await until(pilot, lambda: app.highlight is not None)  # the knee note's highlight
        embedder.release.set()  # the shins note's highlight arrives now, too late
        await settle(pilot)
        assert app.selected is app.results[1]
        assert highlighted(app) == "It hurts only on stairs."

    run(app, script)


def test_a_highlight_further_down_a_long_section_is_scrolled_into_view(tmp_path):
    steps = "\n".join(f"- step {n} done" for n in range(1, 31))
    checklist = f"# Checklist\n\n{steps}\n- the boiler gauge reads low\n"  # one section, 33 lines
    app = BrowseApp(index_of(tmp_path, {"checklist.md": checklist}), FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await pilot.press(*"boiler gauge low")
        await settle(pilot)
        assert highlighted(app) == "- the boiler gauge reads low"
        assert "the boiler gauge reads low" in on_screen(app)

    run(app, script, size=(110, 20))


def test_escape_quits_without_choosing_anything(snapshot):
    app = BrowseApp(snapshot, FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await pilot.press(*"boiler")
        await settle(pilot)
        await pilot.press("escape")

    assert run(app, script) is None


def test_the_status_line_shows_the_index_size_and_the_network_receipt(snapshot):
    app = BrowseApp(snapshot, FakeEmbedder(), guard=NetworkGuard(), debounce=0.05)

    async def script(pilot):
        await settle(pilot)
        screen = on_screen(app)
        assert "3 notes" in screen
        assert "no network connections attempted" in screen.lower()

    run(app, script)


def test_blocked_network_attempts_show_in_the_status_line(snapshot):
    guard = NetworkGuard()
    guard.attempts.append("DNS lookup of telemetry.example.com")
    app = BrowseApp(snapshot, FakeEmbedder(), guard=guard, debounce=0.05)

    async def script(pilot):
        await pilot.press(*"boiler")
        await settle(pilot)
        assert "1 network attempt blocked" in on_screen(app)

    run(app, script)


def test_paths_are_shown_relative_to_the_notes_folder(snapshot, tmp_path):
    app = BrowseApp(snapshot, FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await pilot.press(*"boiler pressure")
        await settle(pilot)
        # The entry's own text, since wrapping on screen could hide a long prefix.
        entry = app.query_one(OptionList).get_option_at_index(0).prompt.plain
        assert entry.split("\n")[0].endswith("  home/boiler.md:5")
        assert str(tmp_path) not in entry  # no long absolute prefix crowding the list

    run(app, script)


def test_each_result_takes_exactly_two_lines_however_long_its_heading(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    long_heading = "a heading that goes on and on " * 6
    (notes / "long.md").write_text(f"# {long_heading}\n\nboiler pressure gauge\n")
    (notes / "short.md").write_text("# Short\n\nboiler notes\n")
    embedder = FakeEmbedder()
    with Store(tmp_path / "index.db", embedder.model_id, embedder.dimension) as store:
        index_folder(notes, store, embedder)
        index = store.snapshot()
    app = BrowseApp(index, embedder, debounce=0.05)

    async def script(pilot):
        await pilot.press(*"boiler pressure")
        await settle(pilot)
        options = app.query_one(OptionList)
        assert options.option_count == 2
        assert options.virtual_size.height == 4  # cut with an ellipsis, never wrapped

    run(app, script)


def test_the_command_palette_is_off(snapshot):
    # Textual's palette offers actions such as saving screenshots; the browser has one job.
    app = BrowseApp(snapshot, FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await settle(pilot)
        assert "palette" not in on_screen(app)

    run(app, script)


@pytest.mark.model
def test_with_the_real_model_a_paraphrase_finds_the_journal_entry(tmp_path, embedder):
    with Store(tmp_path / "index.db", embedder.model_id, embedder.dimension) as store:
        index_folder(EXAMPLES, store, embedder)
        snapshot = store.snapshot()
    app = BrowseApp(snapshot, embedder, query="felt burned out and needed a break", debounce=0.05)

    async def script(pilot):
        await settle(pilot)
        assert Path(app.results[0].path).relative_to(EXAMPLES).as_posix() == "journal/2024-05-12.md"
        assert "completely drained" in on_screen(app)

    run(app, script)
