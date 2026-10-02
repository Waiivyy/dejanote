"""The interactive browser, driven headlessly: type to search, move with the arrow keys, open."""

from __future__ import annotations

import asyncio
import html
from pathlib import Path

import pytest

from conftest import FakeEmbedder
from dejanote.indexer import index_folder
from dejanote.privacy import NetworkGuard
from dejanote.store import Store
from dejanote.tui import BrowseApp

EXAMPLES = Path(__file__).parent.parent / "examples" / "notes"

# With the fake embedder, similarity is word overlap, so these rank predictably.
NOTES = {
    "boiler.md": "# Boiler\n\n## Pressure\n\nboiler pressure gauge low, top up with the filling loop\n\n"
    "## Radiators\n\nbleed the radiators with the key in the drawer\n",
    "sourdough.md": "# Sourdough\n\nfeed the sourdough starter flour and water\n",
    "japan.md": "# Japan\n\nput a suica card in the phone for trains in tokyo\n",
}


@pytest.fixture
def snapshot(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    for name, text in NOTES.items():
        (notes / name).write_text(text)
    embedder = FakeEmbedder()
    with Store(tmp_path / "index.db", embedder.model_id, embedder.dimension) as store:
        index_folder(notes, store, embedder)
        return store.snapshot()


def run(app, script):
    """Run the app headless, let script(pilot) drive it, and return what the app exited with."""

    async def main():
        async with app.run_test(size=(110, 30)) as pilot:
            await script(pilot)
        return app.return_value

    return asyncio.run(main())


async def settle(pilot):
    """Wait for the debounce timer, the search it starts, and the results reaching the screen."""
    await pilot.pause(0.3)
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def on_screen(app) -> str:
    return html.unescape(app.export_screenshot()).replace("\xa0", " ")


def test_typing_lists_the_best_matching_note_first_and_previews_it(snapshot):
    app = BrowseApp(snapshot, FakeEmbedder(), debounce=0.05)

    async def script(pilot):
        await pilot.press(*"boiler pressure")
        await settle(pilot)
        assert Path(app.results[0].path).name == "boiler.md"
        screen = on_screen(app)
        assert "boiler.md:5" in screen  # the results list shows where the match is
        assert "filling loop" in screen  # the preview shows the matching section

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
