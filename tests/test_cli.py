"""CLI behaviour, run in-process under the network guard."""

from __future__ import annotations

import os
import re
import socket
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner
from watchfiles import Change

from dejanote import cli
from dejanote.cli import app
from dejanote.embedding import DEFAULT_MODEL
from dejanote.privacy import block_network
from dejanote.store import Store

runner = CliRunner()

EXAMPLES = Path(__file__).parent.parent / "examples" / "notes"
EXAMPLE_COUNT = len(list(EXAMPLES.rglob("*.md"))) + len(list(EXAMPLES.rglob("*.txt")))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DEJANOTE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def home_with_model(home, model_dir):
    (home / "models").mkdir(parents=True)
    (home / "models" / model_dir.name).symlink_to(model_dir)
    return home


def test_setup_declined_touches_neither_network_nor_disk(home):
    with block_network() as guard:
        result = runner.invoke(app, ["setup"], input="n\n")
    assert "huggingface.co" in result.output  # told where it would download from, before consenting
    assert result.exit_code == 1  # aborted by the user, not a usage error
    assert guard.attempts == []
    assert not (home / "models").exists() or not any((home / "models").iterdir())


@pytest.mark.model
def test_setup_with_verified_model_present_is_an_offline_no_op(home_with_model):
    with block_network() as guard:
        result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0, result.output
    assert "already" in result.output
    assert guard.attempts == []


def test_index_without_the_model_explains_how_to_get_it(home, tmp_path):
    (tmp_path / "notes").mkdir()
    with block_network() as guard:
        result = runner.invoke(app, ["index", str(tmp_path / "notes")])
    assert result.exit_code == 1
    assert "dejanote setup" in result.output
    assert guard.attempts == []


@pytest.mark.model
def test_index_builds_the_index_offline_and_says_so(home_with_model):
    with block_network() as guard:
        result = runner.invoke(app, ["index", str(EXAMPLES)])
    assert result.exit_code == 0, result.output
    assert guard.attempts == []
    assert (home_with_model / "index.db").is_file()
    assert f"{EXAMPLE_COUNT} notes" in result.output
    assert "no network" in result.output.lower()
    assert result.output.startswith("Indexed")  # no progress-bar residue when output is not a terminal


@pytest.mark.model
def test_a_reindex_with_nothing_changed_does_not_even_load_the_model(home_with_model):
    runner.invoke(app, ["index", str(EXAMPLES)])
    # Swap the real model for files that cannot load: from now on any attempt to load it fails.
    model = home_with_model / "models" / DEFAULT_MODEL.name
    model.unlink()
    for name in DEFAULT_MODEL.files:
        (model / name).parent.mkdir(parents=True, exist_ok=True)
        (model / name).write_text("not a model")
    result = runner.invoke(app, ["index", str(EXAMPLES)])
    assert result.exit_code == 0, result.output
    assert f"{EXAMPLE_COUNT} unchanged" in result.output


@pytest.mark.model
def test_index_rebuilds_an_index_made_by_an_older_chunker(home_with_model):
    stale = Store(home_with_model / "index.db", DEFAULT_MODEL.model_id, DEFAULT_MODEL.dimension, chunker_version=0)
    stale.replace_file("/old/chunking/rules.md", "hash", [], np.empty((0, DEFAULT_MODEL.dimension)))
    stale.close()
    result = runner.invoke(app, ["index", str(EXAMPLES)])
    assert result.exit_code == 0, result.output
    assert "rebuilt" in result.output.lower()
    with Store(home_with_model / "index.db", DEFAULT_MODEL.model_id, DEFAULT_MODEL.dimension) as store:
        assert "/old/chunking/rules.md" not in store.file_hashes()
        assert store.counts()[0] == EXAMPLE_COUNT


@pytest.mark.model
def test_index_reports_files_it_had_to_skip(home_with_model, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "fine.md").write_text("A perfectly normal note.")
    (notes / "corrupt.md").write_bytes(b"\x00\x01 garbage")
    result = runner.invoke(app, ["index", str(notes)])
    assert result.exit_code == 0, result.output
    assert "corrupt.md" in result.output and "binary" in result.output


@pytest.mark.model
def test_index_warns_when_something_tries_the_network(home_with_model, monkeypatch):
    real_index_folder = cli.index_folder

    def misbehaving_index_folder(*args, **kwargs):
        try:  # a dependency phoning home in the middle of indexing
            socket.getaddrinfo("telemetry.example.com", 443)
        except OSError:
            pass
        return real_index_folder(*args, **kwargs)

    monkeypatch.setattr(cli, "index_folder", misbehaving_index_folder)
    with block_network() as outer_guard:  # so a regression fails the test instead of leaking
        result = runner.invoke(app, ["index", str(EXAMPLES)])
    assert result.exit_code == 0, result.output
    assert "blocked" in result.output.lower()
    assert "telemetry.example.com" in result.output
    assert outer_guard.attempts == []  # the command's own guard caught it first


@pytest.mark.model
def test_search_without_an_index_says_how_to_build_one_and_creates_nothing(home_with_model):
    with block_network() as guard:
        result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code == 1
    assert "dejanote index" in result.output
    assert not (home_with_model / "index.db").exists()
    assert guard.attempts == []


@pytest.mark.model
def test_search_finds_the_relevant_note_and_shows_where_it_is(home_with_model):
    runner.invoke(app, ["index", str(EXAMPLES)])
    with block_network() as guard:
        result = runner.invoke(app, ["search", "my bread starter smells like nail polish"])
    assert result.exit_code == 0, result.output
    first = result.output.split("\n\n")[0]
    assert re.search(r"cooking/sourdough-starter\.md:\d+", first)
    assert "When it goes wrong" in first
    assert guard.attempts == []
    assert "no network" in result.output.lower()
    assert not [line for line in result.output.splitlines() if line.endswith(" ")]  # clean when piped


@pytest.mark.model
def test_search_shows_as_many_results_as_asked_for(home_with_model):
    runner.invoke(app, ["index", str(EXAMPLES)])
    result = runner.invoke(app, ["search", "cooking dinner", "--limit", "3"])
    assert result.exit_code == 0, result.output
    assert len(re.findall(r"^-?\d\.\d\d  \S+:\d+$", result.output, flags=re.MULTILINE)) == 3


@pytest.mark.model
def test_search_on_an_index_built_for_another_model_says_to_reindex(home_with_model):
    Store(home_with_model / "index.db", "some-other-model@1", DEFAULT_MODEL.dimension).close()
    result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code == 1
    assert "dejanote index" in result.output


@pytest.mark.model
def test_search_lists_a_note_once_with_its_other_matching_sections(home_with_model):
    runner.invoke(app, ["index", str(EXAMPLES)])
    result = runner.invoke(app, ["search", "the heating stopped and the gauge is low"])
    assert result.exit_code == 0, result.output
    assert result.output.count("home/boiler-pressure.md:") == 1
    boiler_block = next(block for block in result.output.split("\n\n") if "boiler-pressure.md" in block)
    assert "also:" in boiler_block


@pytest.mark.model
def test_verify_passes_on_a_working_install_without_touching_the_network(home_with_model):
    with block_network() as guard:
        result = runner.invoke(app, ["verify"])
    assert result.exit_code == 0, result.output
    assert "attempted: 0" in result.output
    assert guard.attempts == []
    assert not (home_with_model / "index.db").exists()  # your own index is left alone


@pytest.mark.model
def test_verify_fails_when_something_tries_the_network(home_with_model, monkeypatch):
    real_index_folder = cli.index_folder

    def misbehaving_index_folder(*args, **kwargs):
        try:
            socket.getaddrinfo("telemetry.example.com", 443)
        except OSError:
            pass
        return real_index_folder(*args, **kwargs)

    monkeypatch.setattr(cli, "index_folder", misbehaving_index_folder)
    with block_network():
        result = runner.invoke(app, ["verify"])
    assert result.exit_code == 1
    assert "telemetry.example.com" in result.output


@pytest.mark.model
def test_verify_fails_on_a_tampered_model_file(home, model_dir):
    # Link every model file except one config file, which gets altered.
    copy = home / "models" / model_dir.name
    for name in DEFAULT_MODEL.files:
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        if name == "config.json":
            (copy / name).write_text((model_dir / name).read_text().replace("384", "385"))
        else:
            (copy / name).symlink_to(model_dir / name)
    result = runner.invoke(app, ["verify"])
    assert result.exit_code == 1
    assert "config.json" in result.output


def test_verify_without_the_model_explains_how_to_get_it(home):
    result = runner.invoke(app, ["verify"])
    assert result.exit_code == 1
    assert "dejanote setup" in result.output


def test_browse_without_the_model_explains_how_to_get_it(home):
    result = runner.invoke(app, ["browse"])
    assert result.exit_code == 1
    assert "dejanote setup" in result.output


@pytest.mark.model
def test_browse_without_an_index_says_how_to_build_one(home_with_model):
    result = runner.invoke(app, ["browse"])
    assert result.exit_code == 1
    assert "dejanote index" in result.output
    assert not (home_with_model / "index.db").exists()


@pytest.mark.model
def test_browse_runs_the_browser_inside_the_network_guard_without_textual_devtools(home_with_model, monkeypatch):
    runner.invoke(app, ["index", str(EXAMPLES)])
    monkeypatch.setenv("TEXTUAL", "devtools")  # would make Textual connect to a local devtools server
    seen = {}

    def probe(self):  # stands in for the interactive session, which needs a real terminal
        seen["devtools setting"] = os.environ.get("TEXTUAL")
        seen["query"] = self.initial_query
        try:
            socket.getaddrinfo("example.com", 443)
            seen["network"] = "open"
        except OSError:
            seen["network"] = "blocked"
        return "examples/notes/cooking/sourdough-starter.md:7"

    monkeypatch.setattr("dejanote.tui.BrowseApp.run", probe)
    with block_network() as outer_guard:  # so a regression fails the test instead of reaching the network
        result = runner.invoke(app, ["browse", "sourdough"])
    assert result.exit_code == 0, result.output
    assert seen == {"devtools setting": None, "query": "sourdough", "network": "blocked"}
    assert outer_guard.attempts == []  # the command's own guard did the blocking
    assert result.stdout.strip() == "examples/notes/cooking/sourdough-starter.md:7"  # clean for scripts
    assert "DNS lookup of example.com" in result.stderr  # the blocked attempt is reported, on stderr


def test_watch_without_the_model_explains_how_to_get_it(home, tmp_path):
    (tmp_path / "notes").mkdir()
    result = runner.invoke(app, ["watch", str(tmp_path / "notes")])
    assert result.exit_code == 1
    assert "dejanote setup" in result.output


@pytest.mark.model
def test_watch_reports_each_change_and_stops_cleanly_on_ctrl_c(home_with_model, tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    note = notes / "boiler.md"
    note.write_text("# Boiler\n\nTop up the pressure with the filling loop.\n")

    def file_events(root, **options):  # stands in for the operating system's file events
        note.write_text("# Boiler\n\nTop up the pressure to 1.2 bar with the filling loop.\n")
        yield {(Change.modified, str(note))}
        raise KeyboardInterrupt  # the user presses Ctrl+C

    monkeypatch.setattr("dejanote.watcher.watch", file_events)
    with block_network() as outer_guard:
        result = runner.invoke(app, ["watch", str(notes)])
    assert result.exit_code == 0, result.output
    assert "updated boiler.md" in result.output
    assert "Stopped watching" in result.output
    assert "no network" in result.output.lower()
    assert outer_guard.attempts == []  # the command's own guard covered the whole session


@pytest.mark.model
def test_watch_summarises_large_batches_instead_of_listing_every_note(home_with_model, tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    for i in range(5):
        (notes / f"note-{i}.md").write_text(f"# Note {i}\n\nSomething worth remembering, part {i}.\n")

    def no_events_then_ctrl_c(root, **options):
        raise KeyboardInterrupt
        yield  # a generator, like the real file events

    monkeypatch.setattr("dejanote.watcher.watch", no_events_then_ctrl_c)
    result = runner.invoke(app, ["watch", str(notes)])
    assert result.exit_code == 0, result.output
    assert "added 5 notes" in result.output
    assert "note-3.md" not in result.output


BREAK_QUERY = "felt burned out and needed a break"


@pytest.mark.model
def test_search_shows_the_matching_passage_even_deep_inside_a_long_section(home_with_model):
    runner.invoke(app, ["index", str(EXAMPLES)])
    result = runner.invoke(app, ["search", BREAK_QUERY, "--limit", "1"])
    assert result.exit_code == 0, result.output
    # The best passage opens the note's last paragraph; the snippet starts there instead of at the top.
    assert "... Decided: taking Monday off, no laptop." in result.output
    assert "journal/2024-05-12.md:5" in result.output  # the line of the match, not of the section


@pytest.mark.model
def test_search_highlights_the_matching_passage_in_a_terminal(home_with_model, monkeypatch):
    runner.invoke(app, ["index", str(EXAMPLES)])
    terminal = Console(force_terminal=True, color_system="standard", width=100, highlight=False, soft_wrap=True)
    monkeypatch.setattr(cli, "console", terminal)
    result = runner.invoke(app, ["search", BREAK_QUERY, "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "\x1b[1mDecided: taking Monday off, no laptop.\x1b[0m" in result.output  # bold is exactly the sentence
