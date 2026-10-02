"""CLI behaviour, run in-process under the network guard."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

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
