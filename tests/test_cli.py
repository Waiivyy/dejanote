"""CLI behaviour, run in-process under the network guard."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dejanote.cli import app
from dejanote.privacy import block_network

runner = CliRunner()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DEJANOTE_HOME", str(tmp_path))
    return tmp_path


def test_setup_declined_touches_neither_network_nor_disk(home):
    with block_network() as guard:
        result = runner.invoke(app, ["setup"], input="n\n")
    assert "huggingface.co" in result.output  # told where it would download from, before consenting
    assert result.exit_code == 1  # aborted by the user, not a usage error
    assert guard.attempts == []
    assert not (home / "models").exists() or not any((home / "models").iterdir())


@pytest.mark.model
def test_setup_with_verified_model_present_is_an_offline_no_op(home, model_dir):
    (home / "models").mkdir()
    (home / "models" / model_dir.name).symlink_to(model_dir)
    with block_network() as guard:
        result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0, result.output
    assert "already" in result.output
    assert guard.attempts == []
