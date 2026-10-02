from __future__ import annotations

from dejanote import config


def test_data_home_honours_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DEJANOTE_HOME", str(tmp_path / "custom"))
    assert config.data_home() == tmp_path / "custom"


def test_data_home_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DEJANOTE_HOME", "~/dn")
    assert config.data_home() == tmp_path / "dn"


def test_models_live_inside_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DEJANOTE_HOME", str(tmp_path))
    assert config.models_dir() == tmp_path / "models"
