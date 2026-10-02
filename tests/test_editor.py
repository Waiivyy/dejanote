"""Opening a note in the user's editor at the matching line."""

from __future__ import annotations

import pytest

from dejanote.editor import configured_editor, editor_command


@pytest.mark.parametrize(
    "editor, expected",
    [
        ("vim", ["vim", "+12", "notes/a b.md"]),
        ("nvim -p", ["nvim", "-p", "+12", "notes/a b.md"]),
        ("nano", ["nano", "+12", "notes/a b.md"]),
        ("emacsclient -t", ["emacsclient", "-t", "+12", "notes/a b.md"]),
        ("code --wait", ["code", "--wait", "-g", "notes/a b.md:12"]),
        ("/Applications/Cursor.app/Contents/Resources/app/bin/cursor", [
            "/Applications/Cursor.app/Contents/Resources/app/bin/cursor", "-g", "notes/a b.md:12",
        ]),
        ("zed --wait", ["zed", "--wait", "notes/a b.md:12"]),
        ("subl -w", ["subl", "-w", "notes/a b.md:12"]),
        ("hx", ["hx", "notes/a b.md:12"]),
    ],
)
def test_each_editor_is_told_the_line_in_the_way_it_understands(editor, expected):
    assert editor_command(editor, "notes/a b.md", 12) == expected


def test_visual_is_preferred_over_editor(monkeypatch):
    monkeypatch.setenv("VISUAL", "code --wait")
    monkeypatch.setenv("EDITOR", "vim")
    assert configured_editor() == "code --wait"


def test_editor_is_used_when_visual_is_not_set(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "vim")
    assert configured_editor() == "vim"


def test_no_editor_configured(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "  ")
    assert configured_editor() is None
