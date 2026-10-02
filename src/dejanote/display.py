"""Small text helpers shared by the command line and the interactive browser."""

from __future__ import annotations

from pathlib import Path


def short_path(path: Path) -> str:
    """A path as short as possible: relative to the current folder, else to the home folder."""
    if path.is_relative_to(Path.cwd()):
        return str(path.relative_to(Path.cwd()))
    if path.is_relative_to(Path.home()):
        return str(Path("~") / path.relative_to(Path.home()))
    return str(path)


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
