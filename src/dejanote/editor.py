"""Open a note in the user's editor at the matching line."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

# Editors that take the line as `-g path:line`, or as `path:line`. Everything
# else is assumed to follow the vi convention of `+line path`, which vim, neovim,
# nano, emacs, micro and kakoune all understand.
_GOTO_FLAG = frozenset({"code", "code-insiders", "codium", "cursor", "windsurf"})
_PATH_COLON_LINE = frozenset({"zed", "subl", "hx", "helix"})


def configured_editor() -> str | None:
    """$VISUAL, else $EDITOR, else None."""
    for variable in ("VISUAL", "EDITOR"):
        editor = os.environ.get(variable, "").strip()
        if editor:
            return editor
    return None


def editor_command(editor: str, path: str, line: int) -> list[str]:
    """The command that opens path at line in editor, given as $VISUAL or $EDITOR would be."""
    words = shlex.split(editor)
    name = Path(words[0]).name
    if name in _GOTO_FLAG:
        return [*words, "-g", f"{path}:{line}"]
    if name in _PATH_COLON_LINE:
        return [*words, f"{path}:{line}"]
    return [*words, f"+{line}", path]
