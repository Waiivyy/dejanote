"""Record the README demo: real dejanote commands on the example notes, saved as an SVG terminal.

Runs in a throwaway home folder with the model linked in, so it starts from an
empty index and never touches your own. Needs `dejanote setup` to have run.

    python scripts/record_demo.py      # writes docs/demo.svg and prints the same output as text
"""

from __future__ import annotations

import io
import os
import re
import shlex
import tempfile
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = [
    "dejanote index examples/notes",
    'dejanote search "felt burned out and needed a break" --limit 3',
]


def main() -> None:
    from dejanote import cli
    from dejanote.embedding import DEFAULT_MODEL, default_model_dir

    model = default_model_dir(DEFAULT_MODEL)  # resolve before HOME changes
    with tempfile.TemporaryDirectory() as home:
        (Path(home) / ".dejanote" / "models").mkdir(parents=True)
        (Path(home) / ".dejanote" / "models" / model.name).symlink_to(model)
        os.environ["HOME"] = home
        os.environ.pop("DEJANOTE_HOME", None)
        os.chdir(ROOT)

        recording = Console(record=True, width=88, file=io.StringIO(), highlight=False, soft_wrap=True)
        cli.console = recording
        for command in COMMANDS:
            recording.print(f"[bold green]$[/] [bold]{escape(command)}[/]")
            result = CliRunner().invoke(cli.app, shlex.split(command)[1:])
            if result.exit_code != 0:
                raise SystemExit(f"{command!r} failed:\n{result.output}")
            recording.print()

    text = recording.export_text(clear=False)
    svg = _without_remote_fonts(recording.export_svg(title="dejanote"))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "demo.svg").write_text(svg)
    print(text)


def _without_remote_fonts(svg: str) -> str:
    """Drop Rich's @font-face rules, which would fetch a font from a CDN; viewers use a local monospace."""
    return re.sub(r"@font-face\s*\{[^}]*\}\s*", "", svg)


if __name__ == "__main__":
    main()
