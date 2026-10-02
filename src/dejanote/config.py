"""Where dejanote keeps its files.

Everything lives in one folder (the model and the index), so it is easy to
inspect, back up or delete. Set DEJANOTE_HOME to move it.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_home() -> Path:
    """$DEJANOTE_HOME if set, otherwise ~/.dejanote."""
    override = os.environ.get("DEJANOTE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".dejanote"


def models_dir() -> Path:
    return data_home() / "models"


def index_path() -> Path:
    return data_home() / "index.db"
