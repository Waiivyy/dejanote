"""Watch mode: keep the index current while notes change.

Each batch of file events runs the same incremental indexing as `dejanote
index`, so only notes whose content changed are embedded again, and the
model stays loaded between batches.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from watchfiles import Change, DefaultFilter, watch

from dejanote.indexer import DocumentEmbedder, IndexReport, index_folder
from dejanote.store import Store

# Editors often save in several steps (write a temporary file, rename it, update
# metadata); waiting this long turns one save into one reindex.
DEBOUNCE_MS = 300


class NoteChangeFilter(DefaultFilter):
    """Accept the file events that can change what `dejanote index` sees under root.

    Hidden files and folders are ignored, as indexing ignores them, and so are
    editor swap and backup files. Folder events count: moving or deleting a
    folder moves or deletes the notes inside it.
    """

    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def __call__(self, change: Change, path: str) -> bool:
        try:
            parts = Path(path).relative_to(self.root).parts
        except ValueError:
            return False
        return super().__call__(change, path) and not any(part.startswith(".") for part in parts)


def watch_folder(
    root: Path,
    store: Store,
    embedder: DocumentEmbedder,
    on_update: Callable[[IndexReport, float], None],
    *,
    changes: Iterable[object] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Bring the index up to date with root, then again after every batch of changes.

    on_update receives each run's report and how long it took. With real file
    events this runs until Ctrl+C or until stop_event is set; `changes` can stand
    in for the file events with any iterable that yields once per batch.
    """
    root = root.resolve()
    _update(root, store, embedder, on_update)
    if changes is None:
        changes = watch(root, watch_filter=NoteChangeFilter(root), debounce=DEBOUNCE_MS, stop_event=stop_event)
    for _batch in changes:
        _update(root, store, embedder, on_update)


def _update(
    root: Path, store: Store, embedder: DocumentEmbedder, on_update: Callable[[IndexReport, float], None]
) -> None:
    started = time.perf_counter()
    report = index_folder(root, store, embedder)
    on_update(report, time.perf_counter() - started)
