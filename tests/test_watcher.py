"""Watch mode: the index follows the folder as notes change."""

from __future__ import annotations

import threading
import time

import pytest
from watchfiles import Change

from dejanote.store import Store
from dejanote.watcher import NoteChangeFilter, watch_folder


@pytest.fixture
def store(tmp_path, fake_embedder):
    with Store(tmp_path / "index.db", fake_embedder.model_id, fake_embedder.dimension) as store:
        yield store


def test_the_index_is_brought_up_to_date_before_watching_starts(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("alpha")
    updates = []
    watch_folder(notes, store, fake_embedder, lambda report, seconds: updates.append(report), changes=[])
    assert [update.added for update in updates] == [1]
    assert set(store.file_hashes()) == {str(notes / "a.md")}


def test_each_batch_of_changes_reindexes_just_what_changed(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    notes.mkdir()
    edited, deleted = notes / "edited.md", notes / "deleted.md"
    edited.write_text("first draft")
    deleted.write_text("short lived")

    def file_events():  # stands in for the operating system: change the folder, then report it
        edited.write_text("second draft")
        yield {(Change.modified, str(edited))}
        deleted.unlink()
        yield {(Change.deleted, str(deleted))}

    updates = []
    watch_folder(notes, store, fake_embedder, lambda report, seconds: updates.append(report), changes=file_events())
    assert [update.changes for update in updates[1:]] == [[("updated", str(edited))], [("removed", str(deleted))]]
    assert fake_embedder.embedded_texts[-1].endswith("second draft")


@pytest.mark.parametrize(
    "relative, counts",
    [
        ("note.md", True),
        ("sub/folder/note.txt", True),
        ("sub", True),  # a folder: moving or deleting one moves or deletes the notes inside it
        (".obsidian/workspace.json", False),
        ("sub/.trash/old.md", False),
        (".draft.md", False),
        ("note.md.swp", False),  # an editor's swap file
        ("note.md~", False),  # an editor's backup copy
    ],
)
def test_which_file_events_count(tmp_path, relative, counts):
    root = tmp_path / "notes"
    assert NoteChangeFilter(root)(Change.modified, str(root / relative)) is counts


def _wait_until(condition, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_real_file_events_keep_the_index_current(tmp_path, fake_embedder):
    notes = tmp_path / "notes"
    notes.mkdir()
    note = notes / "a.md"
    note.write_text("first draft")
    updates, stop = [], threading.Event()

    def watch_in_background():  # SQLite connections belong to their thread, so it opens its own
        with Store(tmp_path / "index.db", fake_embedder.model_id, fake_embedder.dimension) as store:
            watch_folder(notes, store, fake_embedder, lambda report, seconds: updates.append(report), stop_event=stop)

    watcher = threading.Thread(target=watch_in_background, daemon=True)
    watcher.start()
    try:
        assert _wait_until(lambda: len(updates) == 1, 10), "the first indexing never finished"
        time.sleep(0.5)  # let the operating system's watcher settle in
        note.write_text("second draft")
        assert _wait_until(lambda: any(("updated", str(note.resolve())) in u.changes for u in updates), 15)
    finally:
        stop.set()
        watcher.join(10)
    assert not watcher.is_alive(), "the watcher did not stop when asked"
