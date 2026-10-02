"""Indexing pipeline: the right files, chunked, embedded with context, stored per folder."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeEmbedder
from dejanote.chunking import Chunk
from dejanote.indexer import index_folder
from dejanote.privacy import block_network
from dejanote.store import IndexedChunk, Store

EXAMPLES = Path(__file__).parent.parent / "examples" / "notes"


@pytest.fixture
def store(tmp_path, fake_embedder):
    with Store(tmp_path / "index.db", fake_embedder.model_id, fake_embedder.dimension) as store:
        yield store


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def _stored_chunks(store: Store) -> list[IndexedChunk]:
    ids, _ = store.vectors()
    return store.chunks(ids)


def test_every_note_under_the_folder_is_indexed_recursively(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    expected = {
        _write(notes, "a.md", "alpha"),
        _write(notes, "sub/b.markdown", "beta"),
        _write(notes, "sub/deeper/c.txt", "gamma"),
    }
    _write(notes, "sub/photo.png", b"\x89PNG")
    _write(notes, "sub/data.json", "{}")
    report = index_folder(notes, store, fake_embedder)
    assert set(store.file_hashes()) == {str(p) for p in expected}
    assert report.added == 3


def test_hidden_folders_and_files_are_skipped(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    keep = _write(notes, "keep.md", "kept")
    _write(notes, ".obsidian/workspace.md", "editor state")
    _write(notes, ".trash/deleted.md", "old")
    _write(notes, ".draft.md", "hidden")
    index_folder(notes, store, fake_embedder)
    assert set(store.file_hashes()) == {str(keep)}


def test_each_chunk_is_stored_with_its_file_heading_path_and_lines(tmp_path, store, fake_embedder):
    note = _write(tmp_path / "notes", "sourdough.md", "# Sourdough\n\n## Feeding\n\nFlour and water.\n\n## Hooch\n\nPour it off.\n")
    index_folder(tmp_path / "notes", store, fake_embedder)
    assert _stored_chunks(store) == [
        IndexedChunk(str(note), Chunk("Flour and water.", ("Sourdough", "Feeding"), 5, 5)),
        IndexedChunk(str(note), Chunk("Pour it off.", ("Sourdough", "Hooch"), 9, 9)),
    ]


def test_chunks_are_embedded_together_with_their_heading_path(tmp_path, store, fake_embedder):
    _write(tmp_path / "notes", "sourdough.md", "# Sourdough\n\n## Feeding\n\nFlour and water.\n")
    index_folder(tmp_path / "notes", store, fake_embedder)
    [embedded] = fake_embedder.embedded_texts
    assert "Feeding" in embedded and "Flour and water." in embedded


def test_notes_deleted_from_disk_are_removed_from_the_index(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    keep = _write(notes, "keep.md", "kept")
    gone = _write(notes, "gone.md", "soon deleted")
    index_folder(notes, store, fake_embedder)
    gone.unlink()
    report = index_folder(notes, store, fake_embedder)
    assert set(store.file_hashes()) == {str(keep)}
    assert report.removed == 1


def test_indexing_one_folder_leaves_other_folders_alone(tmp_path, store, fake_embedder):
    work = _write(tmp_path / "work", "standup.md", "decisions")
    home = _write(tmp_path / "home", "boiler.md", "pressure")
    index_folder(tmp_path / "work", store, fake_embedder)
    index_folder(tmp_path / "home", store, fake_embedder)
    index_folder(tmp_path / "work", store, fake_embedder)
    assert set(store.file_hashes()) == {str(work), str(home)}


def test_binary_files_are_skipped_with_a_reason(tmp_path, store, fake_embedder):
    _write(tmp_path / "notes", "corrupt.md", b"\x00\x01\x02 not text")
    report = index_folder(tmp_path / "notes", store, fake_embedder)
    assert store.file_hashes() == {}
    assert [(Path(path).name, reason) for path, reason in report.skipped] == [("corrupt.md", "binary file")]


def test_a_note_with_invalid_utf8_is_still_indexed(tmp_path, store, fake_embedder):
    note = _write(tmp_path / "notes", "latin1.md", "caf\xe9 au lait".encode("latin-1"))
    index_folder(tmp_path / "notes", store, fake_embedder)
    [stored] = _stored_chunks(store)
    assert stored.path == str(note)
    assert stored.chunk.text.startswith("caf") and stored.chunk.text.endswith("au lait")


def test_an_empty_note_is_recorded_without_chunks(tmp_path, store, fake_embedder):
    note = _write(tmp_path / "notes", "empty.md", "")
    index_folder(tmp_path / "notes", store, fake_embedder)
    assert set(store.file_hashes()) == {str(note)}
    assert store.counts() == (1, 0)


def _snapshot(store: Store):
    """Everything search could ever see: files and hashes, chunk ids and contents, vectors."""
    ids, matrix = store.vectors()
    return store.file_hashes(), list(zip(ids.tolist(), store.chunks(ids))), matrix.copy()


def test_unchanged_notes_are_not_embedded_again(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    for name in ("a.md", "b.md", "sub/c.txt"):
        _write(notes, name, f"contents of {name}")
    index_folder(notes, store, fake_embedder)
    fake_embedder.batches.clear()
    report = index_folder(notes, store, fake_embedder)
    assert fake_embedder.batches == []
    assert (report.added, report.updated, report.unchanged) == (0, 0, 3)


def test_only_new_and_changed_notes_are_embedded(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    _write(notes, "same.md", "this note never changes")
    edited = _write(notes, "edited.md", "first draft")
    index_folder(notes, store, fake_embedder)
    fake_embedder.batches.clear()
    edited.write_text("second draft")
    _write(notes, "new.md", "a brand new note")
    report = index_folder(notes, store, fake_embedder)
    embedded = " ".join(fake_embedder.embedded_texts)
    assert "second draft" in embedded and "a brand new note" in embedded
    assert "never changes" not in embedded
    assert (report.added, report.updated, report.unchanged) == (1, 1, 1)


def test_a_change_is_detected_even_when_the_modification_time_is_unchanged(tmp_path, store, fake_embedder):
    # Restoring from a backup or some sync tools keep the old mtime; only content counts.
    note = _write(tmp_path / "notes", "n.md", "original text")
    original = note.stat()
    index_folder(tmp_path / "notes", store, fake_embedder)
    note.write_text("edited text")
    os.utime(note, ns=(original.st_atime_ns, original.st_mtime_ns))
    report = index_folder(tmp_path / "notes", store, fake_embedder)
    assert report.updated == 1
    assert [c.chunk.text for c in _stored_chunks(store)] == ["edited text"]


def test_an_edited_note_leaves_no_trace_of_its_old_text(tmp_path, store, fake_embedder):
    note = _write(tmp_path / "notes", "n.md", "# N\n\n## One\n\nold one\n\n## Two\n\nold two\n")
    index_folder(tmp_path / "notes", store, fake_embedder)
    note.write_text("# N\n\n## One\n\nnew one\n")
    index_folder(tmp_path / "notes", store, fake_embedder)
    assert [c.chunk.text for c in _stored_chunks(store)] == ["new one"]


def test_reindexing_is_idempotent(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    _write(notes, "a.md", "# A\n\n## One\n\nfirst\n\n## Two\n\nsecond\n")
    _write(notes, "sub/b.txt", "a plain text note")
    _write(notes, "empty.md", "")
    index_folder(notes, store, fake_embedder)
    hashes, chunks, vectors = _snapshot(store)
    for _ in range(2):
        report = index_folder(notes, store, fake_embedder)
        again_hashes, again_chunks, again_vectors = _snapshot(store)
        assert again_hashes == hashes
        assert again_chunks == chunks  # same ids too: nothing was rewritten
        np.testing.assert_array_equal(again_vectors, vectors)
        assert (report.added, report.updated, report.removed) == (0, 0, 0)


def test_notes_share_embedding_calls_without_mixing_up_their_vectors(tmp_path, store, fake_embedder):
    notes = tmp_path / "notes"
    for i in range(10):
        _write(notes, f"n{i}.md", f"# Note {i}\n\n## A\n\nfirst part of note {i}\n\n## B\n\nsecond part of note {i}\n")
    index_folder(notes, store, fake_embedder)
    assert len(fake_embedder.batches) == 1  # all 20 chunks in one model call, not one call per note
    ids, matrix = store.vectors()
    for vector, stored in zip(matrix, store.chunks(ids)):
        own_vector = FakeEmbedder().embed_documents([stored.chunk.embedding_text])[0]
        np.testing.assert_array_equal(vector, own_vector)


class _CrashingEmbedder(FakeEmbedder):
    """Fails on its second call, like a run interrupted halfway through."""

    def embed_documents(self, texts):
        if self.batches:
            raise KeyboardInterrupt
        return super().embed_documents(texts)


def test_an_interrupted_run_keeps_finished_notes_and_the_next_run_completes_the_rest(tmp_path, store):
    notes = tmp_path / "notes"
    for i in range(300):  # enough chunks for more than one embedding batch
        _write(notes, f"n{i:03}.md", f"note number {i}")
    with pytest.raises(KeyboardInterrupt):
        index_folder(notes, store, _CrashingEmbedder())
    kept = len(store.file_hashes())
    assert 0 < kept < 300
    assert store.counts() == (kept, kept)  # every stored note is complete

    resumed = FakeEmbedder()
    report = index_folder(notes, store, resumed)
    assert store.counts() == (300, 300)
    assert (report.unchanged, report.added) == (kept, 300 - kept)
    assert len(resumed.embedded_texts) == 300 - kept


def test_a_note_that_stops_being_text_drops_out_of_the_index(tmp_path, store, fake_embedder):
    note = _write(tmp_path / "notes", "n.md", "readable for now")
    index_folder(tmp_path / "notes", store, fake_embedder)
    note.write_bytes(b"\x00\x01 binary now")
    index_folder(tmp_path / "notes", store, fake_embedder)
    assert store.file_hashes() == {}


@pytest.mark.model
def test_indexing_the_example_notes_makes_no_network_attempts(tmp_path, embedder):
    expected_files = len(list(EXAMPLES.rglob("*.md"))) + len(list(EXAMPLES.rglob("*.txt")))
    with Store(tmp_path / "index.db", embedder.model_id, embedder.dimension) as store:
        with block_network() as guard:
            report = index_folder(EXAMPLES, store, embedder)
        files, chunks = store.counts()
    assert guard.attempts == []
    assert files == report.added == expected_files
    assert chunks >= files
