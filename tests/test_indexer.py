"""Indexing pipeline: the right files, chunked, embedded with context, stored per folder."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    assert report.indexed == 3


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


@pytest.mark.model
def test_indexing_the_example_notes_makes_no_network_attempts(tmp_path, embedder):
    expected_files = len(list(EXAMPLES.rglob("*.md"))) + len(list(EXAMPLES.rglob("*.txt")))
    with Store(tmp_path / "index.db", embedder.model_id, embedder.dimension) as store:
        with block_network() as guard:
            report = index_folder(EXAMPLES, store, embedder)
        files, chunks = store.counts()
    assert guard.attempts == []
    assert files == report.indexed == expected_files
    assert chunks >= files
