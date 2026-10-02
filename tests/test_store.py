"""The SQLite index: chunks and vectors in, the same chunks and vectors out."""

from __future__ import annotations

import numpy as np
import pytest

from dejanote.chunking import Chunk
from dejanote.store import IndexedChunk, IndexMismatchError, Store

FEEDING = Chunk("Discard all but 50 g, then feed.", ("Sourdough", "Feeding"), 5, 6)
HOOCH = Chunk("Grey liquid means it is hungry.", ("Sourdough", "When it goes wrong"), 9, 9)
VECTORS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.6, 0.8, 0.0]], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "index.db", model_id="test-model@1", dimension=4) as store:
        yield store


def _all_chunks(store: Store) -> list[IndexedChunk]:
    ids, _ = store.vectors()
    return store.chunks(ids)


def test_chunks_and_vectors_round_trip(store):
    store.replace_file("/notes/sourdough.md", "hash-1", [FEEDING, HOOCH], VECTORS)
    ids, matrix = store.vectors()
    np.testing.assert_array_equal(matrix, VECTORS)
    assert store.chunks(ids) == [
        IndexedChunk("/notes/sourdough.md", FEEDING),
        IndexedChunk("/notes/sourdough.md", HOOCH),
    ]


def test_chunks_come_back_in_the_order_the_ids_were_asked_for(store):
    store.replace_file("/notes/sourdough.md", "hash-1", [FEEDING, HOOCH], VECTORS)
    ids, _ = store.vectors()
    assert [c.chunk for c in store.chunks(ids[::-1])] == [HOOCH, FEEDING]


def test_replacing_a_file_drops_its_old_chunks(store):
    store.replace_file("/notes/sourdough.md", "hash-1", [FEEDING, HOOCH], VECTORS)
    store.replace_file("/notes/sourdough.md", "hash-2", [HOOCH], VECTORS[1:])
    assert _all_chunks(store) == [IndexedChunk("/notes/sourdough.md", HOOCH)]
    assert store.file_hashes() == {"/notes/sourdough.md": "hash-2"}


def test_removing_a_file_removes_its_chunks(store):
    store.replace_file("/notes/sourdough.md", "hash-1", [FEEDING], VECTORS[:1])
    store.replace_file("/notes/dal.md", "hash-2", [HOOCH], VECTORS[1:])
    store.remove_files(["/notes/sourdough.md"])
    assert _all_chunks(store) == [IndexedChunk("/notes/dal.md", HOOCH)]
    assert store.counts() == (1, 1)


def test_the_index_survives_reopening(tmp_path):
    with Store(tmp_path / "index.db", model_id="test-model@1", dimension=4) as store:
        store.replace_file("/notes/sourdough.md", "hash-1", [FEEDING, HOOCH], VECTORS)
    with Store(tmp_path / "index.db", model_id="test-model@1", dimension=4) as store:
        assert store.counts() == (1, 2)
        np.testing.assert_array_equal(store.vectors()[1], VECTORS)


def test_an_index_built_with_another_model_is_refused(tmp_path):
    Store(tmp_path / "index.db", model_id="test-model@1", dimension=4).close()
    with pytest.raises(IndexMismatchError, match="test-model@1"):
        Store(tmp_path / "index.db", model_id="other-model@7", dimension=4)


def test_an_empty_index_has_no_vectors(store):
    ids, matrix = store.vectors()
    assert len(ids) == 0
    assert matrix.shape == (0, 4)
