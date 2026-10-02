"""The index: one SQLite file holding every chunk and its embedding.

Embeddings are stored as little-endian float32 blobs next to the chunk text and
its location, so the whole index is a single portable file that the Python
standard library can read.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dejanote.chunking import CHUNKER_VERSION, Chunk

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    headings   TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_file ON chunks(file_id);
"""

_FLOAT32 = np.dtype("<f4")


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk together with the file it came from."""

    path: str
    chunk: Chunk


class IndexMismatchError(RuntimeError):
    """The index on disk was built with a different model, chunker or format."""


class Store:
    """Open (or create) the index for one model and chunker version.

    An index built for anything else is refused, or wiped and started afresh
    when rebuild_if_incompatible is set, which is what indexing wants: its
    vectors or chunks would be useless with the current model and rules.
    """

    def __init__(
        self,
        path: Path,
        model_id: str,
        dimension: int,
        *,
        chunker_version: int = CHUNKER_VERSION,
        rebuild_if_incompatible: bool = False,
    ):
        self.path = path
        self.dimension = dimension
        self.rebuilt = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(_SCHEMA)
            self._check_meta(model_id, dimension, chunker_version, rebuild_if_incompatible)
        except BaseException:
            self._db.close()
            raise

    def _check_meta(self, model_id: str, dimension: int, chunker_version: int, rebuild: bool) -> None:
        expected = {
            "schema_version": str(SCHEMA_VERSION),
            "model": model_id,
            "dimension": str(dimension),
            "chunker": str(chunker_version),
        }
        stored = dict(self._db.execute("SELECT key, value FROM meta"))
        if stored == expected:
            return
        if stored and not rebuild:
            raise IndexMismatchError(
                f"The index at {self.path} was built with {_describe(stored)}, "
                f"but dejanote now uses {_describe(expected)}. "
                "Run `dejanote index` on your notes to rebuild it."
            )
        with self._db:
            self._db.execute("DELETE FROM files")  # their chunks go with them
            self._db.execute("DELETE FROM meta")
            self._db.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", expected.items())
        self.rebuilt = bool(stored)

    def replace_file(self, path: str, content_hash: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Store a file's chunks, replacing whatever was stored for it before, in one transaction."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with self._db:
            self._db.execute("DELETE FROM files WHERE path = ?", (path,))
            file_id = self._db.execute(
                "INSERT INTO files (path, content_hash) VALUES (?, ?)", (path, content_hash)
            ).lastrowid
            self._db.executemany(
                "INSERT INTO chunks (file_id, headings, start_line, end_line, text, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (file_id, json.dumps(c.headings), c.start_line, c.end_line, c.text,
                     np.asarray(v, dtype=_FLOAT32).tobytes())
                    for c, v in zip(chunks, vectors)
                ],
            )

    def remove_files(self, paths: Iterable[str]) -> None:
        with self._db:
            self._db.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])

    def file_hashes(self) -> dict[str, str]:
        """Content hash of every indexed file, by path."""
        return dict(self._db.execute("SELECT path, content_hash FROM files"))

    def counts(self) -> tuple[int, int]:
        """(files, chunks) in the index."""
        files = self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return files, chunks

    def vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Chunk ids and the matching rows of the embedding matrix."""
        rows = self._db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        ids = np.array([row[0] for row in rows], dtype=np.int64)
        matrix = np.frombuffer(b"".join(row[1] for row in rows), dtype=_FLOAT32)
        return ids, matrix.reshape(len(rows), self.dimension)

    def chunks(self, ids: Sequence[int]) -> list[IndexedChunk]:
        """The chunks with these ids, in the same order."""
        ids = [int(i) for i in ids]
        placeholders = ",".join("?" * len(ids))
        rows = self._db.execute(
            "SELECT c.id, f.path, c.headings, c.start_line, c.end_line, c.text"
            " FROM chunks c JOIN files f ON f.id = c.file_id"
            f" WHERE c.id IN ({placeholders})",
            ids,
        )
        found = {
            row[0]: IndexedChunk(row[1], Chunk(row[5], tuple(json.loads(row[2])), row[3], row[4]))
            for row in rows
        }
        return [found[i] for i in ids]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _describe(meta: dict[str, str]) -> str:
    return (
        f"{meta.get('model', 'an unknown model')} "
        f"(chunker {meta.get('chunker', 'unknown')}, format {meta.get('schema_version', 'unknown')})"
    )
