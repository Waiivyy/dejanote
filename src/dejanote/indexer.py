"""Indexing pipeline: find notes, chunk them, embed the chunks, store the result."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from dejanote.chunking import Chunk, chunk_note
from dejanote.store import Store

NOTE_SUFFIXES = frozenset({".md", ".markdown", ".txt"})

# Chunks per model call. Batching across notes is faster than one call per note.
EMBED_BATCH = 256


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class IndexReport:
    added: int = 0  # new notes, chunked, embedded and stored
    updated: int = 0  # notes whose content changed, embedded again
    unchanged: int = 0  # notes whose content hash matched, left exactly as they were
    removed: int = 0  # notes dropped from the index because they were deleted
    chunks: int = 0  # chunks embedded this run
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    changes: list[tuple[str, str]] = field(default_factory=list)  # ("added" | "updated" | "removed", path)

    @property
    def notes(self) -> int:
        """Notes from the folder that are in the index after this run."""
        return self.added + self.updated + self.unchanged


def find_notes(root: Path) -> list[Path]:
    """Markdown and text files under root, in a stable order, skipping hidden files and folders."""
    notes = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = sorted(name for name in subfolders if not name.startswith("."))
        for name in sorted(files):
            if not name.startswith(".") and Path(name).suffix.lower() in NOTE_SUFFIXES:
                notes.append(Path(folder) / name)
    return notes


def index_folder(
    root: Path,
    store: Store,
    embedder: DocumentEmbedder,
    on_progress: Callable[[int, int, Path], None] | None = None,
) -> IndexReport:
    """Bring the index up to date with the notes under root.

    A note is only chunked and embedded when it is new or its content hash has
    changed, so a repeat run over an unchanged folder embeds nothing. Notes
    under root that were deleted or are no longer readable text are dropped;
    notes indexed from other folders are left alone.
    """
    root = root.resolve()
    report = IndexReport()
    notes = find_notes(root)
    previous = {path: digest for path, digest in store.file_hashes().items() if Path(path).is_relative_to(root)}

    on_disk = {str(path) for path in notes}
    deleted = [path for path in previous if path not in on_disk]
    store.remove_files(deleted)
    report.removed = len(deleted)
    report.changes.extend(("removed", path) for path in deleted)

    batch = _EmbeddingBatch(store, embedder, report)
    for done, path in enumerate(notes, start=1):
        _check_note(path, previous.get(str(path)), store, batch, report)
        if on_progress:
            on_progress(done, len(notes), path)
    batch.flush()
    return report


def _check_note(
    path: Path, previous_hash: str | None, store: Store, batch: _EmbeddingBatch, report: IndexReport
) -> None:
    """Queue a note for embedding if it is new or changed; otherwise just record what happened."""
    try:
        data = path.read_bytes()
        problem = "binary file" if b"\0" in data else None
    except OSError as err:
        data, problem = b"", f"unreadable ({err.strerror})"
    if problem:
        report.skipped.append((str(path), problem))
        store.remove_files([str(path)])  # the index must only hold what is on disk now
        return

    digest = hashlib.sha256(data).hexdigest()
    if digest == previous_hash:
        report.unchanged += 1
        return
    chunks = chunk_note(data.decode("utf-8", errors="replace"), path.name)
    batch.add(_PendingNote(path, digest, chunks, is_new=previous_hash is None))


@dataclass
class _PendingNote:
    path: Path
    digest: str
    chunks: list[Chunk]
    is_new: bool


class _EmbeddingBatch:
    """Collects chunks from several notes so the model embeds them in large batches.

    One model call per note spends much of its time on per-call overhead; shared
    batches index about 1.4 times faster and give the same vectors. Each note is
    still stored in its own transaction, so an interrupted run only loses the
    notes in the current batch, and those are picked up again on the next run.
    """

    def __init__(self, store: Store, embedder: DocumentEmbedder, report: IndexReport, size: int = EMBED_BATCH):
        self.store = store
        self.embedder = embedder
        self.report = report
        self.size = size
        self.notes: list[_PendingNote] = []
        self.chunk_count = 0

    def add(self, note: _PendingNote) -> None:
        self.notes.append(note)
        self.chunk_count += len(note.chunks)
        if self.chunk_count >= self.size:
            self.flush()

    def flush(self) -> None:
        texts = [chunk.embedding_text for note in self.notes for chunk in note.chunks]
        if texts:
            vectors = self.embedder.embed_documents(texts)
        else:
            vectors = np.empty((0, self.store.dimension), dtype=np.float32)
        start = 0
        for note in self.notes:
            end = start + len(note.chunks)
            self.store.replace_file(str(note.path), note.digest, note.chunks, vectors[start:end])
            start = end
            self.report.chunks += len(note.chunks)
            if note.is_new:
                self.report.added += 1
            else:
                self.report.updated += 1
            self.report.changes.append(("added" if note.is_new else "updated", str(note.path)))
        self.notes, self.chunk_count = [], 0
