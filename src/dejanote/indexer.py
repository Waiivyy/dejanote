"""Indexing pipeline: find notes, chunk them, embed the chunks, store the result."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from dejanote.chunking import chunk_note
from dejanote.store import Store

NOTE_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


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

    for done, path in enumerate(notes, start=1):
        _update_note(path, previous.get(str(path)), store, embedder, report)
        if on_progress:
            on_progress(done, len(notes), path)
    return report


def _update_note(
    path: Path, previous_hash: str | None, store: Store, embedder: DocumentEmbedder, report: IndexReport
) -> None:
    """Bring one note's entry up to date, embedding it only if its content changed."""
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
    if chunks:
        vectors = embedder.embed_documents([chunk.embedding_text for chunk in chunks])
    else:
        vectors = np.empty((0, store.dimension), dtype=np.float32)
    store.replace_file(str(path), digest, chunks, vectors)
    report.chunks += len(chunks)
    if previous_hash is None:
        report.added += 1
    else:
        report.updated += 1
