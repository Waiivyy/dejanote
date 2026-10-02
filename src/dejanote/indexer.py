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
    indexed: int = 0  # files chunked, embedded and stored this run
    chunks: int = 0  # chunks embedded this run
    removed: int = 0  # files dropped from the index because they no longer exist
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


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
    """Index every note under root and drop notes under root that no longer exist.

    Notes indexed from other folders are left alone.
    """
    root = root.resolve()
    report = IndexReport()
    notes = find_notes(root)

    on_disk = {str(path) for path in notes}
    gone = [path for path in store.file_hashes() if Path(path).is_relative_to(root) and path not in on_disk]
    store.remove_files(gone)
    report.removed = len(gone)

    for done, path in enumerate(notes, start=1):
        try:
            data = path.read_bytes()
        except OSError as err:
            report.skipped.append((str(path), f"unreadable ({err.strerror})"))
            continue
        if b"\0" in data:
            report.skipped.append((str(path), "binary file"))
            continue
        chunks = chunk_note(data.decode("utf-8", errors="replace"), path.name)
        if chunks:
            vectors = embedder.embed_documents([chunk.embedding_text for chunk in chunks])
        else:
            vectors = np.empty((0, store.dimension), dtype=np.float32)
        store.replace_file(str(path), hashlib.sha256(data).hexdigest(), chunks, vectors)
        report.indexed += 1
        report.chunks += len(chunks)
        if on_progress:
            on_progress(done, len(notes), path)
    return report
