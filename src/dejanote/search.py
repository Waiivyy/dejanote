"""Search: rank stored chunks by cosine similarity to the query."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dejanote.chunking import LINE_ITEM, Chunk
from dejanote.store import Store

_FENCE = re.compile(r"^(```|~~~)")


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class SearchResult:
    score: float  # cosine similarity between the query and the chunk, at most 1
    path: str
    chunk: Chunk


def search(store: Store, embedder: QueryEmbedder, query: str, limit: int = 5) -> list[SearchResult]:
    """The chunks most similar in meaning to the query, best first.

    Every stored vector has unit length, so one matrix-vector product gives the
    cosine similarity between the query and every chunk at once.
    """
    ids, matrix = store.vectors()
    if len(ids) == 0 or limit <= 0:
        return []
    scores = matrix @ embedder.embed_query(query)
    best = _top(scores, limit)
    found = store.chunks(ids[best])
    return [SearchResult(float(scores[i]), hit.path, hit.chunk) for i, hit in zip(best, found)]


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, highest first."""
    if k < len(scores):
        candidates = np.argpartition(-scores, k - 1)[:k]
    else:
        candidates = np.arange(len(scores))
    return candidates[np.argsort(-scores[candidates], kind="stable")]


def snippet(text: str, max_chars: int = 240) -> str:
    """A chunk's text condensed for display, one line per paragraph, list item or line of code.

    Hard-wrapped prose is joined up, blank lines and code fence markers are
    dropped, and anything past max_chars is cut at a word boundary with "...".
    """
    lines: list[str] = []
    in_code = False
    starts_new_line = True
    for raw in text.split("\n"):
        line = " ".join(raw.split())
        if _FENCE.match(line):
            in_code = not in_code
            starts_new_line = True
        elif not line:
            starts_new_line = True
        elif starts_new_line or in_code or LINE_ITEM.match(raw):
            lines.append(line)
            starts_new_line = False
        else:
            lines[-1] += " " + line
    return _cut("\n".join(lines), max_chars)


def _cut(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 3]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    return (cut[:boundary] if boundary > 0 else cut) + "..."
