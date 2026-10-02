"""Search: rank stored chunks by cosine similarity to the query."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dejanote.chunking import Chunk
from dejanote.store import Store

_FENCE_LINE = re.compile(r"^\s*(```|~~~).*$", re.MULTILINE)


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
    """A chunk's text on one line, without code fence markers, cut at a word boundary if too long."""
    flat = " ".join(_FENCE_LINE.sub("", text).split())
    if len(flat) <= max_chars:
        return flat
    cut = flat[: max_chars - 3]
    if " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut + "..."
