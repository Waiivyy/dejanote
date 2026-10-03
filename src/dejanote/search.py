"""Search: rank stored chunks by cosine similarity to the query."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from dejanote.chunking import LINE_ITEM, Chunk
from dejanote.store import IndexedChunk

_FENCE = re.compile(r"^(```|~~~)")
_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+")


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


class PassageEmbedder(QueryEmbedder, Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


class ChunkSource(Protocol):
    """Where search reads from: a Store, or an IndexSnapshot held in memory."""

    def vectors(self) -> tuple[np.ndarray, np.ndarray]: ...

    def chunks(self, ids: Sequence[int]) -> list[IndexedChunk]: ...


@dataclass(frozen=True)
class SearchResult:
    score: float  # cosine similarity between the query and the chunk, at most 1
    path: str
    chunk: Chunk
    also: tuple[SearchResult, ...] = ()  # other strong sections of the same note, best first


def search(
    store: ChunkSource, embedder: QueryEmbedder, query: str, limit: int = 5, related: int = 2
) -> list[SearchResult]:
    """The notes that best match the query, one result per note, best first.

    Each result is the note's best-matching section. Up to `related` other
    sections of the same note that rank among the top chunks are listed under
    it instead of crowding other notes out of the results.
    """
    ranked = rank_chunks(store, embedder, query, limit=limit * 8)  # wide enough to find `limit` notes
    strong = limit * 2  # chunks ranked this high are worth a mention under their note
    by_note: dict[str, SearchResult] = {}
    for position, hit in enumerate(ranked):
        best = by_note.get(hit.path)
        if best is None:
            if len(by_note) < limit:
                by_note[hit.path] = hit
        elif position < strong and len(best.also) < related:
            by_note[hit.path] = replace(best, also=(*best.also, hit))
    return list(by_note.values())


def rank_chunks(store: ChunkSource, embedder: QueryEmbedder, query: str, limit: int) -> list[SearchResult]:
    """The chunks most similar in meaning to the query, best first, however many share a note.

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


@dataclass(frozen=True)
class Passage:
    """A sentence of prose, a list item or a line of code: a span of a chunk's text."""

    start: int
    end: int
    new_line: bool  # starts a new line when the text is condensed for display


def passages(text: str) -> list[Passage]:
    """Split a chunk into passages: sentences of prose, list items, and lines of code.

    Hard-wrapped lines of one sentence stay one passage, a list item keeps its
    indented continuation lines, and code fence markers are not passages.
    """
    found: list[Passage] = []
    prose: list[int] | None = None  # [start, end] of the prose being collected
    item: list[int] | None = None  # [start, end] of the list item being collected
    in_code = False

    def flush() -> None:
        nonlocal prose, item
        if item:
            found.append(Passage(item[0], item[1], new_line=True))
        if prose:
            for i, (start, end) in enumerate(_sentences(text, *prose)):
                found.append(Passage(start, end, new_line=i == 0))
        prose = item = None

    offset = 0
    for raw in text.split("\n"):
        start, end = offset + len(raw) - len(raw.lstrip()), offset + len(raw.rstrip())
        offset += len(raw) + 1
        if _FENCE.match(raw.strip()):
            flush()
            in_code = not in_code
        elif start == end:
            flush()
        elif in_code:
            flush()
            found.append(Passage(start, end, new_line=True))
        elif LINE_ITEM.match(raw):
            flush()
            item = [start, end]
        elif item and raw[0].isspace():
            item[1] = end  # an indented continuation of the list item
        elif prose:
            prose[1] = end
        else:
            flush()
            prose = [start, end]
    flush()
    return found


def passage_line(chunk: Chunk, passage: Passage | None) -> int:
    """The line of the note a passage starts on; without a passage, the chunk's first line."""
    if passage is None:
        return chunk.start_line
    return chunk.start_line + chunk.text.count("\n", 0, passage.start)


def best_passages(embedder: PassageEmbedder, query: str, texts: list[str]) -> list[Passage | None]:
    """For each text, the passage closest in meaning to the query: the one worth highlighting.

    Each passage is scored on its own and together with its neighbours, then the
    two scores are added. On its own, a short sentence can win on one shared
    word, and "Do it every few uses." means little without the sentence before it.
    A text that is a single passage gets None, since highlighting all of it says
    nothing. Everything is embedded together in one batch.
    """
    found = [passages(text) for text in texts]
    alone: list[str] = []
    together: list[str] = []
    for text, ps in zip(texts, found):
        if len(ps) > 1:
            words = [" ".join(text[p.start : p.end].split()) for p in ps]
            alone += words
            together += [" ".join(words[max(0, i - 1) : i + 2]) for i in range(len(words))]
    if not alone:
        return [None] * len(texts)
    similarity = embedder.embed_documents(alone + together) @ embedder.embed_query(query)
    scores = similarity[: len(alone)] + similarity[len(alone) :]
    best: list[Passage | None] = []
    position = 0
    for ps in found:
        if len(ps) > 1:
            best.append(ps[int(np.argmax(scores[position : position + len(ps)]))])
            position += len(ps)
        else:
            best.append(None)
    return best


def _sentences(text: str, start: int, end: int):
    position = start
    for gap in _SENTENCE_GAP.finditer(text, start, end):
        if gap.start() > position:
            yield position, gap.start()
        position = gap.end()
    if position < end:
        yield position, end


def snippet_lines(text: str, max_chars: int = 240, focus: Passage | None = None) -> list[list[tuple[str, bool]]]:
    """A chunk condensed for display: one line per paragraph, list item or line of code.

    Each line is a list of (text, is_focus) parts, so the focus passage can be
    highlighted. Text past max_chars is cut at a word boundary with "...", and
    when the focus would be cut off, the snippet starts at it after "... ".
    """
    parts = [(" ".join(text[p.start : p.end].split()), p == focus, p.new_line) for p in passages(text)]
    first = _first_part(parts, max_chars, focus)
    budget = max_chars - (len(_SKIPPED) if first else 0)
    kept: list[tuple[str, str, bool]] = []  # (separator, words, is_focus)
    used = 0
    for words, is_focus, new_line in parts[first:]:
        separator = ("\n" if new_line else " ") if kept else ""
        if used + len(separator) + len(words) > budget:
            words = _whole_words(words, budget - used - len(separator) - 3)
            if words:
                kept.append((separator, words + "...", is_focus))
            elif kept:
                kept[-1] = (kept[-1][0], kept[-1][1] + "...", kept[-1][2])
            break
        kept.append((separator, words, is_focus))
        used += len(separator) + len(words)
    return _as_lines(kept, skipped=first > 0)


def snippet(text: str, max_chars: int = 240, focus: Passage | None = None) -> str:
    """The same as snippet_lines, as plain text."""
    return "\n".join("".join(part for part, _ in line) for line in snippet_lines(text, max_chars, focus))


_SKIPPED = "... "


def _first_part(parts: list[tuple[str, bool, bool]], max_chars: int, focus: Passage | None) -> int:
    """Index of the passage the snippet starts at: the focus, if it would not fit otherwise."""
    if focus is None:
        return 0
    length = 0
    for index, (words, is_focus, _) in enumerate(parts):
        length += (1 if index else 0) + len(words)
        if is_focus:
            return index if length > max_chars - 3 else 0
    return 0


def _whole_words(words: str, room: int) -> str:
    """The longest start of words, in whole words, that fits in room characters."""
    if room <= 0:
        return ""
    if len(words) <= room:
        return words
    cut = words[:room]
    if words[room] != " " and " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip()


def _as_lines(kept: list[tuple[str, str, bool]], skipped: bool) -> list[list[tuple[str, bool]]]:
    """Group parts into lines, keep separators out of the focus, and merge neighbours alike."""
    lines: list[list[tuple[str, bool]]] = [[(_SKIPPED, False)] if skipped else []]
    for separator, words, is_focus in kept:
        if separator == "\n":
            lines.append([])
        elif separator:
            lines[-1].append((separator, False))
        lines[-1].append((words, is_focus))
    return [_merged(line) for line in lines if line]


def _merged(line: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    merged: list[tuple[str, bool]] = []
    for part, is_focus in line:
        if merged and merged[-1][1] == is_focus:
            merged[-1] = (merged[-1][0] + part, is_focus)
        else:
            merged.append((part, is_focus))
    return merged
