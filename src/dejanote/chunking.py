"""Split notes into sections small enough to embed well.

Markdown is split at headings first, so every chunk belongs to exactly one
section and carries its heading path. Sections over the word budget are split
again at paragraph boundaries. Only a paragraph that is too long on its own is
split further: at sentence boundaries for prose, at line breaks for lists,
tables and code. Text is never cut at an arbitrary character count.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePath

# all-MiniLM-L6-v2 reads at most 256 tokens. 150 words of English prose is
# roughly 200 tokens, which leaves room for the heading path.
MAX_WORDS = 150

# Bump whenever chunk boundaries or chunk text change. The index records it, so
# existing indexes get rebuilt instead of keeping chunks made by the old rules.
CHUNKER_VERSION = 1

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
LINE_ITEM = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)])\s|\|)")
_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+")
_LINE_BREAK = re.compile(r"\n")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """A piece of a note: its text as written, the section it sits in, and its lines."""

    text: str
    headings: tuple[str, ...]
    start_line: int  # 1-based, inclusive
    end_line: int

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the heading path gives the section text its context."""
        return " > ".join(self.headings) + "\n\n" + self.text


@dataclass(frozen=True)
class _Heading:
    level: int
    title: str


@dataclass(frozen=True)
class _Block:
    """A paragraph, list or table (consecutive non-blank lines), or a fenced code block."""

    text: str
    start_line: int
    split_by_line: bool  # lists, tables and code split between lines, prose between sentences


@dataclass(frozen=True)
class _Piece:
    """A span of one block's text: the whole block, or one of its sentences or lines."""

    block: int
    start: int
    end: int
    words: int


def chunk_note(text: str, filename: str, max_words: int = MAX_WORDS) -> list[Chunk]:
    """Chunk a note. Markdown is recognised by the file suffix; anything else is plain text."""
    path = PurePath(filename)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    items = _parse(lines, markdown=path.suffix.lower() in MARKDOWN_SUFFIXES)
    chunks = []
    for stack, blocks in _sections(items):
        headings = _breadcrumb(stack, title=path.stem)
        pieces = [p for i, block in enumerate(blocks) for p in _pieces(block, i, max_words)]
        chunks.extend(_assemble(group, blocks, headings) for group in _pack(pieces, max_words))
    return chunks


def _parse(lines: list[str], markdown: bool) -> list[_Heading | _Block]:
    items: list[_Heading | _Block] = []
    paragraph: list[str] = []
    start = 0
    fence: str | None = None

    def flush(code: bool = False) -> None:
        if paragraph:
            split_by_line = code or LINE_ITEM.match(paragraph[0]) is not None
            items.append(_Block("\n".join(paragraph), start, split_by_line))
            paragraph.clear()

    for index in range(_front_matter_end(lines) if markdown else 0, len(lines)):
        line, number = lines[index], index + 1
        if fence is not None:
            paragraph.append(line)
            if _closes_fence(line, fence):
                flush(code=True)
                fence = None
            continue
        if markdown and (match := _FENCE.match(line)):
            flush()
            fence, start = match.group(1), number
            paragraph.append(line)
            continue
        if markdown and (match := _HEADING.match(line)):
            flush()
            items.append(_Heading(len(match.group(1)), match.group(2).strip()))
            continue
        if not line.strip():
            flush()
            continue
        if not paragraph:
            start = number
        paragraph.append(line)
    flush(code=fence is not None)  # an unclosed fence runs to the end of the note
    return items


def _front_matter_end(lines: list[str]) -> int:
    """Index of the first line after YAML front matter, or 0 when there is none."""
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() in ("---", "..."):
                return index + 1
    return 0


def _closes_fence(line: str, fence: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(fence) and set(stripped) == {fence[0]}


def _sections(items: list[_Heading | _Block]) -> Iterator[tuple[tuple[_Heading, ...], list[_Block]]]:
    """Group blocks by section, yielding the heading stack each section sits under."""
    stack: list[_Heading] = []
    blocks: list[_Block] = []
    for item in items:
        if isinstance(item, _Block):
            blocks.append(item)
            continue
        if blocks:
            yield tuple(stack), blocks
            blocks = []
        while stack and stack[-1].level >= item.level:
            stack.pop()
        stack.append(item)
    if blocks:
        yield tuple(stack), blocks


def _breadcrumb(stack: tuple[_Heading, ...], title: str) -> tuple[str, ...]:
    """Heading path for a section, led by the note's title (its H1, else its file name)."""
    headings = tuple(heading.title for heading in stack)
    if stack and stack[0].level == 1:
        return headings
    return (title, *headings)


def _pieces(block: _Block, index: int, max_words: int) -> list[_Piece]:
    words = len(block.text.split())
    if words <= max_words:
        return [_Piece(index, 0, len(block.text), words)]
    pieces = []
    separator = _LINE_BREAK if block.split_by_line else _SENTENCE_GAP
    for start, end in _spans(block.text, separator):
        words = len(block.text[start:end].split())
        if words <= max_words:
            pieces.append(_Piece(index, start, end, words))
        else:  # a single sentence or line longer than the budget: last resort
            pieces.extend(_word_windows(block.text, start, end, index, max_words))
    return pieces


def _spans(text: str, separator: re.Pattern[str]) -> Iterator[tuple[int, int]]:
    """(start, end) offsets of the non-empty stretches of text between separators."""
    start = 0
    for match in separator.finditer(text):
        if match.start() > start:
            yield start, match.start()
        start = match.end()
    if start < len(text):
        yield start, len(text)


def _word_windows(text: str, start: int, end: int, index: int, max_words: int) -> Iterator[_Piece]:
    words = list(_WORD.finditer(text, start, end))
    for offset in range(0, len(words), max_words):
        window = words[offset : offset + max_words]
        yield _Piece(index, window[0].start(), window[-1].end(), len(window))


def _pack(pieces: list[_Piece], max_words: int) -> list[list[_Piece]]:
    """Greedily group consecutive pieces while they fit in the word budget."""
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    count = 0
    for piece in pieces:
        if current and count + piece.words > max_words:
            groups.append(current)
            current, count = [], 0
        current.append(piece)
        count += piece.words
    if current:
        groups.append(current)
    return groups


def _assemble(group: list[_Piece], blocks: list[_Block], headings: tuple[str, ...]) -> Chunk:
    """Rebuild a chunk's text from the original source, keeping its exact formatting."""
    parts = []
    for block_index, run in itertools.groupby(group, key=lambda piece: piece.block):
        run = list(run)
        parts.append(blocks[block_index].text[run[0].start : run[-1].end])
    first, last = group[0], group[-1]
    return Chunk(
        text="\n\n".join(parts),
        headings=headings,
        start_line=_line_of(blocks[first.block], first.start),
        end_line=_line_of(blocks[last.block], last.end),
    )


def _line_of(block: _Block, offset: int) -> int:
    return block.start_line + block.text.count("\n", 0, offset)
