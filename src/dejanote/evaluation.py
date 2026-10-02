"""Measure how well search finds the right note for a set of example queries.

Each case names the note that answers a query and, usually, the section within
it (the last heading on the chunk's heading path). Scores:

    note@1     the expected note is the best result
    note@3     the expected note is among the three best distinct notes
    section@1  the best chunk is the expected section
    section@3  the expected section is among the three best chunks
    MRR        mean of 1/rank of the expected note, 0 when it is missing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dejanote.search import QueryEmbedder, rank_chunks
from dejanote.store import Store


@dataclass(frozen=True)
class Case:
    query: str
    note: str  # path relative to the notes folder, with forward slashes
    section: str | None = None


@dataclass(frozen=True)
class Outcome:
    case: Case
    note_rank: int | None  # 1-based rank among distinct notes, None if not found
    section_rank: int | None  # 1-based rank among chunks, None if not found or not asked for
    top_note: str  # what came first instead, for explaining misses


@dataclass(frozen=True)
class Scores:
    note_at_1: float
    note_at_3: float
    section_at_1: float
    section_at_3: float
    mrr: float


def load_cases(path: Path) -> list[Case]:
    return [Case(item["query"], item["note"], item.get("section")) for item in json.loads(path.read_text())]


def run_case(store: Store, embedder: QueryEmbedder, notes_root: Path, case: Case) -> Outcome:
    """Rank every chunk for the case's query and find where the expected note and section land."""
    everything = store.counts()[1]
    results = rank_chunks(store, embedder, case.query, limit=everything)
    notes: list[str] = []
    section_rank = None
    for rank, result in enumerate(results, start=1):
        note = Path(result.path).relative_to(notes_root).as_posix()
        if note not in notes:
            notes.append(note)
        if section_rank is None and case.section and note == case.note and result.chunk.headings[-1] == case.section:
            section_rank = rank
    note_rank = notes.index(case.note) + 1 if case.note in notes else None
    return Outcome(case, note_rank, section_rank, top_note=notes[0] if notes else "")


def score(outcomes: list[Outcome]) -> Scores:
    with_section = [o for o in outcomes if o.case.section]
    return Scores(
        note_at_1=_share(outcomes, lambda o: o.note_rank == 1),
        note_at_3=_share(outcomes, lambda o: o.note_rank is not None and o.note_rank <= 3),
        section_at_1=_share(with_section, lambda o: o.section_rank == 1),
        section_at_3=_share(with_section, lambda o: o.section_rank is not None and o.section_rank <= 3),
        mrr=sum(1 / o.note_rank for o in outcomes if o.note_rank) / len(outcomes) if outcomes else 0.0,
    )


def _share(outcomes: list[Outcome], hit) -> float:
    return sum(1 for o in outcomes if hit(o)) / len(outcomes) if outcomes else 0.0
