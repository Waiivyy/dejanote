"""Measure how well search finds the right note for a set of example queries.

Each case names the note that answers a query and, usually, the section within
it (the last heading on the chunk's heading path). Some also list the passages
that answer it, word for word, to judge the highlight. Scores:

    note@1     the expected note is the best result
    note@3     the expected note is among the three best distinct notes
    section@1  the best chunk is the expected section
    section@3  the expected section is among the three best chunks
    MRR        mean of 1/rank of the expected note, 0 when it is missing
    highlight  in the expected section, the highlighted passage is an expected one
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dejanote.search import PassageEmbedder, best_passages, rank_chunks
from dejanote.store import Store


@dataclass(frozen=True)
class Case:
    query: str
    note: str  # path relative to the notes folder, with forward slashes
    section: str | None = None
    passages: tuple[str, ...] = ()  # any of these, inside the highlight, makes it right


@dataclass(frozen=True)
class Outcome:
    case: Case
    note_rank: int | None  # 1-based rank among distinct notes, None if not found
    section_rank: int | None  # 1-based rank among chunks, None if not found or not asked for
    top_note: str  # what came first instead, for explaining misses
    highlight: str | None = None  # the passage highlighted in the expected section
    highlight_right: bool | None = None  # None when the case lists no passages


@dataclass(frozen=True)
class Scores:
    note_at_1: float
    note_at_3: float
    section_at_1: float
    section_at_3: float
    mrr: float
    highlight: float


def load_cases(path: Path) -> list[Case]:
    return [
        Case(item["query"], item["note"], item.get("section"), tuple(item.get("passages", ())))
        for item in json.loads(path.read_text())
    ]


def run_case(store: Store, embedder: PassageEmbedder, notes_root: Path, case: Case) -> Outcome:
    """Rank every chunk for the case's query and find where the expected note and section land.

    The highlight is judged in the best-ranked chunk of the expected section (of
    the expected note, without a section), wherever it ranks, so a ranking miss
    does not count against the highlight as well.
    """
    everything = store.counts()[1]
    results = rank_chunks(store, embedder, case.query, limit=everything)
    notes: list[str] = []
    section_rank = None
    expected = None  # the best-ranked chunk of the expected section
    for rank, result in enumerate(results, start=1):
        note = Path(result.path).relative_to(notes_root).as_posix()
        if note not in notes:
            notes.append(note)
        in_section = note == case.note and (case.section is None or result.chunk.headings[-1] == case.section)
        if expected is None and in_section:
            expected = result.chunk
            if case.section:
                section_rank = rank
    note_rank = notes.index(case.note) + 1 if case.note in notes else None
    highlight = highlight_right = None
    if case.passages:
        highlight = ""
        if expected is not None:
            [passage] = best_passages(embedder, case.query, [expected.text])
            highlight = " ".join(expected.text[passage.start : passage.end].split()) if passage else ""
        highlight_right = any(wanted in highlight for wanted in case.passages)
    return Outcome(case, note_rank, section_rank, notes[0] if notes else "", highlight, highlight_right)


def score(outcomes: list[Outcome]) -> Scores:
    with_section = [o for o in outcomes if o.case.section]
    return Scores(
        note_at_1=_share(outcomes, lambda o: o.note_rank == 1),
        note_at_3=_share(outcomes, lambda o: o.note_rank is not None and o.note_rank <= 3),
        section_at_1=_share(with_section, lambda o: o.section_rank == 1),
        section_at_3=_share(with_section, lambda o: o.section_rank is not None and o.section_rank <= 3),
        mrr=sum(1 / o.note_rank for o in outcomes if o.note_rank) / len(outcomes) if outcomes else 0.0,
        highlight=_share([o for o in outcomes if o.highlight_right is not None], lambda o: o.highlight_right),
    )


def _share(outcomes: list[Outcome], hit) -> float:
    return sum(1 for o in outcomes if hit(o)) / len(outcomes) if outcomes else 0.0
