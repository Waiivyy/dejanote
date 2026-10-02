"""Evaluation: ranks of the expected note and section, and the scores built from them."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dejanote.chunking import Chunk
from dejanote.evaluation import Case, Outcome, load_cases, run_case, score
from dejanote.store import Store


class FixedQuery:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed_query(self, text):
        return self.vector


@pytest.fixture
def store(tmp_path):
    # Similarity to the query [1, 0, 0] runs: tea > coffee section B > coffee section A > water.
    with Store(tmp_path / "index.db", "test-model@1", 3) as store:
        store.replace_file(
            str(tmp_path / "notes/drinks/coffee.md"),
            "h1",
            [Chunk("a", ("Coffee", "A"), 1, 1), Chunk("b", ("Coffee", "B"), 3, 3)],
            np.array([[0.6, 0.8, 0.0], [0.8, 0.6, 0.0]]),
        )
        store.replace_file(str(tmp_path / "notes/tea.md"), "h2", [Chunk("t", ("Tea",), 1, 1)], np.array([[1.0, 0.0, 0.0]]))
        store.replace_file(str(tmp_path / "notes/water.md"), "h3", [Chunk("w", ("Water",), 1, 1)], np.array([[0.0, 0.0, 1.0]]))
        yield store


def test_ranks_count_distinct_notes_and_individual_sections(tmp_path, store):
    outcome = run_case(store, FixedQuery([1, 0, 0]), tmp_path / "notes", Case("q", "drinks/coffee.md", "A"))
    assert outcome.note_rank == 2  # tea first, then coffee
    assert outcome.section_rank == 3  # tea, coffee B, then coffee A
    assert outcome.top_note == "tea.md"


def test_a_case_without_a_section_has_no_section_rank(tmp_path, store):
    outcome = run_case(store, FixedQuery([1, 0, 0]), tmp_path / "notes", Case("q", "water.md"))
    assert (outcome.note_rank, outcome.section_rank) == (3, None)


def test_a_note_that_is_not_in_the_index_has_no_rank(tmp_path, store):
    outcome = run_case(store, FixedQuery([1, 0, 0]), tmp_path / "notes", Case("q", "missing.md", "X"))
    assert (outcome.note_rank, outcome.section_rank) == (None, None)


def test_scores_from_known_ranks():
    outcomes = [
        Outcome(Case("q1", "a.md", "s"), note_rank=1, section_rank=1, top_note="a.md"),
        Outcome(Case("q2", "b.md", "s"), note_rank=2, section_rank=4, top_note="x.md"),
        Outcome(Case("q3", "c.md"), note_rank=None, section_rank=None, top_note="x.md"),
        Outcome(Case("q4", "d.md"), note_rank=4, section_rank=None, top_note="x.md"),
    ]
    scores = score(outcomes)
    assert scores.note_at_1 == pytest.approx(1 / 4)
    assert scores.note_at_3 == pytest.approx(2 / 4)
    assert scores.mrr == pytest.approx((1 + 1 / 2 + 0 + 1 / 4) / 4)
    assert scores.section_at_1 == pytest.approx(1 / 2)  # only q1 and q2 name a section
    assert scores.section_at_3 == pytest.approx(1 / 2)


def test_cases_load_from_json(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([{"query": "q", "note": "a.md", "section": "S"}, {"query": "r", "note": "b.md"}]))
    assert load_cases(path) == [Case("q", "a.md", "S"), Case("r", "b.md")]
