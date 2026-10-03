"""Search quality on the example notes with the real model: the README's claims, enforced."""

from __future__ import annotations

from pathlib import Path

import pytest

from dejanote.evaluation import load_cases, run_case, score
from dejanote.indexer import index_folder
from dejanote.search import best_passages, search
from dejanote.store import Store

EXAMPLES = Path(__file__).parent.parent / "examples" / "notes"
QUERIES = Path(__file__).parent.parent / "examples" / "queries.json"


@pytest.fixture(scope="session")
def example_index(tmp_path_factory, embedder):
    store = Store(tmp_path_factory.mktemp("quality") / "index.db", embedder.model_id, embedder.dimension)
    index_folder(EXAMPLES, store, embedder)
    yield store
    store.close()


@pytest.mark.model
@pytest.mark.parametrize(
    "query, note",
    [
        ("feeding my sourdough starter", "cooking/sourdough-starter.md"),
        ("boiler pressure is too low", "home/boiler-pressure.md"),
        ("undo the last git commit", "tech/git-recipes.md"),
        ("our trip to Japan", "travel/japan-2024.md"),
    ],
)
def test_an_obvious_query_finds_the_note_it_is_about(example_index, embedder, query, note):
    best = search(example_index, embedder, query)[0]
    assert Path(best.path).relative_to(EXAMPLES).as_posix() == note


@pytest.mark.model
def test_the_highlight_is_the_passage_that_answers_the_query(example_index, embedder):
    # "Gratuity" appears nowhere in the note: the right line has to be found by meaning.
    best = search(example_index, embedder, "is it rude to leave a gratuity")[0]
    [passage] = best_passages(embedder, "is it rude to leave a gratuity", [best.chunk.text])
    assert best.chunk.headings[-1] == "Etiquette I want to remember"
    assert best.chunk.text[passage.start : passage.end] == "- no tipping anywhere; it comes across as awkward or even rude"



@pytest.mark.model
def test_a_highlight_is_judged_together_with_the_sentences_around_it(example_index, embedder):
    # On its own, "First time in weeks my head felt quiet." scores highest, partly for sharing
    # "felt" with the query. Read with its neighbours, the decision to take a day off wins.
    query = "felt burned out and needed a break"
    best = search(example_index, embedder, query)[0]
    [passage] = best_passages(embedder, query, [best.chunk.text])
    assert best.chunk.text[passage.start : passage.end] == "Decided: taking Monday off, no laptop."

@pytest.mark.model
def test_search_quality_on_the_example_queries_does_not_regress(example_index, embedder):
    # Measured with bge-small-en-v1.5: note@1 0.98, note@3 1.00, section@1 0.79, MRR 0.99.
    # The floors leave room for one borderline query but not for real regressions: dropping
    # the query instruction or the heading paths costs section@1 0.08 to 0.11, and
    # all-MiniLM-L6-v2 only reaches note@1 0.91.
    scores = score([run_case(example_index, embedder, EXAMPLES, case) for case in load_cases(QUERIES)])
    assert scores.note_at_1 >= 0.93
    assert scores.note_at_3 >= 0.97
    assert scores.section_at_1 >= 0.75
    assert scores.mrr >= 0.95
