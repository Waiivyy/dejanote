"""Search: rank stored chunks by cosine similarity to the query."""

from __future__ import annotations

import numpy as np
import pytest

from dejanote.chunking import Chunk
from dejanote.search import search, snippet
from dejanote.store import Store


class FixedQuery:
    """Embeds every query as the same hand-picked vector, so expected rankings can be worked out by hand."""

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed_query(self, text):
        return self.vector


def _chunk(name: str) -> Chunk:
    return Chunk(f"text of {name}", (name,), 1, 1)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "index.db", model_id="test-model@1", dimension=3) as store:
        yield store


def test_results_are_ranked_by_cosine_similarity_with_their_scores(store):
    store.replace_file("/notes/far.md", "h1", [_chunk("far")], np.array([[0.0, 1.0, 0.0]]))
    store.replace_file("/notes/exact.md", "h2", [_chunk("exact")], np.array([[1.0, 0.0, 0.0]]))
    store.replace_file("/notes/close.md", "h3", [_chunk("close")], np.array([[0.8, 0.6, 0.0]]))
    results = search(store, FixedQuery([1.0, 0.0, 0.0]), "anything", limit=3)
    assert [r.path for r in results] == ["/notes/exact.md", "/notes/close.md", "/notes/far.md"]
    assert [round(r.score, 6) for r in results] == [1.0, 0.8, 0.0]
    assert results[0].chunk == _chunk("exact")


def test_only_the_best_results_up_to_the_limit_come_back_in_order(store):
    # 30 chunks at 0, 3, ..., 87 degrees from the query, stored shuffled; the smallest angles win.
    # A limit of 10 matters: numpy hands back the top 5 or more in arbitrary order unless sorted.
    degrees = np.random.default_rng(7).permutation(np.arange(0, 90, 3))
    angles = np.radians(degrees)
    vectors = np.stack([np.cos(angles), np.sin(angles), np.zeros(len(degrees))], axis=1)
    store.replace_file("/notes/fan.md", "h", [_chunk(f"angle {d}") for d in degrees], vectors)
    results = search(store, FixedQuery([1.0, 0.0, 0.0]), "anything", limit=10)
    assert [r.chunk.headings[0] for r in results] == [f"angle {d}" for d in range(0, 30, 3)]


def test_an_empty_index_has_no_results(store):
    assert search(store, FixedQuery([1.0, 0.0, 0.0]), "anything") == []


def test_snippets_collapse_whitespace():
    assert snippet("Grey liquid\non top   means\n\nit is hungry.") == "Grey liquid on top means it is hungry."


def test_snippets_cut_long_text_at_a_word_boundary_and_say_so():
    text = "word " * 100
    cut = snippet(text, max_chars=42)
    assert len(cut) <= 42
    assert cut.endswith("...")
    assert cut[:-3].split() == ["word"] * len(cut[:-3].split())  # no word cut in half


def test_snippets_leave_out_code_fence_markers():
    assert snippet("Keep the changes staged:\n\n```bash\ngit reset --soft HEAD~1\n```") == (
        "Keep the changes staged: git reset --soft HEAD~1"
    )
