"""Search: rank stored chunks by cosine similarity to the query."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeEmbedder
from dejanote.chunking import Chunk
from dejanote.search import best_passages, passage_line, passages, rank_chunks, search, snippet, snippet_lines
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
    results = rank_chunks(store, FixedQuery([1.0, 0.0, 0.0]), "anything", limit=10)
    assert [r.chunk.headings[0] for r in results] == [f"angle {d}" for d in range(0, 30, 3)]


def test_an_empty_index_has_no_results(store):
    assert search(store, FixedQuery([1.0, 0.0, 0.0]), "anything") == []


def _scoring(score: float) -> list[float]:
    """A unit vector whose cosine similarity with the query [1, 0, 0] is exactly score."""
    return [score, (1 - score**2) ** 0.5, 0.0]


@pytest.fixture
def notes_with_sections(store):
    # Ranking for the query [1, 0, 0]: a1, a2, a3, a4, b1, c1, then a5 far behind.
    store.replace_file(
        "/notes/a.md", "ha",
        [_chunk("a1"), _chunk("a2"), _chunk("a3"), _chunk("a4"), _chunk("a5")],
        np.array([_scoring(s) for s in (0.9, 0.85, 0.8, 0.75, 0.1)]),
    )
    store.replace_file("/notes/b.md", "hb", [_chunk("b1")], np.array([_scoring(0.7)]))
    store.replace_file("/notes/c.md", "hc", [_chunk("c1")], np.array([_scoring(0.6)]))
    return store


def test_each_note_appears_once_with_its_best_section_and_other_strong_sections_under_it(notes_with_sections):
    results = search(notes_with_sections, FixedQuery([1.0, 0.0, 0.0]), "anything", limit=3, related=2)
    assert [(r.path, r.chunk.headings[0]) for r in results] == [("/notes/a.md", "a1"), ("/notes/b.md", "b1"), ("/notes/c.md", "c1")]
    assert [s.chunk.headings[0] for s in results[0].also] == ["a2", "a3"]
    assert results[1].also == ()


def test_the_limit_counts_notes_not_sections(notes_with_sections):
    results = search(notes_with_sections, FixedQuery([1.0, 0.0, 0.0]), "anything", limit=2)
    assert [r.path for r in results] == ["/notes/a.md", "/notes/b.md"]


def test_weak_sections_are_not_listed_under_their_note(store):
    store.replace_file("/notes/a.md", "ha", [_chunk("a1"), _chunk("a5")], np.array([_scoring(0.9), _scoring(0.1)]))
    store.replace_file("/notes/b.md", "hb", [_chunk("b1")], np.array([_scoring(0.7)]))
    store.replace_file("/notes/c.md", "hc", [_chunk("c1")], np.array([_scoring(0.6)]))
    [best] = search(store, FixedQuery([1.0, 0.0, 0.0]), "anything", limit=1)
    assert best.also == ()  # a5 ranks below other notes' sections, so it is not worth a mention


def _passage_texts(text: str) -> list[str]:
    return [text[p.start : p.end] for p in passages(text)]


def test_prose_is_split_into_sentences():
    text = "Grey liquid on top means it is hungry. Pour it off!\nThen feed it? Yes."
    assert _passage_texts(text) == ["Grey liquid on top means it is hungry.", "Pour it off!", "Then feed it?", "Yes."]


def test_a_hard_wrapped_sentence_is_one_passage():
    text = "Grey liquid\non top means hunger. Feed it."
    assert _passage_texts(text) == ["Grey liquid\non top means hunger.", "Feed it."]


def test_list_items_and_lines_of_code_are_passages_of_their_own():
    text = "Signs:\n- doubled in size\n- smells tangy\n  like yoghurt\n\n```bash\ngit status\ngit log\n```"
    assert _passage_texts(text) == ["Signs:", "- doubled in size", "- smells tangy\n  like yoghurt", "git status", "git log"]


def _focus(text: str, wanted: str):
    return next(p for p in passages(text) if text[p.start : p.end] == wanted)


def test_the_focus_passage_is_marked_in_the_snippet():
    text = "Grey liquid means hunger. A nail polish smell means the same. Pink streaks mean contamination."
    focus = _focus(text, "A nail polish smell means the same.")
    assert snippet_lines(text, focus=focus) == [
        [("Grey liquid means hunger. ", False), ("A nail polish smell means the same.", True),
         (" Pink streaks mean contamination.", False)],
    ]


def test_a_snippet_starts_at_a_focus_that_would_otherwise_be_cut_off():
    filler = " ".join(f"Filler sentence number {i} about nothing much." for i in range(10))
    text = f"{filler} The sentence that matched is here. And one more after it."
    focus = _focus(text, "The sentence that matched is here.")
    [line] = snippet_lines(text, max_chars=80, focus=focus)
    assert line[0] == ("... ", False)
    assert line[1] == ("The sentence that matched is here.", True)
    assert sum(len(part) for part, _ in line) <= 80


def test_a_passage_knows_which_line_of_the_note_it_starts_on():
    chunk = Chunk("First paragraph.\n\nSecond one.\nStill second. Third sentence.", ("n",), 10, 13)
    assert passage_line(chunk, _focus(chunk.text, "Third sentence.")) == 13
    assert passage_line(chunk, _focus(chunk.text, "Second one.")) == 12
    assert passage_line(chunk, None) == 10  # no passage: the section's first line


def test_the_passage_to_highlight_is_the_one_closest_to_the_query():
    # The fake embedder's similarity is word overlap, so the answer can be worked out by hand.
    text = "The boiler gauge reads low. Feed the starter with flour and water. Trains in Tokyo."
    [best] = best_passages(FakeEmbedder(), "starter flour water", [text])
    assert text[best.start : best.end] == "Feed the starter with flour and water."


def test_a_text_that_is_one_passage_gets_no_highlight():
    texts = ["Just one sentence here.", "Two sentences. Here they are."]
    best = best_passages(FakeEmbedder(), "sentences", texts)
    assert best[0] is None
    assert best[1] is not None


def test_snippets_join_hard_wrapped_prose_into_one_line_per_paragraph():
    text = "Grey liquid\non top   means\nit is hungry.\n\nPour it off."
    assert snippet(text) == "Grey liquid on top means it is hungry.\nPour it off."


def test_snippets_keep_list_items_on_their_own_lines():
    text = "Signs:\n- doubled in size\n- smells tangy\n  like yoghurt\n- floats"
    assert snippet(text) == "Signs:\n- doubled in size\n- smells tangy like yoghurt\n- floats"


def test_snippets_cut_long_text_at_a_word_boundary_and_say_so():
    text = "word " * 100
    cut = snippet(text, max_chars=42)
    assert len(cut) <= 42
    assert cut.endswith("...")
    assert cut[:-3].split() == ["word"] * len(cut[:-3].split())  # no word cut in half


def test_snippets_keep_code_lines_apart_and_leave_out_fence_markers():
    text = "Keep the changes staged:\n\n```bash\ngit reset --soft HEAD~1\ngit status\n```"
    assert snippet(text) == "Keep the changes staged:\ngit reset --soft HEAD~1\ngit status"
