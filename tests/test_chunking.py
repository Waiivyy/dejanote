"""Chunking: sections first, then paragraphs, then sentences; never arbitrary cuts."""

from __future__ import annotations

from dejanote.chunking import Chunk, chunk_note

JAPAN = """\
# Trip to Japan

Two weeks in April, mostly trains.

## Kyoto

Stayed near Gion. Temples are best before 8am.

### Food

Nishiki market for lunch.

## Tokyo ##

Shibuya at night.
"""


def test_sections_follow_headings_and_carry_the_heading_path():
    assert chunk_note(JAPAN, "japan.md") == [
        Chunk("Two weeks in April, mostly trains.", ("Trip to Japan",), 3, 3),
        Chunk("Stayed near Gion. Temples are best before 8am.", ("Trip to Japan", "Kyoto"), 7, 7),
        Chunk("Nishiki market for lunch.", ("Trip to Japan", "Kyoto", "Food"), 11, 11),
        Chunk("Shibuya at night.", ("Trip to Japan", "Tokyo"), 15, 15),
    ]


def test_a_note_without_a_top_level_heading_is_titled_by_its_file_name():
    assert chunk_note("## Feeding\n\nFlour and water.\n", "sourdough-starter.md") == [
        Chunk("Flour and water.", ("sourdough-starter", "Feeding"), 3, 3),
    ]


def test_text_before_the_first_heading_is_its_own_chunk():
    assert chunk_note("Intro line.\n\n## Details\n\nMore.\n", "note.md") == [
        Chunk("Intro line.", ("note",), 1, 1),
        Chunk("More.", ("note", "Details"), 5, 5),
    ]


def test_headings_without_content_produce_no_chunks():
    assert chunk_note("# A\n## B\n\ntext\n", "n.md") == [Chunk("text", ("A", "B"), 4, 4)]


def test_hashtags_are_not_headings():
    assert chunk_note("#todo buy milk\n#errands\n", "n.md") == [
        Chunk("#todo buy milk\n#errands", ("n",), 1, 2),
    ]


def test_lines_inside_code_fences_are_never_headings():
    note = "## Undo the last commit\n\nKeep the changes staged:\n\n```bash\n# a shell comment\ngit reset --soft HEAD~1\n```\n"
    assert chunk_note(note, "git.md") == [
        Chunk(
            "Keep the changes staged:\n\n```bash\n# a shell comment\ngit reset --soft HEAD~1\n```",
            ("git", "Undo the last commit"),
            3,
            8,
        ),
    ]


def test_content_after_a_closed_code_fence_is_parsed_normally():
    note = "## Before\n\n```\ncode\n```\n\n## After\n\nprose\n"
    assert chunk_note(note, "n.md") == [
        Chunk("```\ncode\n```", ("n", "Before"), 3, 5),
        Chunk("prose", ("n", "After"), 9, 9),
    ]


def test_an_oversized_code_block_splits_between_lines_never_inside_one():
    lines = [f'print("Step {i} done. Next step follows here now")' for i in range(10)]  # ". " mid-line
    note = "## Script\n\n```python\n" + "\n".join(lines) + "\n```\n"
    chunks = chunk_note(note, "s.md", max_words=20)
    source_lines = note.split("\n")
    for chunk in chunks:
        assert chunk.text.split("\n") == source_lines[chunk.start_line - 1 : chunk.end_line]


def test_a_code_block_with_blank_lines_stays_in_one_piece():
    note = "## Snippet\n\n```python\ndef f():\n    return 1\n\n\nprint(f())\n```\n"
    assert chunk_note(note, "s.md") == [
        Chunk("```python\ndef f():\n    return 1\n\n\nprint(f())\n```", ("s", "Snippet"), 3, 9),
    ]


def test_chunk_text_keeps_the_notes_blank_lines_so_line_numbers_can_be_counted():
    note = "## S\n\nfirst paragraph\n\n\n\nsecond paragraph\n"
    [chunk] = chunk_note(note, "n.md")
    assert chunk.text == "first paragraph\n\n\n\nsecond paragraph"
    assert (chunk.start_line, chunk.end_line) == (3, 7)


def test_small_paragraphs_in_a_section_are_packed_together():
    note = "## S\n\none two three\n\nfour five\n\nsix\n"
    assert chunk_note(note, "n.md", max_words=100) == [
        Chunk("one two three\n\nfour five\n\nsix", ("n", "S"), 3, 7),
    ]


def _words(word: str, n: int) -> str:
    return " ".join([word] * n)


def test_paragraphs_that_exactly_fill_the_budget_share_a_chunk():
    note = "## S\n\n" + _words("a", 50) + "\n\n" + _words("b", 50) + "\n"
    assert len(chunk_note(note, "n.md", max_words=100)) == 1


def test_a_long_section_splits_at_paragraph_boundaries():
    note = "## S\n\n" + "\n\n".join([_words("a", 60), _words("b", 60), _words("c", 60)]) + "\n"
    chunks = chunk_note(note, "n.md", max_words=100)
    assert [c.text for c in chunks] == [_words("a", 60), _words("b", 60), _words("c", 60)]
    assert [(c.start_line, c.end_line) for c in chunks] == [(3, 3), (5, 5), (7, 7)]


def _sentence(word: str) -> str:
    return _words(word, 39) + " end."  # 40 words


def test_a_paragraph_too_long_on_its_own_splits_at_sentence_boundaries():
    paragraph = " ".join(_sentence(w) for w in "abcd")  # 160 words on one line
    chunks = chunk_note(f"## S\n\n{paragraph}\n", "n.md", max_words=100)
    assert [c.text for c in chunks] == [
        _sentence("a") + " " + _sentence("b"),
        _sentence("c") + " " + _sentence("d"),
    ]
    assert [(c.start_line, c.end_line) for c in chunks] == [(3, 3), (3, 3)]


def test_sentence_splits_of_a_wrapped_paragraph_keep_their_line_numbers():
    paragraph = "\n".join(_sentence(w) for w in "abcd")  # one sentence per line
    chunks = chunk_note(f"## S\n\n{paragraph}\n", "n.md", max_words=100)
    assert [c.text for c in chunks] == [
        _sentence("a") + "\n" + _sentence("b"),
        _sentence("c") + "\n" + _sentence("d"),
    ]
    assert [(c.start_line, c.end_line) for c in chunks] == [(3, 4), (5, 6)]


def test_a_list_too_long_on_its_own_splits_between_items():
    items = [f"- item {i} " + _words("word", 20) for i in range(6)]  # 23 words each
    chunks = chunk_note("## List\n\n" + "\n".join(items) + "\n", "l.md", max_words=50)
    assert [c.text for c in chunks] == ["\n".join(items[0:2]), "\n".join(items[2:4]), "\n".join(items[4:6])]
    assert [(c.start_line, c.end_line) for c in chunks] == [(3, 4), (5, 6), (7, 8)]


def test_a_single_endless_sentence_falls_back_to_word_windows():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_note(f"## S\n\n{text}\n", "n.md", max_words=100)
    assert [len(c.text.split()) for c in chunks] == [100, 100, 50]
    assert chunks[0].text.startswith("w0 w1 ") and chunks[-1].text.endswith(" w249")


def test_front_matter_is_skipped_but_line_numbers_still_match_the_file():
    note = "---\ntags: [cooking]\n---\n# Weeknight dal\n\nRed lentils, cumin, garlic.\n"
    assert chunk_note(note, "dal.md") == [Chunk("Red lentils, cumin, garlic.", ("Weeknight dal",), 6, 6)]


def test_plain_text_files_have_no_markdown_headings():
    note = "# not a heading here\nsecond line\n\nNext paragraph.\n"
    assert chunk_note(note, "packing.txt") == [
        Chunk("# not a heading here\nsecond line\n\nNext paragraph.", ("packing",), 1, 4),
    ]


def test_windows_line_endings_are_normalised():
    assert chunk_note("## S\r\n\r\nline one\r\nline two\r\n", "n.md") == [
        Chunk("line one\nline two", ("n", "S"), 3, 4),
    ]


def test_an_empty_note_has_no_chunks():
    assert chunk_note("", "n.md") == []
    assert chunk_note("\n  \n\n", "n.md") == []


def test_the_embedded_text_carries_the_heading_path():
    chunk = Chunk("Discard all but 50 g, then feed.", ("Sourdough starter", "Feeding"), 7, 7)
    assert "Sourdough starter" in chunk.embedding_text
    assert "Feeding" in chunk.embedding_text
    assert chunk.text in chunk.embedding_text
