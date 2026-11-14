from ragdemia.parsing import Block, _join_lines, _looks_like_heading


def test_detects_title_and_sections(parsed):
    assert parsed.title == "A Tiny Paper About Retrieval"
    titles = [s.title for s in parsed.sections]
    assert "1 Introduction" in titles and "2 Method" in titles and "3 Results" in titles


def test_skips_references(parsed):
    assert parsed.skipped_sections == ["References"]
    assert all("cited paper" not in p.text for s in parsed.sections for p in s.paragraphs)


def test_pages_are_one_based(parsed):
    assert min(p.page for s in parsed.sections for p in s.paragraphs) == 1


def test_dehyphenation():
    assert _join_lines(["impres-", "sive results"]) == "impressive results"
    assert _join_lines(["state-of-the-art", "Model"]) == "state-of-the-art Model"


def _block(text, size=10.0, bold=0.0, lines=1):
    return Block(text, 1, size, bold, lines, 0, 10, 800, len(text))


def test_heading_heuristics():
    assert _looks_like_heading(_block("3.2 HyDE"), 10, set())  # numbered, short title
    assert _looks_like_heading(_block("II. OVERVIEW OF RAG", size=8.5), 10, set())  # small caps
    assert not _looks_like_heading(_block("20 Total Retrieved Documents"), 10, set())  # figure text
    assert not _looks_like_heading(_block("We use a batch size of 128."), 10, set())  # sentence
    assert not _looks_like_heading(_block("A SELF-RAG Details 17"), 10, set())  # TOC entry
    assert _looks_like_heading(_block("REFERENCES", size=8), 10, {"references"})
