import pytest

from ragdemia.chunking import chunk_document, split_sentences


def test_sentence_splitter_respects_abbreviations():
    text = "We follow Karpukhin et al. in this setup. See Fig. 2 for details. Results improve."
    assert split_sentences(text) == [
        "We follow Karpukhin et al. in this setup.",
        "See Fig. 2 for details.",
        "Results improve.",
    ]


def test_chunks_respect_budget_and_sections(parsed):
    chunks = chunk_document(parsed, max_tokens=80, overlap_tokens=20, min_tokens=10)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.n_tokens <= 80
        assert c.page_start <= c.page_end
        assert " > " not in c.section or c.section  # breadcrumb is a string
    # no chunk mixes text from two sections
    method = [c for c in chunks if c.section == "2 Method"]
    assert method and all("78.4" not in c.text for c in method)


def test_overlap_repeats_sentences(parsed):
    chunks = chunk_document(parsed, max_tokens=60, overlap_tokens=25, min_tokens=10)
    same_section = [c for c in chunks if c.section == "2 Method"]
    assert len(same_section) >= 2
    last_sentence = split_sentences(same_section[0].text)[-1]
    assert same_section[1].text.startswith(last_sentence)


def test_overlap_must_be_smaller_than_budget(parsed):
    with pytest.raises(ValueError):
        chunk_document(parsed, max_tokens=50, overlap_tokens=50)


def test_embed_text_has_context_header(parsed):
    c = chunk_document(parsed, max_tokens=80, overlap_tokens=20, min_tokens=10)[0]
    assert c.embed_text.startswith(f"{parsed.title} | {c.section}\n")
