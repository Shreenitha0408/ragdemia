from ragdemia.citations import check_citations, extract_citations, is_abstention

SOURCES = [
    "DPR uses two independent BERT encoders and the CLS token representation.",
    "The model is trained with a batch size of 128 and one BM25 negative per question.",
]


def test_extract_citation_formats():
    assert extract_citations("A [1]. B [2][3]. C [1, 4]. D [2-3].") == [1, 2, 3, 4]


def test_invalid_citations_are_flagged():
    rep = check_citations("Training used a batch size of 128 [2]. It also used magic [7].", SOURCES)
    assert rep.cited == [2] and rep.invalid == [7]


def test_trailing_citation_after_period_counts():
    rep = check_citations("Training used a batch size of 128 and one BM25 negative. [2]", SOURCES)
    assert rep.coverage == 1.0


def test_auto_attribution_for_uncited_supported_sentence():
    rep = check_citations("DPR uses two independent BERT encoders.", SOURCES)
    assert rep.cited == [] and rep.auto_attributed == [1]
    assert rep.groundedness == 1.0


def test_unsupported_sentence_lowers_groundedness():
    rep = check_citations(
        "Quantum chromodynamics explains hadron confinement completely [1].", SOURCES
    )
    assert rep.groundedness == 0.0


def test_abstention_detection():
    assert is_abstention("I couldn't find this in your documents.")
    assert is_abstention("I don't know.")
    assert not is_abstention("The batch size was 128 [2].")
