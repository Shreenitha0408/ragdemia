from ragdemia.evaluation import GoldRef, fact_recall, retrieval_metrics
from ragdemia.llm import EchoLLM
from ragdemia.pipeline import RAGPipeline
from ragdemia.prompts import ABSTAIN_RAG


def test_pipeline_builds_numbered_prompt_and_parses_answer(retriever):
    llm = EchoLLM("The encoder was trained with a batch size of 128 [1].")
    pipe = RAGPipeline(retriever, llm, top_k=3)
    ans = pipe.ask("What batch size was used?")
    user_turn = llm.calls[0][-1]["content"]
    assert "[1] Source: A Tiny Paper About Retrieval" in user_turn
    assert "What batch size was used?" in user_turn
    assert ans.citations.cited == [1]
    assert ans.to_dict()["sources"][0]["cited"] is True


def test_closed_book_has_no_sources(retriever):
    pipe = RAGPipeline(retriever, EchoLLM("I don't know."))
    ans = pipe.ask_closed_book("Anything?")
    assert ans.sources == [] and ans.citations.abstained


def test_abstention_is_detected(retriever):
    ans = RAGPipeline(retriever, EchoLLM(ABSTAIN_RAG)).ask("Unrelated question?")
    assert ans.citations.abstained


def test_fact_recall_aliases():
    score, hits = fact_recall(
        "It uses a **batch size of 128** and one BM25 negative.",
        [["128"], ["one bm25", "single bm25"], ["contriever"]],
    )
    assert hits == [True, True, False] and round(score, 2) == 0.67


def test_retrieval_metrics_page_level(retriever):
    res = retriever.retrieve("batch size in-batch negatives", top_k=3)
    m = retrieval_metrics(res, [GoldRef(res[0].chunk.doc_id, [res[0].chunk.page_start])])
    assert m["hit"] == 1.0 and m["mrr"] == 1.0
