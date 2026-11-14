from ragdemia.llm import EchoLLM
from ragdemia.pipeline import RAGPipeline
from ragdemia.prompts import clean_rewrite, rewrite_messages


class ScriptedLLM(EchoLLM):
    """Returns queued responses in order (rewrite call first, then the answer)."""

    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)

    def generate(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_no_history_means_no_rewrite_call(retriever):
    llm = ScriptedLLM(["Batch size 128 [1]."])
    ans = RAGPipeline(retriever, llm).ask("What batch size was used?")
    assert len(llm.calls) == 1 and ans.search_query is None


def test_follow_up_is_rewritten_and_used_for_retrieval(retriever):
    llm = ScriptedLLM(["What batch size was used to train the encoder?", "It was 128 [1]."])
    pipe = RAGPipeline(retriever, llm, top_k=2)
    history = [("How is the encoder trained?", "With in-batch negatives [1].")]
    ans = pipe.ask("and what batch size?", history=history)
    assert ans.search_query == "What batch size was used to train the encoder?"
    # the answer prompt sees the standalone question, not the vague follow-up
    assert "What batch size was used to train the encoder?" in llm.calls[1][-1]["content"]
    assert ans.sources[0].chunk.section == "2 Method"


def test_rewrite_prompt_contains_history_and_truncates():
    msgs = rewrite_messages("why?", [("Q1", "A" * 1000)])
    assert "User: Q1" in msgs[-1]["content"] and "Follow-up: why?" in msgs[-1]["content"]
    assert len(msgs[-1]["content"]) < 700


def test_clean_rewrite_strips_labels_and_falls_back():
    assert clean_rewrite('Standalone question: "How is BPE trained?"\nextra', "x") == (
        "How is BPE trained?"
    )
    assert clean_rewrite("", "original?") == "original?"
