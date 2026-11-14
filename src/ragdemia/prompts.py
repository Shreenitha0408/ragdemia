"""Prompt templates.  Kept in one file so prompt changes show up as clean diffs.

Prompt-design notes:

* Sources are numbered ``[1]..[k]`` and each carries its citation string, so the model
  only has to emit a short integer - small models copy integers far more reliably than
  titles - and we map integers back to (document, section, page) in code.
* A fixed abstention sentence makes "I don't know" machine-detectable for evaluation.
* The closed-book prompt is deliberately symmetric (same style rules, same abstention
  option), so the RAG-vs-no-RAG comparison isolates the effect of retrieval.
"""

from __future__ import annotations

from .llm import Message
from .retrieval import RetrievedChunk

ABSTAIN_RAG = "I couldn't find this in your documents."
ABSTAIN_CLOSED = "I don't know."

RAG_SYSTEM = f"""You are RAGDemia, a research assistant that answers questions about the \
user's academic documents (papers, lecture notes, slides).

Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. Cite every factual sentence with the number of the source(s) that support it, \
like [1] or [2][3]. Put the citation at the end of the sentence.
3. If the sources do not contain the answer, reply exactly: "{ABSTAIN_RAG}"
4. Be precise and concise: at most 6 sentences. Quote numbers and names exactly as written."""

CLOSED_BOOK_SYSTEM = f"""You are a research assistant answering questions about academic \
papers on NLP and machine learning.

Rules:
1. Answer from your own knowledge.
2. If you are not confident, reply exactly: "{ABSTAIN_CLOSED}"
3. Be precise and concise: at most 6 sentences. Quote numbers and names exactly."""


def format_sources(results: list[RetrievedChunk]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] Source: {r.chunk.citation}\n{r.chunk.text}")
    return "\n\n".join(blocks)


# One worked example, shown to the model as a previous conversation turn ("few-shot").
# Small models copy a demonstrated format far more reliably than a described one.  The
# example is deliberately from an unrelated domain so its content cannot leak into answers.
_FEWSHOT_SOURCES = """[1] Source: Plant Biology Notes — 2 Photosynthesis, p. 4
Photosynthesis converts light energy into chemical energy stored in glucose. It takes place \
in the chloroplasts, mainly in the mesophyll cells of leaves.

[2] Source: Plant Biology Notes — 3 Respiration, p. 7
Cellular respiration releases the energy stored in glucose. It occurs in the mitochondria."""
_FEWSHOT_QUESTION = "Where does photosynthesis take place and what does it produce?"
_FEWSHOT_ANSWER = (
    "Photosynthesis takes place in the chloroplasts, mainly in the mesophyll cells of leaves [1]. "
    "It converts light energy into chemical energy stored in glucose [1]."
)


def _rag_user_turn(sources: str, question: str) -> str:
    return (
        f"Sources:\n\n{sources}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above and end each sentence with its source number "
        f'in square brackets. If the sources do not answer the question, reply exactly: "{ABSTAIN_RAG}"'
    )


def rag_messages(question: str, results: list[RetrievedChunk]) -> list[Message]:
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": _rag_user_turn(_FEWSHOT_SOURCES, _FEWSHOT_QUESTION)},
        {"role": "assistant", "content": _FEWSHOT_ANSWER},
        {"role": "user", "content": _rag_user_turn(format_sources(results), question)},
    ]


def closed_book_messages(question: str) -> list[Message]:
    return [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
    ]


# --------------------------------------------------------------------------- multi-turn
# Follow-up questions ("how is it trained?") are useless as search queries: "it" matches
# nothing.  Before retrieval, the LLM rewrites the follow-up into a standalone question using
# the conversation so far (the "condense question" step).  Only the *question* is rewritten;
# the answer is still generated strictly from freshly retrieved sources.

REWRITE_SYSTEM = """You rewrite a follow-up question into a standalone question for a search \
engine. Use the conversation to replace pronouns and vague references ("it", "that", "the \
second one", "this method") with what they refer to. Keep the user's wording otherwise. If the \
follow-up is already standalone, return it unchanged. Output only the question, nothing else."""

_REWRITE_EXAMPLE_USER = """Conversation:
User: What is byte-pair encoding?
Assistant: Byte-pair encoding (BPE) is a subword tokenization algorithm that repeatedly merges \
the most frequent pair of adjacent symbols [1].

Follow-up: how is it trained?
Standalone question:"""
_REWRITE_EXAMPLE_ANSWER = "How is byte-pair encoding (BPE) trained?"

MAX_HISTORY_TURNS = 3
MAX_HISTORY_ANSWER_CHARS = 400


def rewrite_messages(question: str, history: list[tuple[str, str]]) -> list[Message]:
    lines = []
    for q, a in history[-MAX_HISTORY_TURNS:]:
        a = a.strip()
        if len(a) > MAX_HISTORY_ANSWER_CHARS:
            a = a[:MAX_HISTORY_ANSWER_CHARS].rsplit(" ", 1)[0] + " …"
        lines += [f"User: {q.strip()}", f"Assistant: {a}"]
    user = (
        "Conversation:\n"
        + "\n".join(lines)
        + f"\n\nFollow-up: {question.strip()}\nStandalone question:"
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": _REWRITE_EXAMPLE_USER},
        {"role": "assistant", "content": _REWRITE_EXAMPLE_ANSWER},
        {"role": "user", "content": user},
    ]


def clean_rewrite(raw: str, original: str) -> str:
    """Keep the first line, strip quotes/labels; fall back to the original if it looks wrong."""
    line = next((ln.strip() for ln in raw.strip().splitlines() if ln.strip()), "")
    for prefix in ("Standalone question:", "Question:", "Rewritten question:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix) :].strip()
    line = line.strip("\"'“”` ")
    if not line or len(line) > 3 * len(original) + 200:
        return original
    return line
