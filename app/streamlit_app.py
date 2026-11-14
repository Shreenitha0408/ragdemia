"""RAGDemia web UI (Streamlit).

Run with:  ragdemia ui      (or: streamlit run app/streamlit_app.py)

Layout: a question box; the answer streams in; citations [n] are listed beneath with the
exact section/page and the retrieved passage, so every claim can be checked in one click.
A toggle shows the same LLM's answer *without* retrieval for comparison.
"""

from __future__ import annotations

import html
import re

import streamlit as st

from ragdemia.config import load_config
from ragdemia.llm import LLMError
from ragdemia.pipeline import RAGPipeline

st.set_page_config(page_title="RAGDemia", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1100px; padding-top: 2rem;}
      .src-meta {color: var(--text-color); opacity: .65; font-size: .85rem;}
      .badge {display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: .75rem;
              margin-right: 6px; border: 1px solid rgba(128,128,128,.4);}
      .badge.model {background: rgba(46,160,67,.15); border-color: rgba(46,160,67,.6);}
      .badge.auto {background: rgba(210,153,34,.15); border-color: rgba(210,153,34,.6);}
      .cite {font-weight: 600; color: #2e86de;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model, reranker and index…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_config(load_config())


try:
    pipe = get_pipeline()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nAdd PDFs to `data/raw/` and run `ragdemia ingest`.")
    st.stop()

manifest = pipe.retriever.store.manifest

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 RAGDemia")
    st.caption("Cited answers from your own academic documents.")
    top_k = st.slider("Sources to retrieve (top-k)", 1, 10, pipe.top_k)
    mode = st.radio(
        "Retriever",
        ["hybrid", "dense", "bm25"],
        horizontal=True,
        help="hybrid = dense + BM25 fused with Reciprocal Rank Fusion",
    )
    compare = st.toggle("Compare with no-retrieval answer", value=False)
    st.divider()
    st.markdown(f"**LLM:** `{pipe.llm.name}`")
    st.markdown(f"**Embeddings:** `{manifest.get('embedding_model')}`")
    rr = pipe.retriever.reranker
    st.markdown(f"**Reranker:** `{rr.model_name if rr else 'off'}`")
    with st.expander(f"Indexed documents ({len(manifest.get('documents', []))})"):
        for d in manifest.get("documents", []):
            st.markdown(
                f"- {d['title']}  \n  <span class='src-meta'>{d['num_pages']} pages · "
                f"{d['n_chunks']} chunks</span>",
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------- main
st.title("Ask your documents")
examples = [
    "What are the three RAG paradigms described in the survey?",
    "How does HyDE generate a hypothetical document for zero-shot retrieval?",
    "What batch size did DPR use for in-batch negative training?",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples, strict=True):
    if col.button(ex, use_container_width=True):
        st.session_state["q"] = ex

question = st.text_input(
    "Question", key="q", placeholder="e.g. What does Lost in the Middle find about long contexts?"
)


def highlight_citations(text: str) -> str:
    safe = html.escape(text)
    return re.sub(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", r"<span class='cite'>[\1]</span>", safe).replace(
        "\n", "<br>"
    )


if question:
    answer_col, *rest = st.columns([3, 2]) if compare else [st.container()]
    with answer_col:
        st.subheader("Answer")
        try:
            sources, stream = pipe.ask_stream(question, top_k, mode)
            placeholder = st.empty()
            text = ""
            for token in stream:
                text += token
                placeholder.markdown(text + "▌")
            ans = pipe.finalize(question, text, sources)
            placeholder.markdown(highlight_citations(ans.text), unsafe_allow_html=True)
        except LLMError as exc:
            st.error(str(exc))
            st.stop()

        c = ans.citations
        if c.abstained:
            st.info(
                "The documents don't contain an answer, so RAGDemia declined rather than guess."
            )
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Sources cited", len(c.all_sources))
            m2.metric(
                "Grounded sentences",
                f"{c.groundedness:.0%}",
                help="Share of answer sentences whose content words are found in a retrieved source",
            )
            m3.metric(
                "Invalid citations",
                len(c.invalid),
                help="Citations pointing to a source number that was never provided",
            )

    if compare:
        with rest[0]:
            st.subheader("Same LLM, no retrieval")
            with st.spinner("Generating closed-book answer…"):
                cb = pipe.ask_closed_book(question)
            st.markdown(cb.text)
            st.caption("No sources - this answer comes only from the model's parameters.")

    st.subheader("Sources")
    for i, r in enumerate(sources, start=1):
        badge = ""
        if i in c.cited:
            badge = "<span class='badge model'>cited by model</span>"
        elif i in c.auto_attributed:
            badge = "<span class='badge auto'>auto-attributed</span>"
        label = f"[{i}] {r.chunk.doc_title} — {r.chunk.section}"
        with st.expander(label, expanded=i in c.all_sources):
            pages = (
                f"p. {r.chunk.page_start}"
                if r.chunk.page_start == r.chunk.page_end
                else f"pp. {r.chunk.page_start}–{r.chunk.page_end}"
            )
            st.markdown(
                f"{badge}<span class='src-meta'>{pages} · score {r.score:.3f} · "
                f"{r.chunk.source_path}</span>",
                unsafe_allow_html=True,
            )
            st.write(r.chunk.text)
            with st.popover("Retrieval diagnostics"):
                st.json(r.scores)

    if c.sentences:
        with st.expander("Sentence-level attribution"):
            st.dataframe(
                [
                    {
                        "sentence": s.sentence,
                        "cited": s.cited or "-",
                        "best source": s.best_source,
                        "support": f"{s.support:.0%}",
                    }
                    for s in c.sentences
                ],
                use_container_width=True,
                hide_index=True,
            )
