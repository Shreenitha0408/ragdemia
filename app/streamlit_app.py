"""RAGDemia web UI (Streamlit): multi-turn chat over your documents + a document manager.

Run with:  ragdemia ui      (or: streamlit run app/streamlit_app.py)

* Chat: follow-up questions are rewritten into standalone search queries using the
  conversation; every answer shows its cited sources (document, section, page).
* Documents: upload PDFs in the sidebar - only the new file is parsed and embedded, and it is
  searchable immediately.  Uploaded documents can be removed again; files you placed on disk
  are listed but never deleted from the UI.
"""

from __future__ import annotations

import html
import re

import streamlit as st

from ragdemia.chunking import hf_token_counter
from ragdemia.config import load_config
from ragdemia.library import Library, LibraryError
from ragdemia.llm import LLMError
from ragdemia.pipeline import RAGPipeline

st.set_page_config(page_title="RAGDemia", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1100px; padding-top: 2rem;}
      .src-meta {opacity: .65; font-size: .85rem;}
      .badge {display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: .75rem;
              margin-right: 6px; border: 1px solid rgba(128,128,128,.4);}
      .badge.model {background: rgba(46,160,67,.15); border-color: rgba(46,160,67,.6);}
      .badge.auto {background: rgba(210,153,34,.15); border-color: rgba(210,153,34,.6);}
      .cite {font-weight: 600; color: #2e86de;}
      .rewrite {opacity: .7; font-size: .85rem; font-style: italic;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model, reranker and index…")
def get_library() -> Library:
    cfg = load_config()
    pipe = RAGPipeline.from_config(cfg)
    return Library(cfg, pipe, hf_token_counter(cfg.embedding.model))


try:
    library = get_library()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nAdd PDFs to `data/raw/` and run `ragdemia ingest`.")
    st.stop()
pipe = library.pipeline

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "answer"?, "closed_book"?}]


def highlight_citations(text: str) -> str:
    safe = html.escape(text)
    safe = re.sub(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", r"<span class='cite'>[\1]</span>", safe)
    return safe.replace("\n", "<br>")


def render_answer(msg: dict) -> None:
    ans = msg["answer"]
    if ans.search_query:
        st.markdown(
            f"<div class='rewrite'>↳ searched for: {html.escape(ans.search_query)}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(highlight_citations(ans.text), unsafe_allow_html=True)
    c = ans.citations
    if c.abstained:
        st.caption("The documents don't contain an answer, so RAGDemia declined rather than guess.")
    else:
        st.caption(
            f"{len(c.all_sources)} source(s) cited · {c.groundedness:.0%} of sentences supported "
            f"by retrieved text · {len(c.invalid)} invalid citation(s)"
        )
    if msg.get("closed_book"):
        with st.expander("Same LLM without retrieval"):
            st.markdown(msg["closed_book"])
            st.caption("No sources - this answer comes only from the model's parameters.")
    with st.expander(f"Sources ({len(ans.sources)})", expanded=False):
        for i, r in enumerate(ans.sources, start=1):
            badge = ""
            if i in c.cited:
                badge = "<span class='badge model'>cited by model</span>"
            elif i in c.auto_attributed:
                badge = "<span class='badge auto'>auto-attributed</span>"
            pages = (
                f"p. {r.chunk.page_start}"
                if r.chunk.page_start == r.chunk.page_end
                else f"pp. {r.chunk.page_start}–{r.chunk.page_end}"
            )
            st.markdown(
                f"**[{i}] {html.escape(r.chunk.doc_title)}** — {html.escape(r.chunk.section)}<br>"
                f"{badge}<span class='src-meta'>{pages} · score {r.score:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.write(r.chunk.text)
            st.divider()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 RAGDemia")
    st.caption("Cited answers from your own academic documents.")
    if st.button("🗨️ New conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Settings"):
        top_k = st.slider("Sources per answer (top-k)", 1, 10, pipe.top_k)
        mode = st.radio(
            "Retriever",
            ["hybrid", "dense", "bm25"],
            horizontal=True,
            help="hybrid = dense + BM25 fused with Reciprocal Rank Fusion, then reranked",
        )
        compare = st.toggle("Also answer without retrieval", value=False)
        st.markdown(f"**LLM:** `{pipe.llm.name}`")
        rr = pipe.retriever.reranker
        st.markdown(f"**Reranker:** `{rr.model_name if rr else 'off'}`")

    st.subheader("Documents")
    uploads = st.file_uploader(
        "Add PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploads and st.button(f"Index {len(uploads)} file(s)", type="primary", width="stretch"):
        with st.status("Adding documents…", expanded=True) as status:
            try:
                for f in uploads:
                    library.add(f.name, f.getvalue())
                    st.write(f"saved {f.name}")
                report = library.sync(progress=st.write)
                for path, err in report.failures:
                    st.warning(f"{path}: {err}")
                status.update(
                    label=f"Indexed {len(report.added)} new document(s) "
                    f"({report.n_embedded} chunks) in {report.seconds:.1f}s",
                    state="complete",
                )
            except LibraryError as exc:
                status.update(label=str(exc), state="error")

    docs = library.documents
    st.caption(f"{len(docs)} documents · {sum(d['n_chunks'] for d in docs)} chunks")
    for d in docs:
        cols = st.columns([5, 1])
        cols[0].markdown(
            f"**{html.escape(d['title'][:60])}**<br><span class='src-meta'>{d['num_pages']} pages · "
            f"{d['n_chunks']} chunks · {html.escape(d['doc_id'])}</span>",
            unsafe_allow_html=True,
        )
        if d["deletable"]:
            if cols[1].button("🗑", key=f"del-{d['doc_id']}", help="Remove from the library"):
                library.remove(d["doc_id"])
                library.sync()
                st.rerun()
        else:
            cols[1].markdown(
                "<span title='Added on disk - manage the file directly'>🔒</span>",
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------- chat
st.title("Ask your documents")

if not st.session_state.messages:
    st.caption("Try one of these, then ask a follow-up like “how is it trained?”")
    examples = [
        "What is byte-pair encoding?",
        "What are the three RAG paradigms described in the survey?",
        "Who built ELIZA, and when?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples, strict=True):
        if col.button(ex, width="stretch"):
            st.session_state.pending = ex
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg)

question = st.chat_input("Ask about your lecture notes and papers…")
question = question or st.session_state.pop("pending", None)

if question:
    history = [
        (m["answer"].question, m["answer"].text)
        for m in st.session_state.messages
        if m["role"] == "assistant"
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents…"):
                query, sources, stream = pipe.ask_stream(question, top_k, mode, history=history)
            placeholder = st.empty()
            text = ""
            for token in stream:
                text += token
                placeholder.markdown(text + "▌")
            placeholder.empty()
            answer = pipe.finalize(question, text, sources, query)
            msg = {"role": "assistant", "answer": answer}
            if compare:
                with st.spinner("Generating answer without retrieval…"):
                    msg["closed_book"] = pipe.ask_closed_book(query).text
            render_answer(msg)
            st.session_state.messages.append(msg)
        except LLMError as exc:
            st.error(str(exc))
