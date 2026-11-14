"""FastAPI backend: REST API + the RAGDemia web app, served from one process.

    http://localhost:8000/        web app (static HTML/CSS/JS in ``ragdemia/web``)
    http://localhost:8000/docs    interactive API documentation

The pipeline (embedding model, reranker, FAISS index) is loaded once at startup and shared
across requests; loading per request would add seconds of latency.

Streaming: ``POST /ask/stream`` returns **Server-Sent Events** so the browser can render the
answer token by token.  SSE is plain HTTP (``text/event-stream``), one-directional, and works
through proxies without the handshake of WebSockets - all a chat answer needs.  Event order::

    meta         {"search_query", "sources": [...]}      retrieval is done, sources are known
    token        {"t": "..."}                             one per generated text fragment
    done         {...full answer with citation checks}    after the citation verifier ran
    closed_book  {"text": "..."}                          only when compare=true
    error        {"message": "..."}                       LLM unreachable etc.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .library import Library, LibraryError
from .llm import LLMError
from .pipeline import Answer, RAGPipeline

WEB_DIR = Path(__file__).parent / "web"


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    history: list[Turn] = Field(
        default_factory=list,
        description="Previous turns, oldest first; follow-ups are rewritten into standalone queries",
    )
    top_k: int | None = Field(None, ge=1, le=20)
    mode: Literal["dense", "bm25", "hybrid"] | None = None
    compare: bool = Field(False, description="Also return a closed-book (no retrieval) answer")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    mode: Literal["dense", "bm25", "hybrid"] | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _report(report) -> dict:
    return {
        "added": report.added,
        "updated": report.updated,
        "removed": report.removed,
        "n_chunks": report.n_chunks,
        "n_embedded": report.n_embedded,
        "seconds": round(report.seconds, 2),
        "failures": [{"path": p, "error": e} for p, e in report.failures],
    }


def create_app(pipeline: RAGPipeline | None = None, library: Library | None = None) -> FastAPI:
    """Build the app. With no arguments, models are loaded from config at startup; tests pass
    a pipeline built from lightweight fakes instead."""
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            from .chunking import hf_token_counter
            from .config import load_config

            cfg = load_config()
            p = RAGPipeline.from_config(cfg)
            state["pipeline"] = p
            state["library"] = Library(cfg, p, hf_token_counter(cfg.embedding.model))
        else:
            state["pipeline"], state["library"] = pipeline, library
        state["defaults"] = {
            "top_k": state["pipeline"].top_k,
            "mode": state["pipeline"].retriever.mode,
        }
        yield
        state.clear()

    app = FastAPI(
        title="RAGDemia API",
        version=__version__,
        description="Retrieval-augmented Q&A over your academic documents, with section-level citations.",
        lifespan=lifespan,
    )

    def pipe() -> RAGPipeline:
        if "pipeline" not in state:
            raise HTTPException(503, "Pipeline not loaded")
        return state["pipeline"]

    def lib() -> Library:
        pipe()
        if state.get("library") is None:
            raise HTTPException(501, "Document management is not enabled")
        return state["library"]

    # ------------------------------------------------------------------ status
    @app.get("/health")
    def health() -> dict:
        p = pipe()
        m = p.retriever.store.manifest
        return {
            "status": "ok",
            "version": __version__,
            "llm": p.llm.name,
            "embedding_model": m.get("embedding_model"),
            "reranker": p.retriever.reranker.model_name if p.retriever.reranker else None,
            "n_documents": len(m.get("documents", [])),
            "n_chunks": m.get("n_chunks", len(p.retriever.store.chunks)),
            "index_built_at": m.get("built_at"),
            "defaults": state["defaults"],
        }

    # ------------------------------------------------------------------ documents
    @app.get("/documents")
    def documents() -> list[dict]:
        return lib().documents

    @app.post("/documents", status_code=201)
    async def upload_documents(files: list[UploadFile] = File(...)) -> dict:
        """Upload one or more PDFs; they are indexed incrementally and searchable immediately."""
        library_ = lib()
        saved = []
        try:
            for f in files:
                saved.append(str(library_.add(f.filename or "document.pdf", await f.read())))
        except LibraryError as exc:
            for path in saved:  # all-or-nothing
                Path(path).unlink(missing_ok=True)
            raise HTTPException(400, str(exc)) from exc
        return {"saved": saved, **_report(library_.sync())}

    @app.delete("/documents/{doc_id:path}")
    def delete_document(doc_id: str) -> dict:
        """Remove a document that was uploaded through the API or web app."""
        library_ = lib()
        try:
            library_.remove(doc_id)
        except LibraryError as exc:
            raise HTTPException(403 if "on disk" in str(exc) else 404, str(exc)) from exc
        return _report(library_.sync())

    # ------------------------------------------------------------------ retrieval / QA
    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> list[dict]:
        results = pipe().retrieve(req.query, req.top_k, req.mode)
        return [
            {"rank": r.rank, "score": r.score, "scores": r.scores, "citation": r.chunk.citation}
            | r.chunk.to_dict()
            for r in results
        ]

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        p = pipe()
        try:
            history = [(t.question, t.answer) for t in req.history]
            out = {"rag": p.ask(req.question, req.top_k, req.mode, history=history).to_dict()}
            if req.compare:
                out["closed_book"] = p.ask_closed_book(req.question).to_dict()
        except LLMError as exc:
            raise HTTPException(503, str(exc)) from exc
        return out

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        """Same as /ask, streamed as Server-Sent Events (see module docstring)."""
        p = pipe()
        history = [(t.question, t.answer) for t in req.history]

        def events() -> Iterator[str]:
            try:
                t0 = time.perf_counter()
                query, sources, stream = p.ask_stream(req.question, req.top_k, req.mode, history)
                t1 = time.perf_counter()
                preview = Answer(req.question, "", sources, search_query=query).to_dict()
                yield _sse(
                    "meta",
                    {
                        "search_query": query if query != req.question else None,
                        "sources": preview["sources"],
                    },
                )
                text = ""
                for token in stream:
                    text += token
                    yield _sse("token", {"t": token})
                answer = p.finalize(req.question, text, sources, query)
                answer.timings = {
                    "retrieval_s": t1 - t0,
                    "generation_s": time.perf_counter() - t1,
                }
                yield _sse("done", answer.to_dict())
                if req.compare:
                    yield _sse("closed_book", {"text": p.ask_closed_book(query).text})
            except LLMError as exc:
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ web app
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
