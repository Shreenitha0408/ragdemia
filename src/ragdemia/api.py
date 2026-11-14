"""FastAPI backend: REST API + the RAGDemia web app, served from one process.

    http://localhost:8000/        web app (static HTML/CSS/JS in ``ragdemia/web``)
    http://localhost:8000/docs    interactive API documentation

Two operating modes (``server.public`` in config):

* **Local** (default, your laptop): single user; uploads go to the shared library and
  documents uploaded through the app can be deleted.
* **Public** (the hosted demo): visitors get a private, temporary *workspace* for their own
  uploads (header ``X-Workspace: <uuid>``), searched together with the shared library.  Only
  the admin (header ``X-Admin-Token``) can change the shared library, and changes are
  persisted to a private Hugging Face dataset.  Questions and uploads are rate-limited per
  visitor.

Streaming: ``POST /ask/stream`` returns **Server-Sent Events** so the browser can render the
answer token by token.  Event order::

    meta         {"search_query", "sources": [...]}      retrieval is done, sources are known
    token        {"t": "..."}                             one per generated text fragment
    done         {...full answer with citation checks}    after the citation verifier ran
    closed_book  {"text": "..."}                          only when compare=true
    error        {"message": "..."}                       LLM unreachable etc.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import ServerConfig
from .library import Library, LibraryError
from .llm import LLMError
from .pipeline import Answer, RAGPipeline
from .workspaces import WORKSPACE_PREFIX, WorkspaceManager

WEB_DIR = Path(__file__).parent / "web"
log = logging.getLogger("ragdemia.api")


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


class RateLimiter:
    """Sliding-window limit per client (keyed by IP): at most ``per_minute`` hits in 60 s."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                wait = int(60 - (now - q[0])) + 1
                raise HTTPException(
                    429,
                    f"Too many requests - please wait {wait}s. (This public demo is rate-limited.)",
                )
            q.append(now)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _client_key(request: Request) -> str:
    # Behind the Hugging Face / Docker proxy the visitor's IP is the first X-Forwarded-For hop.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


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


def create_app(
    pipeline: RAGPipeline | None = None,
    library: Library | None = None,
    *,
    server: ServerConfig | None = None,
    workspaces: WorkspaceManager | None = None,
) -> FastAPI:
    """Build the app. With no arguments, everything is loaded from config at startup; tests
    pass lightweight fakes instead."""
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            from .chunking import hf_token_counter
            from .config import load_config

            cfg = load_config()
            p = RAGPipeline.from_config(cfg)
            counter = hf_token_counter(cfg.embedding.model)
            storage = None
            if cfg.server.public:
                from .persistence import storage_from_config

                storage = storage_from_config(cfg)
            if storage is not None:
                try:
                    storage.ensure_repo()
                    storage.pull()
                except Exception:  # the demo must still start if storage is unreachable
                    log.exception("could not pull persisted documents")
            lib = Library(cfg, p, counter, storage=storage)
            if storage is not None or cfg.server.public:
                # Index whatever is in the library folder but not yet in the index: documents
                # pulled from storage, or admin uploads on a mounted volume after a restart.
                # Incremental, so a no-op when nothing changed.
                lib.sync()
            state.update(pipeline=p, library=lib, server=cfg.server)
            if cfg.server.public:
                state["workspaces"] = WorkspaceManager.from_config(
                    cfg, p.retriever.embedder, counter
                )
        else:
            state.update(
                pipeline=pipeline,
                library=library,
                server=server or ServerConfig(),
                workspaces=workspaces,
            )
        srv: ServerConfig = state["server"]
        state["limiter"] = RateLimiter(srv.rate_limit_per_minute if srv.public else 0)
        state["admin_token"] = os.environ.get(srv.admin_token_env, "")
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

    # ------------------------------------------------------------------ helpers
    def pipe() -> RAGPipeline:
        if "pipeline" not in state:
            raise HTTPException(503, "Pipeline not loaded")
        return state["pipeline"]

    def lib() -> Library:
        pipe()
        if state.get("library") is None:
            raise HTTPException(501, "Document management is not enabled")
        return state["library"]

    def public() -> bool:
        return bool(state.get("server") and state["server"].public)

    def is_admin(token: str | None) -> bool:
        expected = state.get("admin_token", "")
        return bool(expected and token) and hmac.compare_digest(token.encode(), expected.encode())

    def can_edit_shared(token: str | None) -> bool:
        return not public() or is_admin(token)

    def extra_stores(workspace: str | None) -> list | None:
        ws: WorkspaceManager | None = state.get("workspaces")
        return ws.stores(workspace) if ws is not None else None

    def limit(request: Request) -> None:
        state["limiter"].check(_client_key(request))

    # ------------------------------------------------------------------ status
    @app.get("/health")
    def health(
        x_admin_token: str | None = Header(None), x_workspace: str | None = Header(None)
    ) -> dict:
        p = pipe()
        m = p.retriever.store.manifest
        ws: WorkspaceManager | None = state.get("workspaces")
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
            "public": public(),
            "admin_enabled": bool(state.get("admin_token")),
            "is_admin": is_admin(x_admin_token),
            "can_edit_shared": can_edit_shared(x_admin_token),
            "workspace": (
                {**ws.limits, "n_documents": len(ws.documents(x_workspace))}
                if ws is not None
                else None
            ),
        }

    # ------------------------------------------------------------------ documents
    @app.get("/documents")
    def documents(
        x_admin_token: str | None = Header(None), x_workspace: str | None = Header(None)
    ) -> list[dict]:
        editable = can_edit_shared(x_admin_token)
        shared = [
            {**d, "scope": "shared", "deletable": d["deletable"] and editable}
            for d in lib().documents
        ]
        ws: WorkspaceManager | None = state.get("workspaces")
        return (ws.documents(x_workspace) if ws is not None else []) + shared

    @app.post("/documents", status_code=201)
    async def upload_documents(
        request: Request,
        files: list[UploadFile] = File(...),
        x_admin_token: str | None = Header(None),
        x_workspace: str | None = Header(None),
    ) -> dict:
        """Upload PDFs. Locally (or as admin) they join the shared library; on the public demo
        they go to the visitor's private, temporary workspace."""
        ws: WorkspaceManager | None = state.get("workspaces")
        if not can_edit_shared(x_admin_token):
            if ws is None:
                raise HTTPException(403, "Uploads are disabled on this server")
            limit(request)
            t0 = time.perf_counter()
            added, failures, n = [], [], 0
            for f in files:
                try:
                    entry = ws.add(x_workspace, f.filename or "document.pdf", await f.read())
                    added.append(entry["doc_id"])
                    n += entry["n_chunks"]
                except LibraryError as exc:
                    failures.append({"path": f.filename, "error": str(exc)})
            if not added and failures:
                raise HTTPException(400, failures[0]["error"])
            return {
                "scope": "workspace",
                "added": added,
                "updated": [],
                "removed": [],
                "n_chunks": n,
                "n_embedded": n,
                "seconds": round(time.perf_counter() - t0, 2),
                "failures": failures,
            }

        library_ = lib()
        saved = []
        try:
            for f in files:
                saved.append(str(library_.add(f.filename or "document.pdf", await f.read())))
        except LibraryError as exc:
            for path in saved:  # all-or-nothing
                Path(path).unlink(missing_ok=True)
            raise HTTPException(400, str(exc)) from exc
        return {"scope": "shared", "saved": saved, **_report(library_.sync())}

    @app.delete("/documents/{doc_id:path}")
    def delete_document(
        doc_id: str,
        x_admin_token: str | None = Header(None),
        x_workspace: str | None = Header(None),
    ) -> dict:
        """Remove a document: your own workspace upload, or (admin) a shared-library upload."""
        ws: WorkspaceManager | None = state.get("workspaces")
        if doc_id.startswith(WORKSPACE_PREFIX):
            if ws is None:
                raise HTTPException(404, f"No document with id '{doc_id}'")
            try:
                ws.remove(x_workspace, doc_id)
            except LibraryError as exc:
                raise HTTPException(404, str(exc)) from exc
            return {"scope": "workspace", "removed": [doc_id]}
        if not can_edit_shared(x_admin_token):
            raise HTTPException(403, "Only the admin can change the shared library")
        library_ = lib()
        try:
            library_.remove(doc_id)
        except LibraryError as exc:
            raise HTTPException(403 if "on disk" in str(exc) else 404, str(exc)) from exc
        return {"scope": "shared", **_report(library_.sync())}

    # ------------------------------------------------------------------ retrieval / QA
    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest, x_workspace: str | None = Header(None)) -> list[dict]:
        results = pipe().retrieve(req.query, req.top_k, req.mode, extra_stores(x_workspace))
        return [
            {"rank": r.rank, "score": r.score, "scores": r.scores, "citation": r.chunk.citation}
            | r.chunk.to_dict()
            for r in results
        ]

    @app.post("/ask")
    def ask(request: Request, req: AskRequest, x_workspace: str | None = Header(None)) -> dict:
        p = pipe()
        limit(request)
        try:
            history = [(t.question, t.answer) for t in req.history]
            ans = p.ask(
                req.question,
                req.top_k,
                req.mode,
                history=history,
                extra_stores=extra_stores(x_workspace),
            )
            out = {"rag": ans.to_dict()}
            if req.compare:
                out["closed_book"] = p.ask_closed_book(req.question).to_dict()
        except LLMError as exc:
            raise HTTPException(503, str(exc)) from exc
        return out

    @app.post("/ask/stream")
    def ask_stream(
        request: Request, req: AskRequest, x_workspace: str | None = Header(None)
    ) -> StreamingResponse:
        """Same as /ask, streamed as Server-Sent Events (see module docstring)."""
        p = pipe()
        limit(request)
        history = [(t.question, t.answer) for t in req.history]
        extra = extra_stores(x_workspace)

        def events() -> Iterator[str]:
            try:
                t0 = time.perf_counter()
                query, sources, stream = p.ask_stream(
                    req.question, req.top_k, req.mode, history, extra_stores=extra
                )
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
