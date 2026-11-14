"""Command-line interface.

ragdemia inspect            # audit PDFs before indexing (text, sections, problems)
ragdemia ingest             # parse -> chunk -> embed -> build the index
ragdemia ask "question"     # one-shot answer with sources   (--compare for no-RAG side-by-side)
ragdemia chat               # multi-turn interactive session
ragdemia eval               # retrieval ablation + RAG vs closed-book report
ragdemia serve              # web app + REST API (FastAPI) on :8000
ragdemia ui                 # same, and open the browser
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import load_config

app = typer.Typer(add_completion=False, help="RAGDemia - cited Q&A over your academic documents.")
console = Console()

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config.yaml")


def _quiet_libraries() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    for name in ("transformers", "sentence_transformers", "httpx", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # pragma: no cover
        pass


# ----------------------------------------------------------------------------- inspect


@app.command()
def inspect(
    raw_dir: Path = typer.Argument(None, help="Folder of PDFs (default: paths.raw_dir)"),
    config: Path = ConfigOpt,
    sections: bool = typer.Option(False, "--sections", help="Print the detected section tree"),
) -> None:
    """Audit PDFs before indexing: extractable text, detected sections, likely problems."""
    from .parsing import discover_pdfs, parse_pdf

    cfg = load_config(config)
    root = Path(raw_dir or cfg.paths.raw_dir)
    pdfs = discover_pdfs(root)
    if not pdfs:
        console.print(f"[red]No PDFs found under {root}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Corpus inspection - {root}", show_lines=False)
    for col in ("Document", "Pages", "Chars", "Sections", "Skipped", "Warnings"):
        table.add_column(col, overflow="fold")
    for pdf in pdfs:
        try:
            d = parse_pdf(
                pdf,
                root=root,
                min_font_ratio=cfg.parsing.min_font_ratio,
                skip_sections=cfg.parsing.skip_sections,
            )
        except Exception as exc:
            table.add_row(pdf.name, "-", "-", "-", "-", f"[red]parse error: {exc}[/red]")
            continue
        warns = []
        if d.empty_pages:
            share = len(d.empty_pages) / d.num_pages
            warns.append(
                f"{len(d.empty_pages)} page(s) without text{' - scanned? needs OCR' if share > 0.3 else ''}"
            )
        if d.n_chars / max(1, d.num_pages) < 300:
            warns.append("very little text per page (slides or images?)")
        if len(d.sections) <= 2:
            warns.append("no section structure detected")
        table.add_row(
            f"{d.title[:60]}\n[dim]{d.doc_id}[/dim]",
            str(d.num_pages),
            f"{d.n_chars:,}",
            str(len(d.sections)),
            ", ".join(d.skipped_sections) or "-",
            "; ".join(warns) or "[green]ok[/green]",
        )
        if sections:
            for s in d.sections:
                console.print(f"  [dim]p{s.page_start:>3}[/dim] {'  ' * (s.level - 1)}{s.title}")
    console.print(table)


# ----------------------------------------------------------------------------- ingest


@app.command()
def ingest(
    raw_dir: Path = typer.Argument(None, help="Folder of PDFs (default: paths.raw_dir)"),
    config: Path = ConfigOpt,
    full: bool = typer.Option(False, "--full", help="Re-process every document"),
) -> None:
    """Sync the index with the PDF folder (only new/changed files are processed)."""
    _quiet_libraries()
    from .chunking import hf_token_counter
    from .embeddings import SentenceTransformerEmbedder
    from .ingest import ingest as run_ingest

    cfg = load_config(config)
    with console.status("Loading embedding model…"):
        emb = SentenceTransformerEmbedder(
            cfg.embedding.model,
            cfg.embedding.query_instruction,
            cfg.embedding.batch_size,
            cfg.embedding.device,
        )
        counter = hf_token_counter(cfg.embedding.model)
    console.print(f"Embedding model [bold]{emb.model_name}[/bold] on [bold]{emb.device}[/bold]")
    report = run_ingest(
        cfg, emb, counter, raw_dir=raw_dir, full=full, progress=lambda m: console.print(f"  · {m}")
    )
    if report.full_rebuild:
        console.print(f"[dim]full rebuild ({report.rebuild_reason})[/dim]")

    status = {d: "added" for d in report.added} | {d: "updated" for d in report.updated}
    table = Table(title="Indexed documents")
    for col in ("Document", "Pages", "Sections", "Chunks", "Status"):
        table.add_column(col)
    for d in report.documents:
        st = status.get(d["doc_id"], "unchanged")
        colour = {"added": "green", "updated": "yellow"}.get(st, "dim")
        table.add_row(
            f"{d['title'][:60]}\n[dim]{d['doc_id']}[/dim]",
            str(d["num_pages"]),
            str(d["n_sections"]),
            str(d["n_chunks"]),
            f"[{colour}]{st}[/{colour}]",
        )
    console.print(table)
    for doc_id in report.removed:
        console.print(f"[red]removed {doc_id}[/red] (file no longer in the folder)")
    for path, err in report.failures:
        console.print(f"[yellow]skipped {path}: {err}[/yellow]")
    console.print(
        f"[green]✓ {report.n_chunks} chunks from {len(report.documents)} documents "
        f"({report.n_embedded} newly embedded) in {report.seconds:.1f}s → {report.index_dir}[/green]"
    )


# ----------------------------------------------------------------------------- ask / chat


def _print_answer(ans, show_sources: bool = True, title: str = "RAGDemia") -> None:
    c = ans.citations
    subtitle = (
        f"cited {c.cited or '-'} · auto {c.auto_attributed or '-'} · grounded {c.groundedness:.0%}"
        if ans.mode == "rag" and not c.abstained
        else ("abstained" if c.abstained else "")
    )
    console.print(Panel(Markdown(ans.text), title=title, subtitle=subtitle, border_style="cyan"))
    if show_sources and ans.sources:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("#", justify="right")
        table.add_column("Source")
        table.add_column("Score", justify="right")
        table.add_column("Used")
        for i, r in enumerate(ans.sources, start=1):
            used = "model" if i in c.cited else ("auto" if i in c.auto_attributed else "")
            table.add_row(str(i), r.chunk.citation, f"{r.score:.3f}", used)
        console.print(table)
    if ans.timings:
        console.print(
            "[dim]" + " · ".join(f"{k} {v:.2f}s" for k, v in ans.timings.items()) + "[/dim]"
        )


def _load_pipeline(config: Path | None):
    _quiet_libraries()
    from .pipeline import RAGPipeline

    with console.status("Loading models and index…"):
        return RAGPipeline.from_config(load_config(config))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    config: Path = ConfigOpt,
    top_k: int = typer.Option(None, "--top-k", "-k"),
    mode: str = typer.Option(None, "--mode", help="dense | bm25 | hybrid"),
    compare: bool = typer.Option(False, "--compare", help="Also answer without retrieval"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Answer a single question with cited sources."""
    pipe = _load_pipeline(config)
    ans = pipe.ask(question, top_k, mode)
    if as_json:
        out = {"rag": ans.to_dict()}
        if compare:
            out["closed_book"] = pipe.ask_closed_book(question).to_dict()
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return
    _print_answer(ans)
    if compare:
        _print_answer(pipe.ask_closed_book(question), False, "Same LLM, no retrieval")


@app.command()
def chat(config: Path = ConfigOpt) -> None:
    """Multi-turn Q&A. Follow-ups ("how is it trained?") are rewritten using the conversation.

    Commands: exit · :sources (toggle source table) · :new (forget the conversation)
    """
    pipe = _load_pipeline(config)
    show = True
    history: list[tuple[str, str]] = []
    console.print(
        "[bold]RAGDemia[/bold] - ask about your documents. [dim]exit · :sources · :new[/dim]"
    )
    while True:
        try:
            q = console.input("[bold green]›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in {"exit", "quit", ":q"}:
            break
        if q == ":sources":
            show = not show
            continue
        if q == ":new":
            history.clear()
            console.print("[dim]conversation cleared[/dim]")
            continue
        if not q:
            continue
        ans = pipe.ask(q, history=history)
        if ans.search_query:
            console.print(f"[dim]↳ searched for: {ans.search_query}[/dim]")
        _print_answer(ans, show)
        history.append((q, ans.text))


# ----------------------------------------------------------------------------- eval


@app.command("eval")
def evaluate(
    questions: Path = typer.Option(Path("eval/questions.yaml"), "--questions", "-q"),
    out_dir: Path = typer.Option(Path("eval/results"), "--out"),
    config: Path = ConfigOpt,
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Skip LLM answering"),
) -> None:
    """Run the retrieval ablation and the RAG-vs-closed-book comparison."""
    from .evaluation import run_evaluation

    pipe = _load_pipeline(config)
    res = run_evaluation(
        pipe,
        questions,
        out_dir,
        progress=lambda m: console.print(f"  · {m}"),
        skip_answers=retrieval_only,
    )
    k = res["meta"]["top_k"]
    t = Table(title="Retrieval")
    for col in ("Retriever", f"Hit@{k}", "MRR", "ms/query"):
        t.add_column(col)
    for name, m in res["retrieval"].items():
        t.add_row(name, f"{m[f'hit@{k}']:.0%}", f"{m['mrr']:.3f}", f"{m['ms_per_query']:.0f}")
    console.print(t)
    if res["answers"]:
        a = res["answers"]
        t = Table(title="Answers")
        t.add_column("Metric")
        t.add_column("Closed-book")
        t.add_column("RAG")
        for label, key in [
            ("Key-fact recall", "fact_recall"),
            ("Fully correct", "fully_correct"),
            ("Correct abstention", "correct_abstention"),
        ]:
            t.add_row(label, f"{a['closed_book'][key]:.0%}", f"{a['rag'][key]:.0%}")
        console.print(t)
        console.print(f"[green]Report → {out_dir / 'report.md'}[/green]")


# ----------------------------------------------------------------------------- servers


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Use 0.0.0.0 to expose on your network"),
    port: int = 8000,
    open_browser: bool = typer.Option(False, "--open", help="Open the web app in a browser"),
) -> None:
    """Start the web app and REST API (web app at /, API docs at /docs)."""
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    console.print(
        f"[bold]RAGDemia[/bold] → web app [cyan]{url}[/cyan] · API docs [cyan]{url}/docs[/cyan]"
    )
    if open_browser:
        threading.Timer(4.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("ragdemia.api:app", host=host, port=port, log_level="warning")


@app.command()
def ui(port: int = 8000) -> None:
    """Launch the web app and open it in your browser (same as `serve --open`)."""
    serve(host="127.0.0.1", port=port, open_browser=True)


if __name__ == "__main__":
    app()
