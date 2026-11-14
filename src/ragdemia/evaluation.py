"""Evaluation: retrieval quality per retrieval mode, and RAG vs. closed-book answer quality.

Two questions are answered separately, because they fail for different reasons:

1. **Did we fetch the right evidence?** (retrieval metrics, no LLM involved)
   * *Hit@k*   - at least one top-k chunk comes from a gold document *and* gold page.
   * *MRR*     - mean of 1/rank of the first such chunk (rewards ranking it high).
   * *DocHit@k*- looser: right document, any page.
   Run for dense, BM25, hybrid and hybrid + reranker, i.e. an ablation of each component.

2. **Did the model answer correctly and honestly?**
   * *Key-fact recall* - share of required facts (with accepted phrasings) found in the
     answer.  Deterministic and transparent; misses paraphrases, so it is conservative.
   * *Correct abstention* - on unanswerable questions, did the system refuse instead of
     inventing an answer?
   * *Citation precision* - did the RAG answer cite (or get auto-attributed to) a chunk from
     a gold document?
   * *Groundedness* - share of answer sentences lexically supported by retrieved text.

We deliberately avoid an LLM-as-judge here: with a local 3-7B judge the scores are noisy and
hard to defend; string-level metrics on hand-written key facts are reproducible.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .pipeline import Answer, RAGPipeline
from .retrieval import RetrievedChunk


@dataclass
class GoldRef:
    doc: str
    pages: list[int] = field(default_factory=list)


@dataclass
class EvalQuestion:
    id: str
    question: str
    type: str
    answer: str
    key_facts: list[list[str]]
    gold: list[GoldRef]

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_questions(path: str | Path) -> list[EvalQuestion]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for q in data["questions"]:
        out.append(
            EvalQuestion(
                id=q["id"],
                question=q["question"],
                type=q.get("type", "factual"),
                answer=q.get("answer", ""),
                key_facts=[[str(a) for a in f] for f in q.get("key_facts", [])],
                gold=[GoldRef(g["doc"], list(g.get("pages", []))) for g in q.get("gold", [])],
            )
        )
    return out


# ----------------------------------------------------------------------------- metrics


def _norm(text: str) -> str:
    text = text.lower().replace("**", "")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = text.replace("’", "'")
    return re.sub(r"\s+", " ", text)


def fact_recall(answer: str, key_facts: list[list[str]]) -> tuple[float, list[bool]]:
    if not key_facts:
        return 1.0, []
    a = _norm(answer)
    hits = [any(_norm(alias) in a for alias in fact) for fact in key_facts]
    return sum(hits) / len(hits), hits


def is_relevant(
    chunk_doc: str, page_start: int, page_end: int, gold: list[GoldRef], page_level: bool
) -> bool:
    for g in gold:
        if chunk_doc != g.doc:
            continue
        if not page_level or not g.pages:
            return True
        if any(page_start <= p <= page_end for p in g.pages):
            return True
    return False


def retrieval_metrics(results: list[RetrievedChunk], gold: list[GoldRef]) -> dict[str, float]:
    first_rank = None
    doc_hit = False
    for r in results:
        c = r.chunk
        if is_relevant(c.doc_id, c.page_start, c.page_end, gold, page_level=False):
            doc_hit = True
        if first_rank is None and is_relevant(c.doc_id, c.page_start, c.page_end, gold, True):
            first_rank = r.rank
    return {
        "hit": float(first_rank is not None),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "doc_hit": float(doc_hit),
    }


# ----------------------------------------------------------------------------- runners


RETRIEVAL_CONFIGS = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid (RRF)", "hybrid", False),
    ("hybrid + rerank", "hybrid", True),
]


def evaluate_retrieval(
    pipeline: RAGPipeline, questions: list[EvalQuestion], top_k: int
) -> dict[str, dict[str, float]]:
    retriever = pipeline.retriever
    saved_reranker = retriever.reranker
    summary: dict[str, dict[str, float]] = {}
    try:
        for name, mode, rerank in RETRIEVAL_CONFIGS:
            if rerank and saved_reranker is None:
                continue
            retriever.reranker = saved_reranker if rerank else None
            rows = []
            t0 = time.perf_counter()
            for q in questions:
                if not q.answerable:
                    continue
                rows.append(retrieval_metrics(retriever.retrieve(q.question, top_k, mode), q.gold))
            elapsed = time.perf_counter() - t0
            summary[name] = {
                f"hit@{top_k}": statistics.mean(r["hit"] for r in rows),
                "mrr": statistics.mean(r["mrr"] for r in rows),
                f"doc_hit@{top_k}": statistics.mean(r["doc_hit"] for r in rows),
                "ms_per_query": 1000 * elapsed / max(1, len(rows)),
            }
    finally:
        retriever.reranker = saved_reranker
    return summary


def _score_answer(q: EvalQuestion, ans: Answer) -> dict:
    recall, hits = fact_recall(ans.text, q.key_facts)
    row = {
        "answer": ans.text,
        "abstained": ans.citations.abstained,
        "fact_recall": recall if q.answerable else None,
        "fact_hits": hits,
        "seconds": round(sum(ans.timings.values()), 2),
    }
    if ans.mode == "rag":
        cited = ans.cited_sources
        row.update(
            {
                "cited_sources": [s.chunk.citation for _, s in cited],
                "model_citations": ans.citations.cited,
                "auto_citations": ans.citations.auto_attributed,
                "invalid_citations": ans.citations.invalid,
                "citation_correct": (
                    any(is_relevant(s.chunk.doc_id, 0, 0, q.gold, False) for _, s in cited)
                    if q.answerable and cited
                    else None
                ),
                "groundedness": round(ans.citations.groundedness, 3),
                "retrieved": [r.chunk.citation for r in ans.sources],
                **{f"retrieval_{k}": v for k, v in retrieval_metrics(ans.sources, q.gold).items()},
            }
        )
    return row


def evaluate_answers(
    pipeline: RAGPipeline,
    questions: list[EvalQuestion],
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, start=1):
        if progress:
            progress(f"[{i}/{len(questions)}] {q.id}")
        rag = pipeline.ask(q.question)
        closed = pipeline.ask_closed_book(q.question)
        rows.append(
            {
                "id": q.id,
                "type": q.type,
                "question": q.question,
                "reference": q.answer,
                "rag": _score_answer(q, rag),
                "closed_book": _score_answer(q, closed),
            }
        )
    return rows


def summarise_answers(rows: list[dict]) -> dict:
    def block(system: str) -> dict:
        ans = [r[system] for r in rows if r["type"] != "unanswerable"]
        una = [r[system] for r in rows if r["type"] == "unanswerable"]
        out = {
            "n_answerable": len(ans),
            "n_unanswerable": len(una),
            "fact_recall": statistics.mean(a["fact_recall"] for a in ans) if ans else 0.0,
            "fully_correct": statistics.mean(float(a["fact_recall"] == 1.0) for a in ans)
            if ans
            else 0.0,
            "abstain_on_answerable": statistics.mean(float(a["abstained"]) for a in ans)
            if ans
            else 0.0,
            "correct_abstention": statistics.mean(float(a["abstained"]) for a in una)
            if una
            else 0.0,
            "mean_seconds": statistics.mean(a["seconds"] for a in ans + una) if rows else 0.0,
        }
        if system == "rag":
            cc = [a["citation_correct"] for a in ans if a["citation_correct"] is not None]
            out["citation_precision"] = statistics.mean(float(c) for c in cc) if cc else 0.0
            answered = [a for a in ans if not a["abstained"]]
            out["groundedness"] = (
                statistics.mean(a["groundedness"] for a in answered) if answered else 0.0
            )
            out["model_cited_share"] = (
                statistics.mean(float(bool(a["model_citations"])) for a in answered)
                if answered
                else 0.0
            )
        return out

    by_type: dict[str, dict] = {}
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        if t == "unanswerable":
            by_type[t] = {
                s: statistics.mean(float(r[s]["abstained"]) for r in sub)
                for s in ("rag", "closed_book")
            }
        else:
            by_type[t] = {
                s: statistics.mean(r[s]["fact_recall"] for r in sub) for s in ("rag", "closed_book")
            }
    return {"rag": block("rag"), "closed_book": block("closed_book"), "by_type": by_type}


# ----------------------------------------------------------------------------- report


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _quote(text: str, limit: int = 700) -> str:
    text = text.strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " …"
    return "\n".join("> " + line if line.strip() else ">" for line in text.splitlines())


def render_report(meta: dict, retrieval: dict, answers: list[dict], summary: dict) -> str:
    k = meta["top_k"]
    L: list[str] = []
    L.append("# RAGDemia evaluation report\n")
    L.append(
        f"*Generated {meta['generated_at']} · LLM `{meta['llm']}` · embeddings "
        f"`{meta['embedding_model']}` · reranker `{meta['reranker'] or 'off'}` · "
        f"top-k = {k} · {meta['n_questions']} questions ({summary['rag']['n_answerable']} "
        f"answerable, {summary['rag']['n_unanswerable']} unanswerable) · corpus: "
        f"{meta['n_documents']} documents / {meta['n_chunks']} chunks*\n"
    )

    rag, cb = summary["rag"], summary["closed_book"]
    L.append("## Headline: does grounding help?\n")
    L.append("| Metric | Closed-book LLM | RAGDemia | Δ |")
    L.append("|---|---:|---:|---:|")
    for label, key in [
        ("Key-fact recall (answerable)", "fact_recall"),
        ("Fully correct answers", "fully_correct"),
        ("Correct abstention (unanswerable)", "correct_abstention"),
    ]:
        d = rag[key] - cb[key]
        L.append(
            f"| {label} | {_pct(cb[key])} | **{_pct(rag[key])}** | {'+' if d >= 0 else ''}{100 * d:.0f} pp |"
        )
    L.append(
        f"| Refused an answerable question | {_pct(cb['abstain_on_answerable'])} | {_pct(rag['abstain_on_answerable'])} | |"
    )
    L.append(
        f"| Mean latency per question (this machine) | {cb['mean_seconds']:.1f}s | {rag['mean_seconds']:.1f}s | |"
    )
    L.append("")
    L.append("RAG-only answer diagnostics:\n")
    L.append(
        f"- **Citation precision** - answers whose cited sources include a gold document: **{_pct(rag['citation_precision'])}**"
    )
    L.append(
        f"- **Groundedness** - answer sentences lexically supported by a retrieved chunk: **{_pct(rag['groundedness'])}**"
    )
    L.append(
        f"- **Model wrote its own `[n]` citations** in {_pct(rag['model_cited_share'])} of non-refused answers (the rest were auto-attributed)\n"
    )

    L.append("By question type (fact recall; for *unanswerable* = correct-abstention rate):\n")
    L.append("| Type | Closed-book | RAG |")
    L.append("|---|---:|---:|")
    for t, v in summary["by_type"].items():
        L.append(f"| {t} | {_pct(v['closed_book'])} | {_pct(v['rag'])} |")
    L.append("")

    L.append("## Retrieval ablation\n")
    L.append("Gold = the document *and page* that contain the answer. Answerable questions only.\n")
    L.append(f"| Retriever | Hit@{k} | MRR | DocHit@{k} | ms / query |")
    L.append("|---|---:|---:|---:|---:|")
    for name, m in retrieval.items():
        L.append(
            f"| {name} | {_pct(m[f'hit@{k}'])} | {m['mrr']:.3f} | {_pct(m[f'doc_hit@{k}'])} | {m['ms_per_query']:.0f} |"
        )
    L.append("")

    # ---- case studies
    corrected = [
        r
        for r in answers
        if r["type"] != "unanswerable" and r["rag"]["fact_recall"] > r["closed_book"]["fact_recall"]
    ]
    corrected.sort(
        key=lambda r: r["rag"]["fact_recall"] - r["closed_book"]["fact_recall"], reverse=True
    )
    refused_right = [
        r
        for r in answers
        if r["type"] == "unanswerable"
        and r["rag"]["abstained"]
        and not r["closed_book"]["abstained"]
    ]
    failures = [
        r
        for r in answers
        if (r["type"] != "unanswerable" and r["rag"]["fact_recall"] < 1.0)
        or (r["type"] == "unanswerable" and not r["rag"]["abstained"])
    ]

    def case(r: dict, n: int) -> None:
        L.append(f"### {n}. {r['question']}\n")
        L.append(f"*Reference:* {r['reference']}\n")
        cb_r, rag_r = r["closed_book"], r["rag"]
        cb_score = (
            "abstained"
            if cb_r["abstained"]
            else (
                f"fact recall {_pct(cb_r['fact_recall'])}"
                if cb_r["fact_recall"] is not None
                else "answered"
            )
        )
        rag_score = (
            "abstained"
            if rag_r["abstained"]
            else (
                f"fact recall {_pct(rag_r['fact_recall'])}"
                if rag_r["fact_recall"] is not None
                else "answered"
            )
        )
        L.append(f"**Closed-book LLM** ({cb_score}):\n")
        L.append(_quote(cb_r["answer"]) + "\n")
        L.append(f"**RAGDemia** ({rag_score}):\n")
        L.append(_quote(rag_r["answer"]) + "\n")
        if rag_r.get("cited_sources"):
            L.append("Sources: " + "; ".join(f"`{c}`" for c in rag_r["cited_sources"]) + "\n")

    L.append("## Case studies: where grounding changed the answer\n")
    for i, r in enumerate(corrected[:6], start=1):
        case(r, i)
    if refused_right:
        L.append("## Case studies: refusing instead of hallucinating\n")
        L.append(
            "Questions whose answer is *not* in the corpus. A trustworthy assistant should say so.\n"
        )
        for i, r in enumerate(refused_right[:3], start=1):
            case(r, i)
    L.append("## Failure analysis\n")
    if not failures:
        L.append("No failures on this set.\n")
    else:
        L.append(
            "Every question where RAGDemia was not fully correct, with the retrieval outcome - "
            "this separates *retrieval misses* from *generation misses*.\n"
        )
        L.append("| Question | Fact recall | Gold chunk retrieved? | Abstained | Diagnosis |")
        L.append("|---|---:|:---:|:---:|---|")
        for r in failures:
            rr = r["rag"]
            hit = rr.get("retrieval_hit")
            if r["type"] == "unanswerable":
                diag, fr = "answered an unanswerable question", "-"
            else:
                fr = _pct(rr["fact_recall"])
                if not hit:
                    diag = "retrieval miss"
                elif rr["abstained"]:
                    diag = "over-cautious refusal (evidence was retrieved)"
                else:
                    diag = "generation miss / paraphrase (evidence was retrieved)"
            L.append(
                f"| {r['id']} | {fr} | {'yes' if hit else 'no'} | {'yes' if rr['abstained'] else 'no'} | {diag} |"
            )
        L.append("")
    L.append("## Method notes\n")
    L.append(
        "- Both systems use the **same LLM, decoding settings and answer-style rules**; the only "
        "difference is whether retrieved sources are in the prompt.\n"
        "- Key-fact matching is string-based (with accepted aliases), so a correct paraphrase can "
        "be scored as a miss. Numbers make this conservative rather than generous.\n"
        "- The sample corpus is public, and the closed-book model may have seen these papers in "
        "pre-training; on private lecture notes and unpublished work the gap is expected to be "
        "larger, because the closed-book model has *no* way to know the content.\n"
    )
    return "\n".join(L)


def run_evaluation(
    pipeline: RAGPipeline,
    questions_path: str | Path,
    out_dir: str | Path,
    progress: Callable[[str], None] | None = None,
    skip_answers: bool = False,
) -> dict:
    questions = load_questions(questions_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    top_k = pipeline.top_k

    if progress:
        progress("retrieval ablation")
    retrieval = evaluate_retrieval(pipeline, questions, top_k)
    answers = [] if skip_answers else evaluate_answers(pipeline, questions, progress)
    summary = summarise_answers(answers) if answers else {}

    store = pipeline.retriever.store
    reranker = pipeline.retriever.reranker
    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "llm": pipeline.llm.name,
        "embedding_model": store.manifest.get("embedding_model"),
        "reranker": reranker.model_name if reranker else None,
        "top_k": top_k,
        "n_questions": len(questions),
        "n_documents": len(store.manifest.get("documents", [])),
        "n_chunks": len(store.chunks),
    }
    (out / "retrieval.json").write_text(json.dumps(retrieval, indent=2))
    if answers:
        with open(out / "answers.jsonl", "w", encoding="utf-8") as fh:
            for r in answers:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / "summary.json").write_text(
            json.dumps({"meta": meta, "retrieval": retrieval, "answers": summary}, indent=2)
        )
        (out / "report.md").write_text(
            render_report(meta, retrieval, answers, summary), encoding="utf-8"
        )
    return {"meta": meta, "retrieval": retrieval, "answers": summary}
