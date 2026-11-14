# RAGDemia: design notes and interview preparation

This document explains *why* every component is built the way it is, and lists the questions
an interviewer (or admissions panel) is likely to ask, with answers grounded in this codebase.
File references point to the code that implements each idea.

---

## 1. The one-paragraph pitch

> RAGDemia is a retrieval-augmented question-answering system over my own academic documents.
> It parses PDFs into section-aware chunks, retrieves evidence with a hybrid of dense
> embeddings and BM25 fused by Reciprocal Rank Fusion, reranks with a cross-encoder, and
> generates answers with a locally hosted open LLM that must cite the section and page of
> every claim. A verification layer checks those citations and measures how much of the
> answer is supported by the retrieved text. I evaluated it against the same LLM without
> retrieval on a hand-written test set: grounding raised key-fact recall from X% to Y% and
> turned confident hallucinations on unanswerable questions into explicit refusals. *(Fill in
> X and Y from `eval/results/report.md`. On the sample corpus with the 3B model: key-fact
> recall 15% → 75%, fully correct 7% → 67%, Hit@5 81% → 96% with reranking.)*

---

## 2. Component by component

### 2.1 PDF parsing (`src/ragdemia/parsing.py`)

**What it does.** Reads every character with its font, size and position (PyMuPDF
`rawdict`), rebuilds lines, removes noise, detects headings, and outputs a section tree whose
paragraphs carry page numbers.

**Why not `page.get_text()`?** Plain text loses page and section boundaries, which are
exactly what citations need. It also inherits PDF quirks. On the IEEE-format survey, words
came out glued together ("RetrievalAugmentedGeneration") because that PDF positions words
without emitting space characters. Rebuilding lines from glyph gaps fixes it.

**How headings are detected (no ML):** the most common font size (weighted by characters) is
the body size. A block is a heading if it is short, does not end like a sentence, and is
larger than body text, bold, or numbered ("3.2 Results", "II. OVERVIEW"). Several failure
modes came from the real corpus and each has a rule:

| Observed failure | Rule that fixes it |
|---|---|
| Section number and title emitted as separate blocks ("2.1" / "Models") | merge a number-only block with the next block |
| Figure text "20 Total Retrieved Documents" taken as section 20 | top-level section numbers must be ≤ 15 |
| Equation "C E((x,y),r)∼…" taken as appendix C | reject math symbols and low letter ratio |
| Table-of-contents entries "A SELF-RAG Details 17" | reject headings ending in a page number |
| Small-caps "REFERENCES" not detected → bibliography indexed | skip-list headings are always detected |
| Drop cap "L" glued to "I. INTRODUCTION" | move the letter back into the paragraph |

**Why drop the References section?** Bibliography text contains every keyword of the field
("retrieval", "dense", "language model"), so it matches almost any query lexically while
answering none of them.

**Limitation:** scanned PDFs have no text layer. `ragdemia inspect` flags them; OCR
(e.g. Tesseract via `ocrmypdf`) is the fix.

### 2.2 Chunking (`src/ragdemia/chunking.py`)

**Choices:**
1. **Chunks never cross section boundaries**, so each chunk has one topic and one citation.
2. **Sentences are atomic**: a rule-based splitter that knows "et al.", "Fig.", "e.g.".
3. **~320-token budget, ~64-token overlap.** BGE truncates at 512 tokens; leaving headroom
   for the context header avoids silent truncation. Overlap keeps a fact that spans a
   boundary complete in at least one chunk.
4. **Token counts use the embedding model's own tokenizer**, not word counts.
5. **Contextual header**: each chunk is embedded as `title | section path \n text`. A
   paragraph saying "our method improves recall" means nothing alone; the header tells the
   embedding *which* method.

**Q: Why not fixed 500-character windows?** They cut sentences in half, mix sections, and
make citations ambiguous. The cost of my approach is variable chunk length, which is fine.

**Q: Why not "semantic chunking" with embeddings?** Academic documents already have
author-defined semantic units: sections. Using the document's own structure is cheaper,
deterministic, and gives better citations.

### 2.3 Embeddings (`src/ragdemia/embeddings.py`)

`BAAI/bge-base-en-v1.5`: a 110M-parameter BERT encoder, 768-dimensional output, strong on
the MTEB retrieval benchmark, free, runs on CPU.

**What happens underneath:** WordPiece tokenisation → 12 transformer layers → the `[CLS]`
token's final hidden state is the sentence vector → L2 normalisation. With unit vectors,
inner product = cosine similarity.

**Asymmetric prefix:** BGE was trained with an instruction prefix on *queries only*
("Represent this sentence for searching relevant passages: "). Forgetting it measurably
lowers retrieval quality.

**Q: Why not OpenAI embeddings?** Per-query cost, documents leave the machine, and
results would not be reproducible if the model changed behind the API.

### 2.4 Vector store (`src/ragdemia/store.py`)

FAISS `IndexFlatIP` (exact search). At 400–5,000 chunks, brute force over 768-d vectors is
a single matrix-vector product: under a millisecond. Approximate indexes (HNSW, IVF-PQ)
trade recall for speed and only pay off around 10⁶ vectors. The index is saved with a
**manifest** (embedding model, config, document checksums), and the pipeline refuses to
query an index built with a different embedding model than the one configured (which would
silently return garbage).

### 2.5 BM25 (`src/ragdemia/bm25.py`, written from scratch)

```
score(d,q) = Σ_t IDF(t) · tf(t,d)(k1+1) / (tf(t,d) + k1(1 − b + b·|d|/avgdl))
```
- `tf` saturates (k1 = 1.5): the 10th occurrence adds little.
- `b = 0.75` normalises for chunk length.
- `IDF` up-weights rare terms ("HyDE", "TriviaQA", "128").

**Q: Why keep a 1990s algorithm next to a neural retriever?** Dense embeddings are weak at
exact matches: acronyms, dataset names, numbers. In the evaluation, BM25 alone matched dense
retrieval (Hit@5 81% each) and found the right *document* for 100% of questions, because
academic questions are full of rare technical terms.

### 2.6 Hybrid fusion and reranking (`src/ragdemia/retrieval.py`)

**Reciprocal Rank Fusion:** `RRF(d) = Σ_r 1/(60 + rank_r(d))`. It fuses *ranks*, not
scores, because BM25 scores (0–30) and cosine scores (−1 to 1) are not comparable.

**Cross-encoder reranker** (`BAAI/bge-reranker-base`): the bi-encoder embeds query and
passage *separately* (fast, so it can scan everything). The cross-encoder reads them
*together* through one transformer (accurate, but one forward pass per pair), so it only
reranks the top-15 fused candidates. This is the classic retrieve-then-rerank cascade.

**A concrete case from development:** for *"What are the three RAG paradigms described in
the survey?"*, dense retrieval returned five Self-RAG chunks (lexical/semantic overlap on
"RAG" and "types"). BM25 found the survey; the reranker put the survey's introduction and
conclusion on top. The ablation table in the README quantifies this across the full set.

### 2.7 Generation (`src/ragdemia/llm.py`, `prompts.py`)

- **Ollama** serves a 4-bit quantised Qwen2.5-7B-Instruct (≈4.7 GB VRAM). 4-bit
  quantisation stores weights in 4 bits plus per-block scales, cutting memory ~4× vs fp16
  with a small quality loss, which is what makes a 7B model fit a laptop GPU.
- Sources are numbered `[1]..[k]`; the model only emits integers, which code maps back to
  (document, section, page). Small models copy integers far more reliably than titles.
- **Few-shot demonstration from an unrelated domain** (plant biology). During development,
  describing the format made the 3B test model copy the template literally, and a
  same-domain example leaked into answers. A worked example from another domain fixed both.
- **Fixed abstention sentence** so "I don't know" is machine-detectable.
- `temperature = 0.1`, fixed seed: answers are reproducible for evaluation.

### 2.8 Citation verification (`src/ragdemia/citations.py`)

The model is *asked* to cite; this module *checks*:
- citations to non-existent sources → hallucination flag,
- **groundedness**: for each answer sentence, the share of its content words found in the
  best-matching source; ≥ 60% counts as supported,
- **auto-attribution**: a supported sentence the model forgot to cite gets its best
  source attached, *labelled as automatic*, never disguised as the model's own citation.

**Q: Why lexical overlap and not an NLI model?** Transparent, instant, deterministic.
Known blind spot: paraphrases score low, so groundedness is a *lower bound*. An NLI
model (e.g. DeBERTa-MNLI) is the natural upgrade.

### 2.9 Evaluation (`src/ragdemia/evaluation.py`, `eval/questions.yaml`)

- **Separate retrieval from generation.** Hit@5 and MRR tell you whether the evidence was
  found; key-fact recall tells you whether the model used it. The failure table labels each
  miss as *retrieval miss* or *generation miss*, which tells you what to fix.
- **Key facts with accepted aliases** instead of LLM-as-judge: a 3–7B local judge is noisy;
  string matching is reproducible and conservative (paraphrases count as misses, never the
  reverse).
- **Unanswerable questions** test the most important property of a trustworthy assistant:
  saying "not in my documents" instead of inventing.
- **Same LLM, same rules, with vs. without retrieval**: the only variable is grounding.
- **Honesty about bias**: two gold page labels were widened after the first run (documented
  in the YAML header); no question was rewritten. One question ("the RAG paper") is
  genuinely ambiguous in a corpus that also contains a RAG survey, and it is kept as a
  real failure.

### 2.10 Two document families: LaTeX papers vs. Word/PowerPoint course packs

The first real lecture pack I indexed (an 83-page course pack exported from PowerPoint) broke
three assumptions that held for every paper:

| Observed failure | Fix |
|---|---|
| Chapter "1. Introduction to NLP" and its subsection "1. What is NLP?" have the *same* number; only font size differs (14 pt vs 12 pt), because Word's multi-level numbering was lost on export | If single-number headings occur in two or more sizes, numbering is untrustworthy → levels come from **font-size rank** instead |
| Bold header cells of the lecture-plan table ("TOPIC", "PROPOSED LECTURE") became headings | Run PyMuPDF's **table detector** on pages with ≥3 heading candidates and ban headings inside tables. Its cell text is *not* used: on LaTeX tables it garbles numbers (59.1 → "591 ."). |
| Huge administrative page titles ("Lecture Plan", 45 pt) became ancestors of every later chapter | A display-size heading is only a parent for headings **on its own page** |

The lesson I would tell an interviewer: heuristics generalise only as far as the documents
you have tested them on, so `ragdemia inspect --sections` exists to make the parse visible
before anything is indexed, and every rule is a regression test.

### 2.11 Incremental indexing (`src/ragdemia/ingest.py`)

The folder is the source of truth; ingest *reconciles* the index with it. Each file's SHA-256
is stored in the manifest. Unchanged files reuse their chunks **and vectors** (read back out
of FAISS with `reconstruct_n`, so nothing is re-embedded); new/changed files are parsed and
embedded; vanished files are dropped. FAISS and BM25 are rebuilt from the final list (ms at
this scale), which keeps row *i* of every structure aligned.

**The subtle part - cache invalidation.** Reusing old chunks is only correct if today's code
would produce identical chunks. The manifest stores a *build signature*: a hash of the parser
version, parsing/chunking settings and embedding model. If it differs, the run becomes a
full rebuild automatically. Without this, changing `max_tokens` would silently mix two
chunking schemes in one index.

**Q: Why not just re-embed everything each time?** It is fine at 10 documents, but cost grows
with the library while the change is usually one file. On the current corpus: full build
~2 minutes on CPU, no-change sync 0.1 s.

### 2.12 Multi-turn chat: query rewriting (`pipeline.condense`, `prompts.py`)

A follow-up like "how is it trained?" retrieves garbage - "it" matches nothing. Before
retrieval, the LLM rewrites the follow-up into a standalone question using the last 3 turns
("How is byte-pair encoding (BPE) trained?"), with a one-shot example of the task. Design
choices:
- Only the *query* is rewritten; the answer is still generated from freshly retrieved
  sources, so conversation history cannot smuggle ungrounded claims into an answer.
- The rewrite is shown to the user ("↳ searched for: …") - if it misreads the conversation,
  you can see why the answer is off.
- No history → no extra LLM call. A topic switch ("Who built ELIZA?") is returned unchanged.

### 2.13 Uploads as an attack surface (`src/ragdemia/library.py`)

Accepting files from a browser means validating them: filenames are reduced to a safe
basename (`../../etc/passwd` → `passwd.pdf`), content must start with `%PDF`, size is capped
(50 MB), duplicates are detected by checksum, and only files uploaded through the app can be
deleted through it - documents placed on disk are never removed by the UI or API.

### 2.14 The web app (`src/ragdemia/web/`, `api.py`)

**Why not Streamlit?** The first UI was Streamlit. It re-runs the whole script on every
interaction (answers visibly flicker), and layout control is limited, so a three-pane
"read the evidence" design was not possible. The current app is plain HTML/CSS/JS served by
FastAPI: one process, one port, no build step, no framework to justify.

**Streaming with Server-Sent Events.** `POST /ask/stream` emits `meta` (sources, rewritten
query) as soon as retrieval finishes, then one `token` event per generated fragment, then
`done` with the verified answer. SSE is plain HTTP (`text/event-stream`), one-directional,
and proxy-friendly, which is all a chat answer needs; WebSockets would add a handshake and
two-way state for no benefit. The browser reads it with `fetch` plus a stream reader,
because `EventSource` only supports GET.

**Designed for verification, not just reading.** Each `[n]` is a button. Clicking it opens
the passage with the answer's content words highlighted, the section breadcrumb, the page,
which answer sentences it supports (with support %), and its dense/BM25/rerank ranks. Cited
sources are shown first; uncited ones are one click away. A groundedness chip turns
amber or red when the answer drifts from the evidence.

**Testability.** `create_app(pipeline, library)` is an app factory: tests inject a pipeline
built from a hashing embedder and a stub LLM, then check the SSE event order, uploads and
static serving without downloading a model.

---

## 3. Likely interview questions

**Q: What is the difference between RAG and fine-tuning?**
Fine-tuning changes the weights, so knowledge is baked in, hard to update, and cannot be cited.
RAG keeps knowledge in an external index that is swapped instantly, and every answer
points to its evidence. They are complementary: fine-tune for *style/format*, retrieve for
*facts*. (The Gao et al. survey in the corpus has a section on exactly this.)

**Q: Your system gives a wrong answer. How do you debug it?**
Check the retrieval first: is the gold chunk in the top-k (`ragdemia ask --json` shows
per-retriever ranks)? If not, the problem is retrieval (chunking, query phrasing, hybrid
weights, reranker). If yes, the problem is generation (prompt, model size, context order).
The evaluation report automates exactly this split.

**Q: How would you scale this to 1 million documents?**
Switch FAISS to HNSW or IVF-PQ, shard the index, embed in batches on GPU, move metadata to a
database, add incremental ingestion keyed on the stored SHA-256 checksums, and cache
query embeddings. BM25 would move to an inverted-index engine (Elasticsearch/OpenSearch).

**Q: Why top-k = 5?**
Precision/recall trade-off within the context window. *Lost in the Middle* (in the corpus)
shows that models under-use evidence in the middle of long contexts, so more chunks is not
free. The reranker makes the top-5 count.

**Q: How do you know the citations are right?**
Three layers: invalid-number detection, sentence-level lexical support, and, in
evaluation, citation precision against gold documents.

**Q: What would you do next?**
NLI-based faithfulness checking, OCR
for scanned notes, multimodal retrieval for figures and slides (a vision-language model such
as Qwen2-VL), and fine-tuning the embedding model on synthetic question-chunk pairs
generated from my own corpus.

---

## 4. Statement-of-purpose language you can adapt

> To understand where large language models fail, I built RAGDemia, a retrieval-augmented
> question-answering system over my own lecture notes and the two conference papers I have
> authored. Rather than relying on a framework, I implemented each stage myself:
> layout-aware PDF parsing that recovers section structure from font statistics, sentence-level
> chunking, a from-scratch BM25 index fused with dense BGE embeddings through Reciprocal
> Rank Fusion, cross-encoder reranking, and citation verification that measures how much of
> each answer is supported by retrieved evidence.

> Evaluating the system taught me as much as building it. On a hand-written test set, the
> same 7B model answered [X]% of factual questions correctly with retrieval versus [Y]%
> without, and on questions whose answers were not in my documents, retrieval turned
> confident fabrications into explicit refusals. Separating retrieval failures from
> generation failures showed me that most remaining errors were [retrieval / generation]
> errors, which is why I want to study [dense retrieval / faithful generation / evaluation of
> LLM systems] at the Master's level.

Replace the bracketed parts with your own numbers from `eval/results/report.md` after you
run it on your documents.
