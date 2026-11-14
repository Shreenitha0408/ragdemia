---
title: RAGDemia
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cited answers from academic PDFs - retrieval-augmented QA
---

# RAGDemia - live demo

Ask questions about research papers and get answers that cite the exact **document, section
and page**. Click any `[n]` citation to see the passage it came from.

- The shared library holds seven open-access papers on retrieval-augmented generation
  (DPR, RAG, HyDE, Lost in the Middle, Ragas, Self-RAG, and a RAG survey).
- **Upload your own PDF** (up to 5, 20 MB each). It is indexed in seconds, searched together
  with the shared library, and visible **only to you**. The file is deleted right after
  indexing and your workspace is forgotten after 6 hours of inactivity.
- If the answer is not in the documents, RAGDemia says so instead of guessing.

Retrieval (BGE embeddings + BM25 + cross-encoder reranking) and citation checking run in this
Space; answers are written by an open model (Llama 3.1 8B) through Groq's API.

Source code, architecture and evaluation: https://github.com/Shreenitha0408/ragdemia
