# RAGDemia evaluation report

*Generated 2026-11-14 08:41 UTC · LLM `hf:Qwen/Qwen2.5-3B-Instruct` · embeddings `BAAI/bge-base-en-v1.5` · reranker `BAAI/bge-reranker-base` · top-k = 5 · 31 questions (27 answerable, 4 unanswerable) · corpus: 7 documents / 395 chunks*

## Headline: does grounding help?

| Metric | Closed-book LLM | RAGDemia | Δ |
|---|---:|---:|---:|
| Key-fact recall (answerable) | 15% | **75%** | +60 pp |
| Fully correct answers | 7% | **67%** | +59 pp |
| Correct abstention (unanswerable) | 50% | **75%** | +25 pp |
| Refused an answerable question | 19% | 11% | |
| Mean latency per question (this machine) | 6.1s | 16.7s | |

RAG-only answer diagnostics:

- **Citation precision** - answers whose cited sources include a gold document: **96%**
- **Groundedness** - answer sentences lexically supported by a retrieved chunk: **98%**
- **Model wrote its own `[n]` citations** in 100% of non-refused answers (the rest were auto-attributed)

By question type (fact recall; for *unanswerable* = correct-abstention rate):

| Type | Closed-book | RAG |
|---|---:|---:|
| factual | 17% | 77% |
| multi_doc | 50% | 0% |
| numeric | 0% | 80% |
| unanswerable | 50% | 75% |

## Retrieval ablation

Gold = the document *and page* that contain the answer. Answerable questions only.

| Retriever | Hit@5 | MRR | DocHit@5 | ms / query |
|---|---:|---:|---:|---:|
| dense | 81% | 0.608 | 93% | 80 |
| bm25 | 81% | 0.591 | 100% | 0 |
| hybrid (RRF) | 81% | 0.575 | 96% | 35 |
| hybrid + rerank | 96% | 0.861 | 96% | 2683 |

## Case studies: where grounding changed the answer

### 1. How was Wikipedia split into retrieval units in the DPR paper, and how many passages resulted?

*Reference:* Articles were split into disjoint 100-word blocks, giving 21,015,324 passages.

**Closed-book LLM** (fact recall 0%):

> In the DPR (Dual-Reader) paper, Wikipedia articles were split into retrieval units by sentence. The number of passages resulting from this splitting process varied depending on the specific experiment, but typically each sentence became a separate passage for retrieval purposes.

**RAGDemia** (fact recall 100%):

> Each article was split into multiple, disjoint text blocks of 100 words as passages [1]. This resulted in 21,015,324 passages in the end [1].

Sources: `Dense Passage Retrieval for Open-Domain Question Answering — 4 Experimental Setup > 4.1 Wikipedia Data Pre-processing, p. 4`

### 2. What are the two RAG model formulations proposed by Lewis et al., and how do they differ?

*Reference:* RAG-Sequence uses the same retrieved document for the whole output sequence; RAG-Token can use a different document for each target token.

**Closed-book LLM** (fact recall 0%):

> Lewis et al. proposed two RAG (Retrieval-Augmented Generation) model formulations: the Retrieval-Augmented Decoder (RAD) and the Retrieval-Augmented Encoder-Decoder (RAED). The RAD uses a retrieval component to select relevant documents for each query, then feeds these selections into a decoder for generation. RAED, on the other hand, integrates the retrieval component directly into the encoder-decoder architecture, allowing it to generate responses conditioned on both the input query and retrieved information.

**RAGDemia** (fact recall 100%):

> The two RAG model formulations proposed by Lewis et al. are RAG-Sequence and RAG-Token [4]. RAG-Sequence uses the same document to predict each target token, whereas RAG-Token can predict each target token based on a different document [4].

Sources: `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — 2 Methods, p. 3`

### 3. According to the HyDE paper, how are incorrect (hallucinated) details in the generated document handled?

*Reference:* The encoder's dense bottleneck filters out the incorrect details, grounding the vector to real documents in the corpus.

**Closed-book LLM** (fact recall 0%):

> The HyDE paper does not explicitly detail how incorrect or hallucinated details in the generated documents are handled.

**RAGDemia** (fact recall 100%):

> The generated document, which is not real and can contain factual errors, is used to capture relevance patterns. The encoder’s dense bottleneck filters out the incorrect details [1].

Sources: `Precise Zero-Shot Dense Retrieval without Relevance Labels — 1 Introduction, p. 2`

### 4. Which web search query sets does HyDE evaluate on?

*Reference:* TREC DL19 and TREC DL20 (based on MS-MARCO).

**Closed-book LLM** (fact recall 0%):

> HyDE evaluates on the WebQuestions-QA dev set and test set.

**RAGDemia** (fact recall 100%):

> HyDE evaluates on TREC DL19 and TREC DL20 [4].

Sources: `Precise Zero-Shot Dense Retrieval without Relevance Labels — 4 Experiments > 4.2 Web Search, pp. 4-5`

### 5. What shape does performance follow in "Lost in the Middle" as the position of the relevant document changes, and why?

*Reference:* A U-shaped curve - best when relevant information is at the beginning (primacy bias) or end (recency bias), worst in the middle.

**Closed-book LLM** (abstained):

> I don't know. The question seems to reference a specific paper or experiment not commonly known by this title or details.

**RAGDemia** (fact recall 100%):

> Performance follows a U-shaped curve as the position of the relevant document changes [1]. This is because language model performance is highest when relevant information occurs at the very start or end of the context, and significantly degrades when models must access and use information in the middle of their input context [1].

Sources: `Lost in the Middle: How Language Models Use Long Contexts — 1 Introduction, p. 1`

### 6. Which two tasks are used in "Lost in the Middle" to study how language models use long contexts?

*Reference:* Multi-document question answering and key-value retrieval.

**Closed-book LLM** (fact recall 0%):

> In "Lost in the Middle," the two tasks used to study language model behavior with long contexts are sentence similarity and next word prediction.

**RAGDemia** (fact recall 100%):

> The two tasks used in "Lost in the Middle" to study how language models use long contexts are multi-document question answering and key-value retrieval [1][5].

Sources: `Lost in the Middle: How Language Models Use Long Contexts — Abstract, p. 1`; `Lost in the Middle: How Language Models Use Long Contexts — 1 Introduction, p. 2`

## Case studies: refusing instead of hallucinating

Questions whose answer is *not* in the corpus. A trustworthy assistant should say so.

### 1. What accuracy does Self-RAG achieve on the MMLU benchmark?

*Reference:* Not in the documents - Self-RAG is not evaluated on MMLU.

**Closed-book LLM** (answered):

> Self-RAG achieves an accuracy of 85.3% on the MMLU benchmark.

**RAGDemia** (abstained):

> I couldn't find this in your documents.

### 2. Which Chinese question-answering dataset was used to evaluate DPR?

*Reference:* Not in the documents - DPR is evaluated on English datasets only.

**Closed-book LLM** (answered):

> The Chinese question-answering dataset used to evaluate DPR (Dual-Readers) is CNN/DailyMail.

**RAGDemia** (abstained):

> I couldn't find this in your documents.

## Failure analysis

Every question where RAGDemia was not fully correct, with the retrieval outcome - this separates *retrieval misses* from *generation misses*.

| Question | Fact recall | Gold chunk retrieved? | Abstained | Diagnosis |
|---|---:|:---:|:---:|---|
| dpr-batch | 50% | yes | no | generation miss / paraphrase (evidence was retrieved) |
| dpr-top20-nq | 50% | yes | no | generation miss / paraphrase (evidence was retrieved) |
| rag-generator | 0% | yes | no | generation miss / paraphrase (evidence was retrieved) |
| rag-index | 0% | no | no | retrieval miss |
| hyde-method | 67% | yes | no | generation miss / paraphrase (evidence was retrieved) |
| litm-models | 50% | yes | no | generation miss / paraphrase (evidence was retrieved) |
| ragas-embedding | 0% | yes | yes | over-cautious refusal (evidence was retrieved) |
| survey-paradigms | 0% | yes | yes | over-cautious refusal (evidence was retrieved) |
| multi-hyde-vs-dpr | 0% | yes | yes | over-cautious refusal (evidence was retrieved) |
| none-ragas-gpus | - | - | no | answered an unanswerable question |

## Method notes

- Both systems use the **same LLM, decoding settings and answer-style rules**; the only difference is whether retrieved sources are in the prompt.
- Key-fact matching is string-based (with accepted aliases), so a correct paraphrase can be scored as a miss. Numbers make this conservative rather than generous.
- The sample corpus is public, and the closed-book model may have seen these papers in pre-training; on private lecture notes and unpublished work the gap is expected to be larger, because the closed-book model has *no* way to know the content.
