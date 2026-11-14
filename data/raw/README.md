# Corpus folder

Put PDFs anywhere under this folder. Sub-folders become part of the document id
(`private/ml_lecture_03.pdf` → `private/ml_lecture_03`).

| Folder | What goes here | Committed to git? |
|---|---|---|
| `sample/` | Open-access papers fetched by `scripts/download_sample_corpus.py` | No (arXiv licence) |
| `private/` | Your lecture notes, slides, and your own papers | **No**: never commit course material you don't own |

After adding or removing files, rebuild the index: `ragdemia ingest`.
Run `ragdemia inspect` first to catch scanned PDFs (no text layer) or slides with very little text.
