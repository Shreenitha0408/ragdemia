"""Download the sample corpus: seven open-access arXiv papers on retrieval-augmented generation.

PDFs are fetched at setup time rather than committed, because arXiv's default licence does not
permit redistribution.  Usage:  python scripts/download_sample_corpus.py [--out data/raw/sample]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

PAPERS = {
    "2004.04906": "Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)",
    "2212.10496": "Precise Zero-Shot Dense Retrieval without Relevance Labels - HyDE (Gao et al., 2022)",
    "2307.03172": "Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)",
    "2309.15217": "Ragas: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)",
    "2310.11511": "Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023)",
    "2312.10997": "Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2023)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/sample")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        follow_redirects=True, timeout=60, headers={"User-Agent": "ragdemia/0.1"}
    ) as client:
        for arxiv_id, title in PAPERS.items():
            target = out / f"{arxiv_id}.pdf"
            if target.exists() and target.read_bytes()[:4] == b"%PDF":
                print(f"✓ {arxiv_id} already present")
                continue
            r = client.get(f"https://arxiv.org/pdf/{arxiv_id}")
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise RuntimeError(f"{arxiv_id}: response is not a PDF")
            target.write_bytes(r.content)
            print(f"↓ {arxiv_id}  {title}  ({len(r.content) / 1e6:.1f} MB)")
            time.sleep(3)  # arXiv asks automated clients to rate-limit themselves


if __name__ == "__main__":
    main()
