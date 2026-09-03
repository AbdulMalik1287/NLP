# NLP — Cross-Domain Complaint Understanding via Multilingual Encoders

Prerequisite-collection stage for the B.Tech AI/ML capstone proposal
*"Cross-Domain Complaint Understanding via Multilingual Encoders"* (IIIT Nagpur, 16 weeks).

This repository currently contains **only the prerequisites** — no training code, no data.
It is the Week-1 deliverable set: environment spec, dataset inventory, and literature base.

## What the project will be

| Piece | Choice |
|---|---|
| Core encoder | XLM-R (`xlm-roberta-base`, `-large` if VRAM allows) |
| Task | Complaint-intent classification under a unified intent ontology |
| Research variant | XLM-R + contrastive learning |
| Baselines | TF-IDF+LR, TF-IDF+SVM, BERT, mBERT, XLM-R |
| Languages | English, Hindi, Hinglish (code-mixed), Other |
| Domains | Banking, Telecom, E-commerce, Education, Government |
| Retrieval | FAISS over complaint embeddings (TF-IDF cosine as baseline) |
| Summarization | mT5 (supporting component, not the research claim) |
| Explainability | SHAP — intent classifier only |
| Serving | FastAPI + Uvicorn, Streamlit prototype UI |

## Contents

- [`requirements.txt`](requirements.txt) — pinned-ish Python deps for the full pipeline
- [`requirements-cpu.txt`](requirements-cpu.txt) — laptop/dev subset (no GPU, no training)
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) — Python/CUDA/GPU/Slurm setup, model downloads, VRAM notes
- [`docs/DATASETS.md`](docs/DATASETS.md) — candidate datasets, per-domain and per-language, with links + licences
- [`docs/LITERATURE.md`](docs/LITERATURE.md) — papers to read before Week 4, grouped by component
- [`docs/PREREQ_CHECKLIST.md`](docs/PREREQ_CHECKLIST.md) — tick-list to close out Week 1

## Status

Prerequisites only. Nothing has been downloaded, trained, or annotated yet.
Dataset licences in `docs/DATASETS.md` are recorded from public documentation and
**must be re-verified before any data is redistributed** — the repo will ship
preparation scripts, not restricted data.
