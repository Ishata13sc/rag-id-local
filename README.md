# RAG-ID Local

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-cpu-blue)](https://github.com/facebookresearch/faiss)
[![SQLite](https://img.shields.io/badge/SQLite-embedded-003B57)](https://www.sqlite.org/)
[![HTMX](https://img.shields.io/badge/HTMX-1.x-1461db)](https://htmx.org/)
[![Tailwind](https://img.shields.io/badge/Tailwind-via%20CDN-38bdf8)](https://tailwindcss.com/)

Lightweight, offline-first **RAG (Retrieval-Augmented Generation)** for Indonesian documents. CPU-only, 8 GB RAM friendly.  
Pipeline: Upload → Split → Embed → FAISS → Retrieve → (optional) Rerank → Cite & highlight.

## Why

- Indo documents without cloud cost or data egress
- Runs on modest laptops (CPU + 8 GB)
- Simple stack: **FastAPI + FAISS-cpu + SQLite + Tailwind/HTMX**

## Features

- Upload PDF/DOCX (server-side parsing), source citations & context viewer
- Offline mode toggles (`RAG_OFFLINE`, `HF_HUB_OFFLINE`)
- Embeddings:
  - `tfidf` (zero-download, fastest on low RAM)
  - sentence-transformers (optional; int8 friendly)
- Optional reranker (Cross-Encoder) with easy on/off
- Minimal UI with Tailwind via CDN and HTMX interactions

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_dev.sh
# open http://127.0.0.1:8000
```
