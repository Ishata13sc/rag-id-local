# RAG-ID Local (FAISS + SQLite)

Offline-friendly, RAM-efficient RAG for Indonesian documents.

## Features
- Local CPU embeddings (Sentence-Transformers, small multilingual model)
- FAISS index on CPU
- SQLite store for documents, chunks, and metadata
- Minimal web UI (HTMX + Alpine)
- Upload PDFs/DOCX, search, citations (WIP)
- Ready for reranker & disk cache (stretch)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# dev
chmod +x run_dev.sh
./run_dev.sh
# visit: http://127.0.0.1:8000
```

## Layout
```
src/
  ingest/        # loaders & splitters
  index/         # build & retrieval
  models/        # embedding model wrapper
  storage/       # SQLite helpers
api/             # FastAPI app
web/             # templates & static assets
tests/           # smoke tests
```

## Environment
- Python 3.10+
- CPU only
- Works on 8 GB RAM

## Roadmap
- [ ] Reranker (bge-reranker-mini)
- [ ] Disk cache for embeddings
- [ ] Highlighting + paragraph-level citations
- [ ] Batch ingest CLI

## License
MIT
