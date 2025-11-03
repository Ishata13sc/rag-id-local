#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

export RAG_EMBEDDER_BACKEND=tfidf
export RAG_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAG_TRUST_REMOTE_CODE=1

mkdir -p data/uploads index

exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

[[ -f ".venv/bin/activate" ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export WATCHFILES_FORCE_POLLING=0  # pakai inotify, lebih ringan di Linux

exec python -m uvicorn api.main:app \
  --reload --host 127.0.0.1 --port 8000 \
  --reload-dir api --reload-dir src \
  --reload-exclude ".venv/*" \
  --reload-exclude ".git/*" \
  --reload-exclude "web/static/*"

