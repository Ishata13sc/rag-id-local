from fastapi import FastAPI, UploadFile, File, Form, Request, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import html
import re
import time
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi.templating import Jinja2Templates

from src.storage.db import (
    init_db,
    get_chunks_with_docs_by_ids,
    get_chunk_with_neighbors,
    get_all_chunks_texts,
    list_documents,
    delete_document_by_title_or_path,
)
from src.ingest.ingest import ingest_path
from src.index.build_index import ensure_index, rebuild_index
from src.index.retriever import search_query
from src.models.embedder import Embedder
from src.models.reranker import Reranker
from src.storage.emb_cache import text_hash, get_many, put_many

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

app = FastAPI(title="RAG-ID Local", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")

_embedder = Embedder()
_reranker: Optional[Reranker] = None

def get_reranker() -> Optional[Reranker]:
    global _reranker
    if _reranker is None:
        try:
            _reranker = Reranker()
        except Exception:
            _reranker = None
    return _reranker
templates = Jinja2Templates(directory="web/templates")

@app.on_event("startup")
def _startup():
    init_db()
    ensure_index()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/version")
def version():
    return {"version": app.version}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"uploaded": file.filename}

@app.post("/upload_multi")
async def upload_multi(files: List[UploadFile] = File(...)):
    saved = []
    for uf in files:
        dest = UPLOAD_DIR / uf.filename
        with open(dest, "wb") as f:
            f.write(await uf.read())
        saved.append(uf.filename)
    return {"uploaded": saved}

@app.post("/ingest", response_class=HTMLResponse)
def ingest(filename: str = Form(...)):
    path = UPLOAD_DIR / filename
    if not path.exists():
        return HTMLResponse(f"<div class='muted'>File not found: <code>{html.escape(filename)}</code></div>")
    try:
        n = ingest_path(path, _embedder)
        return HTMLResponse(f"<div>Ingested <strong>{n}</strong> chunks from <code>{html.escape(filename)}</code>.</div>")
    except Exception as e:
        return HTMLResponse(f"<div class='muted'>Error: {html.escape(repr(e))}</div>")

@app.get("/ingest", response_class=HTMLResponse)
def ingest_get(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists():
        return HTMLResponse(f"<div class='muted'>File not found: <code>{html.escape(filename)}</code></div>")
    try:
        n = ingest_path(path, _embedder)
        return HTMLResponse(f"<div>Ingested <strong>{n}</strong> chunks from <code>{html.escape(filename)}</code>.</div>")
    except Exception as e:
        return HTMLResponse(f"<div class='muted'>Error: {html.escape(repr(e))}</div>")

_ID_STOPWORDS = {"dan","atau","yang","untuk","dari","pada","adalah","itu","ini","dengan","ke","di","sebagai","yg","tp","dgn","kpd","dlm","pd","sebuah","karena","agar"}

def _regex_highlight(text: str, query: str, max_len: int = 700) -> str:
    t = html.escape(text)
    if len(t) > max_len:
        t = t[:max_len] + "…"
    tokens = [w for w in re.findall(r"[\w-]+", query, flags=re.IGNORECASE) if len(w) >= 3 and w.lower() not in _ID_STOPWORDS]
    if not tokens:
        return t
    tokens = sorted(set(tokens), key=len, reverse=True)
    for tok in tokens:
        pattern = re.compile(rf"\b({re.escape(tok)})\b", re.IGNORECASE)
        t = pattern.sub(r"<mark>\\1</mark>", t)
    return t

def _search_core(q: str, k: int, rerank: int, pool: int, source_filter: str = ""):
    pool = max(int(pool), int(k))
    hits = search_query(q, top_k=pool)
    chunk_ids = [h["chunk_id"] for h in hits]
    metadatas = get_chunks_with_docs_by_ids(chunk_ids)
    md_by_id = {m["chunk_id"]: m for m in metadatas}
    items = []
    for h in hits:
        cid = h["chunk_id"]
        md = md_by_id.get(cid)
        if not md:
            continue
        items.append({"chunk_id": cid, "faiss_score": float(h["score"]), "text": md["text"], "chunk_index": md["chunk_index"], "doc_title": md["doc_title"], "doc_path": md["doc_path"]})
    if source_filter:
        s = source_filter.lower()
        items = [it for it in items if s in (it["doc_title"] or "").lower() or s in (it["doc_path"] or "").lower()]
    if rerank:
        rr = get_reranker()
        if rr is not None and items:
            passages = [it["text"] for it in items]
            try:
                scores = rr.score(q, passages).tolist()
                for it, sc in zip(items, scores):
                    it["rerank_score"] = float(sc)
                items.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception:
                items.sort(key=lambda x: x["faiss_score"], reverse=True)
        else:
            items.sort(key=lambda x: x["faiss_score"], reverse=True)
    else:
        items.sort(key=lambda x: x["faiss_score"], reverse=True)
    return items[: int(k)]

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = Form(...), k: int = Form(5), rerank: int = Form(1), pool: int = Form(25), source: str = Form("")):
    t0 = time.time()
    items = _search_core(q, int(k), int(rerank), int(pool), source)
    results = []
    for it in items:
        results.append({
            "score": it.get("rerank_score", it["faiss_score"]),
            "chunk_id": it["chunk_id"],
            "chunk_index": it["chunk_index"],
            "doc_title": it["doc_title"],
            "doc_path": it["doc_path"],
            "highlight": _regex_highlight(it["text"], q)
        })
    elapsed = int((time.time() - t0) * 1000)
    return TEMPLATES.TemplateResponse("search_results.html", {"request": request, "results": results, "elapsed_ms": elapsed, "q": q, "k": int(k), "rerank": int(rerank), "pool": int(pool), "source": source})

@app.get("/chunk", response_class=HTMLResponse)
def chunk(cid: int = Query(..., alias="cid")):
    pack = get_chunk_with_neighbors(cid, window=1)
    if not pack:
        return HTMLResponse("<div class='muted'>Chunk not found.</div>", status_code=404)
    center = pack["center"]
    prev_id = None
    next_id = None
    for n in pack["neighbors"]:
        if n["chunk_index"] == center["chunk_index"] - 1:
            prev_id = n["chunk_id"]
        elif n["chunk_index"] == center["chunk_index"] + 1:
            next_id = n["chunk_id"]
    return TEMPLATES.TemplateResponse("chunk_card.html", {
        "doc_title": pack["doc_title"],
        "doc_path": pack["doc_path"],
        "center_index": center["chunk_index"],
        "center_text": center["text"],
        "prev_id": prev_id,
        "next_id": next_id,
        "request": None
    })

@app.post("/admin/reindex")
def admin_reindex():
    rows = get_all_chunks_texts()
    if not rows:
        rebuild_index(np.array([], dtype="int64"), np.zeros((0, 384), dtype="float32"))
        return {"ok": True, "indexed": 0}
    ids = [rid for rid, _ in rows]
    texts = [t for _, t in rows]
    hashes = [text_hash(t) for t in texts]
    cached = get_many(hashes)
    need_idx = [i for i, h in enumerate(hashes) if h not in cached]
    if need_idx:
        to_encode = [texts[i] for i in need_idx]
        vecs = _embedder.encode(to_encode)
        new_hashes = [hashes[i] for i in need_idx]
        put_many(new_hashes, vecs, 384)
        for h, v in zip(new_hashes, vecs):
            cached[h] = v
    ordered = np.vstack([cached[h] for h in hashes]).astype("float32")
    rebuild_index(np.array(ids, dtype="int64"), ordered)
    return {"ok": True, "indexed": int(len(ids))}

@app.get("/admin/list_docs")
def admin_list_docs():
    rows = list_documents()
    return {"documents": rows}

@app.post("/admin/delete_doc")
def admin_delete_doc(key: str = Form(...)):
    deleted = delete_document_by_title_or_path(key)
    if not deleted:
        return JSONResponse({"ok": False, "error": "document not found"}, status_code=404)
    admin_reindex()
    return {"ok": True, "deleted": key}

@app.get("/admin/list_uploads", response_class=HTMLResponse)
def admin_list_uploads(request: Request):
    up = Path("data/uploads")
    files = []
    if up.exists():
        for p in sorted(up.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file():
                files.append(p.name)
    return templates.TemplateResponse(
        "upload_list.html",
        {"request": request, "files": files}
    )

def _mk_md_export(q: str, items: list) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append(f"# Answer with Citations")
    lines.append("")
    lines.append(f"**Query:** {q}")
    lines.append(f"**Generated:** {ts}")
    lines.append("")
    if items:
        best = items[0]["text"].strip().replace("\n", " ")
        lines.append("## Short Answer (extractive)")
        lines.append("")
        lines.append(f"{best[:600]}{'…' if len(best)>600 else ''}")
        lines.append("")
    lines.append("## Evidence")
    lines.append("")
    for i, it in enumerate(items, 1):
        snippet = it["text"].strip().replace("\n", " ")
        snippet = snippet[:450] + ("…" if len(snippet) > 450 else "")
        lines.append(f"{i}. {snippet}")
        lines.append(f"   - Source: {it['doc_title']} · chunk {it['chunk_index']} · {it['doc_path']}")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    seen = set()
    for it in items:
        key = (it["doc_title"], it["doc_path"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {it['doc_title']} ({it['doc_path']})")
    lines.append("")
    return "\n".join(lines)

@app.post("/export_md")
def export_md(q: str = Form(...), k: int = Form(5), rerank: int = Form(1), pool: int = Form(25), source: str = Form("")):
    items = _search_core(q, int(k), int(rerank), int(pool), source)
    md = _mk_md_export(q, items)
    headers = {"Content-Disposition": "attachment; filename=answer_with_citations.md"}
    return Response(content=md, media_type="text/markdown", headers=headers)

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
