from pathlib import Path
from typing import List
import re

from src.ingest.splitter import split_text
from src.storage.db import upsert_document, insert_chunks, mark_vectors

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_md(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"`{1,3}.*?`{1,3}", " ", raw, flags=re.S)
    raw = re.sub(r"^#{1,6}\s*", "", raw, flags=re.M)
    return raw

def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("pypdf is required for PDF") from e
    reader = PdfReader(str(path))
    pages: List[str] = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
    return "\n\n".join(pages)

def _read_docx(path: Path) -> str:
    try:
        import docx
    except Exception as e:
        raise RuntimeError("python-docx is required for DOCX") from e
    d = docx.Document(str(path))
    return "\n".join([p.text for p in d.paragraphs])

def _norm_ext(name: str) -> str:
    s = name.strip()
    while "  " in s:
        s = s.replace("  ", " ")
    s = s.replace("\u00A0", " ")
    return Path(s).suffix.lower()

def _read_any(path: Path) -> str:
    ext = _norm_ext(path.name)
    if ext in (".txt",):
        return _read_txt(path)
    if ext in (".md", ".markdown"):
        return _read_md(path)
    if ext in (".pdf",):
        return _read_pdf(path)
    if ext in (".docx",):
        return _read_docx(path)
    raise ValueError(f"Unsupported file type: {ext or '<none>'}")

def ingest_path(path: Path, embedder) -> int:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = _read_any(path)
    chunks = split_text(text)
    if not chunks:
        return 0
    doc_id = upsert_document(str(path), title=path.name)
    chunk_ids = insert_chunks(doc_id, chunks)
    vecs = embedder.encode(chunks)
    mark_vectors(chunk_ids, vecs.shape[1])
    from src.index.build_index import append_to_index
    append_to_index(chunk_ids, vecs)
    return len(chunks)
