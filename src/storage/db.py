from pathlib import Path
import sqlite3
from typing import List, Optional, Dict, Any, Tuple

DB_PATH = Path("rag.sqlite")

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT UNIQUE,
  title TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
    cur.execute("""
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER,
  chunk_index INTEGER,
  text TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(id)
);
""")
    cur.execute("""
CREATE TABLE IF NOT EXISTS vectors(
  chunk_id INTEGER PRIMARY KEY,
  dim INTEGER,
  FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);
""")
    cur.execute("""
CREATE TABLE IF NOT EXISTS emb_cache(
  hash TEXT PRIMARY KEY,
  dim INTEGER,
  vec BLOB
);
""")
    conn.commit()
    conn.close()

def upsert_document(path: str, title: Optional[str] = None) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO documents(path, title) VALUES(?, ?)", (path, title))
    conn.commit()
    cur.execute("SELECT id FROM documents WHERE path = ?", (path,))
    row = cur.fetchone()
    conn.close()
    return int(row[0])

def insert_chunks(doc_id: int, chunks: List[str]) -> List[int]:
    conn = get_conn()
    cur = conn.cursor()
    ids = []
    for i, text in enumerate(chunks):
        cur.execute("INSERT INTO chunks(doc_id, chunk_index, text) VALUES(?,?,?)", (doc_id, i, text))
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    return [int(x) for x in ids]

def mark_vectors(chunk_ids: List[int], dim: int):
    if not chunk_ids:
        return
    conn = get_conn()
    cur = conn.cursor()
    for cid in chunk_ids:
        cur.execute("INSERT OR REPLACE INTO vectors(chunk_id, dim) VALUES(?, ?)", (cid, dim))
    conn.commit()
    conn.close()

def get_chunks_with_docs_by_ids(ids: List[int]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    conn = get_conn()
    cur = conn.cursor()
    qmarks = ",".join("?" for _ in ids)
    order_case = " ".join([f"WHEN ? THEN {i}" for i, _ in enumerate(ids)])
    cur.execute(f"""
SELECT c.id, c.text, c.chunk_index, d.title, d.path, d.id
FROM chunks c
JOIN documents d ON c.doc_id = d.id
WHERE c.id IN ({qmarks})
ORDER BY CASE c.id {order_case} END
""", ids + ids)
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "chunk_id": int(r[0]),
            "text": r[1],
            "chunk_index": int(r[2]),
            "doc_title": r[3],
            "doc_path": r[4],
            "doc_id": int(r[5]),
        })
    return out

def get_chunk_with_neighbors(chunk_id: int, window: int = 1) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
SELECT c.id, c.doc_id, c.chunk_index, c.text, d.title, d.path
FROM chunks c
JOIN documents d ON c.doc_id = d.id
WHERE c.id = ?
""", (chunk_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return {}
    cid, doc_id, idx, text, title, path = int(row[0]), int(row[1]), int(row[2]), row[3], row[4], row[5]
    cur.execute("""
SELECT id, chunk_index, text
FROM chunks
WHERE doc_id = ? AND chunk_index BETWEEN ? AND ?
ORDER BY chunk_index
""", (doc_id, max(0, idx - window), idx + window))
    neighbors = [{"chunk_id": int(r[0]), "chunk_index": int(r[1]), "text": r[2]} for r in cur.fetchall()]
    conn.close()
    return {
        "doc_id": doc_id,
        "doc_title": title,
        "doc_path": path,
        "center": {"chunk_id": cid, "chunk_index": idx, "text": text},
        "neighbors": neighbors
    }

def get_all_chunks_texts() -> List[Tuple[int, str]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM chunks ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [(int(r[0]), r[1]) for r in rows]

def list_documents() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
SELECT d.id, d.title, d.path, COUNT(c.id)
FROM documents d
LEFT JOIN chunks c ON c.doc_id = d.id
GROUP BY d.id, d.title, d.path
ORDER BY d.id
""")
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({"id": int(r[0]), "title": r[1], "path": r[2], "chunks": int(r[3])})
    return out

def delete_document_by_title_or_path(key: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE title = ? OR path = ?", (key, key))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False
    doc_id = int(row[0])
    cur.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_ids = [int(r[0]) for r in cur.fetchall()]
    if chunk_ids:
        qmarks = ",".join("?" for _ in chunk_ids)
        cur.execute(f"DELETE FROM vectors WHERE chunk_id IN ({qmarks})", chunk_ids)
        cur.execute(f"DELETE FROM chunks WHERE id IN ({qmarks})", chunk_ids)
    cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return True
