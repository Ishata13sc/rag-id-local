import hashlib
from typing import List, Dict
import numpy as np
from .db import get_conn

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_many(hashes: List[str]) -> Dict[str, np.ndarray]:
    if not hashes:
        return {}
    conn = get_conn()
    cur = conn.cursor()
    qmarks = ",".join("?" for _ in hashes)
    cur.execute(f"SELECT hash, dim, vec FROM emb_cache WHERE hash IN ({qmarks})", hashes)
    out = {}
    for h, dim, blob in cur.fetchall():
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size == dim:
            out[h] = arr
    conn.close()
    return out

def put_many(hashes: List[str], vectors: np.ndarray, dim: int):
    if len(hashes) == 0:
        return
    conn = get_conn()
    cur = conn.cursor()
    for h, vec in zip(hashes, vectors):
        cur.execute("INSERT OR REPLACE INTO emb_cache(hash, dim, vec) VALUES(?,?,?)", (h, dim, vec.astype("float32").tobytes()))
    conn.commit()
    conn.close()
