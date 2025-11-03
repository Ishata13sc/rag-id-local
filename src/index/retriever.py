from typing import List, Dict
import numpy as np
import faiss
from pathlib import Path
from src.models.embedder import Embedder

INDEX_PATH = Path("index/faiss.index")

_embedder = Embedder()

def _load():
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return None

def _norm(vec: np.ndarray) -> np.ndarray:
    v = vec.astype("float32", copy=False)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def _fit_dim(v: np.ndarray, d: int) -> np.ndarray:
    if v.shape[1] == d:
        return v
    out = np.zeros((v.shape[0], d), dtype="float32")
    m = min(d, v.shape[1])
    out[:, :m] = v[:, :m]
    return out

def search_query(q: str, top_k: int = 5) -> List[Dict]:
    idx = _load()
    if idx is None or idx.ntotal == 0:
        return []
    v = _embedder.encode([q])
    v = _fit_dim(v, int(idx.d))
    v = _norm(v)
    dists, ids = idx.search(v, max(top_k, 10))
    out: List[Dict] = []
    for cid, sc in zip(ids[0].tolist(), dists[0].tolist()):
        if int(cid) == -1:
            continue
        out.append({"chunk_id": int(cid), "score": float(sc)})
    return out[:top_k]
