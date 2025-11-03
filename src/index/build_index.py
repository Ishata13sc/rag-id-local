from pathlib import Path
from typing import Sequence
import numpy as np
import faiss

INDEX_DIR = Path("index")
INDEX_PATH = INDEX_DIR / "faiss.index"
DIM_PATH = INDEX_DIR / "dim.txt"

def _mk_index(dim: int):
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap(base)

def _save_index(index):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

def _load_index():
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return None

def _save_dim(dim: int):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DIM_PATH.write_text(str(dim))

def _load_dim(default: int = 384) -> int:
    if DIM_PATH.exists():
        try:
            return int(DIM_PATH.read_text().strip())
        except Exception:
            return default
    return default

def ensure_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        dim = _load_dim(384)
        idx = _mk_index(dim)
        _save_index(idx)
        _save_dim(dim)

def append_to_index(ids: Sequence[int], vecs: np.ndarray):
    if vecs.ndim != 2 or len(ids) != vecs.shape[0]:
        raise ValueError("shape mismatch")
    dim = vecs.shape[1]
    ids_np = np.asarray(ids, dtype="int64")
    v = vecs.astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    v = v / n
    idx = _load_index()
    if idx is None:
        idx = _mk_index(dim)
    else:
        if idx.d != dim:
            idx = _mk_index(dim)
    idx.add_with_ids(v, ids_np)
    _save_index(idx)
    _save_dim(dim)

def rebuild_index(ids: np.ndarray, vecs: np.ndarray):
    if vecs.size == 0 or ids.size == 0:
        dim = 384
        idx = _mk_index(dim)
        _save_index(idx)
        _save_dim(dim)
        return
    if vecs.ndim != 2 or ids.ndim != 1 or vecs.shape[0] != ids.shape[0]:
        raise ValueError("shape mismatch")
    dim = vecs.shape[1]
    ids64 = ids.astype("int64", copy=False)
    v = vecs.astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    v = v / n
    idx = _mk_index(dim)
    idx.add_with_ids(v, ids64)
    _save_index(idx)
    _save_dim(dim)
