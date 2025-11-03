import os
from typing import List
import numpy as np

BACKEND = os.getenv("RAG_EMBEDDER_BACKEND", "sbert").lower()

if BACKEND == "tfidf":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import joblib
    from pathlib import Path
    from src.storage.db import get_all_chunks_texts

    MODEL_DIR = Path("data") / "tfidf_model"
    VEC_PATH = MODEL_DIR / "tfidf.joblib"
    SVD_PATH = MODEL_DIR / "svd.joblib"
    DIM = int(os.getenv("RAG_TFIDF_DIM", "384"))

    class Embedder:
        def __init__(self, *args, **kwargs):
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.vec = None
            self.svd = None
            if VEC_PATH.exists() and SVD_PATH.exists():
                self.vec = joblib.load(VEC_PATH)
                self.svd = joblib.load(SVD_PATH)

        def _ensure_fitted(self):
            if self.vec is not None and self.svd is not None:
                return
            texts = [t for _, t in get_all_chunks_texts()]
            if not texts:
                texts = ["dummy"]
            self.vec = TfidfVectorizer(max_features=50000)
            X = self.vec.fit_transform(texts)
            k = min(DIM, X.shape[1]-1) if X.shape[1] > 1 else 1
            self.svd = TruncatedSVD(n_components=k, random_state=0)
            self.svd.fit(X)
            joblib.dump(self.vec, VEC_PATH)
            joblib.dump(self.svd, SVD_PATH)

        def encode(self, texts: List[str]) -> np.ndarray:
            self._ensure_fitted()
            X = self.vec.transform(texts)
            Z = self.svd.transform(X)
            n = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
            return (Z / n).astype("float32")
else:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if os.getenv("RAG_OFFLINE", "").strip() == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = os.getenv("RAG_EMBEDDER_LOCAL_PATH", "").strip() or "intfloat/multilingual-e5-small"

    class Embedder:
        def __init__(self, model_name: str = None, device: str = "cpu"):
            name = model_name or MODEL_NAME
            self.model = SentenceTransformer(name, device=device)
            try:
                self.model.max_seq_length = 512
            except Exception:
                pass

        def encode(self, texts: List[str]) -> np.ndarray:
            return self.model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
