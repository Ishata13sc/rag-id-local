import os
from typing import List
import numpy as np
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = os.getenv("RAG_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

class Reranker:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        if os.getenv("RAG_OFFLINE", "").strip() == "1":
            os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        name = (os.getenv("RAG_RERANKER_LOCAL_PATH", "").strip() or None) or (model_name or DEFAULT_MODEL)
        trust = True if os.getenv("RAG_TRUST_REMOTE_CODE", "1").strip() == "1" else False
        self.model = CrossEncoder(name, device=device, trust_remote_code=trust)

    def score(self, query: str, passages: List[str]) -> np.ndarray:
        if not passages:
            return np.zeros((0,), dtype=np.float32)
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        return scores.astype("float32")
