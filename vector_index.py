# vector_index.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from settings import MODEL_NAME

# ---------- small utils ----------

def _as_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype="float32", order="C")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = _as_float32(mat)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

@dataclass
class IndexConfig:
    dim: int
    nlist: int = 1024               # number of IVF clusters
    total: int = 0
    # ---- Add PQ parameters ----
    pq_m: Optional[int] = None                # how many sub-vectors
    pq_bits: int = 8                # bits per sub-vector (default 8)
    extras: dict = None             # extra model info

    def __post_init__(self):
        if self.pq_m is None:
            raise ValueError("pq_m must be set.")
        if self.dim % self.pq_m != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by pq_m ({self.pq_m}).")
        if self.pq_bits not in (4,5,6,7,8):
            raise ValueError("pq_bits must be in {4,5,6,7,8} (8 is typical).")

    def to_json(self) -> str:
        return json.dumps({
            "dim": self.dim,
            "nlist": self.nlist,
            "total": self.total,
            "pq_m": self.pq_m,
            "pq_bits": self.pq_bits,
            "extras": self.extras or {},
        })

    @staticmethod
    def from_json(s: str) -> "IndexConfig":
        obj = json.loads(s)
        return IndexConfig(
            dim=int(obj["dim"]),
            nlist=int(obj["nlist"]),
            total=int(obj["total"]),
            pq_m=obj.get("pq_m"),
            pq_bits=obj.get("pq_bits", 8),
            extras=obj.get("extras", {}),
        )


# ---------- Embedder (swap-friendly) ----------

class Embedder:
    """
    Minimal wrapper around Sentence-Transformers.
    Choose cosine by default; we unit-normalize when metric="cosine".
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        embs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embs

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text])

# ---------- FAISS IVF index with explicit ID mapping ----------

class FaissIVFIndex:
    """
    - Cosine similarity is implemented as L2 on unit-normalized vectors.
    - Uses IndexIDMap2 so returned IDs are your item_ids (int64).
    """

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        quantizer = faiss.IndexFlatL2(cfg.dim)
        base = faiss.IndexIVFPQ(quantizer, cfg.dim, cfg.nlist, cfg.pq_m, cfg.pq_bits)
        # we want to store our own IDs, so we wrap the index in an IDMap2.
        # ivf manages vectors internally with rowId, different from our item_ids.
        self.index = faiss.IndexIDMap2(base)

    # ---- training / add ----

    def train(self, train_vectors: np.ndarray) -> None:
        x = _as_float32(train_vectors)
        x = _l2_normalize(x)   # defensively normalize here
        ivf_pq = faiss.downcast_index(self.index.index)
        if not self.is_trained():
            ivf_pq.train(x)

    def add(self, item_ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(item_ids, dtype="int64")
        x = _as_float32(vectors)
        x = _l2_normalize(x)   # defensively normalize here
        if not self.is_trained():
            raise RuntimeError("Index not trained. Call train() first.")
        self.index.add_with_ids(x, ids)
        self.cfg.total = int(self.index.ntotal)

    # ---- search ----

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        nprobe: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Minimal search:
        - Set nprobe if provided,
        - Optionally restrict to a small whitelist of IDs via IDSelectorBatch (minimal stand-in for bitsets),
        - Return (distances, item_ids).
        """
        q = _as_float32(query_vec)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q = _l2_normalize(q)

        ivf_pq = faiss.downcast_index(self.index.index)
        old_nprobe = ivf_pq.nprobe
        ivf_pq.nprobe = 16 if nprobe is None else int(nprobe)
        try:
            distances, indices = self.index.search(q, k)
        finally:
            ivf_pq.nprobe = old_nprobe
        return distances, indices


    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "vectors.index"))
        with open(os.path.join(dirpath, "index_meta.json"), "w") as f:
            f.write(self.cfg.to_json())

    @staticmethod
    def load(dirpath: str) -> "FaissIVFIndex":
        index = faiss.read_index(os.path.join(dirpath, "vectors.index"))
        with open(os.path.join(dirpath, "index_meta.json"), "r") as f:
            cfg = IndexConfig.from_json(f.read())
        obj = FaissIVFIndex.__new__(FaissIVFIndex)  # no __init__
        obj.cfg = cfg
        obj.index = index
        obj.cfg.total = int(obj.index.ntotal)
        return obj

    # ---- info ----

    def count(self) -> int:
        return int(self.index.ntotal)

    def is_trained(self) -> bool:
        return bool(faiss.downcast_index(self.index.index).is_trained)

