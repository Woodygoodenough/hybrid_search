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
    metric: str = "cosine"          # "cosine" (via L2 on unit vectors) or "l2"
    index_type: str = "ivf_flat"    # "ivf_flat", "ivf_pq"
    nlist: int = 1024               # number of IVF clusters
    trained: bool = False
    total: int = 0

    # ---- Add PQ parameters ----
    pq_m: int = None                # how many sub-vectors
    pq_bits: int = 8                # bits per sub-vector (default 8)
    extras: dict = None             # extra model info

    def to_json(self) -> str:
        return json.dumps({
            "dim": self.dim,
            "metric": self.metric,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "trained": self.trained,
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
            metric=obj["metric"],
            index_type=obj["index_type"],
            nlist=int(obj["nlist"]),
            trained=bool(obj["trained"]),
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
    def __init__(self, model_name: str = MODEL_NAME, metric: str = "cosine"):
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()
        self.metric = metric

    @property
    def dim(self) -> int:
        return self._dim

    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        embs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=(self.metric == "cosine"),
        )
        return embs

    def encode_query(self, text: str) -> np.ndarray:
        v = self.encode_texts([text])
        return v  # shape (1, dim)


# ---------- FAISS IVF index with explicit ID mapping ----------

class FaissIVFIndex:
    """
    Minimal IVF-Flat index with stable ID mapping and persistence.
    - Cosine similarity is implemented as L2 on unit-normalized vectors.
    - Uses IndexIDMap2 so returned IDs are your item_ids (int64).
    """

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        # Choose metric
        if cfg.metric == "cosine":
            self.faiss_metric = faiss.METRIC_L2
        elif cfg.metric == "l2":
            self.faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError("metric must be 'cosine' or 'l2' (minimal implementation).")

        # Build the coarse quantizer (flat) and IVF-Flat container
        quantizer = faiss.IndexFlatL2(cfg.dim)
        ivf = faiss.IndexIVFFlat(quantizer, cfg.dim, cfg.nlist, self.faiss_metric)

        # Wrap in IDMap2 to store your IDs explicitly
        self.index = faiss.IndexIDMap2(ivf)

    # ---- training / add ----

    def train(self, train_vectors: np.ndarray) -> None:
        x = _as_float32(train_vectors)
        if self.cfg.metric == "cosine":
            x = _l2_normalize(x)
        ivf = faiss.downcast_index(self.index.index)  # unwrap to IVF piece
        if not ivf.is_trained:
            ivf.train(x)
        self.cfg.trained = bool(ivf.is_trained)

    def add(self, item_ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(item_ids, dtype="int64")
        x = _as_float32(vectors)
        if self.cfg.metric == "cosine":
            x = _l2_normalize(x)
        if not faiss.downcast_index(self.index.index).is_trained:
            raise RuntimeError("Index not trained. Call train() first.")
        self.index.add_with_ids(x, ids)
        self.cfg.total = int(self.index.ntotal)

    # ---- search ----

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        nprobe: Optional[int] = None,
        allow_ids: Optional[Iterable[int]] = None,
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
        if self.cfg.metric == "cosine":
            q = _l2_normalize(q)

        ivf = faiss.downcast_index(self.index.index)

        # Set nprobe
        old_nprobe = ivf.nprobe
        if nprobe is not None:
            ivf.nprobe = int(nprobe)

        # Optional allowlist via IDSelectorBatch (good for small sets; for big masks, switch to bitsets later)
        params = None
        selector = None
        if allow_ids is not None:
            allow_ids = np.asarray(list(allow_ids), dtype="int64")
            selector = faiss.IDSelectorBatch(allow_ids.size, faiss.swig_ptr(allow_ids))
            params = faiss.SearchParametersIVF()
            params.sel = selector

        try:
            if params is None:
                D, I = self.index.search(q, k)
            else:
                # use the IVF search with parameters (ID selector)
                D, I = ivf.search(q, k, params)
        finally:
            # restore original nprobe
            ivf.nprobe = old_nprobe

        return D, I  # distances, item_ids (these are your IDs via IDMap2)

    # ---- persistence ----

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
        obj = FaissIVFIndex(cfg)
        obj.index = index  # already an IDMap2 if saved that way
        return obj

    # ---- info ----

    def count(self) -> int:
        return int(self.index.ntotal)

    def is_trained(self) -> bool:
        return bool(faiss.downcast_index(self.index.index).is_trained)


# ---------- Flat index for small datasets ----------

class IndexFlat:
    """
    Simple flat index for small datasets (no clustering needed).
    """

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        # Choose metric
        if cfg.metric == "cosine":
            self.faiss_metric = faiss.METRIC_L2
        elif cfg.metric == "l2":
            self.faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError("metric must be 'cosine' or 'l2'")

        # Create flat index
        if cfg.metric == "cosine":
            # For cosine, we'll normalize manually and use L2
            index = faiss.IndexFlatL2(cfg.dim)
        else:
            index = faiss.IndexFlatL2(cfg.dim)

        # Wrap in IDMap2 to store IDs
        self.index = faiss.IndexIDMap2(index)

    def train(self, train_vectors: np.ndarray) -> None:
        # Flat index doesn't need training
        pass

    def add(self, item_ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(item_ids, dtype="int64")
        x = _as_float32(vectors)
        if self.cfg.metric == "cosine":
            x = _l2_normalize(x)
        self.index.add_with_ids(x, ids)
        self.cfg.total = int(self.index.ntotal)

    def search(self, query_vec: np.ndarray, k: int, nprobe: Optional[int] = None, allow_ids: Optional[Iterable[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        q = _as_float32(query_vec)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if self.cfg.metric == "cosine":
            q = _l2_normalize(q)

        # For flat index, ignore nprobe and allow_ids for simplicity
        D, I = self.index.search(q, k)
        return D, I

    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "vectors.index"))
        with open(os.path.join(dirpath, "index_meta.json"), "w") as f:
            f.write(self.cfg.to_json())

    @staticmethod
    def load(dirpath: str) -> "IndexFlat":
        index = faiss.read_index(os.path.join(dirpath, "vectors.index"))
        with open(os.path.join(dirpath, "index_meta.json"), "r") as f:
            cfg = IndexConfig.from_json(f.read())
        obj = IndexFlat(cfg)
        obj.index = index
        return obj

    def count(self) -> int:
        return int(self.index.ntotal)
