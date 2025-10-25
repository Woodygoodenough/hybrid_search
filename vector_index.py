# vector_index.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from settings import MODEL_NAME, NPROBE

@dataclass
class FilterIds:
    ids: np.ndarray
    @staticmethod
    def from_list(ids: List[int]) -> "FilterIds":
        return FilterIds(ids=np.asarray(ids, dtype="int64"))
    def to_faiss(self) -> faiss.IDSelector:
        return faiss.IDSelectorBatch(self.ids)

@dataclass
class AnnSearchResults:
    """
    Container for Approximate Nearest Neighbor search results.

    This class handles the results from FAISS vector similarity search, including
    proper handling of empty results and dimension management.

    Attributes:
        distances: np.ndarray of shape (n,) containing similarity distances (lower is better)
        item_ids: np.ndarray of shape (n,) containing item IDs from the database

    Dimension Behavior:
        - When FAISS finds results: distances.shape = (k,), item_ids.shape = (k,)
        - When FAISS finds 1 result: distances.shape = (), item_ids.shape = () [0-dimensional]
        - When no results found: distances.shape = (0,), item_ids.shape = (0,) [empty 1-dimensional]

    Length Behavior:
        - len(result) returns the number of valid (non -1) results
        - Empty results (all -1) have length 0
        - 0-dimensional arrays (single result) have length 1
        - 1-dimensional arrays have length equal to their size
    """
    distances: np.ndarray
    item_ids: np.ndarray

    def __post_init__(self):
        """
        Post-initialization processing of search results.

        Handles dimension normalization and filtering of invalid results:
        1. Reshapes 0-dimensional arrays to 1-dimensional for consistency
        2. Filters out results with item_id = -1 (FAISS convention for "no result")
        3. Maintains parallel arrays between distances and item_ids
        """
        # Handle 0-dimensional arrays (single result from FAISS)
        if self.distances.ndim == 0:
            self.distances = self.distances.reshape(1)
        if self.item_ids.ndim == 0:
            self.item_ids = self.item_ids.reshape(1)

        # Create mask for valid (non -1) item ids before filtering
        # FAISS uses -1 to indicate "no result found at this position"
        valid_mask = self.item_ids != -1

        # Keep only the valid results (where item_id != -1)
        self.item_ids = self.item_ids[valid_mask]
        self.distances = self.distances[valid_mask]

    @classmethod
    def empty(cls) -> "AnnSearchResults":
        """
        Create an empty AnnSearchResults instance representing no search results.

        This is the preferred way to create empty results instead of using
        np.zeros(0) which can cause dimension confusion.

        Returns:
            AnnSearchResults: Empty instance with no results
        """
        return cls(
            distances=np.array([], dtype=np.float32),
            item_ids=np.array([], dtype=np.int64)
        )

    def is_empty(self) -> bool:
        """
        Check if this result set contains any valid results.

        Returns:
            bool: True if no valid results found, False otherwise
        """
        return len(self.item_ids) == 0

    def to_dict(self) -> Dict[int, float]:
        """
        Convert results to a dictionary mapping item_id to similarity score.

        This method handles all dimension cases correctly:
        - Empty results: returns empty dict {}
        - Single result (0-dim): returns {item_id: similarity}
        - Multiple results (1-dim): returns {item_id1: sim1, item_id2: sim2, ...}

        Returns:
            Dict[int, float]: Mapping of item_id to similarity score
        """
        # All -1 values should already be filtered out in __post_init__

        if self.is_empty():
            return {}

        # Handle 0-dimensional arrays (when k=1, FAISS returns shape ())
        if self.item_ids.ndim == 0:
            item_ids = [int(self.item_ids)]
            distances = [float(self.distances)]
        else:
            # Handle 1-dimensional arrays (normal case)
            item_ids = self.item_ids.tolist()
            distances = self.distances.tolist()

        return {item_id: similarity for item_id, similarity in zip(item_ids, distances)}
    
    def __len__(self) -> int:
        """
        Return the number of valid results found.

        This method handles different numpy array dimensions correctly:
        - Empty arrays (shape (0,)): return 0
        - 0-dimensional arrays (shape ()): return 1 (single result)
        - 1-dimensional arrays (shape (n,)): return n

        Returns:
            int: Number of valid results (item_ids != -1)
        """
        return len(self.item_ids)


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
    - Cosine similarity using inner product on unit-normalized vectors.
    - Uses IndexIDMap2 so returned IDs are your item_ids (int64).
    - Returns values in range [-1, 1] where higher = more similar.
    """

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        # Use inner product (cosine similarity) for normalized vectors
        quantizer = faiss.IndexFlatIP(cfg.dim)
        self.index = faiss.IndexIVFPQ(quantizer, cfg.dim, cfg.nlist, cfg.pq_m, cfg.pq_bits, faiss.METRIC_INNER_PRODUCT)
        self.nprobe = NPROBE
    # ---- training / add ----

    def train(self, train_vectors: np.ndarray) -> None:
        x = _as_float32(train_vectors)
        x = _l2_normalize(x)   # defensively normalize here
        if not self.index.is_trained:
            self.index.train(x)

    def add(self, item_ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(item_ids, dtype="int64")
        x = _as_float32(vectors)
        x = _l2_normalize(x)   # defensively normalize here
        if not self.index.is_trained:
            raise RuntimeError("Index not trained. Call train() first.")
        self.index.add_with_ids(x, ids)
        self.cfg.total = int(self.index.ntotal)

    # ---- search ----

    # search should support multiple queries. we disable it for now.
    def search(self, query_vec: np.ndarray, k: int,
            nprobe: Optional[int] = None,
            item_ids: Optional[List[int]] = None) -> AnnSearchResults:
        
        q = _l2_normalize(_as_float32(query_vec))
        if q.ndim == 1:
            q = q.reshape(1, -1)
        filter_ids = FilterIds.from_list(item_ids) if item_ids else None
        id_selector = filter_ids.to_faiss() if filter_ids else None
        if nprobe is None:
            nprobe = self.nprobe
        params = faiss.IVFPQSearchParameters(sel=id_selector, nprobe=nprobe)
        distances, indices = self.index.search(q, k, params=params)
        return AnnSearchResults(distances=distances.squeeze(), item_ids=indices.squeeze())


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
        obj.nprobe = 16  # Initialize nprobe
        obj.cfg.total = int(obj.index.ntotal)
        return obj

    # ---- info ----

    def count(self) -> int:
        return int(self.index.ntotal)

    