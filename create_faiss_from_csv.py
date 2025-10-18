#!/usr/bin/env python3
"""
Minimal streaming builder: Create FAISS IVF-PQ index from a CSV with [item_id, text].
Assumes your own modules:
  - Embedder:  .dim (int), .encode_texts(list[str], batch_size:int) -> np.ndarray [n, dim]
  - FaissIVFIndex: .train(np.ndarray), .add(np.ndarray[int64], np.ndarray), .save(str), .count() -> int
  - IndexConfig: constructor supports (dim, metric="cosine", nlist=..., pq_m=32, pq_bits=8)
"""

import sys
import csv
import random
import numpy as np
import pandas as pd
from pathlib import Path

from vector_index import Embedder, FaissIVFIndex, IndexConfig

# --------------------------- utils ---------------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32, order="C")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms

def reservoir_sample_texts(csv_path: str, text_col: str, k: int, chunksize: int = 50_000):
    """Reservoir sample k texts from a large CSV without loading it fully."""
    sample, seen = [], 0
    for chunk in pd.read_csv(csv_path, usecols=[text_col], chunksize=chunksize):
        for t in chunk[text_col].fillna(""):
            seen += 1
            if len(sample) < k:
                sample.append(t)
            else:
                j = random.randint(1, seen)  # 1..seen
                if j <= k:
                    sample[j - 1] = t
    return sample

def embed_texts_in_batches(embedder: Embedder, texts, batch_size: int = 64):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        v = embedder.encode_texts(batch, batch_size=batch_size)
        vecs.append(v)
    return np.vstack(vecs)

# --------------------------- builder ---------------------------

def build_ivfpq_from_csv(
    csv_path: str,
    text_col: str = "text",
    id_col: str = "item_id",
    train_samples: int = 50_000,
    nlist: int = 4096,
    pq_m: int = 32,
    pq_bits: int = 8,
    add_chunksize: int = 10_000,
    embed_batch: int = 64,
    out_prefix: str = "faiss_index_from_csv",
):
    # quick sanity
    if not Path(csv_path).exists():
        raise FileNotFoundError(csv_path)

    # ---------- pass 1: sample for training ----------
    print(f"[1/4] Sampling {train_samples} training texts from {csv_path} …")
    train_texts = reservoir_sample_texts(csv_path, text_col, train_samples)
    print(f"Sampled {len(train_texts)} texts")

    # ---------- embed + normalize training ----------
    print("[2/4] Embedding training texts …")
    embedder = Embedder()
    train_vecs = embed_texts_in_batches(embedder, train_texts, batch_size=embed_batch)
    train_vecs = l2_normalize(train_vecs)  # cosine → normalize
    dim = train_vecs.shape[1]
    print(f"Training matrix: {train_vecs.shape}")

    # heuristic warn (optional)
    need = 30 * nlist
    if train_vecs.shape[0] < need:
        print(f"Warning: train_samples={train_vecs.shape[0]} < ~30*nlist={need}. "
              f"Index will still work, but more training data may improve quality.")

    # ---------- build + train index ----------
    print("[3/4] Creating & training IVF-PQ index …")
    cfg = IndexConfig(dim=dim, metric="cosine", nlist=nlist, pq_m=pq_m, pq_bits=pq_bits)
    index = FaissIVFIndex(cfg)
    index.train(train_vecs)

    # free training buffers
    del train_texts, train_vecs

    # ---------- pass 2: add all items in chunks ----------
    print("[4/4] Adding all vectors to index …")
    total = 0
    usecols = [id_col, text_col]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=add_chunksize):
        ids = chunk[id_col].astype("int64").to_numpy(copy=False)
        texts = chunk[text_col].fillna("").tolist()

        vecs = embed_texts_in_batches(embedder, texts, batch_size=embed_batch)
        vecs = l2_normalize(vecs)
        index.add(ids, vecs)
        total += len(ids)
        print(f"  +{len(ids)} → ntotal={total}")

    # save
    out_path = f"{out_prefix}_nlist{nlist}_pq{pq_m}x{pq_bits}"
    index.save(out_path)
    print(f"Done. Saved index at: {out_path}")
    print(f"ntotal={index.count()}")

    return index

# --------------------------- cli ---------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_faiss_ivfpq.py <csv_path> [train_samples] [nlist]")
        sys.exit(1)

    csv_path = sys.argv[1]
    train_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    nlist = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

    build_ivfpq_from_csv(
        csv_path=csv_path,
        train_samples=train_samples,
        nlist=nlist,
        # tweak below only if needed:
        pq_m=32,
        pq_bits=8,
        add_chunksize=10_000,
        embed_batch=64,
    )
