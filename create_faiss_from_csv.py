#!/usr/bin/env python3
import faiss # do not remove this import, there is an apple silicon issue with faiss's importing order
import random
import numpy as np
import pandas as pd
from pathlib import Path
from settings import FAISS_PATH, SEED, TEXT_COL, TRAIN_MAX, NLIST, PQ_M, PQ_BITS, ITEM_COL_ID, OUT_CSV, CHUNK_SIZE, EMBED_BATCH
from vector_index import Embedder, FaissIVFIndex, IndexConfig

# --------------------------- utils ---------------------------

def reservoir_sample_texts(csv_path: str, text_col: str, k: int, chunksize: int = 10_000):
    """Reservoir sample k texts from a large CSV without loading it fully."""
    sample, seen = [], 0
    for idx, chunk in enumerate(pd.read_csv(csv_path, usecols=[text_col], chunksize=chunksize)):
        print(f"Processing chunk {idx}")
        for t in chunk[text_col].fillna(""):
            seen += 1
            if len(sample) < k:
                sample.append(t)
            else:
                j = random.randint(1, seen)  # 1..seen
                if j <= k:
                    sample[j - 1] = t
    return sample

# --------------------------- builder ---------------------------

def build_ivfpq_from_csv(
    csv_path: str,
    text_col: str,
    id_col: str,
    train_samples: int,
    nlist: int,
    pq_m: int,
    pq_bits: int,
    add_chunksize: int,
    embed_batch: int,
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
    train_vecs = embedder.encode_texts(train_texts, batch_size=embed_batch)
    dim = train_vecs.shape[1]
    print(f"Training matrix: {train_vecs.shape}")

    # heuristic warn (optional)
    need = 100 * nlist
    if train_vecs.shape[0] < need:
        print(f"Warning: train_samples={train_vecs.shape[0]} < ~100*nlist={need}. "
              f"Index will still work, but more training data may improve quality.")

    # ---------- build + train index ----------
    print("[3/4] Creating & training IVF-PQ index …")
    cfg = IndexConfig(dim=dim, nlist=nlist, pq_m=pq_m, pq_bits=pq_bits)
    index = FaissIVFIndex(cfg)
    index.train(train_vecs)

    # free training buffers
    del train_texts, train_vecs

    # ---------- pass 2: add all items in chunks ----------
    print("[4/4] Adding all vectors to index …")
    total = 0
    usecols = [id_col, text_col]
    for idx, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=add_chunksize)):
        print(f"Processing chunk {idx}")
        ids = chunk[id_col].astype("int64").to_numpy(copy=False)
        texts = chunk[text_col].fillna("").tolist()
        vecs = embedder.encode_texts(texts, batch_size=embed_batch)
        index.add(ids, vecs)
        total += len(ids)
        print(f"  +{len(ids)} → ntotal={total}")

    # save
    index.save(FAISS_PATH)
    print(f"Done. Saved index at: {FAISS_PATH}")
    print(f"ntotal={index.count()}")

    return index

# --------------------------- cli ---------------------------

if __name__ == "__main__":
    random.seed(SEED)   
    np.random.seed(SEED)
    
    build_ivfpq_from_csv(
        csv_path=OUT_CSV,
        text_col=TEXT_COL,
        id_col=ITEM_COL_ID,
        train_samples=TRAIN_MAX,
        nlist=NLIST,
        pq_m=PQ_M,
        pq_bits=PQ_BITS,
        add_chunksize=CHUNK_SIZE,
        embed_batch=EMBED_BATCH,
    )
