"""Encode every table row once and cache the vectors (Stage 2's input).

Stage 2 compares each query against the rows of its candidate tables. Encoding
those rows for every query set would be wasteful, so we encode them once here
and memory-map the result at query time.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from craft_tabqa.core.dense import encode_texts
from craft_tabqa.core.files import load_pickle, save_pickle


def encode_rows(
    row_texts: List[str],
    row_meta: List[Dict],
    output_dir: str,
    model_id: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 256,
    hf_cache: str = "",
    trust_remote_code: bool = False,
    logger=None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Encode ``row_texts`` and save ``row_embeddings.npy`` + ``row_meta.pkl``.

    Returns ``(embeddings, row_meta)``. If the cache already exists it is loaded
    (embeddings memory-mapped) instead of recomputed.
    """
    log = logger.info if logger else print
    output_dir = Path(output_dir)
    emb_path = output_dir / "row_embeddings.npy"
    meta_path = output_dir / "row_meta.pkl"

    if emb_path.exists() and meta_path.exists():
        log(f"[rows] Cached row embeddings found at {emb_path}")
        return np.load(emb_path, mmap_mode="r"), load_pickle(meta_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[rows] Encoding {len(row_texts):,} rows with {model_id} ...")

    start = time.time()
    embeddings = encode_texts(
        row_texts,
        model_id=model_id,
        batch_size=batch_size,
        hf_cache=hf_cache,
        trust_remote_code=trust_remote_code,
    )
    log(f"[rows] Encoded in {time.time() - start:.1f}s (shape {embeddings.shape})")

    np.save(emb_path, embeddings)
    save_pickle(meta_path, row_meta)
    log(f"[rows] Saved embeddings to {emb_path} and meta to {meta_path}")
    return embeddings, row_meta
