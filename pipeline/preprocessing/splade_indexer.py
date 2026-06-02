"""
Build the SPLADE inverted index for Stage 1 retrieval.

This is an offline step: you run it once per corpus and cache the result.
At query time, Stage 1 only needs the saved index file.

The heavy lifting (multi-GPU sparse encoding, inverted index construction)
is done by the existing utils/splade_utils.py.  This module adds a clean
save/load wrapper around it so preprocess.py stays readable.
"""

import gc
import time
from pathlib import Path
from typing import List


def build_splade_index(
    corpus_texts: List[str],
    output_path: str,
    model_id: str = "naver/splade_v2_distil",
    batch_size: int = 128,
    sparse_top_k: int = 5_000,
    hf_cache: str = "",
    logger=None,
) -> dict:
    """
    Encode all corpus texts with SPLADE and build an inverted index.

    Uses all GPUs visible via CUDA_VISIBLE_DEVICES automatically (multi-GPU
    support is built into utils/splade_utils.build_inverted_index).

    After encoding is done, the SPLADE model is deleted from GPU memory so
    that Stage 2 can load its own model without running out of VRAM.

    Args:
        corpus_texts: One text per table (from build_corpus_texts).
        output_path:  Where to save the index pickle file.
        model_id:     HuggingFace model ID for SPLADE.
        batch_size:   Texts encoded per forward pass.
        sparse_top_k: Max non-zero terms kept per document vector.
        hf_cache:     HuggingFace cache directory.
        logger:       Optional logger; falls back to print.

    Returns:
        The inverted index dict (also saved to output_path).
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from utils.splade_utils import load_splade_model, build_inverted_index
    from utils.io_utils import save_pickle, load_pickle

    log = logger.info if logger else print

    out_path = Path(output_path)
    if out_path.exists():
        log(f"[splade_indexer] Index already exists at {out_path} — loading from cache")
        return load_pickle(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve model path from cache if it exists there
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[splade_indexer] Loading SPLADE model: {model_id}  (device={device})")

    # Use the model snapshot path when available (avoids re-downloading)
    model_path = model_id
    if hf_cache:
        import os
        # Snapshot dirs are nested: models--org--name/snapshots/sha/
        safe_name = model_id.replace("/", "--")
        snap_parent = Path(hf_cache) / f"models--{safe_name}" / "snapshots"
        if snap_parent.exists():
            snaps = sorted(snap_parent.iterdir())
            if snaps:
                model_path = str(snaps[-1])  # use most recent snapshot
                log(f"[splade_indexer] Using cached snapshot: {model_path}")

    tokenizer, model, device_obj = load_splade_model(model_id, device)

    t0 = time.time()
    log(f"[splade_indexer] Building inverted index for {len(corpus_texts):,} documents …")

    index = build_inverted_index(
        corpus_texts=corpus_texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        top_k=sparse_top_k,
        device=device,
        model_id=model_path,  # always pass — enables multi-GPU when >1 GPU visible
        logger=logger,
    )

    elapsed = time.time() - t0
    log(f"[splade_indexer] Done in {elapsed:.1f}s  |  {len(index):,} unique terms")

    # Free GPU memory before Stage 2 loads its model
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log("[splade_indexer] SPLADE model unloaded from GPU")

    save_pickle(out_path, index)
    log(f"[splade_indexer] Index saved to {out_path}")
    return index
