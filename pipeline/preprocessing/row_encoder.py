"""
Encode every table row with the Stage 2 dense model and save the embeddings.

This is a one-time offline step.  The result is a numpy memory-mapped file
(row_embeddings.npy) so retrieval can load only the slices it needs without
pulling the entire matrix into RAM.

Why pre-encode rows?
--------------------
Stage 2 needs to compare each query against every row in top-5000 tables.
Instead of re-encoding the corpus rows for every new query set, we do it
once here and reuse the embeddings at query time.  This is especially useful
when you have many queries but the same corpus.
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _dense_worker(
    rank: int,
    model_id: str,
    shard: List[str],
    batch_size: int,
    result_queue,  # mp.Queue
    hf_cache: str,
    trust_remote_code: bool,
) -> None:
    """One spawned process per GPU: loads SentenceTransformer, encodes its shard."""
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer

    torch.cuda.set_device(rank)

    model_kwargs = {"cache_folder": hf_cache} if hf_cache else {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    model = SentenceTransformer(model_id, **model_kwargs)
    model = model.to(f"cuda:{rank}")

    embeddings = model.encode(
        shard,
        batch_size=batch_size,
        show_progress_bar=(rank == 0),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    result_queue.put((rank, embeddings))
    del model
    torch.cuda.empty_cache()


def encode_rows(
    row_texts: List[str],
    row_meta: List[Dict],
    output_dir: str,
    model_id: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 512,
    hf_cache: str = "",
    trust_remote_code: bool = False,
    logger=None,
) -> Tuple[Any, List[Dict]]:  # returns (np.ndarray, list)
    """
    Encode all row texts with the Stage 2 Sentence Transformer model.

    Multi-GPU: spawns one process per visible GPU (same pattern as SPLADE DDP),
    each loading an independent model copy and encoding its own shard. This
    avoids the IPC overhead of encode_multi_process and the tokenize() breakage
    of DataParallel.

    After encoding, the model is explicitly deleted from GPU memory.

    Args:
        row_texts:         One string per table row (from build_row_texts).
        row_meta:          Same-length list of {table_id, row_idx} dicts.
        output_dir:        Directory where row_embeddings.npy and row_meta.pkl
                           will be saved.
        model_id:          HuggingFace Sentence Transformer ID.
        batch_size:        Rows encoded per forward pass per GPU.
        hf_cache:          HuggingFace cache directory.
        trust_remote_code: Required for some models (e.g. JINA v3).
        logger:            Optional logger.

    Returns:
        (embeddings, row_meta) where embeddings[i] is the L2-normalised
        vector for row_texts[i].
    """
    import numpy as np
    import torch
    import torch.multiprocessing as mp
    from sentence_transformers import SentenceTransformer
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from utils.io_utils import save_pickle, load_pickle

    log = logger.info if logger else print

    out_dir = Path(output_dir)
    emb_path = out_dir / "row_embeddings.npy"
    meta_path = out_dir / "row_meta.pkl"

    if emb_path.exists() and meta_path.exists():
        log(f"[row_encoder] Row embeddings already exist at {emb_path} — loading from cache")
        embs = np.load(emb_path, mmap_mode="r")
        meta = load_pickle(meta_path)
        return embs, meta

    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[row_encoder] Loading model: {model_id}")
    model_kwargs = {"cache_folder": hf_cache} if hf_cache else {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    t0 = time.time()
    log(f"[row_encoder] Encoding {len(row_texts):,} rows (batch_size={batch_size}) …")

    n_gpus = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if n_gpus > 1:
        # Pre-shard data and spawn one process per GPU. Each process loads its
        # own model copy and encodes its shard independently — no IPC bottleneck.
        total = len(row_texts)
        shard_size = (total + n_gpus - 1) // n_gpus
        shards = [row_texts[r * shard_size:(r + 1) * shard_size] for r in range(n_gpus)]
        log(f"[row_encoder] DDP-style encoding across {n_gpus} GPUs "
            f"(shard sizes: {[len(s) for s in shards]})")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        procs = [
            ctx.Process(
                target=_dense_worker,
                args=(rank, model_id, shards[rank], batch_size, result_queue, hf_cache, trust_remote_code),
            )
            for rank in range(n_gpus)
        ]
        for p in procs:
            p.start()

        raw: Dict[int, np.ndarray] = {}
        for _ in range(n_gpus):
            rank, embs = result_queue.get()
            raw[rank] = embs
            log(f"[row_encoder] GPU {rank} finished — {len(embs):,} rows encoded")

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Row encoder worker (pid={p.pid}) exited with code {p.exitcode}")

        embeddings = np.vstack([raw[r] for r in range(n_gpus)])

    else:
        log(f"[row_encoder] Single device: {device}")
        model = SentenceTransformer(model_id, **model_kwargs)
        model = model.to(device)
        embeddings = model.encode(
            row_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - t0
    log(f"[row_encoder] Encoded {len(embeddings):,} rows in {elapsed:.1f}s  "
        f"| shape={embeddings.shape}  dtype={embeddings.dtype}")

    np.save(emb_path, embeddings)
    save_pickle(meta_path, row_meta)
    log(f"[row_encoder] Embeddings saved to {emb_path}")
    log(f"[row_encoder] Row meta saved to   {meta_path}")

    gc.collect()
    log("[row_encoder] Stage-2 model unloaded from GPU")

    return embeddings, row_meta
