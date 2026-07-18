"""Dense sentence encoding, single- or multi-GPU, shared across the pipeline.

Both the offline row encoder and the Stage 2 reranker need to encode a large
list of texts with a SentenceTransformer. This module provides one function for
that so the multi-GPU logic lives in exactly one place.

Multi-GPU strategy: pre-shard the texts, spawn one process per visible GPU (each
loads its own model copy), and concatenate the shards back in order. This avoids
the IPC overhead and tokenizer issues of the built-in multi-process encoders.
"""

import os
import tempfile
from typing import List, Optional

import numpy as np


def encode_texts(
    texts: List[str],
    model_id: str,
    batch_size: int = 256,
    hf_cache: str = "",
    trust_remote_code: bool = False,
    task: Optional[str] = None,
    show_progress: bool = True,
    model=None,
) -> np.ndarray:
    """Encode ``texts`` into L2-normalised float32 vectors.

    Uses every visible GPU when more than one is present. When a single device
    is used and a preloaded ``model`` is supplied, it is reused (handy for the
    Stage 2 reranker, which keeps its model resident).
    """
    import torch

    num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        owned_model = model is None
        if owned_model:
            model = _load_model(model_id, hf_cache, trust_remote_code)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
        vectors = _encode_one_device(model, texts, batch_size, task, show_progress)
        if owned_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return vectors

    return _encode_multi_gpu(texts, model_id, batch_size, hf_cache, trust_remote_code, task, num_gpus)


def _load_model(model_id: str, hf_cache: str, trust_remote_code: bool):
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if hf_cache:
        kwargs["cache_folder"] = hf_cache
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return SentenceTransformer(model_id, **kwargs)


def _encode_one_device(model, texts, batch_size, task, show_progress) -> np.ndarray:
    kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if task:
        kwargs["task"] = task
    return model.encode(texts, **kwargs)


def _encode_multi_gpu(texts, model_id, batch_size, hf_cache, trust_remote_code, task, num_gpus) -> np.ndarray:
    import torch.multiprocessing as mp

    shard_size = (len(texts) + num_gpus - 1) // num_gpus
    shards = [texts[r * shard_size : (r + 1) * shard_size] for r in range(num_gpus)]

    tmp_dir = tempfile.mkdtemp()
    out_paths = [os.path.join(tmp_dir, f"shard_{r}.npy") for r in range(num_gpus)]

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_encode_worker,
            args=(rank, model_id, shards[rank], batch_size, hf_cache, trust_remote_code, task, out_paths[rank]),
        )
        for rank in range(num_gpus)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Dense encode worker (pid={p.pid}) exited with code {p.exitcode}")

    vectors = np.vstack([np.load(p) for p in out_paths])
    for p in out_paths:
        os.unlink(p)
    os.rmdir(tmp_dir)
    return vectors


def _encode_worker(rank, model_id, shard, batch_size, hf_cache, trust_remote_code, task, out_path):
    import torch

    torch.cuda.set_device(rank)
    model = _load_model(model_id, hf_cache, trust_remote_code).to(f"cuda:{rank}")
    vectors = _encode_one_device(model, shard, batch_size, task, show_progress=(rank == 0))
    np.save(out_path, vectors)
    del model
    torch.cuda.empty_cache()
