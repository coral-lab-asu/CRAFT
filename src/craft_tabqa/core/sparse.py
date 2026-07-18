"""SPLADE sparse encoding and the inverted index it feeds (Stage 1).

SPLADE turns each text into a sparse bag of weighted vocabulary terms. We build
an inverted index (term -> list of ``(doc_id, weight)``) over the corpus once,
then score a query by summing the products of shared-term weights.

Encoding the corpus is the expensive part, so it runs multi-GPU when more than
one device is visible: one process per GPU, each encoding an equal shard.
"""

import os
import socket
import threading
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# A sparse vector maps term id -> weight. The inverted index maps term id to the
# documents that contain it.
SparseVector = Dict[int, float]
InvertedIndex = Dict[int, List[Tuple[int, float]]]


def load_splade_model(model_id: str, device: str = "cuda"):
    """Load a SPLADE tokenizer + masked-LM onto ``device``."""
    resolved = torch.device(device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).eval().to(resolved)
    return tokenizer, model, resolved


def encode_sparse(
    texts: List[str],
    tokenizer,
    model,
    top_k: int,
    device: torch.device,
) -> List[SparseVector]:
    """Encode a batch of texts into SPLADE sparse vectors (top ``top_k`` terms each)."""
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # SPLADE pooling: ReLU then max over the sequence dimension.
    weights = torch.max(F.relu(logits), dim=1).values

    vectors: List[SparseVector] = []
    for row in weights:
        values, indices = torch.topk(row, top_k)
        vectors.append(
            {int(i): float(v) for i, v in zip(indices, values) if v.item() > 0}
        )
    return vectors


def build_inverted_index(
    corpus_texts: List[str],
    tokenizer,
    model,
    model_id: str,
    batch_size: int,
    top_k: int,
    device: torch.device,
    logger=None,
) -> InvertedIndex:
    """Encode the whole corpus and assemble the inverted index.

    Uses every visible GPU when there is more than one; otherwise encodes on a
    background thread while the main thread assembles the index concurrently.
    """
    log = logger.info if logger else print
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1 and str(device).startswith("cuda"):
        vectors = _encode_corpus_multi_gpu(
            corpus_texts, model, model_id, batch_size, top_k, num_gpus, log
        )
        return _assemble_index(vectors, log)

    return _encode_and_assemble_single_gpu(
        corpus_texts, tokenizer, model, batch_size, top_k, device, log
    )


def rank_against_index(
    index: InvertedIndex,
    query_vectors: List[SparseVector],
    top_k: int,
) -> List[List[Tuple[int, float]]]:
    """Score each query vector against the index, returning top ``top_k`` docs."""
    return [_rank_one(index, vec, top_k) for vec in query_vectors]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _rank_one(index: InvertedIndex, query_vec: SparseVector, top_k: int):
    scores: Dict[int, float] = defaultdict(float)
    for term_id, q_weight in query_vec.items():
        for doc_id, d_weight in index.get(term_id, ()):
            scores[doc_id] += q_weight * d_weight
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Index assembly
# ---------------------------------------------------------------------------

def _assemble_index(vectors: List[SparseVector], log) -> InvertedIndex:
    log("[sparse] Assembling inverted index ...")
    index: InvertedIndex = defaultdict(list)
    for doc_id, vector in enumerate(tqdm(vectors, desc="Inverted index")):
        for term_id, weight in vector.items():
            index[term_id].append((doc_id, weight))
    log(f"[sparse] Index built - {len(index):,} unique terms")
    return dict(index)


# ---------------------------------------------------------------------------
# Single-GPU path: encode on a worker thread, assemble on the main thread.
# ---------------------------------------------------------------------------

def _encode_and_assemble_single_gpu(
    corpus_texts, tokenizer, model, batch_size, top_k, device, log
) -> InvertedIndex:
    total = len(corpus_texts)
    n_batches = (total + batch_size - 1) // batch_size
    log(f"[sparse] Single-GPU encoding - {total:,} docs, {n_batches} batches")

    queue: Queue = Queue(maxsize=6)

    def encode_worker():
        try:
            for start in range(0, total, batch_size):
                batch = corpus_texts[start : start + batch_size]
                queue.put((start, encode_sparse(batch, tokenizer, model, top_k, device)))
        finally:
            queue.put(None)

    thread = threading.Thread(target=encode_worker, daemon=True)
    thread.start()

    index: InvertedIndex = defaultdict(list)
    with tqdm(total=n_batches, desc="Inverted index") as pbar:
        while True:
            item = queue.get()
            if item is None:
                break
            start, vectors = item
            for offset, vector in enumerate(vectors):
                doc_id = start + offset
                for term_id, weight in vector.items():
                    index[term_id].append((doc_id, weight))
            pbar.update(1)

    thread.join()
    return dict(index)


# ---------------------------------------------------------------------------
# Multi-GPU path: DistributedDataParallel, one process per device.
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _encode_corpus_multi_gpu(
    corpus_texts, model, model_id, batch_size, top_k, num_gpus, log
) -> List[SparseVector]:
    log(f"[sparse] Encoding across {num_gpus} GPUs")
    # Free the already-loaded model so each worker can load a fresh copy.
    model.cpu()
    torch.cuda.empty_cache()

    shard_size = (len(corpus_texts) + num_gpus - 1) // num_gpus
    shards = [corpus_texts[r * shard_size : (r + 1) * shard_size] for r in range(num_gpus)]

    port = _free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    procs = [
        ctx.Process(
            target=_ddp_encode_worker,
            args=(rank, num_gpus, model_id, shards[rank], batch_size, top_k, result_queue, port),
        )
        for rank in range(num_gpus)
    ]
    for p in procs:
        p.start()

    shards_out: Dict[int, List[SparseVector]] = {}
    for _ in range(num_gpus):
        rank, vectors = result_queue.get()
        shards_out[rank] = vectors
        log(f"[sparse] GPU {rank} encoded {len(vectors):,} docs")

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"SPLADE worker (pid={p.pid}) exited with code {p.exitcode}")

    # Reassemble in original corpus order.
    return [vec for rank in range(num_gpus) for vec in shards_out[rank]]


def _ddp_encode_worker(rank, world_size, model_id, shard, batch_size, top_k, result_queue, port):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).eval().to(device)
    model = DDP(model, device_ids=[rank])

    vectors: List[SparseVector] = []
    for start in tqdm(range(0, len(shard), batch_size), desc=f"GPU {rank}", leave=False):
        batch = shard[start : start + batch_size]
        vectors.extend(encode_sparse(batch, tokenizer, model, top_k, device))

    result_queue.put((rank, vectors))
    dist.destroy_process_group()
