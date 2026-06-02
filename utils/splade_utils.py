from collections import defaultdict
import os
from queue import Queue
import socket
import threading
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def _resolve_device(device: str) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def load_splade_model(model_id: str, device: str) -> Tuple[AutoTokenizer, AutoModelForMaskedLM, torch.device]:
    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()
    model.to(resolved_device)
    print("Model Loaded.")
    return tokenizer, model, resolved_device


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _ddp_encode_worker(
    rank: int,
    world_size: int,
    model_id: str,
    corpus_shard: List[str],
    batch_size: int,
    top_k: int,
    result_queue,  # mp.Queue
    port: int,
) -> None:
    """One DDP worker process per GPU: loads model, encodes its shard, enqueues results."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()
    model.to(device)
    ddp_model = DDP(model, device_ids=[rank])

    encodings: List[Dict[int, float]] = []
    for i in tqdm(
        range(0, len(corpus_shard), batch_size),
        desc=f"GPU {rank}",
        leave=False,
    ):
        batch = corpus_shard[i : i + batch_size]
        encodings.extend(splade_encode(batch, tokenizer, ddp_model, top_k, device))

    result_queue.put((rank, encodings))
    dist.destroy_process_group()


def splade_encode(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    top_k: int,
    device: torch.device,
) -> List[Dict[int, float]]:
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits
    sparse_scores = torch.max(F.relu(outputs), dim=1).values
    encoded = []
    for i in range(sparse_scores.size(0)):
        values, indices = torch.topk(sparse_scores[i], top_k)
        encoded.append({idx.item(): val.item() for idx, val in zip(indices, values) if val.item() > 0})
    return encoded


def encode_texts_in_batches(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> List[Dict[int, float]]:
    encodings: List[Dict[int, float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with SPLADE", leave=False):
        batch = texts[i : i + batch_size]
        encodings.extend(splade_encode(batch, tokenizer, model, top_k, device))
    return encodings


def build_inverted_index(
    corpus_texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    batch_size: int,
    top_k: int,
    device: torch.device,
    model_id: str = None,
    logger=None,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build a SPLADE inverted index over corpus_texts.

    Multi-GPU strategy: DistributedDataParallel — one spawned process per GPU,
    each loads its own model copy and encodes an equal shard of the corpus in
    parallel.  This gives symmetric GPU utilisation (no primary-device gather
    overhead) and avoids the memory imbalance of DataParallel.

    Single-GPU strategy: a background thread encodes on GPU while the main
    thread builds the inverted index concurrently.
    """
    log = logger.info if logger else print

    num_gpus = torch.cuda.device_count()
    use_cuda = str(device).startswith("cuda")

    # ------------------------------------------------------------------
    # Multi-GPU path: DistributedDataParallel
    # ------------------------------------------------------------------
    if num_gpus > 1 and use_cuda and model_id:
        log(f"[splade_indexer] DDP encoding across {num_gpus} GPUs")

        # Move the already-loaded model off GPU so workers can load fresh
        # copies on each device without hitting OOM on cuda:0.
        model.cpu()
        torch.cuda.empty_cache()

        total_docs = len(corpus_texts)
        shard_size = (total_docs + num_gpus - 1) // num_gpus
        shards = [
            corpus_texts[r * shard_size : (r + 1) * shard_size]
            for r in range(num_gpus)
        ]
        log(f"[splade_indexer] {total_docs:,} docs split across {num_gpus} GPUs "
            f"(shard sizes: {[len(s) for s in shards]})")

        port = _find_free_port()
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

        # Collect one result per worker (blocks until each enqueues its shard)
        raw: Dict[int, List[Dict[int, float]]] = {}
        for _ in range(num_gpus):
            rank, encodings = result_queue.get()
            raw[rank] = encodings
            log(f"[splade_indexer] GPU {rank} finished — {len(encodings):,} docs encoded")

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"DDP worker (pid={p.pid}) exited with code {p.exitcode}")

        # Reassemble in original corpus order: shard 0, then 1, …
        all_encodings = [enc for rank in range(num_gpus) for enc in raw[rank]]

        log("[splade_indexer] Building inverted index …")
        index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for doc_id, sparse_vec in enumerate(tqdm(all_encodings, desc="Build Inverted Index")):
            for term_id, weight in sparse_vec.items():
                index[term_id].append((doc_id, weight))

        log(f"[splade_indexer] Done — {len(index):,} unique terms")
        return dict(index)

    # ------------------------------------------------------------------
    # Single-GPU path: background thread encodes; main thread builds index.
    # CUDA ops release the GIL so both sides run truly in parallel.
    # ------------------------------------------------------------------
    log(f"[splade_indexer] Single GPU encoding (device={device})")
    total_docs = len(corpus_texts)
    total_batches = (total_docs + batch_size - 1) // batch_size
    log(f"[splade_indexer] {total_docs:,} docs  |  batch={batch_size}  |  {total_batches} batches")

    batch_queue: Queue = Queue(maxsize=6)

    def _encode_worker():
        try:
            for i in range(0, total_docs, batch_size):
                batch = corpus_texts[i : i + batch_size]
                encodings = splade_encode(batch, tokenizer, model, top_k, device)
                batch_queue.put((i, encodings))
        finally:
            batch_queue.put(None)

    encode_thread = threading.Thread(target=_encode_worker, daemon=True)
    encode_thread.start()

    index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    next_milestone = 10
    batches_done = 0

    pbar = tqdm(total=total_batches, desc="Build Inverted Index")
    while True:
        item = batch_queue.get()
        if item is None:
            break
        start_idx, doc_encodings = item
        for j, sparse_vec in enumerate(doc_encodings):
            doc_id = start_idx + j
            for term_id, weight in sparse_vec.items():
                index[term_id].append((doc_id, weight))
        batches_done += 1
        pbar.update(1)
        pct = batches_done * 100 // total_batches
        if pct >= next_milestone:
            log(
                f"[splade_indexer] SPLADE index: {pct}% complete "
                f"({batches_done}/{total_batches} batches, "
                f"{len(index):,} unique terms so far)"
            )
            next_milestone = pct + 10
    pbar.close()

    encode_thread.join()
    return dict(index)


def rank_query(
    index: Dict[int, List[Tuple[int, float]]],
    query_vec: Dict[int, float],
    top_k: int,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = defaultdict(float)
    for term_id, q_weight in query_vec.items():
        postings = index.get(term_id, [])
        for doc_id, d_weight in postings:
            scores[doc_id] += q_weight * d_weight
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def rank_queries(
    index: Dict[int, List[Tuple[int, float]]],
    query_vectors: Iterable[Dict[int, float]],
    top_k: int,
) -> List[List[Tuple[int, float]]]:
    return [rank_query(index, query_vec, top_k) for query_vec in query_vectors]
