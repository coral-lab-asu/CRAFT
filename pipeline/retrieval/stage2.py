"""
Stage 2: Dense semantic reranking.

Takes Stage 1's top-K candidates and reranks them to top-100 using a
Sentence Transformer over mini-tables (top-5 rows per table).

Two modes
---------
mini_table (default, paper method)
    1. Encode all queries at once.
    2. For every (query, candidate-table) pair, score pre-encoded rows and
       select the top TOP_K_ROWS rows to build a mini-table string.
    3. Encode ALL mini-tables in one batched multi-GPU pass (one spawned
       process per GPU, each loading its own model copy).
    4. Compute cosine similarity between each query vec and its mini-table
       vecs via vectorised numpy dot-products.
    5. Rank and return top-K tables per query.

fast
    Skip steps 2-4. Use the best row score (dot-product with pre-encoded
    row embeddings) as the table score.  3-5x faster, slightly lower recall.
"""

import gc
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Module-level worker (must be picklable → defined at top level)
# ---------------------------------------------------------------------------

def _encode_worker(
    rank: int,
    model_id: str,
    shard: List[str],
    batch_size: int,
    hf_cache: str,
    trust_remote_code: bool,
    out_path: str,          # write embeddings to this .npy file
) -> None:
    """Spawned process: load model on cuda:{rank}, encode shard, save to file."""
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer

    torch.cuda.set_device(rank)
    model_kwargs = {"cache_folder": hf_cache} if hf_cache else {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    model = SentenceTransformer(model_id, **model_kwargs)
    model.to(f"cuda:{rank}")

    embs = model.encode(
        shard,
        batch_size=batch_size,
        show_progress_bar=(rank == 0),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(out_path, embs)
    del model
    torch.cuda.empty_cache()


class DenseReranker:
    """
    Stage 2 dense reranker.

    Typical usage
    -------------
        reranker = DenseReranker(
            row_emb_path="results/row_embeddings.npy",
            row_meta_path="results/row_meta.pkl",
            model_id="sentence-transformers/all-mpnet-base-v2",
        )
        stage2_results = reranker.rerank(stage1_results, top_k=100)
        reranker.unload()
    """

    def __init__(
        self,
        row_emb_path: str,
        row_meta_path: str,
        model_id: str = "sentence-transformers/all-mpnet-base-v2",
        hf_cache: str = "",
        trust_remote_code: bool = False,
        logger=None,
    ):
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import torch
        from utils.io_utils import load_pickle

        self.log = logger.info if logger else print

        # Persist these so the multi-GPU encode path can spawn fresh workers
        self._model_id = model_id
        self._hf_cache = hf_cache
        self._trust_remote_code = trust_remote_code

        # --- Load pre-encoded row embeddings (memory-mapped for low RAM use) ---
        self.log(f"[stage2] Loading row embeddings from {row_emb_path} …")
        self.row_embs = np.load(row_emb_path, mmap_mode="r")  # (N_rows, D)
        self.row_meta = load_pickle(row_meta_path)
        self.log(f"[stage2] Row embeddings: {self.row_embs.shape}  ({len(self.row_meta):,} rows)")

        # table_id → list of absolute row indices in row_embs
        self._table_to_rows: Dict[str, List[int]] = {}
        for abs_idx, meta in enumerate(self.row_meta):
            tid = meta["table_id"]
            self._table_to_rows.setdefault(tid, []).append(abs_idx)

        # --- Load Stage 2 dense model (used for query encoding only) ---
        from sentence_transformers import SentenceTransformer
        self.log(f"[stage2] Loading dense model: {model_id}")
        model_kwargs = {"cache_folder": hf_cache} if hf_cache else {}
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        self._n_gpus = torch.cuda.device_count()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_id, **model_kwargs)
        self.model = self.model.to(device)
        self.device = device
        self.log(f"[stage2] Dense model ready (device={device}, {self._n_gpus} GPU(s) visible)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        stage1_results: List[Dict],
        top_k: int = 100,
        top_k_rows: int = 5,
        batch_size: int = 512,
        score_chunk_size: int = 64,
        mode: str = "mini_table",
        query_task: Optional[str] = None,
        passage_task: Optional[str] = None,
    ) -> List[Dict]:
        """
        Rerank Stage 1 candidates to produce top-K per query.

        mini_table mode batches ALL mini-table encoding across every query
        in a single multi-GPU pass, then computes similarities with numpy.

        Args:
            stage1_results:   Output of SpladeRetriever.retrieve().
            top_k:            Tables to keep per query.
            top_k_rows:       Rows per table used to form the mini-table.
            batch_size:       Texts encoded per forward pass per GPU.
            score_chunk_size: Unused (kept for API compatibility).
            mode:             "mini_table" or "fast".
            query_task:       Task prompt for models that support it (e.g. JINA v3).
            passage_task:     Passage task prompt (e.g. JINA v3).

        Returns:
            List of result dicts in the same format as Stage 1.
        """
        self.log(f"[stage2] Reranking {len(stage1_results)} queries  |  mode={mode}")

        # --- 1. Encode all queries at once (fast: only ~966 texts) ---
        questions = [r["question"] for r in stage1_results]
        encode_kwargs = dict(
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if query_task:
            encode_kwargs["task"] = query_task

        self.log(f"[stage2] Encoding {len(questions):,} queries …")
        t0 = time.time()
        query_vecs = self.model.encode(questions, **encode_kwargs)  # (N_q, D)
        self.log(f"[stage2] Queries encoded in {time.time()-t0:.1f}s")

        if mode == "fast":
            return self._rerank_fast(stage1_results, query_vecs, top_k, top_k_rows)
        else:
            return self._rerank_mini_table(
                stage1_results, query_vecs, top_k, top_k_rows,
                batch_size, passage_task,
            )

    def unload(self):
        """Delete the dense model from GPU memory."""
        import torch
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.log("[stage2] Dense model unloaded from GPU")

    # ------------------------------------------------------------------
    # Fast mode
    # ------------------------------------------------------------------

    def _rerank_fast(
        self,
        stage1_results: List[Dict],
        query_vecs: np.ndarray,
        top_k: int,
        top_k_rows: int,
    ) -> List[Dict]:
        """Best pre-encoded row score = table score.  No re-encoding."""
        results = []
        for item, q_vec in tqdm(
            zip(stage1_results, query_vecs),
            total=len(stage1_results),
            desc="Stage 2 fast",
            unit="query",
        ):
            candidate_ids = [r["table_id"] for r in item["retrieved"]]
            table_scores = {}
            for tid in candidate_ids:
                row_idxs = self._table_to_rows.get(tid, [])
                if not row_idxs:
                    table_scores[tid] = -999.0
                    continue
                scores = self.row_embs[row_idxs] @ q_vec
                table_scores[tid] = float(scores.max())

            ranked = sorted(candidate_ids, key=lambda t: table_scores[t], reverse=True)
            results.append({
                "qid": item["qid"],
                "question": item["question"],
                "gold_table_ids": item["gold_table_ids"],
                "retrieved": [
                    {"rank": r + 1, "table_id": tid, "score": table_scores[tid]}
                    for r, tid in enumerate(ranked[:top_k])
                ],
            })
        return results

    # ------------------------------------------------------------------
    # Mini-table mode (batched across all queries)
    # ------------------------------------------------------------------

    def _rerank_mini_table(
        self,
        stage1_results: List[Dict],
        query_vecs: np.ndarray,
        top_k: int,
        top_k_rows: int,
        batch_size: int,
        passage_task: Optional[str],
    ) -> List[Dict]:
        """
        Build every mini-table, encode them all in one batched multi-GPU
        pass, then score against query vecs with vectorised dot-products.
        """

        # --- 2. Build ALL mini-table texts (CPU, numpy) ---
        t0 = time.time()
        all_texts: List[str] = []
        # parallel arrays: which query and table each text belongs to
        text_query_idx: List[int] = []
        text_table_id: List[str] = []
        # per-query bookkeeping: (start, end) slice into the above lists
        query_slices: List[Tuple[int, int]] = []

        for q_idx, (item, q_vec) in enumerate(zip(stage1_results, query_vecs)):
            start = len(all_texts)
            for cand in item["retrieved"]:
                tid = cand["table_id"]
                row_idxs = self._table_to_rows.get(tid)
                if not row_idxs:
                    continue
                # pick top-K rows for this query
                row_scores = self.row_embs[row_idxs] @ q_vec
                top_local = np.argsort(-row_scores)[:top_k_rows]
                row_parts = [
                    self.row_meta[row_idxs[i]].get("text", "")
                    for i in top_local
                    if self.row_meta[row_idxs[i]].get("text", "")
                ]
                mini_text = " ".join(row_parts) if row_parts else tid.replace("_", " ")
                all_texts.append(mini_text)
                text_query_idx.append(q_idx)
                text_table_id.append(tid)
            query_slices.append((start, len(all_texts)))

        self.log(
            f"[stage2] Built {len(all_texts):,} mini-table texts for "
            f"{len(stage1_results)} queries in {time.time()-t0:.1f}s"
        )

        # --- 3. Encode ALL mini-tables in one batched multi-GPU pass ---
        t0 = time.time()
        self.log(
            f"[stage2] Encoding {len(all_texts):,} mini-tables "
            f"(batch_size={batch_size}, {self._n_gpus} GPU(s)) …"
        )
        mt_vecs = self._encode_all(all_texts, batch_size, passage_task)
        self.log(f"[stage2] Mini-tables encoded in {time.time()-t0:.1f}s")

        # --- 4. Per-query similarity + ranking (vectorised numpy) ---
        results = []
        for q_idx, item in enumerate(stage1_results):
            start, end = query_slices[q_idx]
            if start == end:
                results.append({
                    "qid": item["qid"],
                    "question": item["question"],
                    "gold_table_ids": item["gold_table_ids"],
                    "retrieved": [],
                })
                continue

            q_vec = query_vecs[q_idx]
            slice_vecs = mt_vecs[start:end]                  # (M, D)
            scores = slice_vecs @ q_vec                       # (M,)
            ranked_idx = np.argsort(-scores)

            retrieved = [
                {
                    "rank": r + 1,
                    "table_id": text_table_id[start + i],
                    "score": float(scores[i]),
                }
                for r, i in enumerate(ranked_idx[:top_k])
            ]
            results.append({
                "qid": item["qid"],
                "question": item["question"],
                "gold_table_ids": item["gold_table_ids"],
                "retrieved": retrieved,
            })

        return results

    def _encode_all(
        self,
        texts: List[str],
        batch_size: int,
        task: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode texts using all visible GPUs (spawned-process DDP if >1 GPU,
        single model.encode() otherwise).  Returns float32 L2-normalised array.
        """
        import torch
        import torch.multiprocessing as mp

        if self._n_gpus <= 1:
            encode_kwargs = dict(
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if task:
                encode_kwargs["task"] = task
            return self.model.encode(texts, **encode_kwargs)

        # Multi-GPU: pre-shard, spawn one process per GPU, write to temp .npy files
        total = len(texts)
        shard_size = (total + self._n_gpus - 1) // self._n_gpus
        shards = [texts[r * shard_size:(r + 1) * shard_size] for r in range(self._n_gpus)]

        tmp_dir = tempfile.mkdtemp()
        tmp_paths = [os.path.join(tmp_dir, f"shard_{r}.npy") for r in range(self._n_gpus)]

        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(
                target=_encode_worker,
                args=(
                    rank,
                    self._model_id,
                    shards[rank],
                    batch_size,
                    self._hf_cache,
                    self._trust_remote_code,
                    tmp_paths[rank],
                ),
            )
            for rank in range(self._n_gpus)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"Stage2 encode worker rank={p.pid} exited with code {p.exitcode}"
                )

        embeddings = np.vstack([np.load(p) for p in tmp_paths])
        for p in tmp_paths:
            os.unlink(p)
        os.rmdir(tmp_dir)
        return embeddings
