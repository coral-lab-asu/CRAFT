"""Stage 2: dense reranking of Stage 1 candidates down to the top ``top_k``.

Two modes:

``representative_row`` (default)
    A row stands in for its whole table: the table's score is its best-matching
    row's cosine similarity to the query. Rows are pre-encoded during
    preprocessing, so this stage only encodes the (few) queries - no table or
    mini-table encoding happens here. Fast, and it lets Stage 3 be the single
    place that ever encodes a mini-table.

``mini_table``
    For each candidate, pick its top rows, join them into a mini-table string,
    and re-encode that string at query time (the original paper method). More
    expensive; kept for reproducibility.
"""

import gc
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from craft_tabqa.core.dense import encode_texts
from craft_tabqa.core.files import load_pickle


class DenseReranker:
    """Reranks Stage 1 candidates using pre-encoded row embeddings."""

    def __init__(
        self,
        row_emb_path: str,
        row_meta_path: str,
        model_id: str = "sentence-transformers/all-mpnet-base-v2",
        hf_cache: str = "",
        trust_remote_code: bool = False,
        logger=None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.log = logger.info if logger else print
        self.model_id = model_id
        self.hf_cache = hf_cache
        self.trust_remote_code = trust_remote_code

        self.log(f"[stage2] Loading row embeddings from {row_emb_path}")
        self.row_embs = np.load(row_emb_path, mmap_mode="r")
        self.row_meta = load_pickle(row_meta_path)
        self.log(f"[stage2] {self.row_embs.shape[0]:,} rows, dim {self.row_embs.shape[1]}")

        # table_id -> absolute row indices, so a candidate's rows are one lookup.
        self.rows_of_table: Dict[str, List[int]] = {}
        for idx, meta in enumerate(self.row_meta):
            self.rows_of_table.setdefault(meta["table_id"], []).append(idx)

        model_kwargs = {}
        if hf_cache:
            model_kwargs["cache_folder"] = hf_cache
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_id, **model_kwargs).to(device)
        self.log(f"[stage2] Dense model ready on {device}")

    def rerank(
        self,
        stage1_results: List[Dict],
        top_k: int = 100,
        top_k_rows: int = 5,
        batch_size: int = 256,
        mode: str = "representative_row",
        query_task: Optional[str] = None,
        passage_task: Optional[str] = None,
    ) -> List[Dict]:
        """Return the top ``top_k`` tables per query under the chosen ``mode``."""
        self.log(f"[stage2] Reranking {len(stage1_results)} queries (mode={mode})")
        questions = [r["question"] for r in stage1_results]

        encode_kwargs = dict(
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if query_task:
            encode_kwargs["task"] = query_task
        query_vecs = self.model.encode(questions, **encode_kwargs)

        if mode == "mini_table":
            return self._rerank_mini_table(
                stage1_results, query_vecs, top_k, top_k_rows, batch_size, passage_task
            )
        return self._rerank_representative_row(stage1_results, query_vecs, top_k)

    def unload(self) -> None:
        """Release the dense model from GPU memory."""
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.log("[stage2] Dense model unloaded")

    # ------------------------------------------------------------------
    # representative_row: best row score = table score (no re-encoding).
    # ------------------------------------------------------------------

    def _rerank_representative_row(self, stage1_results, query_vecs, top_k) -> List[Dict]:
        results = []
        for item, q_vec in tqdm(
            zip(stage1_results, query_vecs),
            total=len(stage1_results),
            desc="Stage 2",
            unit="query",
        ):
            scored = []
            for candidate in item["retrieved"]:
                table_id = candidate["table_id"]
                row_idxs = self.rows_of_table.get(table_id)
                if not row_idxs:
                    continue
                best = float((self.row_embs[row_idxs] @ q_vec).max())
                scored.append((table_id, best))

            scored.sort(key=lambda kv: kv[1], reverse=True)
            results.append(_format(item, scored[:top_k]))
        return results

    # ------------------------------------------------------------------
    # mini_table: build + re-encode a top-rows mini-table per candidate.
    # ------------------------------------------------------------------

    def _rerank_mini_table(
        self, stage1_results, query_vecs, top_k, top_k_rows, batch_size, passage_task
    ) -> List[Dict]:
        # Build every mini-table text, tracking which query/table each belongs to.
        texts: List[str] = []
        text_table_id: List[str] = []
        query_spans = []  # (start, end) into texts for each query

        for q_idx, (item, q_vec) in enumerate(zip(stage1_results, query_vecs)):
            start = len(texts)
            for candidate in item["retrieved"]:
                table_id = candidate["table_id"]
                row_idxs = self.rows_of_table.get(table_id)
                if not row_idxs:
                    continue
                row_scores = self.row_embs[row_idxs] @ q_vec
                top_rows = np.argsort(-row_scores)[:top_k_rows]
                mini_table = " ".join(
                    self.row_meta[row_idxs[i]]["text"] for i in top_rows
                ) or table_id.replace("_", " ")
                texts.append(mini_table)
                text_table_id.append(table_id)
            query_spans.append((start, len(texts)))

        self.log(f"[stage2] Encoding {len(texts):,} mini-tables")
        mini_vecs = encode_texts(
            texts,
            model_id=self.model_id,
            batch_size=batch_size,
            hf_cache=self.hf_cache,
            trust_remote_code=self.trust_remote_code,
            task=passage_task,
            model=self.model,
        )

        results = []
        for item, q_vec, (start, end) in zip(stage1_results, query_vecs, query_spans):
            if start == end:
                results.append(_format(item, []))
                continue
            scores = mini_vecs[start:end] @ q_vec
            order = np.argsort(-scores)[:top_k]
            scored = [(text_table_id[start + i], float(scores[i])) for i in order]
            results.append(_format(item, scored))
        return results


def _format(item: Dict, scored) -> Dict:
    return {
        "qid": item["qid"],
        "question": item["question"],
        "gold_table_ids": item.get("gold_table_ids", []),
        "retrieved": [
            {"rank": rank + 1, "table_id": table_id, "score": score}
            for rank, (table_id, score) in enumerate(scored)
        ],
    }
