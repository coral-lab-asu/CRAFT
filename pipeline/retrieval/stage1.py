"""
Stage 1: SPLADE sparse retrieval.

Takes the pre-built inverted index and a set of queries and returns the
top-K most relevant tables per query.

All queries are encoded together in a single batched pass — never one by one.
"""

import gc
import time
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from pipeline.loaders.base import QueryEntry


class SpladeRetriever:
    """
    Wraps the SPLADE model + inverted index for Stage 1 retrieval.

    Typical usage
    -------------
        retriever = SpladeRetriever(index_path="results/splade_index.pkl")
        results   = retriever.retrieve(queries, top_k=5000)

    The retriever keeps the model in memory between calls so you can run
    multiple query batches without reloading.  Call .unload() when done to
    free GPU memory for Stage 2.
    """

    def __init__(
        self,
        index_path: str,
        model_id: str = "naver/splade_v2_distil",
        hf_cache: str = "",
        logger=None,
    ):
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        import torch
        from utils.io_utils import load_pickle
        from utils.splade_utils import load_splade_model

        self.log = logger.info if logger else print
        self.model_id = model_id

        self.log(f"[stage1] Loading inverted index from {index_path} …")
        self.index = load_pickle(index_path)
        self.log(f"[stage1] Index loaded  |  {len(self.index):,} unique terms")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log(f"[stage1] Loading SPLADE model: {model_id}  (device={device})")
        self.tokenizer, self.model, self.device = load_splade_model(model_id, device)
        self.log("[stage1] SPLADE model ready")

    def retrieve(
        self,
        queries: List[QueryEntry],
        table_ids: List[str],
        top_k: int = 5_000,
        sparse_top_k: int = 5_000,
        batch_size: int = 128,
        query_type: str = "query+subquestion",
    ) -> List[Dict]:
        """
        Encode all queries and retrieve top-K tables from the index.

        Args:
            queries:     List of QueryEntry dicts.
            table_ids:   Ordered list of table IDs (position = corpus index).
            top_k:       Tables returned per query.
            sparse_top_k: Max non-zero terms in each query vector.
            batch_size:  Queries encoded per batch.
            query_type:  Which query fields to use (same options as Stage 1 notebook).

        Returns:
            List of result dicts (one per query):
            {
              "qid": "...",
              "question": "...",
              "gold_table_ids": [...],
              "retrieved": [{"rank": 1, "table_id": "...", "score": 12.3}, ...]
            }
        """
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from utils.corpus_builder import build_query_texts
        from utils.splade_utils import splade_encode, rank_queries

        # Build the text for each query (handles subquestion concatenation)
        # build_query_texts expects the same dict format as load_nq_questions
        query_dicts = [
            {
                "qid": q["qid"],
                "question": q["question"],
                "subquestion": q.get("subquestion", ""),
                "query_description": q.get("query_description", ""),
            }
            for q in queries
        ]
        query_texts = build_query_texts(query_dicts, query_type=query_type)

        # --- Encode all queries in batches ---
        self.log(f"[stage1] Encoding {len(query_texts):,} queries …")
        t0 = time.time()
        query_vectors = []
        for i in tqdm(range(0, len(query_texts), batch_size), desc="Encoding queries (SPLADE)", unit="batch"):
            batch = query_texts[i : i + batch_size]
            vecs = splade_encode(
                batch,
                tokenizer=self.tokenizer,
                model=self.model,
                top_k=sparse_top_k,
                device=self.device,
            )
            query_vectors.extend(vecs)
        self.log(f"[stage1] Queries encoded in {time.time()-t0:.1f}s")

        # --- Score against inverted index ---
        self.log(f"[stage1] Ranking {len(queries):,} queries against {len(self.index):,}-term index …")
        t0 = time.time()
        ranked_lists = rank_queries(self.index, query_vectors, top_k=top_k)
        self.log(f"[stage1] Ranking done in {time.time()-t0:.1f}s")

        # --- Build output format ---
        results = []
        for query, ranked in zip(queries, ranked_lists):
            retrieved = [
                {
                    "rank": rank + 1,
                    "table_id": table_ids[doc_id],
                    "score": float(score),
                }
                for rank, (doc_id, score) in enumerate(ranked)
            ]
            results.append(
                {
                    "qid": query["qid"],
                    "question": query["question"],
                    "gold_table_ids": query["gold_table_ids"],
                    "retrieved": retrieved,
                }
            )
        return results

    def unload(self):
        """Delete the SPLADE model from GPU memory."""
        import torch
        del self.model, self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.log("[stage1] SPLADE model unloaded from GPU")
