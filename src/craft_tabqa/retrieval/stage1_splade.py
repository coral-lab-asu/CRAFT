"""Stage 1: SPLADE sparse retrieval over the full corpus.

Encodes every query with SPLADE and scores it against the pre-built inverted
index, returning the top ``top_k`` tables per query. All queries are encoded in
batches in a single pass.
"""

import gc
import time
from typing import Dict, List

import torch
from tqdm import tqdm

from craft_tabqa.core.files import load_pickle
from craft_tabqa.core.sparse import encode_sparse, load_splade_model, rank_against_index
from craft_tabqa.core.text import build_query_text
from craft_tabqa.loaders.schema import Query


class SpladeRetriever:
    """Loads the SPLADE model + inverted index and retrieves candidate tables.

    The model stays resident between :meth:`retrieve` calls; call :meth:`unload`
    to free GPU memory before Stage 2 loads its own model.
    """

    def __init__(self, index_path: str, model_id: str = "naver/splade_v2_distil", logger=None):
        self.log = logger.info if logger else print
        self.model_id = model_id

        self.log(f"[stage1] Loading index from {index_path}")
        self.index = load_pickle(index_path)
        self.log(f"[stage1] Index loaded ({len(self.index):,} terms)")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model, self.device = load_splade_model(model_id, device)
        self.log(f"[stage1] SPLADE model ready on {self.device}")

    def retrieve(
        self,
        queries: List[Query],
        table_ids: List[str],
        top_k: int = 5_000,
        sparse_top_k: int = 5_000,
        batch_size: int = 128,
        query_type: str = "query+subquestion",
    ) -> List[Dict]:
        """Return the top ``top_k`` tables for each query as ranked result dicts."""
        query_texts = [build_query_text(q, query_type) for q in queries]

        self.log(f"[stage1] Encoding {len(query_texts):,} queries")
        start = time.time()
        query_vectors = []
        for i in tqdm(range(0, len(query_texts), batch_size), desc="SPLADE queries", unit="batch"):
            batch = query_texts[i : i + batch_size]
            query_vectors.extend(
                encode_sparse(batch, self.tokenizer, self.model, sparse_top_k, self.device)
            )
        self.log(f"[stage1] Queries encoded in {time.time() - start:.1f}s")

        self.log("[stage1] Ranking against index")
        start = time.time()
        ranked = rank_against_index(self.index, query_vectors, top_k)
        self.log(f"[stage1] Ranked in {time.time() - start:.1f}s")

        return [_format(q, r, table_ids) for q, r in zip(queries, ranked)]

    def unload(self) -> None:
        """Release the SPLADE model from GPU memory."""
        del self.model, self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.log("[stage1] SPLADE model unloaded")


def _format(query: Query, ranked, table_ids: List[str]) -> Dict:
    return {
        "qid": query["qid"],
        "question": query["question"],
        "gold_table_ids": query.get("gold_table_ids", []),
        "retrieved": [
            {"rank": rank + 1, "table_id": table_ids[doc_id], "score": float(score)}
            for rank, (doc_id, score) in enumerate(ranked)
        ],
    }
