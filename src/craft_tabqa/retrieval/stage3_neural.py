"""Stage 3 (optional): rerank the Stage 2 shortlist with an embedding API.

For each query we build a mini-table (top rows) for every candidate, embed the
question and all mini-tables in one batched API call, and rank by cosine
similarity. This is the only stage that builds and encodes mini-tables.

Skipped automatically when no API key is available.
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


class NeuralReranker:
    """Reranks candidates using OpenAI or Gemini text embeddings."""

    def __init__(
        self,
        provider: str = "openai",
        model_id: str = "text-embedding-3-large",
        api_batch: int = 100,
        api_sleep: float = 0.5,
        logger=None,
    ):
        self.provider = provider
        self.model_id = model_id
        self.api_batch = api_batch
        self.api_sleep = api_sleep
        self.log = logger.info if logger else print
        self._client = self._make_client()

    def _make_client(self):
        if self.provider == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise EnvironmentError("OPENAI_API_KEY is not set")
            from openai import OpenAI

            self.log(f"[stage3] OpenAI ready (model={self.model_id})")
            return OpenAI(api_key=key)

        if self.provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise EnvironmentError("GEMINI_API_KEY is not set")
            from google import genai

            self.log(f"[stage3] Gemini ready (model={self.model_id})")
            return genai.Client(api_key=key)

        raise ValueError(f"Unknown provider {self.provider!r} (use 'openai' or 'gemini')")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed ``texts`` into L2-normalised vectors, batching and rate-limiting."""
        vectors: List[np.ndarray] = []
        for i in range(0, len(texts), self.api_batch):
            batch = texts[i : i + self.api_batch]
            if self.provider == "openai":
                resp = self._client.embeddings.create(input=batch, model=self.model_id)
                vectors.extend(np.array(e.embedding, dtype=np.float32) for e in resp.data)
            else:
                resp = self._client.models.embed_content(model=self.model_id, contents=batch)
                vectors.extend(np.array(e.values, dtype=np.float32) for e in resp.embeddings)
            if i + self.api_batch < len(texts):
                time.sleep(self.api_sleep)

        mat = np.vstack(vectors)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-10)

    def rerank(
        self,
        stage2_results: List[Dict],
        row_meta: Optional[List[Dict]] = None,
        top_k: int = 50,
        top_k_rows: int = 5,
    ) -> List[Dict]:
        """Return the top ``top_k`` tables per query, reranked by API embeddings."""
        self.log(f"[stage3] Reranking {len(stage2_results)} queries via {self.provider}")
        rows_of_table = _index_rows_by_table(row_meta)

        results = []
        for item in tqdm(stage2_results, desc="Stage 3", unit="query"):
            candidate_ids = [r["table_id"] for r in item["retrieved"]]
            if not candidate_ids:
                results.append(_format(item, []))
                continue

            mini_tables = [
                " ".join(rows_of_table.get(tid, [])[:top_k_rows]) or tid.replace("_", " ")
                for tid in candidate_ids
            ]
            vectors = self.embed([item["question"], *mini_tables])
            scores = vectors[1:] @ vectors[0]
            order = np.argsort(-scores)[:top_k]
            scored = [(candidate_ids[i], float(scores[i])) for i in order]
            results.append(_format(item, scored))

        self.log(f"[stage3] Done ({len(results)} queries)")
        return results


def _index_rows_by_table(row_meta: Optional[List[Dict]]) -> Dict[str, List[str]]:
    rows_of_table: Dict[str, List[str]] = {}
    for meta in row_meta or []:
        text = meta.get("text")
        if text:
            rows_of_table.setdefault(meta["table_id"], []).append(text)
    return rows_of_table


def _format(item: Dict, scored) -> Dict:
    retrieved = [
        {"rank": rank + 1, "table_id": table_id, "score": score}
        for rank, (table_id, score) in enumerate(scored)
    ]
    ret_ids = [r["table_id"] for r in retrieved]
    gold_rank = next(
        (ret_ids.index(g) + 1 for g in item.get("gold_table_ids", []) if g in ret_ids),
        None,
    )
    return {
        "qid": item["qid"],
        "question": item["question"],
        "gold_table_ids": item.get("gold_table_ids", []),
        "gold_rank": gold_rank,
        "retrieved": retrieved,
    }
