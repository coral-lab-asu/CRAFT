"""
Stage 3: Neural reranking via embedding API (optional).

Reranks Stage 2's top-100 candidates to top-50 using a high-quality
embedding model accessed through an API.

Supported providers
-------------------
openai  — text-embedding-3-small / text-embedding-3-large
gemini  — gemini-embedding-001

Stage 3 is skipped automatically when no API key is set.  You can also
skip it explicitly by setting stage3.enabled = false in your YAML.

The OPENAI_API_KEY / GEMINI_API_KEY values are read from the .env file
(or the environment).  Never hard-code them in your config.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


class NeuralReranker:
    """
    Stage 3 neural reranker backed by an embedding API.

    Typical usage
    -------------
        reranker = NeuralReranker(provider="openai", model_id="text-embedding-3-large")
        stage3_results = reranker.rerank(stage2_results, row_meta, top_k=50)
    """

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

        self._client = None
        self._setup_client()

    def _setup_client(self):
        """Initialise the API client, reading keys from the environment."""
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY not found.  "
                    "Set it in your .env file or environment variables."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self.log(f"[stage3] OpenAI client ready  |  model={self.model_id}")

        elif self.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY not found.  "
                    "Set it in your .env file or environment variables."
                )
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self.log(f"[stage3] Gemini client ready  |  model={self.model_id}")

        else:
            raise ValueError(f"Unknown provider: {self.provider!r}. Use 'openai' or 'gemini'.")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts and return L2-normalised float32 vectors.

        Handles batching and rate limiting internally so callers don't need
        to worry about API limits.
        """
        all_embs: List[np.ndarray] = []

        for i in range(0, len(texts), self.api_batch):
            batch = texts[i : i + self.api_batch]

            if self.provider == "openai":
                resp = self._client.embeddings.create(
                    input=batch, model=self.model_id
                )
                batch_embs = [np.array(e.embedding, dtype=np.float32) for e in resp.data]

            else:  # gemini
                resp = self._client.models.embed_content(
                    model=self.model_id, contents=batch
                )
                batch_embs = [np.array(e.values, dtype=np.float32) for e in resp.embeddings]

            all_embs.extend(batch_embs)

            # Respect rate limits between batches
            if i + self.api_batch < len(texts):
                time.sleep(self.api_sleep)

        mat = np.array(all_embs, dtype=np.float32)   # (N, D)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-10)         # L2-normalised

    def rerank(
        self,
        stage2_results: List[Dict],
        row_meta: Optional[List[Dict]] = None,
        top_k: int = 50,
        top_k_rows: int = 5,
    ) -> List[Dict]:
        """
        Rerank Stage 2 candidates using API embeddings.

        For each query we embed the question and all its top-100 mini-tables
        (built from the Stage 2 pre-computed row meta) in one batched API
        call, then rank by cosine similarity.

        Args:
            stage2_results: Output of DenseReranker.rerank().
            row_meta:       List of {table_id, row_idx, text} dicts from the
                            preprocessing step.  Needed to build mini-tables.
                            If None, the table_id string is used as a fallback.
            top_k:          Tables kept per query.
            top_k_rows:     Rows per table in the mini-table.

        Returns:
            List of result dicts in the unified output format.
        """
        self.log(f"[stage3] Reranking {len(stage2_results)} queries with {self.provider} embeddings …")

        # Build a quick lookup: table_id → list of row texts
        table_to_texts: Dict[str, List[str]] = {}
        if row_meta:
            for meta in row_meta:
                tid = meta["table_id"]
                text = meta.get("text", "")
                if text:
                    table_to_texts.setdefault(tid, []).append(text)

        results = []
        for item in tqdm(stage2_results, desc="Stage 3 reranking", unit="query"):
            candidates = [r["table_id"] for r in item["retrieved"]]

            # Build mini-table text for each candidate
            mini_texts = []
            valid_ids = []
            for tid in candidates:
                rows = table_to_texts.get(tid, [])[:top_k_rows]
                text = " ".join(rows) if rows else tid.replace("_", " ")
                mini_texts.append(text)
                valid_ids.append(tid)

            if not valid_ids:
                results.append({
                    "qid": item["qid"],
                    "question": item["question"],
                    "gold_table_ids": item["gold_table_ids"],
                    "gold_rank": None,
                    "retrieved": [],
                })
                continue

            # Embed query + mini-tables in a single API call
            all_texts = [item["question"]] + mini_texts
            all_vecs = self.embed(all_texts)
            q_vec = all_vecs[0]         # (D,)
            mt_vecs = all_vecs[1:]      # (M, D)

            scores = mt_vecs @ q_vec    # cosine similarity
            ranked_idx = np.argsort(-scores)

            retrieved = [
                {
                    "rank": r + 1,
                    "table_id": valid_ids[i],
                    "score": float(scores[i]),
                }
                for r, i in enumerate(ranked_idx[:top_k])
            ]

            # Compute where the gold table ended up
            ret_ids = [t["table_id"] for t in retrieved]
            gold_rank = None
            for g in item["gold_table_ids"]:
                if g in ret_ids:
                    gold_rank = ret_ids.index(g) + 1
                    break

            results.append(
                {
                    "qid": item["qid"],
                    "question": item["question"],
                    "gold_table_ids": item["gold_table_ids"],
                    "gold_rank": gold_rank,
                    "retrieved": retrieved,
                }
            )

        self.log(f"[stage3] Done reranking {len(results)} queries")
        return results
