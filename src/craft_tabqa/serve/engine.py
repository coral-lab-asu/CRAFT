"""A resident retrieval engine for serving one query at a time.

The batch pipeline in :mod:`craft_tabqa.retrieval` is built for scoring a whole
query set at once. A web server instead needs to keep the index and models
loaded and answer single questions with low latency - that is what this engine
does. It loads the cached artifacts once and reuses the Stage 1 and Stage 2
components across requests.

Stage 3 (the embedding API) is optional per query. It is lazily initialised the
first time it is requested, so it costs nothing unless the caller asks for it -
the web demo leaves it off, while the terminal app can offer it as a choice.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np

from craft_tabqa.config import CraftConfig
from craft_tabqa.core.files import load_pickle
from craft_tabqa.loaders import load_corpus
from craft_tabqa.retrieval.stage1_splade import SpladeRetriever
from craft_tabqa.retrieval.stage2_dense import DenseReranker


class RetrievalEngine:
    """Loads a preprocessed corpus and answers single queries interactively."""

    def __init__(self, cfg: CraftConfig, cache_dir: str, logger=None):
        self.log = logger.info if logger else print
        self.cfg = cfg
        cache_dir = Path(cache_dir)

        self.table_ids = load_pickle(cache_dir / "table_ids.pkl")
        self.tables_by_id = {t["table_id"]: t for t in self._load_tables()}

        self.splade = SpladeRetriever(
            index_path=str(cache_dir / "splade_index.pkl"),
            model_id=cfg.stage1.model_id,
            logger=logger,
        )
        self.dense = DenseReranker(
            row_emb_path=str(cache_dir / "row_embeddings.npy"),
            row_meta_path=str(cache_dir / "row_meta.pkl"),
            model_id=cfg.stage2.model_id,
            hf_cache=cfg.hf_cache,
            trust_remote_code="jina" in cfg.stage2.model_id.lower(),
            logger=logger,
        )
        self._row_meta = None       # loaded lazily for Stage 3 mini-tables
        self._neural = None         # the Stage 3 reranker, created on first use
        self._cache_dir = cache_dir
        self.log("[engine] Ready")

    def _load_tables(self):
        return load_corpus(
            self.cfg.data.dataset,
            self.cfg.data.corpus_file,
            self.cfg.data.descriptions_file,
        )

    def search(self, question: str, top_k: int = 10, use_stage3: bool = False) -> List[Dict]:
        """Return the top ``top_k`` tables for ``question``.

        Runs Stages 1 and 2 always. When ``use_stage3`` is true, the Stage 2
        shortlist is reranked with the embedding API (loaded on first use).
        """
        query = {"qid": "live", "question": question, "gold_table_ids": []}

        stage1 = self.splade.retrieve(
            queries=[query],
            table_ids=self.table_ids,
            top_k=self.cfg.stage1.top_k,
            sparse_top_k=self.cfg.stage1.sparse_top_k,
            batch_size=1,
            query_type="query",
        )
        # Keep a wider Stage 2 shortlist when Stage 3 will rerank it.
        stage2_top_k = max(top_k, self.cfg.stage2.top_k) if use_stage3 else top_k
        stage2 = self.dense.rerank(
            stage1_results=stage1,
            top_k=stage2_top_k,
            top_k_rows=self.cfg.stage2.top_k_rows,
            mode=self.cfg.stage2.mode,
        )

        hits = stage2
        if use_stage3:
            hits = self._rerank_stage3(stage2, top_k)

        return [self._describe(hit) for hit in hits[0]["retrieved"][:top_k]]

    def _rerank_stage3(self, stage2: List[Dict], top_k: int) -> List[Dict]:
        """Rerank the Stage 2 shortlist with the embedding API."""
        if self._neural is None:
            from craft_tabqa.retrieval.stage3_neural import NeuralReranker

            self._row_meta = load_pickle(Path(self._cache_dir) / "row_meta.pkl")
            self._neural = NeuralReranker(
                provider=self.cfg.stage3.provider,
                model_id=self.cfg.stage3.model_id,
                api_batch=self.cfg.stage3.api_batch,
                api_sleep=self.cfg.stage3.api_sleep,
                logger=None,
            )
        return self._neural.rerank(
            stage2_results=stage2,
            row_meta=self._row_meta,
            top_k=top_k,
            top_k_rows=self.cfg.stage2.top_k_rows,
        )

    def _describe(self, hit: Dict) -> Dict:
        """Attach a table's title, headers and a few rows to a ranked hit."""
        table = self.tables_by_id.get(hit["table_id"], {})
        return {
            "rank": hit["rank"],
            "table_id": hit["table_id"],
            "score": round(float(hit["score"]), 4),
            "title": table.get("title", ""),
            "headers": table.get("headers", []),
            "rows": table.get("rows", [])[:5],
        }
