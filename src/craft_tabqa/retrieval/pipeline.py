"""Run the three retrieval stages end to end and record recall at each step.

Reads the cached artifacts from preprocessing, runs Stage 1 -> Stage 2 ->
Stage 3 (Stage 3 only if enabled and an API key is present), and writes each
stage's results plus a recall summary CSV into ``run_dir``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from craft_tabqa.config import CraftConfig
from craft_tabqa.core.files import load_pickle, read_jsonl, write_jsonl
from craft_tabqa.core.metrics import recall_summary
from craft_tabqa.loaders import load_queries
from craft_tabqa.retrieval.report import log_recall, save_recall_row


@dataclass
class RetrievalResults:
    """What :func:`retrieve` returns when ``return_results=True``.

    ``recall`` always holds each stage's Recall@k dict (Stage 3 is ``None`` when
    it was skipped). ``stage1``/``stage2``/``stage3`` hold the ranked result
    dicts per query, each already truncated to the requested per-stage length.
    ``stage3`` is an empty list when Stage 3 was skipped.
    """

    recall: Dict[str, Optional[Dict[int, float]]]
    stage1: List[Dict] = field(default_factory=list)
    stage2: List[Dict] = field(default_factory=list)
    stage3: List[Dict] = field(default_factory=list)


def retrieve(
    cfg: CraftConfig,
    cache_dir: str,
    run_dir: str,
    skip_stage1: bool = False,
    skip_stage2: bool = False,
    no_stage3: bool = False,
    return_results: bool = False,
    stage1_top_k: Optional[int] = None,
    stage2_top_k: Optional[int] = 100,
    stage3_top_k: Optional[int] = 10,
    logger=None,
):
    """Run retrieval for ``cfg``.

    By default returns the ``{"stage1", "stage2", "stage3"}`` recall dict (Stage
    3 is ``None`` when skipped). Set ``return_results=True`` to instead get a
    :class:`RetrievalResults` that also carries the ranked tables per query.

    The ranked lists are truncated per stage by ``stage{1,2,3}_top_k``. The
    defaults (Stage 1 all, Stage 2 top 100, Stage 3 top 10) apply only to the
    in-memory return value; the JSONL files on disk are always the full,
    untruncated stage outputs.
    """
    log = logger.info if logger else print
    cache_dir = Path(cache_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    table_ids = load_pickle(cache_dir / "table_ids.pkl")
    queries = load_queries(cfg.data.dataset, cfg.data.queries_file, logger=logger)
    log(f"[retrieve] {len(queries):,} queries, {len(table_ids):,} tables")

    stage1 = _run_stage1(cfg, cache_dir, run_dir, queries, table_ids, skip_stage1, logger)
    s1_recall = _report(stage1, cfg, run_dir, "stage1", cfg.stage1.model_id, logger)

    stage2 = _run_stage2(cfg, cache_dir, run_dir, stage1, skip_stage2, logger)
    s2_recall = _report(stage2, cfg, run_dir, "stage2", cfg.stage2.model_id, logger, mode=cfg.stage2.mode)

    s3_recall = None
    stage3: List[Dict] = []
    if cfg.stage3.enabled and not no_stage3:
        stage3 = _run_stage3(cfg, cache_dir, run_dir, stage2, logger)
        s3_recall = _report(stage3, cfg, run_dir, "stage3", cfg.stage3.model_id, logger)
    else:
        log("[retrieve] Stage 3 skipped")

    recall = {"stage1": s1_recall, "stage2": s2_recall, "stage3": s3_recall}
    if not return_results:
        return recall

    return RetrievalResults(
        recall=recall,
        stage1=_truncate(stage1, stage1_top_k),
        stage2=_truncate(stage2, stage2_top_k),
        stage3=_truncate(stage3, stage3_top_k),
    )


def _truncate(results: List[Dict], top_k: Optional[int]) -> List[Dict]:
    """Keep only the top ``top_k`` retrieved tables per query (``None`` = all)."""
    if top_k is None:
        return results
    return [
        {**item, "retrieved": item["retrieved"][:top_k]}
        for item in results
    ]


def _run_stage1(cfg, cache_dir, run_dir, queries, table_ids, skip, logger) -> List[Dict]:
    out = run_dir / "stage1_results.jsonl"
    if skip and out.exists():
        (logger.info if logger else print)(f"[retrieve] Reusing {out}")
        return list(read_jsonl(out))

    from craft_tabqa.retrieval.stage1_splade import SpladeRetriever

    retriever = SpladeRetriever(
        index_path=str(cache_dir / "splade_index.pkl"),
        model_id=cfg.stage1.model_id,
        logger=logger,
    )
    results = retriever.retrieve(
        queries=queries,
        table_ids=table_ids,
        top_k=cfg.stage1.top_k,
        sparse_top_k=cfg.stage1.sparse_top_k,
        batch_size=cfg.stage1.batch_size,
        query_type=cfg.data.query_type,
    )
    retriever.unload()
    write_jsonl(out, results)
    return results


def _run_stage2(cfg, cache_dir, run_dir, stage1, skip, logger) -> List[Dict]:
    out = run_dir / "stage2_results.jsonl"
    if skip and out.exists():
        (logger.info if logger else print)(f"[retrieve] Reusing {out}")
        return list(read_jsonl(out))

    from craft_tabqa.retrieval.stage2_dense import DenseReranker

    is_jina = "jina" in cfg.stage2.model_id.lower()
    reranker = DenseReranker(
        row_emb_path=str(cache_dir / "row_embeddings.npy"),
        row_meta_path=str(cache_dir / "row_meta.pkl"),
        model_id=cfg.stage2.model_id,
        hf_cache=cfg.hf_cache,
        trust_remote_code=is_jina,
        logger=logger,
    )
    results = reranker.rerank(
        stage1_results=stage1,
        top_k=cfg.stage2.top_k,
        top_k_rows=cfg.stage2.top_k_rows,
        batch_size=cfg.stage2.batch_size,
        mode=cfg.stage2.mode,
        query_task="retrieval.query" if is_jina else None,
        passage_task="retrieval.passage" if is_jina else None,
    )
    reranker.unload()
    write_jsonl(out, results)
    return results


def _run_stage3(cfg, cache_dir, run_dir, stage2, logger) -> List[Dict]:
    from craft_tabqa.retrieval.stage3_neural import NeuralReranker

    row_meta_path = cache_dir / "row_meta.pkl"
    row_meta = load_pickle(row_meta_path) if row_meta_path.exists() else None

    reranker = NeuralReranker(
        provider=cfg.stage3.provider,
        model_id=cfg.stage3.model_id,
        api_batch=cfg.stage3.api_batch,
        api_sleep=cfg.stage3.api_sleep,
        logger=logger,
    )
    results = reranker.rerank(
        stage2_results=stage2,
        row_meta=row_meta,
        top_k=cfg.stage3.top_k,
        top_k_rows=cfg.stage2.top_k_rows,
    )
    write_jsonl(run_dir / "stage3_results.jsonl", results)
    return results


def _report(results, cfg, run_dir, stage, model_id, logger, mode="") -> Dict[int, float]:
    recall = recall_summary(results, cfg.eval_ks)
    log_recall(stage, recall, logger)
    save_recall_row(
        run_dir / "recall_summary.csv",
        dataset=cfg.data.dataset,
        stage=stage,
        model_id=model_id,
        mode=mode,
        n_queries=len(results),
        recall=recall,
    )
    return recall
