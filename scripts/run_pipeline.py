#!/usr/bin/env python3
"""
CRAFT — Full Pipeline Orchestrator
====================================
Creates a dated, numbered run directory and drives both preprocessing and
retrieval while keeping expensive preprocessing artifacts in a shared cache.

Directory layout (all under cfg.data.output_dir, e.g. results/nq_pipeline/):

    cache/                       ← shared preprocessing artifacts (never deleted)
        corpus_texts.pkl
        table_ids.pkl
        splade_index.pkl
        row_texts.pkl
        row_meta.pkl
        row_embeddings.npy

    {dataset}_run_1_2026-05-31/  ← per-run results + run log
        craft_{dataset}.log
        stage1_results.jsonl
        stage2_results.jsonl
        stage3_results.jsonl     (only if Stage 3 enabled)
        recall_summary.csv

Subsequent runs create _run_2_, _run_3_, etc. with independent result files.
Preprocessing is always skipped if all cache artifacts already exist.

Usage
-----
    cd CRAFT/
    python scripts/run_pipeline.py --config configs/nq_tables.yaml

    # Force-rebuild SPLADE index (all other cache files are reused):
    python scripts/run_pipeline.py --config configs/nq_tables.yaml --rebuild-index

    # Skip preprocessing (cache must already exist):
    python scripts/run_pipeline.py --config configs/nq_tables.yaml --skip-preprocess

    # Skip Stage 3 even when an API key is present:
    python scripts/run_pipeline.py --config configs/nq_tables.yaml --no-stage3
"""

import argparse
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make sure the repo root (CRAFT/) is importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CRAFT full pipeline: preprocessing + retrieval with run isolation"
    )
    p.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/nq_tables.yaml)")
    p.add_argument("--rebuild-index",    action="store_true", help="Delete and rebuild the SPLADE index")
    p.add_argument("--rebuild-rows",     action="store_true", help="Delete and rebuild row embeddings")
    p.add_argument("--skip-preprocess",  action="store_true", help="Skip preprocessing (cache must exist)")
    p.add_argument("--skip-rows",        action="store_true", help="Skip row encoding (Stage 1 only)")
    p.add_argument("--no-stage3",        action="store_true", help="Disable Stage 3 even if API key present")
    p.add_argument("--skip-stage1",      action="store_true", help="Load saved Stage 1 results from run dir")
    p.add_argument("--skip-stage2",      action="store_true", help="Load saved Stage 2 results from run dir")
    p.add_argument("--query-type",       default=None,
                   help="Override query_type from config (e.g. query+subquestion+description)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def make_run_dir(base_dir: Path, dataset: str) -> Path:
    """Create results/{base}/{dataset}_run_{n}_{date}/ and return it."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    prefix = f"{dataset}_run_"
    existing = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ) if base_dir.exists() else []
    run_n = len(existing) + 1
    run_dir = base_dir / f"{prefix}{run_n}_{date_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def run_preprocessing(cfg, cache_dir: Path, args, logger):
    """Run Steps 1-4 of preprocessing, writing artifacts to cache_dir."""
    from utils.io_utils import save_pickle, load_pickle
    from pipeline.preprocessing.corpus_builder import build_corpus_texts, build_row_texts
    from pipeline.preprocessing.splade_indexer import build_splade_index
    from pipeline.preprocessing.row_encoder import encode_rows

    logger.info("=" * 60)
    logger.info("PREPROCESSING")
    logger.info(f"  Cache dir : {cache_dir}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1 — Load corpus
    # -----------------------------------------------------------------------
    logger.info("Step 1/4 — Loading corpus …")
    tables = _load_corpus(cfg, logger)
    logger.info(f"  {len(tables):,} tables loaded")

    # -----------------------------------------------------------------------
    # Step 2 — Corpus texts (Stage 1 input)
    # -----------------------------------------------------------------------
    corpus_texts_path = cache_dir / "corpus_texts.pkl"
    table_ids_path    = cache_dir / "table_ids.pkl"

    if corpus_texts_path.exists() and table_ids_path.exists():
        logger.info("Step 2/4 — Corpus texts cache hit; loading …")
        corpus_texts = load_pickle(corpus_texts_path)
        table_ids    = load_pickle(table_ids_path)
    else:
        logger.info(f"Step 2/4 — Building corpus texts  (fields: {cfg.data.corpus_fields}) …")
        corpus_texts, table_ids = build_corpus_texts(tables, corpus_fields=cfg.data.corpus_fields)
        save_pickle(corpus_texts_path, corpus_texts)
        save_pickle(table_ids_path,    table_ids)
        logger.info(f"  {len(corpus_texts):,} corpus texts → {corpus_texts_path}")

    # -----------------------------------------------------------------------
    # Step 3 — SPLADE inverted index
    # -----------------------------------------------------------------------
    index_path = cache_dir / "splade_index.pkl"
    if args.rebuild_index and index_path.exists():
        logger.info("Step 3/4 — --rebuild-index: removing old index …")
        index_path.unlink()

    logger.info("Step 3/4 — SPLADE index …")
    build_splade_index(
        corpus_texts=corpus_texts,
        output_path=str(index_path),
        model_id=cfg.stage1.model_id,
        batch_size=cfg.stage1.batch_size,
        sparse_top_k=cfg.stage1.sparse_top_k,
        hf_cache=cfg.hf_cache,
        logger=logger,
    )

    # -----------------------------------------------------------------------
    # Step 4 — Row texts + embeddings (Stage 2 input)
    # -----------------------------------------------------------------------
    if args.skip_rows:
        logger.info("Step 4/4 — Skipping row encoding (--skip-rows)")
        return

    row_texts_path = cache_dir / "row_texts.pkl"
    row_meta_path  = cache_dir / "row_meta.pkl"
    row_emb_path   = cache_dir / "row_embeddings.npy"

    if args.rebuild_rows:
        logger.info("Step 4/4 — --rebuild-rows: removing old row files …")
        for p in (row_texts_path, row_meta_path, row_emb_path):
            if p.exists():
                p.unlink()

    if row_texts_path.exists() and row_meta_path.exists():
        logger.info("Step 4/4 — Row texts cache hit; loading …")
        row_texts = load_pickle(row_texts_path)
        row_meta  = load_pickle(row_meta_path)
    else:
        logger.info("Step 4/4 — Building row texts …")
        row_texts, row_meta = build_row_texts(tables)
        for meta, text in zip(row_meta, row_texts):
            meta["text"] = text
        save_pickle(row_texts_path, row_texts)
        save_pickle(row_meta_path,  row_meta)
        logger.info(f"  {len(row_texts):,} row texts → {row_texts_path}")

    trust_rc = "jina" in cfg.stage2.model_id.lower()
    encode_rows(
        row_texts=row_texts,
        row_meta=row_meta,
        output_dir=str(cache_dir),
        model_id=cfg.stage2.model_id,
        batch_size=cfg.stage2.batch_size,
        hf_cache=cfg.hf_cache,
        trust_remote_code=trust_rc,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def run_retrieval(cfg, cache_dir: Path, run_dir: Path, args, logger):
    """Run Stages 1-3, reading cache from cache_dir and writing results to run_dir."""
    from utils.io_utils import load_pickle, write_jsonl, read_jsonl
    from pipeline.evaluation.metrics import compute_recall, print_recall_table, save_recall_csv

    logger.info("=" * 60)
    logger.info("RETRIEVAL")
    logger.info(f"  Cache dir : {cache_dir}")
    logger.info(f"  Run dir   : {run_dir}")
    logger.info("=" * 60)

    # Validate cache
    table_ids_path = cache_dir / "table_ids.pkl"
    if not table_ids_path.exists():
        logger.error(f"table_ids.pkl not found at {table_ids_path}. Run preprocessing first.")
        sys.exit(1)
    table_ids = load_pickle(table_ids_path)
    logger.info(f"Table IDs loaded: {len(table_ids):,}")

    # Load queries
    queries = _load_queries(cfg, logger)
    logger.info(f"Queries loaded: {len(queries):,}")

    # Override query_type from CLI if provided
    query_type = args.query_type or cfg.data.query_type
    logger.info(f"Query type: {query_type}")

    # -----------------------------------------------------------------------
    # Stage 1 — SPLADE sparse retrieval
    # -----------------------------------------------------------------------
    stage1_out = run_dir / "stage1_results.jsonl"

    if args.skip_stage1 and stage1_out.exists():
        logger.info(f"Stage 1 — loading saved results from {stage1_out}")
        stage1_results = list(read_jsonl(stage1_out))
        logger.info(f"  Loaded {len(stage1_results):,} results")
    else:
        logger.info("Stage 1 — SPLADE sparse retrieval …")
        index_path = cache_dir / "splade_index.pkl"
        if not index_path.exists():
            logger.error(f"splade_index.pkl not found at {index_path}. Run preprocessing first.")
            sys.exit(1)

        from pipeline.retrieval.stage1 import SpladeRetriever
        retriever = SpladeRetriever(
            index_path=str(index_path),
            model_id=cfg.stage1.model_id,
            hf_cache=cfg.hf_cache,
            logger=logger,
        )
        stage1_results = retriever.retrieve(
            queries=queries,
            table_ids=table_ids,
            top_k=cfg.stage1.top_k,
            sparse_top_k=cfg.stage1.sparse_top_k,
            batch_size=cfg.stage1.batch_size,
            query_type=query_type,
        )
        retriever.unload()
        write_jsonl(stage1_out, stage1_results)
        logger.info(f"Stage 1 results → {stage1_out}")

    s1_recall = compute_recall(stage1_results, cfg.eval_ks)
    logger.info(
        f"Stage 1  Recall@1={s1_recall.get(1,0):.4f}  "
        f"@10={s1_recall.get(10,0):.4f}  "
        f"@{cfg.stage1.top_k}={s1_recall.get(cfg.stage1.top_k,0):.4f}"
    )
    save_recall_csv(
        str(run_dir / "recall_summary.csv"),
        stage="stage1", dataset=cfg.data.dataset,
        metrics=s1_recall, n_queries=len(stage1_results),
        embed_model=cfg.stage1.model_id,
    )

    # -----------------------------------------------------------------------
    # Stage 2 — Dense mini-table reranking
    # -----------------------------------------------------------------------
    stage2_out = run_dir / "stage2_results.jsonl"

    if args.skip_stage2 and stage2_out.exists():
        logger.info(f"Stage 2 — loading saved results from {stage2_out}")
        stage2_results = list(read_jsonl(stage2_out))
        logger.info(f"  Loaded {len(stage2_results):,} results")
    else:
        logger.info(f"Stage 2 — Dense reranking  (mode={cfg.stage2.mode}) …")
        row_emb_path  = cache_dir / "row_embeddings.npy"
        row_meta_path = cache_dir / "row_meta.pkl"

        if not row_emb_path.exists():
            logger.error("row_embeddings.npy not found. Run preprocessing without --skip-rows.")
            sys.exit(1)

        trust_rc     = "jina" in cfg.stage2.model_id.lower()
        query_task   = "retrieval.query"   if trust_rc else None
        passage_task = "retrieval.passage" if trust_rc else None

        from pipeline.retrieval.stage2 import DenseReranker
        reranker = DenseReranker(
            row_emb_path=str(row_emb_path),
            row_meta_path=str(row_meta_path),
            model_id=cfg.stage2.model_id,
            hf_cache=cfg.hf_cache,
            trust_remote_code=trust_rc,
            logger=logger,
        )
        stage2_results = reranker.rerank(
            stage1_results=stage1_results,
            top_k=cfg.stage2.top_k,
            top_k_rows=cfg.stage2.top_k_rows,
            batch_size=cfg.stage2.batch_size,
            score_chunk_size=cfg.stage2.score_chunk_size,
            mode=cfg.stage2.mode,
            query_task=query_task,
            passage_task=passage_task,
        )
        reranker.unload()
        write_jsonl(stage2_out, stage2_results)
        logger.info(f"Stage 2 results → {stage2_out}")

    s2_recall = compute_recall(stage2_results, cfg.eval_ks)
    logger.info(
        f"Stage 2  Recall@1={s2_recall.get(1,0):.4f}  "
        f"@10={s2_recall.get(10,0):.4f}  "
        f"@{cfg.stage2.top_k}={s2_recall.get(cfg.stage2.top_k,0):.4f}"
    )
    save_recall_csv(
        str(run_dir / "recall_summary.csv"),
        stage="stage2", dataset=cfg.data.dataset,
        metrics=s2_recall, n_queries=len(stage2_results),
        embed_model=cfg.stage2.model_id, mode=cfg.stage2.mode,
    )

    # -----------------------------------------------------------------------
    # Stage 3 — Optional neural API reranking
    # -----------------------------------------------------------------------
    stage3_enabled = cfg.stage3.enabled and not args.no_stage3
    s3_recall = None

    if stage3_enabled:
        logger.info(f"Stage 3 — Neural reranking ({cfg.stage3.provider} / {cfg.stage3.model_id}) …")
        row_meta_path = cache_dir / "row_meta.pkl"
        row_meta = load_pickle(str(row_meta_path)) if row_meta_path.exists() else None

        from pipeline.retrieval.stage3 import NeuralReranker
        neural = NeuralReranker(
            provider=cfg.stage3.provider,
            model_id=cfg.stage3.model_id,
            api_batch=cfg.stage3.api_batch,
            api_sleep=cfg.stage3.api_sleep,
            logger=logger,
        )
        stage3_results = neural.rerank(
            stage2_results=stage2_results,
            row_meta=row_meta,
            top_k=cfg.stage3.top_k,
            top_k_rows=cfg.stage2.top_k_rows,
        )
        stage3_out = run_dir / "stage3_results.jsonl"
        write_jsonl(stage3_out, stage3_results)
        logger.info(f"Stage 3 results → {stage3_out}")

        s3_recall = compute_recall(stage3_results, cfg.eval_ks)
        logger.info(
            f"Stage 3  Recall@1={s3_recall.get(1,0):.4f}  "
            f"@10={s3_recall.get(10,0):.4f}  "
            f"@{cfg.stage3.top_k}={s3_recall.get(cfg.stage3.top_k,0):.4f}"
        )
        save_recall_csv(
            str(run_dir / "recall_summary.csv"),
            stage="stage3", dataset=cfg.data.dataset,
            metrics=s3_recall, n_queries=len(stage3_results),
            embed_model=cfg.stage3.model_id,
        )
    else:
        logger.info("Stage 3 — skipped (no API key or --no-stage3)")

    # -----------------------------------------------------------------------
    # Final recall table printed to terminal + log
    # -----------------------------------------------------------------------
    stage_metrics = {"Stage 1 (SPLADE)": s1_recall, "Stage 2 (Dense)": s2_recall}
    n_queries     = {"Stage 1 (SPLADE)": len(stage1_results), "Stage 2 (Dense)": len(stage2_results)}
    if s3_recall:
        stage_metrics["Stage 3 (Neural)"] = s3_recall
        n_queries["Stage 3 (Neural)"]     = len(stage3_results)

    print_recall_table(
        stage_metrics=stage_metrics,
        n_queries=n_queries,
        ks=[k for k in cfg.eval_ks if k <= max(cfg.stage2.top_k, cfg.stage1.top_k)],
        title=f"CRAFT Results — {cfg.data.dataset.upper()}",
    )

    return s1_recall, s2_recall, s3_recall


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_corpus(cfg, logger):
    dataset = cfg.data.dataset.lower()
    if dataset == "nq":
        from pipeline.loaders.nq_tables import load_nq_corpus
        desc_path = os.environ.get("NQ_DESC_PATH") or None
        if desc_path:
            logger.info(f"  Table descriptions: {desc_path}")
        else:
            logger.warning("  NQ_DESC_PATH not set — loading corpus without LLM descriptions")
        return load_nq_corpus(cfg.data.corpus_file, descriptions_path=desc_path)
    elif dataset == "ottqa":
        from pipeline.loaders.ottqa import load_ottqa_corpus
        return load_ottqa_corpus(cfg.data.corpus_file, logger=logger)
    else:
        from pipeline.loaders.generic import load_corpus
        return load_corpus(cfg.data.corpus_file)


def _load_queries(cfg, logger):
    dataset = cfg.data.dataset.lower()
    if dataset == "nq":
        from pipeline.loaders.nq_tables import load_nq_queries
        return load_nq_queries(cfg.data.queries_file)
    elif dataset == "ottqa":
        from pipeline.loaders.ottqa import load_ottqa_queries
        return load_ottqa_queries(cfg.data.queries_file)
    else:
        from pipeline.loaders.generic import load_queries
        return load_queries(cfg.data.queries_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t_start = time.time()

    # --- Config ---
    from pipeline.config import load_config
    cfg = load_config(args.config)

    # Apply hardware env
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    if cfg.hf_cache:
        os.environ["HF_HOME"] = cfg.hf_cache
    if cfg.hf_token:
        os.environ["HF_TOKEN"] = cfg.hf_token

    # Auto-detect table descriptions for NQ if not already set
    dataset = cfg.data.dataset.lower()
    if dataset == "nq" and not os.environ.get("NQ_DESC_PATH"):
        # Try path from config YAML first, then fall back to conventional location
        raw_desc = getattr(cfg.data, "table_descriptions_file", None)
        repo_root = Path(args.config).resolve().parents[1]
        if raw_desc:
            candidate = (repo_root / raw_desc).resolve()
        else:
            candidate = (repo_root / "datasets" / "nq_table_summary_table_description.jsonl").resolve()
        if candidate.exists():
            os.environ["NQ_DESC_PATH"] = str(candidate)

    # --- Directory setup ---
    base_dir  = Path(cfg.data.output_dir)   # e.g. results/nq_pipeline
    cache_dir = base_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(base_dir, dataset)

    # --- Logger (dual: terminal + run-dir log file) ---
    from pipeline.logger import setup_logger
    log_filename = f"craft_{dataset}.log"
    logger = setup_logger(log_file=str(run_dir / log_filename))

    logger.info("=" * 60)
    logger.info("CRAFT Run Pipeline")
    logger.info(f"  Config    : {args.config}")
    logger.info(f"  Dataset   : {cfg.data.dataset}")
    logger.info(f"  Cache dir : {cache_dir.resolve()}")
    logger.info(f"  Run dir   : {run_dir.resolve()}")
    logger.info(f"  Log file  : {(run_dir / log_filename).resolve()}")
    logger.info(f"  GPUs      : {cfg.cuda_devices}")
    logger.info(f"  Query type: {args.query_type or cfg.data.query_type}")
    nq_desc = os.environ.get("NQ_DESC_PATH", "(not set)")
    logger.info(f"  NQ descs  : {nq_desc}")
    logger.info("=" * 60)

    # --- Preprocessing ---
    if args.skip_preprocess:
        logger.info("Preprocessing skipped (--skip-preprocess); using existing cache.")
    else:
        run_preprocessing(cfg, cache_dir, args, logger)

    # Sanity check: cache must exist before retrieval
    required_cache = [cache_dir / "table_ids.pkl", cache_dir / "splade_index.pkl"]
    missing = [p for p in required_cache if not p.exists()]
    if missing:
        logger.error(f"Missing cache files: {missing}")
        logger.error("Run without --skip-preprocess to build them.")
        sys.exit(1)

    # --- Retrieval ---
    run_retrieval(cfg, cache_dir, run_dir, args, logger)

    # --- Summary ---
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed / 60:.1f} min")
    logger.info(f"Cache    : {cache_dir.resolve()}")
    logger.info(f"Results  : {run_dir.resolve()}")
    logger.info(f"Log      : {(run_dir / log_filename).resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
