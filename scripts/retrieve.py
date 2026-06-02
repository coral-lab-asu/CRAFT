#!/usr/bin/env python3
"""
CRAFT — Step B: Query-Time Retrieval
======================================
Run this after preprocess.py.  It reads your query file and passes every
question through the three-stage pipeline:

  Stage 1 (always)     — SPLADE sparse retrieval → top-5000 candidates
  Stage 2 (always)     — Dense mini-table reranking → top-100
  Stage 3 (optional)   — API-based neural reranking → top-50
                          (auto-skipped when no API key is set)

All queries are encoded together in one pass per stage for maximum throughput.
Results are written to the output directory after each stage so you keep
intermediate files even if Stage 3 fails.

Usage
-----
    python scripts/retrieve.py --config configs/nq_tables.yaml

    # Skip Stage 3 even if an API key is present:
    python scripts/retrieve.py --config configs/nq_tables.yaml --no-stage3

    # Resume from saved Stage 1 results (skip Stage 1 encoding):
    python scripts/retrieve.py --config configs/nq_tables.yaml --skip-stage1
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CRAFT retrieval pipeline: Stage 1 → Stage 2 → (Stage 3)"
    )
    p.add_argument("--config",      required=True, help="Path to YAML config file")
    p.add_argument("--no-stage3",   action="store_true", help="Skip Stage 3 even if API key is set")
    p.add_argument("--skip-stage1", action="store_true", help="Load saved Stage 1 results instead of running SPLADE")
    p.add_argument("--skip-stage2", action="store_true", help="Load saved Stage 2 results (run Stage 3 only)")
    return p.parse_args()


def load_queries_for_config(cfg, logger) -> list:
    """Route to the right query loader based on cfg.data.dataset."""
    dataset = cfg.data.dataset.lower()

    if dataset == "nq":
        from pipeline.loaders.nq_tables import load_nq_queries
        logger.info(f"[retrieve] Loading NQ queries from {cfg.data.queries_file}")
        return load_nq_queries(cfg.data.queries_file)

    elif dataset == "ottqa":
        from pipeline.loaders.ottqa import load_ottqa_queries
        logger.info(f"[retrieve] Loading OTT-QA queries from {cfg.data.queries_file}")
        return load_ottqa_queries(cfg.data.queries_file)

    else:
        from pipeline.loaders.generic import load_queries
        logger.info(f"[retrieve] Loading queries from {cfg.data.queries_file}")
        return load_queries(cfg.data.queries_file)


def main():
    args = parse_args()
    t_start = time.time()

    # --- Config & environment ---
    from pipeline.config import load_config
    cfg = load_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    if cfg.hf_cache:
        os.environ["HF_HOME"] = cfg.hf_cache
    if cfg.hf_token:
        os.environ["HF_TOKEN"] = cfg.hf_token

    # --- Logger ---
    from pipeline.logger import setup_logger
    out_dir = Path(cfg.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(out_dir / cfg.log_file))

    logger.info("=" * 60)
    logger.info("CRAFT Retrieval Pipeline")
    logger.info(f"  Config   : {args.config}")
    logger.info(f"  Dataset  : {cfg.data.dataset}")
    logger.info(f"  Output   : {out_dir}")
    logger.info(f"  GPUs     : {cfg.cuda_devices}")
    logger.info(f"  Stage2   : mode={cfg.stage2.mode}")
    stage3_enabled = cfg.stage3.enabled and not args.no_stage3
    logger.info(f"  Stage3   : {'enabled (' + cfg.stage3.provider + ')' if stage3_enabled else 'disabled'}")
    logger.info("=" * 60)

    from utils.io_utils import load_pickle, write_jsonl, read_jsonl
    from pipeline.evaluation.metrics import compute_recall, print_recall_table, save_recall_csv

    # -----------------------------------------------------------------------
    # Load queries
    # -----------------------------------------------------------------------
    queries = load_queries_for_config(cfg, logger)
    logger.info(f"Queries loaded: {len(queries):,}")

    # Load preprocessing artifacts
    table_ids_path = out_dir / "table_ids.pkl"
    if not table_ids_path.exists():
        logger.error(f"table_ids.pkl not found at {table_ids_path}. Run preprocess.py first.")
        sys.exit(1)
    table_ids = load_pickle(table_ids_path)
    logger.info(f"Table IDs loaded: {len(table_ids):,}")

    # -----------------------------------------------------------------------
    # Stage 1 — SPLADE sparse retrieval
    # -----------------------------------------------------------------------
    stage1_out = out_dir / "stage1_results.jsonl"

    if args.skip_stage1 and stage1_out.exists():
        logger.info(f"Stage 1 — loading saved results from {stage1_out}")
        stage1_results = list(read_jsonl(stage1_out))
        logger.info(f"  Loaded {len(stage1_results):,} Stage 1 results")
    else:
        logger.info("Stage 1 — SPLADE sparse retrieval …")
        index_path = out_dir / "splade_index.pkl"
        if not index_path.exists():
            logger.error(f"splade_index.pkl not found at {index_path}. Run preprocess.py first.")
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
            query_type=cfg.data.query_type,
        )
        retriever.unload()

        write_jsonl(stage1_out, stage1_results)
        logger.info(f"Stage 1 results saved to {stage1_out}")

    # Stage 1 recall
    s1_recall = compute_recall(stage1_results, cfg.eval_ks)
    logger.info(f"Stage 1 Recall@1={s1_recall[1]:.4f}  @10={s1_recall[10]:.4f}  "
                f"@{cfg.stage1.top_k}={s1_recall.get(cfg.stage1.top_k, 0):.4f}")
    save_recall_csv(str(out_dir / "recall_summary.csv"),
                    stage="stage1", dataset=cfg.data.dataset,
                    metrics=s1_recall, n_queries=len(stage1_results),
                    embed_model=cfg.stage1.model_id)

    # -----------------------------------------------------------------------
    # Stage 2 — Dense mini-table reranking
    # -----------------------------------------------------------------------
    stage2_out = out_dir / "stage2_results.jsonl"

    if args.skip_stage2 and stage2_out.exists():
        logger.info(f"Stage 2 — loading saved results from {stage2_out}")
        stage2_results = list(read_jsonl(stage2_out))
        logger.info(f"  Loaded {len(stage2_results):,} Stage 2 results")
    else:
        logger.info(f"Stage 2 — Dense reranking  (mode={cfg.stage2.mode}) …")
        row_emb_path  = out_dir / "row_embeddings.npy"
        row_meta_path = out_dir / "row_meta.pkl"

        if not row_emb_path.exists():
            logger.error(f"row_embeddings.npy not found. Run preprocess.py (without --skip-rows) first.")
            sys.exit(1)

        trust_rc = "jina" in cfg.stage2.model_id.lower()

        # JINA v3 uses task prompts for query vs passage
        query_task   = "retrieval.query"   if "jina" in cfg.stage2.model_id.lower() else None
        passage_task = "retrieval.passage" if "jina" in cfg.stage2.model_id.lower() else None

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
        logger.info(f"Stage 2 results saved to {stage2_out}")

    # Stage 2 recall
    s2_recall = compute_recall(stage2_results, cfg.eval_ks)
    logger.info(f"Stage 2 Recall@1={s2_recall[1]:.4f}  @10={s2_recall.get(10,0):.4f}  "
                f"@{cfg.stage2.top_k}={s2_recall.get(cfg.stage2.top_k,0):.4f}")
    save_recall_csv(str(out_dir / "recall_summary.csv"),
                    stage="stage2", dataset=cfg.data.dataset,
                    metrics=s2_recall, n_queries=len(stage2_results),
                    embed_model=cfg.stage2.model_id, mode=cfg.stage2.mode)

    # -----------------------------------------------------------------------
    # Stage 3 — Neural API reranking (optional)
    # -----------------------------------------------------------------------
    stage3_out = out_dir / "stage3_results.jsonl"

    if stage3_enabled:
        logger.info(f"Stage 3 — Neural reranking  ({cfg.stage3.provider} / {cfg.stage3.model_id}) …")

        row_meta_path = out_dir / "row_meta.pkl"
        row_meta = load_pickle(row_meta_path) if row_meta_path.exists() else None

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

        write_jsonl(stage3_out, stage3_results)
        logger.info(f"Stage 3 results saved to {stage3_out}")

        s3_recall = compute_recall(stage3_results, cfg.eval_ks)
        logger.info(f"Stage 3 Recall@1={s3_recall[1]:.4f}  @10={s3_recall.get(10,0):.4f}  "
                    f"@{cfg.stage3.top_k}={s3_recall.get(cfg.stage3.top_k,0):.4f}")
        save_recall_csv(str(out_dir / "recall_summary.csv"),
                        stage="stage3", dataset=cfg.data.dataset,
                        metrics=s3_recall, n_queries=len(stage3_results),
                        embed_model=cfg.stage3.model_id)
    else:
        s3_recall = None
        logger.info("Stage 3 — skipped (no API key or --no-stage3 flag)")

    # -----------------------------------------------------------------------
    # Final recall comparison table
    # -----------------------------------------------------------------------
    stage_metrics = {"Stage 1 (SPLADE)": s1_recall, "Stage 2 (Dense)": s2_recall}
    n_queries = {"Stage 1 (SPLADE)": len(stage1_results), "Stage 2 (Dense)": len(stage2_results)}
    if s3_recall:
        stage_metrics["Stage 3 (Neural)"] = s3_recall
        n_queries["Stage 3 (Neural)"] = len(stage3_results)

    print_recall_table(
        stage_metrics=stage_metrics,
        n_queries=n_queries,
        ks=[k for k in cfg.eval_ks if k <= max(cfg.stage2.top_k, cfg.stage1.top_k)],
        title=f"CRAFT Results — {cfg.data.dataset.upper()}",
    )

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Retrieval complete in {elapsed / 60:.1f} min")
    logger.info(f"Results in: {out_dir.resolve()}")
    logger.info("  stage1_results.jsonl")
    logger.info("  stage2_results.jsonl")
    if stage3_enabled:
        logger.info("  stage3_results.jsonl")
    logger.info("  recall_summary.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
