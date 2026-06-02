#!/usr/bin/env python3
"""
CRAFT — Step A: Offline Preprocessing
======================================
Run this once per corpus before retrieval.  It builds and caches:

  1. Corpus texts  — one flat string per table (title + headers + desc + cells)
  2. SPLADE index  — inverted index over corpus texts for Stage 1
  3. Row texts     — one string per table row (title + header:value pairs)
  4. Row embeddings — dense vectors for every row (used by Stage 2)

Nothing in this script touches your query file.  All expensive computation
is skipped on subsequent runs if the output files already exist.

Usage
-----
    python scripts/preprocess.py --config configs/nq_tables.yaml

    # Force rebuild of a specific artifact:
    python scripts/preprocess.py --config configs/nq_tables.yaml --rebuild-index
    python scripts/preprocess.py --config configs/nq_tables.yaml --rebuild-rows
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Make sure the repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CRAFT offline preprocessing (corpus + SPLADE index + row embeddings)"
    )
    p.add_argument("--config", required=True, help="Path to a YAML config file (e.g. configs/nq_tables.yaml)")
    p.add_argument("--rebuild-index", action="store_true", help="Delete and rebuild the SPLADE index even if it exists")
    p.add_argument("--rebuild-rows",  action="store_true", help="Delete and rebuild row embeddings even if they exist")
    p.add_argument("--skip-rows",     action="store_true", help="Skip row encoding (if you only need Stage 1)")
    return p.parse_args()


def load_corpus_for_config(cfg, logger) -> list:
    """Load the corpus using the dataset-specific loader or the generic one."""
    dataset = cfg.data.dataset.lower()

    if dataset == "nq":
        from pipeline.loaders.nq_tables import load_nq_corpus
        # corpus_file should point to tables.jsonl; descriptions are optional
        desc_path = os.environ.get("NQ_DESC_PATH") or None
        logger.info(f"[preprocess] Loading NQ-Tables corpus from {cfg.data.corpus_file}")
        return load_nq_corpus(cfg.data.corpus_file, descriptions_path=desc_path)

    elif dataset == "ottqa":
        from pipeline.loaders.ottqa import load_ottqa_corpus
        logger.info(f"[preprocess] Loading OTT-QA corpus from {cfg.data.corpus_file}")
        return load_ottqa_corpus(cfg.data.corpus_file)

    else:
        from pipeline.loaders.generic import load_corpus
        logger.info(f"[preprocess] Loading custom corpus from {cfg.data.corpus_file}")
        return load_corpus(cfg.data.corpus_file)


def main():
    args = parse_args()
    t_start = time.time()

    # --- Config & environment ---
    from pipeline.config import load_config
    cfg = load_config(args.config)

    # Apply GPU and HuggingFace settings from config / .env
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
    logger.info("CRAFT Preprocessing")
    logger.info(f"  Config   : {args.config}")
    logger.info(f"  Dataset  : {cfg.data.dataset}")
    logger.info(f"  Output   : {out_dir}")
    logger.info(f"  GPUs     : {cfg.cuda_devices}")
    logger.info("=" * 60)

    from utils.io_utils import save_pickle, load_pickle
    from pipeline.preprocessing.corpus_builder import build_corpus_texts, build_row_texts
    from pipeline.preprocessing.splade_indexer import build_splade_index
    from pipeline.preprocessing.row_encoder import encode_rows

    # -----------------------------------------------------------------------
    # Step 1 — Load corpus
    # -----------------------------------------------------------------------
    logger.info("Step 1/4 — Loading corpus …")
    tables = load_corpus_for_config(cfg, logger)
    logger.info(f"  {len(tables):,} tables loaded")

    # -----------------------------------------------------------------------
    # Step 2 — Build corpus texts (Stage 1 input)
    # -----------------------------------------------------------------------
    corpus_texts_path = out_dir / "corpus_texts.pkl"
    table_ids_path    = out_dir / "table_ids.pkl"

    if corpus_texts_path.exists() and table_ids_path.exists():
        logger.info("Step 2/4 — Corpus texts already exist, loading from cache …")
        corpus_texts = load_pickle(corpus_texts_path)
        table_ids    = load_pickle(table_ids_path)
    else:
        logger.info(f"Step 2/4 — Building corpus texts  (fields: {cfg.data.corpus_fields}) …")
        corpus_texts, table_ids = build_corpus_texts(tables, corpus_fields=cfg.data.corpus_fields)
        save_pickle(corpus_texts_path, corpus_texts)
        save_pickle(table_ids_path, table_ids)
        logger.info(f"  Corpus texts saved to {corpus_texts_path}")

    # -----------------------------------------------------------------------
    # Step 3 — Build SPLADE inverted index (Stage 1)
    # -----------------------------------------------------------------------
    index_path = out_dir / "splade_index.pkl"
    if args.rebuild_index and index_path.exists():
        logger.info("Step 3/4 — --rebuild-index: deleting existing index …")
        index_path.unlink()

    logger.info("Step 3/4 — Building SPLADE inverted index …")
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
    # Step 4 — Build row texts + encode rows (Stage 2 input)
    # -----------------------------------------------------------------------
    if args.skip_rows:
        logger.info("Step 4/4 — Skipping row encoding (--skip-rows set)")
    else:
        row_texts_path = out_dir / "row_texts.pkl"
        row_meta_path  = out_dir / "row_meta.pkl"
        row_emb_path   = out_dir / "row_embeddings.npy"

        if args.rebuild_rows:
            logger.info("Step 4/4 — --rebuild-rows: deleting existing row files …")
            for p in (row_texts_path, row_meta_path, row_emb_path):
                if Path(p).exists():
                    Path(p).unlink()

        if row_texts_path.exists() and row_meta_path.exists():
            logger.info("Step 4/4 — Row texts already exist, loading from cache …")
            row_texts = load_pickle(row_texts_path)
            row_meta  = load_pickle(row_meta_path)
        else:
            logger.info("Step 4/4 — Building row texts …")
            row_texts, row_meta = build_row_texts(tables)
            # Store the text inside row_meta so Stage 2 can read it without
            # separately loading a row_texts list (saves one file lookup)
            for meta, text in zip(row_meta, row_texts):
                meta["text"] = text
            save_pickle(row_texts_path, row_texts)
            save_pickle(row_meta_path,  row_meta)
            logger.info(f"  {len(row_texts):,} row texts saved to {row_texts_path}")

        # Determine whether trust_remote_code is needed (JINA v3, etc.)
        trust_rc = "jina" in cfg.stage2.model_id.lower()

        encode_rows(
            row_texts=row_texts,
            row_meta=row_meta,
            output_dir=str(out_dir),
            model_id=cfg.stage2.model_id,
            batch_size=cfg.stage2.batch_size,
            hf_cache=cfg.hf_cache,
            trust_remote_code=trust_rc,
            logger=logger,
        )

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Preprocessing complete in {elapsed / 60:.1f} min")
    logger.info(f"Output directory: {out_dir.resolve()}")
    logger.info("  corpus_texts.pkl  — corpus texts for Stage 1")
    logger.info("  table_ids.pkl     — ordered table ID list")
    logger.info("  splade_index.pkl  — SPLADE inverted index")
    if not args.skip_rows:
        logger.info("  row_texts.pkl     — per-row text strings")
        logger.info("  row_meta.pkl      — per-row metadata (table_id, row_idx, text)")
        logger.info("  row_embeddings.npy — dense row vectors (Stage 2 input)")
    logger.info("=" * 60)
    logger.info("Next step:  python scripts/retrieve.py --config " + args.config)


if __name__ == "__main__":
    main()
