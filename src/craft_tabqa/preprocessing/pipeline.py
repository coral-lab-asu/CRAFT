"""Offline preprocessing: turn a raw corpus into the cached artifacts retrieval needs.

Run once per corpus. Every artifact is cached, so re-running skips work that is
already done. Produced files (under ``cache_dir``):

    corpus_texts.pkl     one SPLADE input string per table
    table_ids.pkl        table ids in corpus order (row index -> id)
    splade_index.pkl     the SPLADE inverted index (Stage 1)
    row_texts.pkl        one string per table row
    row_meta.pkl         row -> {table_id, row_idx, text} (Stage 2)
    row_embeddings.npy   dense vector per row (Stage 2)
"""

from pathlib import Path
from typing import List

from craft_tabqa.config import CraftConfig
from craft_tabqa.core.files import load_pickle, save_pickle
from craft_tabqa.core.text import build_corpus_text, build_row_texts
from craft_tabqa.loaders import load_corpus
from craft_tabqa.loaders.schema import Table
from craft_tabqa.preprocessing.row_encoder import encode_rows
from craft_tabqa.preprocessing.splade_index import build_splade_index


def preprocess(
    cfg: CraftConfig,
    cache_dir: str,
    rebuild_index: bool = False,
    rebuild_rows: bool = False,
    skip_rows: bool = False,
    logger=None,
) -> None:
    """Build (or reuse) all preprocessing artifacts for ``cfg`` into ``cache_dir``."""
    log = logger.info if logger else print
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log("[preprocess] Loading corpus ...")
    tables = load_corpus(
        cfg.data.dataset, cfg.data.corpus_file, cfg.data.descriptions_file, logger=logger
    )
    log(f"[preprocess] {len(tables):,} tables loaded")

    if cfg.generation.enabled and cfg.generation.enrich_tables:
        tables = _enrich_tables(tables, cfg, cache_dir, logger)

    _build_corpus_texts(tables, cfg, cache_dir, log)
    _build_index(cfg, cache_dir, rebuild_index, logger)

    if skip_rows:
        log("[preprocess] Skipping row encoding (skip_rows=True)")
        return
    _build_rows(tables, cfg, cache_dir, rebuild_rows, logger)


def _enrich_tables(tables: List[Table], cfg: CraftConfig, cache_dir: Path, logger) -> List[Table]:
    """Fill in generated titles/descriptions via the LLM step (cached to disk)."""
    from craft_tabqa.preprocessing.enrich import enrich_tables

    return enrich_tables(
        tables,
        cache_file=str(cache_dir / "table_enrichment.jsonl"),
        generation=cfg.generation,
        hf_cache=cfg.hf_cache,
        logger=logger,
    )


def _build_corpus_texts(tables, cfg, cache_dir, log) -> None:
    texts_path = cache_dir / "corpus_texts.pkl"
    ids_path = cache_dir / "table_ids.pkl"
    if texts_path.exists() and ids_path.exists():
        log("[preprocess] Corpus texts cached")
        return

    log(f"[preprocess] Building corpus texts (fields: {cfg.data.corpus_fields})")
    corpus_texts = [build_corpus_text(t, cfg.data.corpus_fields) for t in tables]
    table_ids = [t["table_id"] for t in tables]
    save_pickle(texts_path, corpus_texts)
    save_pickle(ids_path, table_ids)
    log(f"[preprocess] Saved {len(corpus_texts):,} corpus texts")


def _build_index(cfg, cache_dir, rebuild_index, logger) -> None:
    index_path = cache_dir / "splade_index.pkl"
    if rebuild_index and index_path.exists():
        index_path.unlink()

    corpus_texts = load_pickle(cache_dir / "corpus_texts.pkl")
    build_splade_index(
        corpus_texts=corpus_texts,
        output_path=str(index_path),
        model_id=cfg.stage1.model_id,
        batch_size=cfg.stage1.batch_size,
        sparse_top_k=cfg.stage1.sparse_top_k,
        logger=logger,
    )


def _build_rows(tables, cfg, cache_dir, rebuild_rows, logger) -> None:
    log = logger.info if logger else print
    texts_path = cache_dir / "row_texts.pkl"
    meta_path = cache_dir / "row_meta.pkl"
    emb_path = cache_dir / "row_embeddings.npy"

    if rebuild_rows:
        for p in (texts_path, meta_path, emb_path):
            if p.exists():
                p.unlink()

    if texts_path.exists() and meta_path.exists():
        log("[preprocess] Row texts cached")
        row_texts = load_pickle(texts_path)
        row_meta = load_pickle(meta_path)
    else:
        log("[preprocess] Building row texts ...")
        row_texts, row_meta = build_row_texts(tables)
        save_pickle(texts_path, row_texts)
        save_pickle(meta_path, row_meta)
        log(f"[preprocess] Saved {len(row_texts):,} row texts")

    trust_remote_code = "jina" in cfg.stage2.model_id.lower()
    encode_rows(
        row_texts=row_texts,
        row_meta=row_meta,
        output_dir=str(cache_dir),
        model_id=cfg.stage2.model_id,
        batch_size=cfg.stage2.batch_size,
        hf_cache=cfg.hf_cache,
        trust_remote_code=trust_remote_code,
        logger=logger,
    )
