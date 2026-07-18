"""The ``craft`` command line tool: preprocess, retrieve, generate, serve.

    craft preprocess --config configs/nq_tables.yaml
    craft retrieve   --config configs/nq_tables.yaml
    craft generate   --config configs/nq_tables.yaml          # LLM enrichment only
    craft serve      --config configs/nq_tables.yaml --port 8000

Each subcommand loads the same YAML config and shares the same cache directory
(``<output_dir>/cache``), so preprocessing artifacts are built once and reused.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

from craft_tabqa.config import CraftConfig, load_config
from craft_tabqa.logging_setup import setup_logger


def main(argv=None) -> None:
    args = _parse_args(argv)

    # `tui` chooses its own config through an interactive menu, so it does not
    # take --config and is dispatched before any config is loaded.
    if args.command == "tui":
        from craft_tabqa.tui import run_tui

        run_tui(configs_dir=args.configs_dir)
        return

    # `export-web` works directly from stage result files, not a run config.
    if args.command == "export-web":
        from craft_tabqa.webexport import export_dataset

        export_dataset(
            dataset=args.dataset,
            stage1=args.stage1, stage2=args.stage2, stage3=args.stage3,
            out_path=args.out, tables_path=args.tables, descriptions_path=args.descriptions,
        )
        return

    cfg = load_config(args.config)
    _apply_hardware_env(cfg)

    output_dir = Path(cfg.data.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    args.func(args, cfg, cache_dir)


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="craft", description="CRAFT cascaded table retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preprocess", help="Build the SPLADE index and row embeddings")
    pre.add_argument("--config", required=True)
    pre.add_argument("--rebuild-index", action="store_true", help="Rebuild the SPLADE index")
    pre.add_argument("--rebuild-rows", action="store_true", help="Rebuild row embeddings")
    pre.add_argument("--skip-rows", action="store_true", help="Skip row encoding (Stage 1 only)")
    pre.set_defaults(func=_cmd_preprocess)

    gen = sub.add_parser("generate", help="Run only the LLM enrichment step")
    gen.add_argument("--config", required=True)
    gen.set_defaults(func=_cmd_generate)

    ret = sub.add_parser("retrieve", help="Run the three-stage retrieval pipeline")
    ret.add_argument("--config", required=True)
    ret.add_argument("--skip-stage1", action="store_true", help="Reuse saved Stage 1 results")
    ret.add_argument("--skip-stage2", action="store_true", help="Reuse saved Stage 2 results")
    ret.add_argument("--no-stage3", action="store_true", help="Skip Stage 3 even if a key is set")
    ret.set_defaults(func=_cmd_retrieve)

    srv = sub.add_parser("serve", help="Serve a retrieval API over the preprocessed corpus")
    srv.add_argument("--config", required=True)
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)
    srv.set_defaults(func=_cmd_serve)

    tui = sub.add_parser("tui", help="Interactive menu-driven terminal app (pick config in-app)")
    tui.add_argument("--configs-dir", default="configs", help="Directory of .yaml configs to choose from")

    exp = sub.add_parser("export-web", help="Export stage results to the website's JSON")
    exp.add_argument("--dataset", required=True, help="Dataset name (e.g. nq, ottqa)")
    exp.add_argument("--stage1", help="stage1 results .jsonl")
    exp.add_argument("--stage2", help="stage2 results .jsonl")
    exp.add_argument("--stage3", help="stage3 results .jsonl")
    exp.add_argument("--tables", help="table corpus for content resolution (NQ tables.jsonl)")
    exp.add_argument("--descriptions", help="per-table descriptions .jsonl (optional)")
    exp.add_argument("--out", required=True, help="output JSON path (e.g. site/data/nq.json)")

    return parser.parse_args(argv)


def _apply_hardware_env(cfg: CraftConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    if cfg.hf_cache:
        os.environ["HF_HOME"] = cfg.hf_cache
    if cfg.hf_token:
        os.environ["HF_TOKEN"] = cfg.hf_token


def _cmd_preprocess(args, cfg, cache_dir):
    from craft_tabqa.preprocessing import preprocess

    logger = setup_logger(str(cache_dir / cfg.log_file))
    preprocess(
        cfg,
        cache_dir=str(cache_dir),
        rebuild_index=args.rebuild_index,
        rebuild_rows=args.rebuild_rows,
        skip_rows=args.skip_rows,
        logger=logger,
    )


def _cmd_generate(args, cfg, cache_dir):
    from craft_tabqa.loaders import load_corpus, load_queries
    from craft_tabqa.preprocessing.enrich import enrich_tables, expand_queries

    logger = setup_logger(str(cache_dir / cfg.log_file))
    if not cfg.generation.enabled:
        logger.warning("generation.enabled is false in the config; nothing to do")
        return

    if cfg.generation.enrich_tables:
        tables = load_corpus(cfg.data.dataset, cfg.data.corpus_file, cfg.data.descriptions_file, logger=logger)
        enrich_tables(tables, str(cache_dir / "table_enrichment.jsonl"), cfg.generation, cfg.hf_cache, logger)
    if cfg.generation.expand_queries:
        queries = load_queries(cfg.data.dataset, cfg.data.queries_file, logger=logger)
        expand_queries(queries, str(cache_dir / "query_expansion.jsonl"), cfg.generation, cfg.hf_cache, logger)


def _cmd_retrieve(args, cfg, cache_dir):
    from craft_tabqa.retrieval import retrieve

    run_dir = _make_run_dir(Path(cfg.data.output_dir), cfg.data.dataset)
    logger = setup_logger(str(run_dir / cfg.log_file))
    logger.info(f"Run directory: {run_dir}")
    retrieve(
        cfg,
        cache_dir=str(cache_dir),
        run_dir=str(run_dir),
        skip_stage1=args.skip_stage1,
        skip_stage2=args.skip_stage2,
        no_stage3=args.no_stage3,
        logger=logger,
    )


def _cmd_serve(args, cfg, cache_dir):
    from craft_tabqa.serve.app import run_server

    run_server(cfg, cache_dir=str(cache_dir), host=args.host, port=args.port)


def _make_run_dir(output_dir: Path, dataset: str) -> Path:
    """Create a fresh, numbered, dated run directory under ``output_dir``."""
    date = datetime.now().strftime("%Y-%m-%d")
    prefix = f"{dataset}_run_"
    existing = [p for p in output_dir.glob(f"{prefix}*") if p.is_dir()] if output_dir.exists() else []
    run_dir = output_dir / f"{prefix}{len(existing) + 1}_{date}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    main()
