"""
Configuration for the CRAFT pipeline.

We use plain Python dataclasses so there are no extra dependencies.
Values are loaded from a YAML file first, then .env overrides sensitive keys
(API keys, GPU settings) so you never have to hard-code them in version control.

Typical usage
-------------
    cfg = load_config("configs/nq_tables.yaml")
    print(cfg.stage1.top_k)   # 5000
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sub-configs (one per pipeline concern)
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Paths and field choices for a specific dataset / corpus."""

    # --- Required: set in your YAML or pass on the command line ---
    corpus_file: str = ""        # path to corpus.jsonl
    queries_file: str = ""       # path to queries.jsonl
    output_dir: str = "results/pipeline"

    # --- How to build the text fed to SPLADE (Stage 1) ---
    # Available fields: title, headers, description, cells
    corpus_fields: str = "title+headers+description+cells"

    # --- How to build the query text ---
    # Options: "query", "query+subquestion", "query+description"
    query_type: str = "query+subquestion"

    # --- Dataset shorthand (used by built-in loaders) ---
    # Set to "nq", "ottqa", or "custom" (custom = generic JSONL loader)
    dataset: str = "custom"

    # Optional: path to LLM-generated table descriptions file (NQ-Tables only).
    # Also readable from the NQ_DESC_PATH environment variable.
    table_descriptions_file: str = ""


@dataclass
class Stage1Config:
    """SPLADE sparse retrieval — covers the full corpus."""

    model_id: str = "naver/splade_v2_distil"

    # How many tokens per text to keep in the sparse vector
    sparse_top_k: int = 5_000

    # Tables returned to Stage 2 per query
    top_k: int = 5_000

    # Corpus encoding batch size (increase if you have lots of GPU memory)
    batch_size: int = 128


@dataclass
class Stage2Config:
    """Dense semantic reranking using a Sentence Transformer."""

    # Default: all-mpnet-base-v2 works well across datasets.
    # For OTT-QA: jinaai/jina-embeddings-v3 is used in the paper.
    model_id: str = "sentence-transformers/all-mpnet-base-v2"

    # Number of top rows selected per table to form the mini-table
    top_k_rows: int = 5

    # Tables returned to Stage 3 (or final output if Stage 3 is skipped)
    top_k: int = 100

    # Batch size for encoding mini-tables / rows
    batch_size: int = 256

    # "mini_table" (paper default): re-encode mini-tables at query time.
    # "fast": use pre-encoded row embeddings directly; no re-encoding needed.
    mode: str = "mini_table"

    # How many queries to score against row embeddings at once.
    # Tune this down if you hit GPU OOM during Stage 2 scoring.
    score_chunk_size: int = 64


@dataclass
class Stage3Config:
    """
    Optional neural reranking via an embedding API.

    Stage 3 is skipped automatically when no API key is found.
    Set `enabled = true` in your YAML and provide the key in .env.
    """

    enabled: bool = False         # override to true in YAML if you have a key

    # "openai" uses text-embedding-3-{small,large}
    # "gemini" uses gemini-embedding-001
    provider: str = "openai"
    model_id: str = "text-embedding-3-large"

    # Tables in the final output
    top_k: int = 50

    # Number of texts sent per API call (higher = fewer calls, same cost)
    api_batch: int = 100

    # Seconds to wait between API batches to stay within rate limits
    api_sleep: float = 0.5


@dataclass
class CRAFTConfig:
    """Top-level config — merge of all sub-configs plus hardware settings."""

    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)

    # Comma-separated GPU indices; mirrors CUDA_VISIBLE_DEVICES.
    cuda_devices: str = "0,1"

    # Where HuggingFace caches downloaded models.
    hf_cache: str = ""

    # HuggingFace API token (for gated models like JINA v3).
    hf_token: str = ""

    # Log file path; parent dirs are created automatically.
    log_file: str = "craft_run.log"

    # k-values used when printing Recall@k tables
    eval_ks: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 500])


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> "CRAFTConfig":
    """
    Load a YAML config file and overlay .env values on top.

    Priority (highest → lowest):
      1. Environment variables (including those read from .env)
      2. Values in the YAML file
      3. Dataclass defaults

    Args:
        yaml_path: Path to a YAML file (e.g. configs/nq_tables.yaml).

    Returns:
        Fully resolved CRAFTConfig.
    """
    import yaml
    from dotenv import load_dotenv

    # Load .env from repo root (silently ignore if it doesn't exist)
    load_dotenv(dotenv_path=Path(yaml_path).parent.parent / ".env", override=False)

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = CRAFTConfig()

    # --- data ---
    d = raw.get("data", {})
    cfg.data.corpus_file   = d.get("corpus_file",   cfg.data.corpus_file)
    cfg.data.queries_file  = d.get("queries_file",  cfg.data.queries_file)
    cfg.data.output_dir    = d.get("output_dir",    cfg.data.output_dir)
    cfg.data.corpus_fields           = d.get("corpus_fields",           cfg.data.corpus_fields)
    cfg.data.query_type              = d.get("query_type",              cfg.data.query_type)
    cfg.data.dataset                 = d.get("dataset",                 cfg.data.dataset)
    cfg.data.table_descriptions_file = d.get("table_descriptions_file", cfg.data.table_descriptions_file)

    # --- stage1 ---
    s1 = raw.get("stage1", {})
    cfg.stage1.model_id      = s1.get("model_id",      cfg.stage1.model_id)
    cfg.stage1.sparse_top_k  = s1.get("sparse_top_k",  cfg.stage1.sparse_top_k)
    cfg.stage1.top_k         = s1.get("top_k",         cfg.stage1.top_k)
    cfg.stage1.batch_size    = s1.get("batch_size",     cfg.stage1.batch_size)

    # --- stage2 ---
    s2 = raw.get("stage2", {})
    cfg.stage2.model_id         = s2.get("model_id",         cfg.stage2.model_id)
    cfg.stage2.top_k_rows       = s2.get("top_k_rows",       cfg.stage2.top_k_rows)
    cfg.stage2.top_k            = s2.get("top_k",            cfg.stage2.top_k)
    cfg.stage2.batch_size       = s2.get("batch_size",       cfg.stage2.batch_size)
    cfg.stage2.mode             = s2.get("mode",             cfg.stage2.mode)
    cfg.stage2.score_chunk_size = s2.get("score_chunk_size", cfg.stage2.score_chunk_size)

    # --- stage3 ---
    s3 = raw.get("stage3", {})
    cfg.stage3.enabled   = s3.get("enabled",   cfg.stage3.enabled)
    cfg.stage3.provider  = s3.get("provider",  cfg.stage3.provider)
    cfg.stage3.model_id  = s3.get("model_id",  cfg.stage3.model_id)
    cfg.stage3.top_k     = s3.get("top_k",     cfg.stage3.top_k)
    cfg.stage3.api_batch = s3.get("api_batch", cfg.stage3.api_batch)
    cfg.stage3.api_sleep = s3.get("api_sleep", cfg.stage3.api_sleep)

    # --- top-level ---
    cfg.cuda_devices = raw.get("cuda_devices", cfg.cuda_devices)
    cfg.log_file     = raw.get("log_file",     cfg.log_file)
    cfg.eval_ks      = raw.get("eval_ks",      cfg.eval_ks)

    # --- env overrides (sensitive values) ---
    cfg.hf_cache  = os.environ.get("HF_HOME",  raw.get("hf_cache",  cfg.hf_cache))
    cfg.hf_token  = os.environ.get("HF_TOKEN", raw.get("hf_token",  cfg.hf_token))

    # CUDA_VISIBLE_DEVICES from .env wins over YAML
    env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_gpus:
        cfg.cuda_devices = env_gpus

    # Auto-enable Stage 3 if an API key exists in the environment
    if not cfg.stage3.enabled:
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            cfg.stage3.enabled = True

    return cfg
