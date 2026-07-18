"""Typed configuration for the pipeline, loaded from a YAML file.

The precedence when resolving a value is: environment variables (including
anything read from a ``.env`` file) > YAML file > dataclass defaults. Sensitive
values (API keys, GPU selection, HuggingFace cache) come from the environment so
they never live in version control.

    cfg = load_config("configs/nq_tables.yaml")
    cfg.stage1.top_k      # 5000
    cfg.stage2.mode       # "representative_row"
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """Where the corpus/queries live and how their text is assembled."""

    dataset: str = "custom"                 # "nq", "ottqa", or "custom"
    corpus_file: str = ""
    queries_file: str = ""
    output_dir: str = "results/pipeline"

    # Table fields concatenated for the SPLADE index.
    corpus_fields: str = "title+headers+description+cells"
    # Query fields concatenated for search: query [+subquestion] [+description].
    query_type: str = "query+subquestion"
    # Optional per-table generated titles/descriptions (NQ-Tables).
    descriptions_file: str = ""


@dataclass
class GenerationConfig:
    """Optional LLM step that writes titles/descriptions and query expansions.

    Disabled by default. When enabled, an open-source model (served locally via
    vLLM) enriches the corpus and, if requested, expands queries before indexing.
    """

    enabled: bool = False
    backend: str = "vllm"                   # "vllm", "transformers", or "openai"
    model: str = "Qwen/Qwen3-8B"
    enrich_tables: bool = True              # generate table title + description
    expand_queries: bool = False            # generate sub-question + description
    max_new_tokens: int = 512
    batch_size: int = 64
    sample_rows: int = 10                   # rows shown to the model per table
    temperature: float = 0.2


@dataclass
class Stage1Config:
    """SPLADE sparse retrieval over the full corpus."""

    model_id: str = "naver/splade_v2_distil"
    sparse_top_k: int = 5_000               # non-zero terms kept per vector
    top_k: int = 5_000                      # tables passed to Stage 2
    batch_size: int = 128


@dataclass
class Stage2Config:
    """Dense reranking over table rows.

    ``representative_row`` (default): a table's score is its best matching row's
    score - no re-encoding at query time. ``mini_table``: build a top-rows
    mini-table per candidate and re-encode it (the original paper method).
    """

    model_id: str = "sentence-transformers/all-mpnet-base-v2"
    mode: str = "representative_row"        # "representative_row" or "mini_table"
    top_k: int = 100                        # tables passed to Stage 3
    top_k_rows: int = 5                     # rows per mini-table
    batch_size: int = 256


@dataclass
class Stage3Config:
    """Optional reranking via an embedding API (OpenAI or Gemini).

    Auto-enabled when the matching API key is present in the environment.
    """

    enabled: bool = False
    provider: str = "openai"                # "openai" or "gemini"
    model_id: str = "text-embedding-3-large"
    top_k: int = 50
    api_batch: int = 100
    api_sleep: float = 0.5


@dataclass
class CraftConfig:
    """The whole configuration: data, generation, three stages, and hardware."""

    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)

    cuda_devices: str = "0"
    hf_cache: str = ""
    hf_token: str = ""
    log_file: str = "craft.log"
    eval_ks: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 500])


def _apply(section: dict, target) -> None:
    """Copy known keys from a YAML ``section`` onto a dataclass ``target``."""
    for key, value in section.items():
        if hasattr(target, key):
            setattr(target, key, value)


def load_config(yaml_path: str) -> CraftConfig:
    """Load a YAML config, overlay ``.env``, and resolve sensitive values."""
    import yaml
    from dotenv import load_dotenv

    yaml_path = Path(yaml_path)
    load_dotenv(dotenv_path=yaml_path.parent.parent / ".env", override=False)

    raw = yaml.safe_load(yaml_path.read_text()) or {}
    cfg = CraftConfig()

    _apply(raw.get("data", {}), cfg.data)
    _apply(raw.get("generation", {}), cfg.generation)
    _apply(raw.get("stage1", {}), cfg.stage1)
    _apply(raw.get("stage2", {}), cfg.stage2)
    _apply(raw.get("stage3", {}), cfg.stage3)

    for key in ("cuda_devices", "log_file", "eval_ks", "hf_cache", "hf_token"):
        if key in raw:
            setattr(cfg, key, raw[key])

    # Environment wins for hardware and secrets.
    cfg.hf_cache = os.environ.get("HF_HOME", cfg.hf_cache)
    cfg.hf_token = os.environ.get("HF_TOKEN", cfg.hf_token)
    cfg.cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", cfg.cuda_devices)

    # Turn Stage 3 on automatically if a usable API key exists.
    if not cfg.stage3.enabled:
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            cfg.stage3.enabled = True

    return cfg
