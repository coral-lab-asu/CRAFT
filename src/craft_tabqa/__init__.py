"""
CRAFT: Training-Free Cascaded Retrieval for Tabular QA.

A three-stage, zero-shot table retrieval pipeline:

    Stage 1  SPLADE sparse retrieval   (full corpus  -> top 5000)
    Stage 2  Dense row reranking       (top 5000     -> top 100)
    Stage 3  Embedding-API reranking   (top 100      -> top 50, optional)

Quick start
-----------
    from craft_tabqa import load_config
    from craft_tabqa.retrieval import SpladeRetriever, DenseReranker

See the `craft` command line tool (``craft --help``) for the batteries-included
preprocess / retrieve / serve entry points.
"""

from craft_tabqa.config import CraftConfig, load_config

__version__ = "1.0.0"

__all__ = ["CraftConfig", "load_config", "__version__"]
