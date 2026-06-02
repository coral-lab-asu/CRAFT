"""
CRAFT: Training-Free Cascaded Retrieval for Tabular QA
=======================================================
A modular, zero-shot pipeline: SPLADE → Dense Reranking → Neural Reranking

Usage
-----
See scripts/preprocess.py and scripts/retrieve.py for the two entry points,
or import individual stages for custom workflows:

    from pipeline.retrieval.stage1 import SpladeRetriever
    from pipeline.retrieval.stage2 import DenseReranker
    from pipeline.retrieval.stage3 import NeuralReranker
"""

__version__ = "1.0.0"
__authors__ = [
    "Adarsh Singh",
    "Kushal Raj Bhandari",
    "Jianxi Gao",
    "Soham Dan",
    "Vivek Gupta",
]
