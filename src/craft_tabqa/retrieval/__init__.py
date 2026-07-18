"""The three retrieval stages and the orchestrator that runs them in order."""

from craft_tabqa.retrieval.pipeline import RetrievalResults, retrieve
from craft_tabqa.retrieval.stage1_splade import SpladeRetriever
from craft_tabqa.retrieval.stage2_dense import DenseReranker
from craft_tabqa.retrieval.stage3_neural import NeuralReranker

__all__ = [
    "retrieve",
    "RetrievalResults",
    "SpladeRetriever",
    "DenseReranker",
    "NeuralReranker",
]
