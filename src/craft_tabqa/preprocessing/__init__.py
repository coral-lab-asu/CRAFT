"""Offline preprocessing: build the cached artifacts retrieval reads from."""

from craft_tabqa.preprocessing.pipeline import preprocess
from craft_tabqa.preprocessing.row_encoder import encode_rows
from craft_tabqa.preprocessing.splade_index import build_splade_index

__all__ = ["preprocess", "build_splade_index", "encode_rows"]
