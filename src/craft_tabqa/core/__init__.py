"""Low-level building blocks shared across the pipeline.

Nothing in here knows about a specific dataset or pipeline stage: these are the
plain data-in / data-out helpers (text builders, sparse encoder, file I/O,
recall metrics) that the higher-level modules compose.
"""

from craft_tabqa.core.files import (
    load_pickle,
    read_jsonl,
    save_pickle,
    write_jsonl,
)
from craft_tabqa.core.metrics import recall_at_k, recall_summary
from craft_tabqa.core.text import (
    build_corpus_text,
    build_query_text,
    build_row_texts,
)

__all__ = [
    "load_pickle",
    "save_pickle",
    "read_jsonl",
    "write_jsonl",
    "recall_at_k",
    "recall_summary",
    "build_corpus_text",
    "build_query_text",
    "build_row_texts",
]
