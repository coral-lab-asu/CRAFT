"""Dataset loaders and a single dispatch point that picks the right one.

``load_corpus`` / ``load_queries`` route on the ``dataset`` name from the config
(``"nq"``, ``"ottqa"``, or anything else -> the generic JSONL loader), so callers
never branch on dataset themselves.
"""

from typing import List

from craft_tabqa.loaders import jsonl, nq_tables, ottqa
from craft_tabqa.loaders.schema import Query, Table

__all__ = ["Table", "Query", "load_corpus", "load_queries"]


def load_corpus(dataset: str, corpus_file: str, descriptions_file: str = "", logger=None) -> List[Table]:
    """Load the corpus for ``dataset`` using the matching loader."""
    dataset = dataset.lower()
    if dataset == "nq":
        return nq_tables.load_corpus(corpus_file, descriptions_file or None)
    if dataset == "ottqa":
        return ottqa.load_corpus(corpus_file, logger=logger)
    return jsonl.load_corpus(corpus_file, logger=logger)


def load_queries(dataset: str, queries_file: str, logger=None) -> List[Query]:
    """Load the query set for ``dataset`` using the matching loader."""
    dataset = dataset.lower()
    if dataset == "nq":
        return nq_tables.load_queries(queries_file)
    if dataset == "ottqa":
        return ottqa.load_queries(queries_file)
    return jsonl.load_queries(queries_file, logger=logger)
