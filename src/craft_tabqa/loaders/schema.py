"""The two dict shapes that flow through the whole pipeline: tables and queries.

TypedDicts (rather than classes) keep everything JSON-serialisable and cheap to
pass between processes, which matters for the multi-GPU encoding paths.
"""

from typing import List, Optional, TypedDict


class Table(TypedDict, total=False):
    """One table in the corpus.

    ``table_id``, ``title``, ``headers`` and ``rows`` are always present.
    ``description`` and ``generated_title`` are optional enrichments that the
    LLM preprocessing step (or a dataset's own metadata) may fill in.
    """

    table_id: str
    title: str
    headers: List[str]
    rows: List[List[str]]
    description: Optional[str]
    generated_title: Optional[str]


class Query(TypedDict, total=False):
    """One question.

    ``subquestion`` and ``query_description`` are optional query expansions used
    by Stage 1; ``gold_table_ids`` is only needed to compute recall.
    """

    qid: str
    question: str
    subquestion: Optional[str]
    query_description: Optional[str]
    gold_table_ids: List[str]
    answer: Optional[str]


def normalize_gold_ids(value) -> List[str]:
    """Coerce a gold-table field into a list of strings (accepts str or list)."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]
