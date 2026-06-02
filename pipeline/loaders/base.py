"""
Shared data types used throughout the pipeline.

Both loaders (NQ-Tables, OTT-QA, generic JSONL) and retrieval stages
pass these simple dicts around.  Using TypedDicts rather than classes
keeps things light and JSON-serialisable.
"""

from typing import Dict, List, Optional, TypedDict


class CorpusEntry(TypedDict):
    """
    One table in the corpus.

    Fields
    ------
    table_id  : Unique string identifier.  For Wikipedia tables this is
                usually "{Title}_{HexHash}".
    title     : Human-readable table title.
    headers   : Column names, left-to-right.
    rows      : List of rows; each row is a list of cell strings.
    description : Optional LLM-generated summary (used by Stage 1 corpus).
    """
    table_id: str
    title: str
    headers: List[str]
    rows: List[List[str]]
    description: Optional[str]


class QueryEntry(TypedDict):
    """
    One question / query.

    Fields
    ------
    qid              : Unique question identifier.
    question         : The natural-language question text.
    subquestion      : Optional decomposed sub-question (used in Stage 1).
    query_description: Optional LLM-generated question description (used in Stage 1
                       when query_type includes "description").
    gold_table_ids   : List of correct table IDs (usually length 1).
    answer           : Optional gold answer string (used for end-to-end eval).
    """
    qid: str
    question: str
    subquestion: Optional[str]
    query_description: Optional[str]
    gold_table_ids: List[str]
    answer: Optional[str]
