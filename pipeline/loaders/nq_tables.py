"""
NQ-Tables dataset loader.

Thin wrappers over the existing utils/datasets.py functions so that the
new craft/ package stays consistent without duplicating any logic.

Expected file paths (set these in your config YAML)
----------------------------------------------------
  corpus_file  : datasets/NQ_Tables/tables/tables.jsonl
  queries_file : datasets/NQ_Tables/interactions/combined.jsonl
  descriptions : datasets/nq_table_summary_table_description.jsonl  (optional)
"""

from pathlib import Path
from typing import List, Optional

from pipeline.loaders.base import CorpusEntry, QueryEntry


def load_nq_corpus(
    tables_path: str,
    descriptions_path: Optional[str] = None,
) -> List[CorpusEntry]:
    """
    Load NQ-Tables table corpus via the existing utils/datasets.py loader.

    Args:
        tables_path:       Path to NQ_Tables/tables/tables.jsonl
        descriptions_path: Path to nq_table_summary_table_description.jsonl
                           (adds LLM-generated titles and descriptions).

    Returns:
        List of CorpusEntry dicts in the standard pipeline format.
    """
    import sys, os
    # Make sure utils/ is importable when called from any working directory
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.datasets import load_nq_tables

    raw_tables = load_nq_tables(
        tables_path=Path(tables_path),
        desc_path=Path(descriptions_path) if descriptions_path else None,
    )

    return [
        CorpusEntry(
            table_id=t["table_id"],
            title=t.get("title", ""),
            headers=t.get("headers", []),
            rows=t.get("rows", []),
            description=t.get("description"),
        )
        for t in raw_tables
    ]


def load_nq_queries(
    queries_path: str,
    max_questions: Optional[int] = None,
) -> List[QueryEntry]:
    """
    Load NQ-Tables question set.

    Args:
        queries_path:  Path to interactions/combined.jsonl
        max_questions: Truncate to this many questions (useful for debugging).

    Returns:
        List of QueryEntry dicts.
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.datasets import load_nq_questions

    raw_qs = load_nq_questions(
        path=Path(queries_path),
        max_questions=max_questions,
    )

    return [
        QueryEntry(
            qid=q["qid"],
            question=q["question"],
            subquestion=q.get("subquestion"),
            query_description=q.get("query_description"),
            gold_table_ids=q["gold_table_ids"],
            answer=None,
        )
        for q in raw_qs
    ]
