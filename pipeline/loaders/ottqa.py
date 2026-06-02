"""
OTT-QA dataset loader.

Thin wrappers over the existing utils/datasets.py functions.

Expected file paths (set these in your config YAML)
----------------------------------------------------
  corpus_file  : datasets/OTT-QA/all_plain_tables.json
  queries_file : datasets/OTT-QA/released_data/dev.json
"""

from pathlib import Path
from typing import List, Optional

from pipeline.loaders.base import CorpusEntry, QueryEntry


def load_ottqa_corpus(tables_path: str, logger=None) -> List[CorpusEntry]:
    """
    Load OTT-QA table corpus.

    Args:
        tables_path: Path to traindev_tables_tok/ directory (one JSON per table)
                     or traindev_tables.json (single dict file).
        logger:      Optional logger; stats are printed at load time.

    Returns:
        List of CorpusEntry dicts.
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.datasets import load_ott_tables

    log = logger.info if logger else None
    raw_tables = load_ott_tables(tables_path=Path(tables_path), logger=log)

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


def load_ottqa_queries(
    queries_path: str,
    max_questions: Optional[int] = None,
) -> List[QueryEntry]:
    """
    Load OTT-QA dev questions.

    Args:
        queries_path:  Path to released_data/dev.json.
        max_questions: Truncate for debugging.

    Returns:
        List of QueryEntry dicts.
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.datasets import load_ott_questions

    raw_qs = load_ott_questions(
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
