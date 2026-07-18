"""Load a custom corpus and query set from plain JSONL files.

Corpus line::

    {"table_id": "...", "title": "...", "headers": [...], "rows": [[...]],
     "description": "optional"}

Query line::

    {"qid": "...", "question": "...", "gold_table_ids": [...],
     "subquestion": "optional", "answer": "optional"}

Malformed lines are skipped with a warning rather than aborting the run.
"""

import json
from typing import List

from craft_tabqa.loaders.schema import Query, Table, normalize_gold_ids

_REQUIRED_TABLE_FIELDS = ("table_id", "title", "headers", "rows")


def load_corpus(corpus_file: str, logger=None) -> List[Table]:
    """Load a JSONL corpus, skipping lines missing required fields."""
    warn = logger.warning if logger else print
    tables: List[Table] = []
    skipped = 0

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                warn(f"[jsonl] corpus line {line_no}: {e}")
                skipped += 1
                continue
            if not all(field in obj for field in _REQUIRED_TABLE_FIELDS):
                warn(f"[jsonl] corpus line {line_no}: missing a required field, skipping")
                skipped += 1
                continue

            tables.append(
                Table(
                    table_id=str(obj["table_id"]),
                    title=str(obj.get("title", "")),
                    headers=[str(h) for h in obj.get("headers", [])],
                    rows=[[str(c) for c in row] for row in obj.get("rows", [])],
                    description=obj.get("description") or None,
                )
            )

    if skipped:
        warn(f"[jsonl] skipped {skipped} malformed corpus lines")
    return tables


def load_queries(queries_file: str, logger=None) -> List[Query]:
    """Load a JSONL query set, skipping lines missing ``qid`` or ``question``."""
    warn = logger.warning if logger else print
    queries: List[Query] = []
    skipped = 0

    with open(queries_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                warn(f"[jsonl] query line {line_no}: {e}")
                skipped += 1
                continue
            if "qid" not in obj or "question" not in obj:
                warn(f"[jsonl] query line {line_no}: missing 'qid' or 'question', skipping")
                skipped += 1
                continue

            queries.append(
                Query(
                    qid=str(obj["qid"]),
                    question=str(obj["question"]),
                    subquestion=obj.get("subquestion") or obj.get("GeneratedSubQuestion") or None,
                    query_description=obj.get("query_description") or None,
                    gold_table_ids=normalize_gold_ids(
                        obj.get("gold_table_ids") or obj.get("gold_table_id")
                    ),
                    answer=obj.get("answer") or None,
                )
            )

    if skipped:
        warn(f"[jsonl] skipped {skipped} malformed query lines")
    return queries
