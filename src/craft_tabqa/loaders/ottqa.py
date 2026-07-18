"""Load the OTT-QA corpus and questions into the pipeline's dict shapes.

Expected files (set the paths in your config):

    corpus   datasets/OTT-QA/all_plain_tables.json   (uid -> table dict)
    queries  datasets/OTT-QA/released_data/dev.json

OTT-QA cells are ``[text, [links]]`` pairs; we keep just the text. The table
description is built from ``intro + section_title + section_text``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from craft_tabqa.loaders.schema import Query, Table, normalize_gold_ids


def _cell_text(cell) -> str:
    """Extract the text from an OTT-QA ``[text, [links]]`` cell."""
    if isinstance(cell, list) and cell:
        return str(cell[0])
    return str(cell)


def _read_records(tables_path: Path) -> List[Dict[str, Any]]:
    """Read OTT-QA tables from a directory of JSON files or a single dict file."""
    if tables_path.is_dir():
        return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(tables_path.glob("*.json"))]

    with open(tables_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for table_id, record in raw.items():
        record.setdefault("uid", table_id)
        records.append(record)
    return records


def load_corpus(corpus_file: str, logger=None) -> List[Table]:
    """Load OTT-QA tables, combining metadata fields into a description."""
    records = _read_records(Path(corpus_file))
    if logger:
        logger.info(f"[ottqa] Loaded {len(records):,} tables")

    tables: List[Table] = []
    for record in records:
        headers = [
            h[0] if isinstance(h, list) and h else str(h)
            for h in record.get("header", [])
        ]
        rows = [[_cell_text(cell) for cell in row] for row in record.get("data", [])]

        description_parts = [
            record.get("intro", ""),
            record.get("section_title", ""),
            record.get("section_text", ""),
        ]
        description = " ".join(p.strip() for p in description_parts if p and p.strip()) or None

        tables.append(
            Table(
                table_id=str(record.get("uid") or record.get("title") or ""),
                title=record.get("title", ""),
                headers=headers,
                rows=rows,
                description=description,
            )
        )
    return tables


def load_queries(queries_file: str, max_queries: Optional[int] = None) -> List[Query]:
    """Load OTT-QA dev questions."""
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries: List[Query] = []
    for item in data:
        qid = item.get("question_id") or item.get("qid") or item.get("id")
        question = item.get("question")
        if not qid or not question:
            continue
        queries.append(
            Query(
                qid=str(qid),
                question=question,
                subquestion=None,
                gold_table_ids=normalize_gold_ids(item.get("table_id") or item.get("gold_table_id")),
                answer=item.get("answer-text"),
            )
        )
        if max_queries and len(queries) >= max_queries:
            break
    return queries
