"""Load the NQ-Tables corpus and questions into the pipeline's dict shapes.

Expected files (set the paths in your config):

    corpus       datasets/NQ_Tables/tables/tables.jsonl
    queries      datasets/NQ_Tables/interactions/combined.jsonl
    descriptions datasets/nq_table_summary_table_description.jsonl  (optional)

The descriptions file carries LLM-generated titles and descriptions. When
absent, Stage 1 falls back to title + headers + cells.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from craft_tabqa.loaders.schema import Query, Table, normalize_gold_ids


def _load_descriptions(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """Read the per-table description file into ``table_id -> {title, description}``."""
    descriptions: Dict[str, Dict[str, Optional[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            table_id = data.get("table_index") or data.get("tableId") or data.get("table_id")
            if not table_id:
                continue
            descriptions[str(table_id)] = {
                "title": data.get("Title") or data.get("Table Title") or data.get("Table_Title"),
                "description": (
                    data.get("Description")
                    or data.get("Table Description")
                    or data.get("Table_Description")
                ),
            }
    return descriptions


def load_corpus(corpus_file: str, descriptions_file: Optional[str] = None) -> List[Table]:
    """Load NQ-Tables tables, optionally merging in generated titles/descriptions."""
    descriptions = _load_descriptions(Path(descriptions_file)) if descriptions_file else {}

    tables: List[Table] = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            table_id = data.get("tableId") or data.get("table_id")
            if not table_id:
                continue

            extra = descriptions.get(str(table_id), {})
            tables.append(
                Table(
                    table_id=str(table_id),
                    title=data.get("documentTitle") or data.get("title") or "",
                    headers=[col.get("text", "") for col in data.get("columns", [])],
                    rows=[
                        [cell.get("text", "") for cell in row.get("cells", [])]
                        for row in data.get("rows", [])
                    ],
                    description=extra.get("description"),
                    generated_title=extra.get("title"),
                )
            )
    return tables


def load_queries(queries_file: str, max_queries: Optional[int] = None) -> List[Query]:
    """Load NQ-Tables questions, including any pre-generated query expansions."""
    queries: List[Query] = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get("qid") or data.get("question_id") or data.get("id")
            question = data.get("OriginalQuestion") or data.get("question") or data.get("question_text")
            if not qid or not question:
                continue

            queries.append(
                Query(
                    qid=str(qid),
                    question=question,
                    subquestion=data.get("GeneratedSubQuestion"),
                    query_description=(
                        data.get("GeneratedQuestionDescription") or data.get("Table_Description")
                    ),
                    gold_table_ids=normalize_gold_ids(
                        data.get("gold_table_ids") or data.get("gold_table_id") or data.get("table_id")
                    ),
                    answer=None,
                )
            )
            if max_queries and len(queries) >= max_queries:
                break
    return queries
