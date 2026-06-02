import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_gold_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def load_nq_descriptions(desc_path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    desc_map: Dict[str, Dict[str, Optional[str]]] = {}
    with open(desc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            table_id = data.get("table_index") or data.get("tableId") or data.get("table_id")
            if not table_id:
                continue
            desc_map[str(table_id)] = {
                "title": data.get("Title") or data.get("Table Title") or data.get("Table_Title"),
                "description": data.get("Description")
                or data.get("Table Description")
                or data.get("Table_Description"),
            }
    return desc_map


def load_nq_tables(tables_path: Path, desc_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    desc_map = load_nq_descriptions(desc_path) if desc_path else {}
    tables: List[Dict[str, Any]] = []
    with open(tables_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            table_id = data.get("tableId") or data.get("table_id")
            if not table_id:
                continue
            title = data.get("documentTitle") or data.get("title")
            headers = [col.get("text", "") for col in data.get("columns", [])]
            rows = []
            for row in data.get("rows", []):
                row_cells = [cell.get("text", "") for cell in row.get("cells", [])]
                rows.append(row_cells)
            desc = desc_map.get(str(table_id), {})
            tables.append(
                {
                    "table_id": str(table_id),
                    "title": title,
                    "new_title": desc.get("title"),
                    "description": desc.get("description"),
                    "headers": headers,
                    "rows": rows,
                }
            )
    return tables


def load_ott_tables(tables_path: Path, logger=None) -> List[Dict[str, Any]]:
    """
    Load OTT-QA tables from either:
      - a directory of per-table JSON files (traindev_tables_tok/), or
      - a single JSON dict file (traindev_tables.json).

    Description field is built by joining all available text metadata:
      intro + section_title + section_text (whichever are non-empty).
    """
    log = logger if logger else print
    tables_path = Path(tables_path)

    # ------------------------------------------------------------------
    # Read raw records
    # ------------------------------------------------------------------
    raw_records: List[Dict[str, Any]] = []
    if tables_path.is_dir():
        files = sorted(tables_path.glob("*.json"))
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                record = json.load(fh)
            raw_records.append(record)
    else:
        with open(tables_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for table_id, record in raw.items():
            if "uid" not in record:
                record["uid"] = table_id
            raw_records.append(record)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    n = len(raw_records)
    n_intro        = sum(1 for r in raw_records if r.get("intro",        "").strip())
    n_section_title = sum(1 for r in raw_records if r.get("section_title", "").strip())
    n_section_text  = sum(1 for r in raw_records if r.get("section_text",  "").strip())
    log(f"[ottqa] Tables loaded       : {n:,}")
    log(f"[ottqa] With intro          : {n_intro:,}  ({100*n_intro/n:.1f}%)")
    log(f"[ottqa] With section_title  : {n_section_title:,}  ({100*n_section_title/n:.1f}%)")
    log(f"[ottqa] With section_text   : {n_section_text:,}  ({100*n_section_text/n:.1f}%)")

    # ------------------------------------------------------------------
    # Parse into pipeline dicts
    # ------------------------------------------------------------------
    tables: List[Dict[str, Any]] = []
    for record in raw_records:
        headers = [
            h[0] if isinstance(h, list) and h else str(h)
            for h in record.get("header", [])
        ]
        rows = [
            [cell[0] if isinstance(cell, list) and cell else str(cell) for cell in row]
            for row in record.get("data", [])
        ]
        # Combine all text metadata into one description string
        desc_parts = [
            record.get("intro",         ""),
            record.get("section_title", ""),
            record.get("section_text",  ""),
        ]
        description = " ".join(p.strip() for p in desc_parts if p and p.strip()) or None

        tables.append({
            "table_id":    str(record.get("uid", record.get("title", ""))),
            "title":       record.get("title", ""),
            "description": description,
            "headers":     headers,
            "rows":        rows,
        })
    return tables


def load_nq_questions(path: Path, max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get("qid") or data.get("question_id") or data.get("id")
            question = data.get("OriginalQuestion") or data.get("question") or data.get("question_text")
            if not qid or not question:
                continue
            questions.append(
                {
                    "qid": str(qid),
                    "question": question,
                    "subquestion": data.get("GeneratedSubQuestion"),
                    "query_description": data.get("GeneratedQuestionDescription") or data.get("Table_Description"),
                    "gold_table_ids": _normalize_gold_ids(
                        data.get("gold_table_ids") or data.get("gold_table_id") or data.get("table_id")
                    ),
                }
            )
            if max_questions and len(questions) >= max_questions:
                break
    return questions


def load_ott_questions(path: Path, max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions: List[Dict[str, Any]] = []
    for item in data:
        qid = item.get("question_id") or item.get("qid") or item.get("id")
        question = item.get("question")
        if not qid or not question:
            continue
        questions.append(
            {
                "qid": str(qid),
                "question": question,
                "subquestion": None,
                "gold_table_ids": _normalize_gold_ids(item.get("table_id") or item.get("gold_table_id")),
            }
        )
        if max_questions and len(questions) >= max_questions:
            break
    return questions
