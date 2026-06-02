from typing import Any, Dict, Iterable, List

CANONICAL_FIELDS = ["title", "headers", "description", "cells"]
ALIAS_MAP = {
    "header": "headers",
    "headers": "headers",
    "title": "title",
    "desc": "description",
    "description": "description",
    "intro": "description",
    "cell": "cells",
    "cells": "cells",
    "cellvalues": "cells",
}


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def parse_corpus_fields(corpus_fields: Iterable[str]) -> List[str]:
    if isinstance(corpus_fields, str):
        parts = [p.strip().lower() for p in corpus_fields.split("+") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in corpus_fields]
    canonical_set = set(CANONICAL_FIELDS)
    fields: List[str] = []
    seen = set()
    for part in parts:
        mapped = ALIAS_MAP.get(part, part)
        if mapped in canonical_set and mapped not in seen:
            fields.append(mapped)
            seen.add(mapped)
    if not fields:
        raise ValueError("No valid corpus fields provided")
    return fields


def normalize_corpus_fields(corpus_fields: Iterable[str]) -> str:
    return "+".join(parse_corpus_fields(corpus_fields))


def _flatten_cells(rows: List[List[str]]) -> str:
    cells: List[str] = []
    for row in rows:
        for cell in row:
            cell_text = _clean_text(cell)
            if cell_text:
                cells.append(cell_text)
    return " ".join(cells)


def build_corpus_texts(
    dataset: str,
    tables: List[Dict[str, Any]],
    corpus_fields: Iterable[str],
) -> List[str]:
    fields = parse_corpus_fields(corpus_fields)
    corpus_texts: List[str] = []
    table_ids: List[str] = []
    for table in tables:
        parts: List[str] = []
        if "title" in fields:
            title = table.get("title") or table.get("new_title") 
            title = _clean_text(title)
            if title:
                parts.append(title)
        if "headers" in fields:
            headers = " ".join([_clean_text(h) for h in table.get("headers", []) if _clean_text(h)])
            if headers:
                parts.append(headers)
        if "description" in fields:
            description = _clean_text(table.get("description"))
            if description:
                parts.append(description)
        if "cells" in fields:
            cell_text = _flatten_cells(table.get("rows", []))
            if cell_text:
                parts.append(cell_text)
        corpus_texts.append(" ".join(parts))
        table_ids.append(table.get("table_id"))
    return corpus_texts, table_ids


def build_query_texts(questions: List[Dict[str, Any]], query_type: str) -> List[str]:
    query_texts: List[str] = []
    for item in questions:
        question = _clean_text(item.get("question"))
        subquestion = _clean_text(item.get("subquestion"))
        qt = str(query_type).strip().lower().replace(" ", "")
        parts: List[str] = []
        if "query" in qt:
            if question:
                parts.append(question)
        if "subquestion" in qt:
            if subquestion:
                parts.append(subquestion)
        if "description" in qt or "querydescription" in qt:
            # support question-level generated description from NQ loader
            qdesc = _clean_text(item.get("query_description") or item.get("GeneratedQuestionDescription"))
            if qdesc:
                parts.append(qdesc)

        if parts:
            query_texts.append(" ".join(parts))
        else:
            query_texts.append(question)
    return query_texts
