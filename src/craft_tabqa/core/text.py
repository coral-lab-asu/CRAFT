"""Turn structured tables and queries into the flat text each stage indexes.

Three builders live here:

* :func:`build_corpus_text`  - one string per table for the SPLADE index (Stage 1).
* :func:`build_row_texts`    - one string per table row for dense reranking (Stage 2).
* :func:`build_query_text`   - one string per query, combining question fields.

Keeping all text construction in one place means the corpus and the queries are
always tokenised the same way.
"""

from typing import Dict, List, Sequence, Tuple

from craft_tabqa.loaders.schema import Table

# Field names the corpus builder understands, plus the aliases datasets use.
CORPUS_FIELDS = ("title", "headers", "description", "cells")
_FIELD_ALIASES = {
    "title": "title",
    "header": "headers",
    "headers": "headers",
    "desc": "description",
    "description": "description",
    "intro": "description",
    "cell": "cells",
    "cells": "cells",
    "cellvalues": "cells",
}


def _clean(text) -> str:
    """Collapse runs of whitespace and coerce to ``str`` (``None`` -> "")."""
    if text is None:
        return ""
    return " ".join(str(text).split())


def parse_corpus_fields(fields: str) -> List[str]:
    """Resolve a ``"title+headers+cells"`` spec into canonical field names.

    Unknown tokens are ignored; duplicates are dropped while preserving order.
    Raises ``ValueError`` if nothing valid remains.
    """
    resolved: List[str] = []
    for token in fields.split("+"):
        canonical = _FIELD_ALIASES.get(token.strip().lower())
        if canonical and canonical not in resolved:
            resolved.append(canonical)
    if not resolved:
        raise ValueError(f"No valid corpus fields in {fields!r} (choose from {CORPUS_FIELDS})")
    return resolved


def build_corpus_text(table: Table, fields: str = "title+headers+description+cells") -> str:
    """Build the single flat string SPLADE indexes for one table.

    ``fields`` selects which parts of the table to include and in what order.
    """
    selected = parse_corpus_fields(fields)
    parts: List[str] = []

    if "title" in selected:
        title = _clean(table.get("title") or table.get("generated_title"))
        if title:
            parts.append(title)
    if "headers" in selected:
        headers = " ".join(_clean(h) for h in table.get("headers", []) if _clean(h))
        if headers:
            parts.append(headers)
    if "description" in selected:
        description = _clean(table.get("description"))
        if description:
            parts.append(description)
    if "cells" in selected:
        cells = " ".join(
            _clean(cell)
            for row in table.get("rows", [])
            for cell in row
            if _clean(cell)
        )
        if cells:
            parts.append(cells)

    return " ".join(parts)


def build_row_texts(tables: Sequence[Table]) -> Tuple[List[str], List[Dict]]:
    """Build one text string per table row, for dense row encoding.

    Each row is rendered as ``"{title}  {header}: {cell}  ..."`` so the encoder
    sees the table's subject plus the row's own values.

    Returns ``(row_texts, row_meta)`` where ``row_meta[i]`` is
    ``{"table_id", "row_idx", "text"}`` mapping row ``i`` back to its table.
    """
    row_texts: List[str] = []
    row_meta: List[Dict] = []

    for table in tables:
        title = _clean(table.get("title"))
        headers = [_clean(h) for h in table.get("headers", [])]
        table_id = table["table_id"]

        for row_idx, cells in enumerate(table.get("rows", [])):
            values = [_clean(c) for c in cells]
            if headers:
                pairs = "  ".join(
                    f"{h}: {v}" for h, v in zip(headers, values) if v
                )
            else:
                pairs = "  ".join(v for v in values if v)

            text = _clean(f"{title}  {pairs}") if pairs else title
            row_texts.append(text)
            row_meta.append({"table_id": table_id, "row_idx": row_idx, "text": text})

    return row_texts, row_meta


def build_query_text(query: Dict, query_type: str = "query+subquestion") -> str:
    """Combine query fields into one search string.

    ``query_type`` is a ``+``-joined spec; recognised tokens:
    ``query`` (the question), ``subquestion`` (a decomposed follow-up), and
    ``description`` (a generated question description). Falls back to the bare
    question when the requested fields are empty.
    """
    spec = query_type.lower().replace(" ", "")
    question = _clean(query.get("question"))
    parts: List[str] = []

    if "query" in spec and question:
        parts.append(question)
    if "subquestion" in spec:
        subquestion = _clean(query.get("subquestion"))
        if subquestion:
            parts.append(subquestion)
    if "description" in spec:
        description = _clean(query.get("query_description"))
        if description:
            parts.append(description)

    return " ".join(parts) if parts else question
