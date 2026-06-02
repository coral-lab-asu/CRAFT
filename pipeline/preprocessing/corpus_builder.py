"""
Build the text representations that each retrieval stage needs.

Stage 1  — one text per table (title + headers + description + cells).
Stage 2  — one text per table row (title + "header: value" pairs).

These are computed once during preprocessing and cached to disk so they don't
need to be rebuilt every time you run retrieval.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from pipeline.loaders.base import CorpusEntry


# ---------------------------------------------------------------------------
# Stage 1 corpus texts
# ---------------------------------------------------------------------------

def build_corpus_texts(
    tables: List[CorpusEntry],
    corpus_fields: str = "title+headers+description+cells",
) -> Tuple[List[str], List[str]]:
    """
    Build one flat text string per table for SPLADE indexing.

    The fields included are controlled by corpus_fields (same syntax as the
    existing utils/corpus_builder.py, so nothing changes for NQ/OTT-QA runs).

    Args:
        tables:        List of CorpusEntry dicts loaded from your corpus.
        corpus_fields: '+'-separated list of fields to include.
                       Options: title, headers, description, cells

    Returns:
        (corpus_texts, table_ids)
        corpus_texts[i] is the text for table_ids[i].
    """
    import sys
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from utils.corpus_builder import normalize_corpus_fields, parse_corpus_fields

    fields = parse_corpus_fields(corpus_fields.split("+"))

    corpus_texts: List[str] = []
    table_ids: List[str] = []

    for t in tables:
        parts = []
        if "title" in fields and t.get("title"):
            parts.append(t["title"])
        if "headers" in fields and t.get("headers"):
            parts.append(" ".join(str(h) for h in t["headers"]))
        if "description" in fields and t.get("description"):
            parts.append(t["description"])
        if "cells" in fields and t.get("rows"):
            for row in t["rows"]:
                parts.append(" ".join(str(c) for c in row))

        # Clean whitespace and join everything with spaces
        text = " ".join(" ".join(str(p).split()) for p in parts if p)
        corpus_texts.append(text)
        table_ids.append(t["table_id"])

    return corpus_texts, table_ids


# ---------------------------------------------------------------------------
# Stage 2 row texts
# ---------------------------------------------------------------------------

def build_row_texts(
    tables: List[CorpusEntry],
) -> Tuple[List[str], List[Dict]]:
    """
    Build one text string per table row for dense row encoding.

    Format: "{title}  {header1}: {cell1}  {header2}: {cell2}  ..."

    This gives the Stage 2 encoder enough context to score how relevant each
    individual row is to a query, without encoding the entire table at once.

    Args:
        tables: List of CorpusEntry dicts.

    Returns:
        (row_texts, row_meta)

        row_texts[i]  — flat string for row i.
        row_meta[i]   — dict with {table_id, row_idx} so you can map a row
                        embedding back to its parent table.
    """
    row_texts: List[str] = []
    row_meta: List[Dict] = []

    for table in tables:
        title = str(table.get("title", "")).strip()
        headers = [str(h) for h in table.get("headers", [])]
        rows = table.get("rows", [])
        table_id = table["table_id"]

        for row_idx, row_cells in enumerate(rows):
            cells = [str(c) for c in row_cells]

            # Pair header with cell value: "Country: France  Year: 2022"
            if headers:
                pairs = "  ".join(
                    f"{h}: {c}" for h, c in zip(headers, cells) if c.strip()
                )
            else:
                pairs = "  ".join(c for c in cells if c.strip())

            text = f"{title}  {pairs}".strip() if pairs else title
            # Normalise internal whitespace
            text = " ".join(text.split())

            row_texts.append(text)
            row_meta.append({"table_id": table_id, "row_idx": row_idx})

    return row_texts, row_meta
