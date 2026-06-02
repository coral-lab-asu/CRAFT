"""
Generic JSONL loaders for custom datasets.

If you have your own table corpus or question set, format them as JSONL with
the schemas below and these functions will handle the rest.

Corpus schema (one JSON object per line)
-----------------------------------------
{
  "table_id":    "some_unique_id",      # required
  "title":       "Table Title",         # required
  "headers":     ["Col1", "Col2"],      # required
  "rows":        [["v1", "v2"], ...],   # required
  "description": "Optional summary."   # optional
}

Query schema (one JSON object per line)
----------------------------------------
{
  "qid":           "q001",             # required
  "question":      "Who won ...?",     # required
  "subquestion":   "What team ...?",   # optional
  "gold_table_ids": ["table_id_1"],    # required for evaluation
  "answer":        "Real Madrid"       # optional
}
"""

from pathlib import Path
from typing import List

from pipeline.loaders.base import CorpusEntry, QueryEntry


def load_corpus(corpus_file: str) -> List[CorpusEntry]:
    """
    Load a JSONL corpus file into a list of CorpusEntry dicts.

    Rows that are missing required fields ('table_id', 'title', 'headers',
    'rows') are skipped with a warning so one bad line doesn't crash the run.
    """
    import json

    entries: List[CorpusEntry] = []
    skipped = 0

    with open(corpus_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[corpus loader] line {line_no}: JSON error — {e}")
                skipped += 1
                continue

            # Require the four core fields
            if not all(k in obj for k in ("table_id", "title", "headers", "rows")):
                print(f"[corpus loader] line {line_no}: missing required field, skipping")
                skipped += 1
                continue

            entries.append(
                CorpusEntry(
                    table_id=str(obj["table_id"]),
                    title=str(obj.get("title", "")),
                    headers=[str(h) for h in obj.get("headers", [])],
                    rows=[[str(c) for c in row] for row in obj.get("rows", [])],
                    description=obj.get("description") or None,
                )
            )

    if skipped:
        print(f"[corpus loader] skipped {skipped} malformed lines")
    print(f"[corpus loader] loaded {len(entries):,} tables from {corpus_file}")
    return entries


def load_queries(queries_file: str) -> List[QueryEntry]:
    """
    Load a JSONL queries file into a list of QueryEntry dicts.

    Lines missing 'qid' or 'question' are skipped.
    """
    import json

    entries: List[QueryEntry] = []
    skipped = 0

    with open(queries_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[query loader] line {line_no}: JSON error — {e}")
                skipped += 1
                continue

            if "qid" not in obj or "question" not in obj:
                print(f"[query loader] line {line_no}: missing 'qid' or 'question', skipping")
                skipped += 1
                continue

            # gold_table_ids may be a list or a single string
            gold_raw = obj.get("gold_table_ids", obj.get("gold_table_id", []))
            if isinstance(gold_raw, str):
                gold_raw = [gold_raw]

            entries.append(
                QueryEntry(
                    qid=str(obj["qid"]),
                    question=str(obj["question"]),
                    subquestion=obj.get("subquestion") or obj.get("GeneratedSubQuestion") or None,
                    gold_table_ids=[str(g) for g in gold_raw],
                    answer=obj.get("answer") or None,
                )
            )

    if skipped:
        print(f"[query loader] skipped {skipped} malformed lines")
    print(f"[query loader] loaded {len(entries):,} queries from {queries_file}")
    return entries
