"""Tests for the generic JSONL loaders and the dataset dispatch."""

import json

from craft_tabqa.loaders import load_corpus, load_queries


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_jsonl_corpus_loads_valid_and_skips_bad(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write(corpus, [
        {"table_id": "t1", "title": "A", "headers": ["h"], "rows": [["v"]]},
        {"title": "missing id"},  # skipped
    ])
    tables = load_corpus("custom", str(corpus))
    assert len(tables) == 1
    assert tables[0]["table_id"] == "t1"


def test_jsonl_queries_normalize_gold_ids(tmp_path):
    queries = tmp_path / "queries.jsonl"
    _write(queries, [
        {"qid": "q1", "question": "Q?", "gold_table_ids": "t1"},
        {"qid": "q2", "question": "Q2?", "gold_table_ids": ["t2", "t3"]},
    ])
    loaded = load_queries("custom", str(queries))
    assert loaded[0]["gold_table_ids"] == ["t1"]
    assert loaded[1]["gold_table_ids"] == ["t2", "t3"]
