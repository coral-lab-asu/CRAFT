"""Tests for the text builders that feed every retrieval stage."""

from craft_tabqa.core.text import (
    build_corpus_text,
    build_query_text,
    build_row_texts,
    parse_corpus_fields,
)

TABLE = {
    "table_id": "solar_0",
    "title": "Planets",
    "headers": ["Planet", "Moons"],
    "rows": [["Mercury", "0"], ["Saturn", "146"]],
    "description": "Moon counts of the planets.",
}


def test_parse_corpus_fields_resolves_aliases_and_dedupes():
    assert parse_corpus_fields("title+header+cell+cells") == ["title", "headers", "cells"]


def test_parse_corpus_fields_rejects_empty():
    import pytest

    with pytest.raises(ValueError):
        parse_corpus_fields("bogus+unknown")


def test_build_corpus_text_orders_and_includes_selected_fields():
    text = build_corpus_text(TABLE, "title+headers+description+cells")
    assert text.startswith("Planets Planet Moons")
    assert "Moon counts" in text
    assert "Saturn" in text


def test_build_corpus_text_can_drop_cells():
    text = build_corpus_text(TABLE, "title+headers")
    assert "Saturn" not in text


def test_build_row_texts_pairs_headers_with_cells():
    row_texts, row_meta = build_row_texts([TABLE])
    assert row_texts[0] == "Planets Planet: Mercury Moons: 0"
    assert row_meta[1] == {"table_id": "solar_0", "row_idx": 1, "text": row_texts[1]}


def test_build_query_text_combines_and_falls_back():
    q = {"question": "How many moons?", "subquestion": "moon count", "query_description": "counting"}
    assert build_query_text(q, "query+subquestion") == "How many moons? moon count"
    assert build_query_text(q, "query+subquestion+description").endswith("counting")
    assert build_query_text({"question": "Q", "subquestion": ""}, "query+subquestion") == "Q"
