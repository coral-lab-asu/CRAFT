"""Tests for the LLM enrichment step using a stub backend (no model needed)."""

import craft_tabqa.preprocessing.enrich as enrich
from craft_tabqa.config import GenerationConfig
from craft_tabqa.preprocessing.enrich import _parse_pair, _table_prompt, enrich_tables


class _StubBackend:
    def generate(self, prompts):
        return [
            '"Table Title":"Generated Title" | "Table Description":"Generated description."'
            for _ in prompts
        ]


def test_parse_pair_extracts_two_values():
    assert _parse_pair('"A":"first" | "B":"second"') == ("first", "second")
    assert _parse_pair("no format here") == ("", "")


def test_table_prompt_fills_placeholders():
    table = {"table_id": "t1", "headers": ["Planet", "Moons"], "rows": [["Mercury", "0"]]}
    prompt = _table_prompt("H:[TABLE HEADERS] R:[SUBSET OF TABLES]", table, sample_rows=5)
    assert "Planet, Moons" in prompt
    assert "Mercury | 0" in prompt


def test_enrich_tables_fills_missing_fields_and_is_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich, "_make_backend", lambda gen, hf: _StubBackend())
    cache = tmp_path / "enrich.jsonl"
    table = {"table_id": "t1", "title": "", "headers": ["h"], "rows": [["v"]]}

    out = enrich_tables([dict(table)], str(cache), GenerationConfig(enabled=True))
    assert out[0]["description"] == "Generated description."
    assert out[0]["generated_title"] == "Generated Title"

    # Second run must reuse the cache (stub would still work, but nothing pending).
    out2 = enrich_tables([dict(table)], str(cache), GenerationConfig(enabled=True))
    assert out2[0]["description"] == "Generated description."


def test_enrich_tables_skips_tables_with_existing_description(tmp_path, monkeypatch):
    called = {"n": 0}

    class Counting(_StubBackend):
        def generate(self, prompts):
            called["n"] += len(prompts)
            return super().generate(prompts)

    monkeypatch.setattr(enrich, "_make_backend", lambda gen, hf: Counting())
    table = {"table_id": "t1", "title": "T", "headers": ["h"], "rows": [["v"]], "description": "already here"}
    enrich_tables([dict(table)], str(tmp_path / "c.jsonl"), GenerationConfig(enabled=True))
    assert called["n"] == 0
