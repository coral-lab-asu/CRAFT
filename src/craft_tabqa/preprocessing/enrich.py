"""LLM enrichment: generate table titles/descriptions and expand queries.

This is the optional preprocessing step that fills in the ``description`` (and
``generated_title``) fields a corpus may lack, and - if asked - the
``subquestion`` / ``query_description`` fields on queries. Both use a small
open-source model served through one of the :mod:`backends`.

Generation is the slow part, so results are cached to a JSONL file keyed by id.
Re-running only generates what is missing, which makes the step resumable.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from craft_tabqa.config import GenerationConfig
from craft_tabqa.loaders.schema import Query, Table

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# The prompts ask for:  "<Field A>":"..." | "<Field B>":"..."
_PAIR_RE = re.compile(r'"[^"]+"\s*:\s*"(.*?)"', re.DOTALL)


def _load_template(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _parse_pair(text: str) -> Tuple[str, str]:
    """Pull the two quoted values out of a ``"a":"..." | "b":"..."`` response.

    Falls back to empty strings for whichever value the model omitted.
    """
    matches = _PAIR_RE.findall(text)
    first = matches[0].strip() if len(matches) >= 1 else ""
    second = matches[1].strip() if len(matches) >= 2 else ""
    return first, second


def _read_cache(cache_file: Path) -> Dict[str, Dict]:
    """Load previously generated records keyed by their id."""
    if not cache_file.exists():
        return {}
    cached = {}
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                cached[record["id"]] = record
    return cached


def _append_cache(cache_file: Path, records: List[Dict]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _batches(items: List, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ---------------------------------------------------------------------------
# Table enrichment
# ---------------------------------------------------------------------------

def _table_prompt(template: str, table: Table, sample_rows: int) -> str:
    headers = ", ".join(str(h) for h in table.get("headers", []))
    rows = table.get("rows", [])[:sample_rows]
    rendered_rows = "\n".join(" | ".join(str(c) for c in row) for row in rows)
    return template.replace("[TABLE HEADERS]", headers).replace("[SUBSET OF TABLES]", rendered_rows)


def enrich_tables(
    tables: List[Table],
    cache_file: str,
    generation: GenerationConfig,
    hf_cache: str = "",
    logger=None,
) -> List[Table]:
    """Return ``tables`` with generated title/description merged in.

    Tables that already have a description are left untouched. Newly generated
    values are cached to ``cache_file`` so a re-run skips them.
    """
    log = logger.info if logger else print
    cache_file = Path(cache_file)
    cached = _read_cache(cache_file)

    pending = [
        t for t in tables
        if not t.get("description") and t["table_id"] not in cached
    ]
    log(f"[enrich] {len(tables):,} tables, {len(cached):,} cached, {len(pending):,} to generate")

    if pending:
        template = _load_template("table_enrich.txt")
        backend = _make_backend(generation, hf_cache)
        for batch in _batches(pending, generation.batch_size):
            prompts = [_table_prompt(template, t, generation.sample_rows) for t in batch]
            outputs = backend.generate(prompts)
            records = []
            for table, output in zip(batch, outputs):
                title, description = _parse_pair(output)
                records.append({"id": table["table_id"], "title": title, "description": description})
            _append_cache(cache_file, records)
            for record in records:
                cached[record["id"]] = record
            log(f"[enrich] generated {len(records)} (total cached {len(cached):,})")

    for table in tables:
        record = cached.get(table["table_id"])
        if record:
            if record.get("description") and not table.get("description"):
                table["description"] = record["description"]
            if record.get("title"):
                table["generated_title"] = record["title"]
    return tables


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_queries(
    queries: List[Query],
    cache_file: str,
    generation: GenerationConfig,
    hf_cache: str = "",
    logger=None,
) -> List[Query]:
    """Return ``queries`` with generated sub-question / description merged in.

    Queries that already carry a sub-question are left untouched. Cached to
    ``cache_file`` for resumability.
    """
    log = logger.info if logger else print
    cache_file = Path(cache_file)
    cached = _read_cache(cache_file)

    pending = [
        q for q in queries
        if not q.get("subquestion") and q["qid"] not in cached
    ]
    log(f"[expand] {len(queries):,} queries, {len(cached):,} cached, {len(pending):,} to generate")

    if pending:
        template = _load_template("query_expand.txt")
        backend = _make_backend(generation, hf_cache)
        for batch in _batches(pending, generation.batch_size):
            prompts = [template.replace("[QUESTION]", q["question"]) for q in batch]
            outputs = backend.generate(prompts)
            records = []
            for query, output in zip(batch, outputs):
                subquestion, description = _parse_pair(output)
                records.append({"id": query["qid"], "subquestion": subquestion, "description": description})
            _append_cache(cache_file, records)
            for record in records:
                cached[record["id"]] = record
            log(f"[expand] generated {len(records)} (total cached {len(cached):,})")

    for query in queries:
        record = cached.get(query["qid"])
        if record:
            if record.get("subquestion") and not query.get("subquestion"):
                query["subquestion"] = record["subquestion"]
            if record.get("description") and not query.get("query_description"):
                query["query_description"] = record["description"]
    return queries


def _make_backend(generation: GenerationConfig, hf_cache: str):
    from craft_tabqa.preprocessing.backends import make_backend

    return make_backend(
        backend=generation.backend,
        model=generation.model,
        max_new_tokens=generation.max_new_tokens,
        temperature=generation.temperature,
        hf_cache=hf_cache,
    )
