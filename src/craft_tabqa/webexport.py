"""Export precomputed retrieval results into a compact JSON the website loads.

The demo on the website is a result browser: pick a dataset, search a question,
and see the tables CRAFT retrieved at each stage. That needs a self-contained
JSON per dataset — one entry per query with its stage-1/2/3 ranked tables, plus
enough table metadata (title, and where available headers/rows/description) to
preview each result.

This module turns the pipeline's stage result files (``stage{1,2,3}_results``)
into that JSON. Table content is resolved only for the tables that actually
appear in the exported top-k of some query, so the output stays small.

    craft export-web --dataset nq \
        --stage1 results/stage1/nq_stage1.jsonl \
        --stage2 results/stage2/nq_stage2.jsonl \
        --stage3 results/stage3/nq_stage3_large.jsonl \
        --tables datasets/NQ_Tables/tables/tables.jsonl \
        --descriptions datasets/nq_table_summary_table_description.jsonl \
        --out site/data/nq.json

For OTT-QA (no table corpus on hand) omit ``--tables``; each table then carries
a title derived from its id plus the id itself.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# How many tables to keep per stage in the exported file. Stage 3 is CRAFT's
# final output; stages 1-2 are shown on demand, so a shorter list is enough.
STAGE_TOP_N = {"stage1": 10, "stage2": 10, "stage3": 10}

# Keep the website payload small: cap preview rows, and de-duplicate table
# content into a shared map. Descriptions are intentionally NOT shipped to the
# demo (they are large) — this only affects the website; the pipeline and the
# terminal app still generate and use full descriptions.
MAX_PREVIEW_ROWS = 3

_TRAILING_INDEX = re.compile(r"_\d+$")


def title_from_id(table_id: str) -> str:
    """Derive a human-readable title from a table id like ``Nonso_Anozie_1``."""
    base = _TRAILING_INDEX.sub("", table_id)
    # NQ ids look like "Brazos River_8F7B4BA175AC5E8F" — drop a trailing hex hash.
    base = re.sub(r"_[0-9A-F]{8,}$", "", base)
    return base.replace("_", " ").strip() or table_id


def _ranked_list(record: Dict) -> List[Dict]:
    """Return a record's ranked tables regardless of which schema it uses."""
    items = record.get("retrieved") or record.get("ranked_tables") or []
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append({"table_id": it.get("table_id"), "score": it.get("score")})
        else:
            out.append({"table_id": str(it), "score": None})
    return out


def _gold_ids(record: Dict) -> List[str]:
    gold = record.get("gold_table_ids")
    if gold:
        return list(gold)
    single = record.get("gold_table_id")
    return [single] if single else []


def _load_stage(path: Optional[str], top_n: int) -> Dict[str, List[Dict]]:
    """Map qid -> its top-``top_n`` ranked tables from a stage result file."""
    if not path:
        return {}
    by_qid: Dict[str, List[Dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_qid[str(rec["qid"])] = _ranked_list(rec)[:top_n]
    return by_qid


def _collect_needed_ids(*stage_maps: Dict[str, List[Dict]]) -> set:
    ids = set()
    for stage in stage_maps:
        for ranked in stage.values():
            for item in ranked:
                if item["table_id"]:
                    ids.add(item["table_id"])
    return ids


def _load_nq_tables(tables_path: str, needed: set) -> Dict[str, Dict]:
    """Resolve title/headers/rows for the NQ tables we actually export."""
    content: Dict[str, Dict] = {}
    with open(tables_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tid = t.get("tableId") or t.get("table_id")
            if not tid or tid not in needed:
                continue
            content[str(tid)] = {
                "title": t.get("documentTitle") or t.get("title") or title_from_id(tid),
                "headers": [c.get("text", "") for c in t.get("columns", [])],
                "rows": [[c.get("text", "") for c in r.get("cells", [])]
                         for r in t.get("rows", [])][:MAX_PREVIEW_ROWS],
            }
    return content


def _table_entry(table_id: str, content: Dict[str, Dict]) -> Dict:
    """The shared-map entry for a table: preview if resolved, else title only."""
    if table_id in content:
        return content[table_id]
    return {"title": title_from_id(table_id), "headers": [], "rows": []}


def export_dataset(
    dataset: str,
    stage1: Optional[str],
    stage2: Optional[str],
    stage3: Optional[str],
    out_path: str,
    tables_path: Optional[str] = None,
    descriptions_path: Optional[str] = None,  # accepted for CLI compatibility; unused
    logger=None,
) -> None:
    """Write ``out_path`` — the website's JSON for one dataset.

    Layout: a shared ``tables`` map (id -> {title, headers, rows}) stored once,
    and per-query ``stages`` that reference tables by id only. This keeps the
    file small even though the same tables recur across stages and queries.
    """
    log = logger.info if logger else print

    s1 = _load_stage(stage1, STAGE_TOP_N["stage1"])
    s2 = _load_stage(stage2, STAGE_TOP_N["stage2"])
    s3 = _load_stage(stage3, STAGE_TOP_N["stage3"])
    log(f"[export] stages loaded — s1:{len(s1)} s2:{len(s2)} s3:{len(s3)} queries")

    needed = _collect_needed_ids(s1, s2, s3)
    log(f"[export] {len(needed):,} unique tables referenced")

    content = _load_nq_tables(tables_path, needed) if tables_path else {}
    if tables_path:
        log(f"[export] resolved content for {len(content):,} tables")

    # Shared table map: one entry per referenced table.
    tables = {tid: _table_entry(tid, content) for tid in sorted(needed)}

    # Question text + gold come from whichever stage file has each query.
    questions: Dict[str, Dict] = {}
    for path in (stage3, stage2, stage1):
        if not path:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = str(rec["qid"])
                questions.setdefault(qid, {"question": rec.get("question", ""), "gold": _gold_ids(rec)})

    queries = []
    for qid, meta in questions.items():
        stages = {}
        for name, stage in (("stage1", s1), ("stage2", s2), ("stage3", s3)):
            ranked = stage.get(qid)
            if ranked:
                stages[name] = [it["table_id"] for it in ranked]
        queries.append({
            "qid": qid,
            "question": meta["question"],
            "gold_table_ids": meta["gold"],
            "stages": stages,
        })

    payload = {
        "dataset": dataset,
        "has_table_content": bool(content),
        "stage_order": ["stage1", "stage2", "stage3"],
        "num_queries": len(queries),
        "tables": tables,
        "queries": queries,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
    size_mb = out.stat().st_size / 1e6
    log(f"[export] wrote {len(queries):,} queries, {len(tables):,} tables -> {out} ({size_mb:.1f} MB)")
