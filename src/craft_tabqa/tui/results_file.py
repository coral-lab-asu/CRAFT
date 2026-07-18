"""Append retrieval results to a human-readable text file, one query at a time.

The terminal app shows only a few tables inline to save space; the full ranked
list for each query is written here in a plain, easy-to-read layout so the user
can open the file and scroll it while more queries stream in.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List


def start_results_file(path: str, dataset: str, scope: str) -> None:
    """Create (or overwrite) the results file with a small header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("CRAFT retrieval results\n")
        f.write(f"Dataset : {dataset}\n")
        f.write(f"Pipeline: {scope}\n")
        f.write(f"Started : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")


def append_query_results(path: str, question: str, results: List[Dict]) -> None:
    """Append one query and its ranked tables to the results file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\n")
        f.write("-" * 70 + "\n")
        if not results:
            f.write("  (no tables retrieved)\n\n")
            return
        for hit in results:
            f.write(f"  #{hit['rank']:<3} score={hit['score']:<8} {hit.get('title') or hit['table_id']}\n")
            f.write(f"       id: {hit['table_id']}\n")
            headers = hit.get("headers") or []
            if headers:
                f.write(f"       columns: {', '.join(str(h) for h in headers)}\n")
            for row in (hit.get("rows") or [])[:3]:
                f.write(f"         - {' | '.join(str(c) for c in row)}\n")
        f.write("\n")
