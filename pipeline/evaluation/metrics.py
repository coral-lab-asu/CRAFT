"""
Recall@k evaluation helpers.

These functions wrap the existing utils/eval_metric.py and add the
pretty-printing and CSV-logging that the retrieval scripts need.
"""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Core recall computation
# ---------------------------------------------------------------------------

def compute_recall(
    results: List[Dict],
    ks: Iterable[int] = (1, 10, 50, 100, 500),
) -> Dict[int, float]:
    """
    Compute Recall@k for a list of retrieval results.

    Each result dict must have:
      "gold_table_ids" : list of correct table IDs
      "retrieved"      : list of dicts with "table_id" key (or plain strings)

    A query is a hit at k if any gold table appears in the top-k retrieved items.

    Returns a dict {k: recall_value}.
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from utils.eval_metric import evaluate_recall

    return evaluate_recall(results, list(ks))


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_recall_table(
    stage_metrics: Dict[str, Dict[int, float]],
    n_queries: Dict[str, int],
    ks: Iterable[int] = (1, 10, 50, 100, 500),
    title: str = "Recall@k",
) -> None:
    """
    Print a side-by-side recall table for multiple pipeline stages.

    Args:
        stage_metrics: {"Stage 1": {1: 0.34, 10: 0.73, ...}, "Stage 2": ...}
        n_queries:     {"Stage 1": 966, "Stage 2": 966, ...}
        ks:            Cutoffs to show.
        title:         Header line title.
    """
    ks = list(ks)
    stage_names = list(stage_metrics.keys())

    # Header
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    for name in stage_names:
        n = n_queries.get(name, "?")
        print(f"  {name}: n={n}")
    print(f"  Improvement = relative gain  [ (S_new − S_prev) / S_prev × 100 ]")
    print(f"{'=' * 72}")

    # Column header
    col_w = 10
    header = f"{'k':<5}" + "".join(f"{name[:col_w]:>{col_w}}" for name in stage_names)
    if len(stage_names) >= 2:
        header += f"{'Impr.':>{col_w}}"
    print(header)
    print("-" * len(header))

    # Rows
    for k in ks:
        row = f"@{k:<4}"
        values = [stage_metrics[name].get(k, 0.0) for name in stage_names]
        row += "".join(f"{v:>{col_w}.4f}" for v in values)
        if len(values) >= 2:
            s_prev, s_new = values[-2], values[-1]
            if s_prev > 0:
                impr = (s_new - s_prev) / s_prev * 100
                row += f"{impr:>+{col_w}.1f}%"[: col_w]
        print(row)

    print(f"{'=' * 72}\n")


def _pct_improvement(prev: float, new: float) -> str:
    if prev == 0:
        return "N/A"
    return f"{(new - prev) / prev * 100:+.1f}%"


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def save_recall_csv(
    csv_path: str,
    stage: str,
    dataset: str,
    metrics: Dict[int, float],
    n_queries: int,
    mode: str = "computed",
    embed_model: str = "",
) -> None:
    """
    Append a recall row to a summary CSV file.

    Creates the file (with header) if it doesn't exist yet, otherwise appends.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ks = sorted(metrics.keys())
    fieldnames = ["dataset", "stage", "embed_model", "mode", "num_queries"] + [f"recall@{k}" for k in ks]

    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {
            "dataset": dataset,
            "stage": stage,
            "embed_model": embed_model,
            "mode": mode,
            "num_queries": n_queries,
        }
        for k in ks:
            row[f"recall@{k}"] = f"{metrics[k]:.6f}"
        writer.writerow(row)
