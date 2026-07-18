"""Log recall numbers and append them to the run's recall summary CSV."""

import csv
from pathlib import Path
from typing import Dict


def log_recall(stage: str, recall: Dict[int, float], logger=None) -> None:
    """Log a compact one-line recall summary for ``stage``."""
    log = logger.info if logger else print
    highlights = [k for k in (1, 10, 100) if k in recall]
    parts = " ".join(f"@{k}={recall[k]:.4f}" for k in highlights)
    log(f"[{stage}] Recall  {parts}")


def save_recall_row(
    csv_path,
    dataset: str,
    stage: str,
    model_id: str,
    mode: str,
    n_queries: int,
    recall: Dict[int, float],
) -> None:
    """Append one recall row to ``csv_path`` (writing a header on first use)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    ks = sorted(recall)
    fieldnames = ["dataset", "stage", "model", "mode", "num_queries"] + [f"recall@{k}" for k in ks]
    row = {
        "dataset": dataset,
        "stage": stage,
        "model": model_id,
        "mode": mode,
        "num_queries": n_queries,
        **{f"recall@{k}": f"{recall[k]:.6f}" for k in ks},
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
