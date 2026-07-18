"""Recall@k, the retrieval quality metric used at every stage.

A query counts as a hit at cutoff ``k`` when any of its gold tables appears in
the top ``k`` retrieved tables. Recall@k is the fraction of queries that hit.
"""

from typing import Dict, Iterable, List, Sequence


def _retrieved_ids(retrieved: Sequence) -> List[str]:
    """Accept either a list of ``{"table_id": ...}`` dicts or plain id strings."""
    if retrieved and isinstance(retrieved[0], dict):
        return [r.get("table_id") for r in retrieved]
    return list(retrieved)


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Return 1.0 if any gold id is in the top ``k`` retrieved ids, else 0.0."""
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(gold in top_k for gold in gold_ids) else 0.0


def recall_summary(results: Iterable[Dict], ks: Iterable[int]) -> Dict[int, float]:
    """Compute mean Recall@k across all query results for each cutoff in ``ks``.

    Each result must carry ``gold_table_ids`` and ``retrieved`` (a list of
    ``{"table_id": ...}`` dicts, as produced by every retrieval stage).
    """
    ks = sorted(ks)
    results = list(results)
    if not results:
        return {k: 0.0 for k in ks}

    hits = {k: 0.0 for k in ks}
    for item in results:
        gold_ids = item.get("gold_table_ids", [])
        retrieved_ids = _retrieved_ids(item.get("retrieved", []))
        for k in ks:
            hits[k] += recall_at_k(retrieved_ids, gold_ids, k)

    n = float(len(results))
    return {k: hits[k] / n for k in ks}
