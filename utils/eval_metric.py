from typing import Dict, Iterable, List


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(gold_id in top_k for gold_id in gold_ids) else 0.0


def evaluate_recall(results: Iterable[Dict], ks: Iterable[int]) -> Dict[int, float]:
    ks = sorted(ks)
    totals = {k: 0.0 for k in ks}
    results_list = list(results)
    if not results_list:
        return totals
    for item in results_list:
        gold_ids = item.get("gold_table_ids", [])
        retrieved = item.get("retrieved", [])
        if retrieved and isinstance(retrieved[0], dict):
            retrieved_ids = [r.get("table_id") for r in retrieved]
        else:
            retrieved_ids = list(retrieved)
        for k in ks:
            totals[k] += recall_at_k(retrieved_ids, gold_ids, k)
    total = float(len(results_list))
    return {k: totals[k] / total for k in ks}
