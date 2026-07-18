"""Tests for Recall@k."""

from craft_tabqa.core.metrics import recall_at_k, recall_summary


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert recall_at_k(["a"], [], k=1) == 0.0


def test_recall_summary_averages_across_queries():
    results = [
        {"gold_table_ids": ["t1"], "retrieved": [{"table_id": "t1"}, {"table_id": "t2"}]},
        {"gold_table_ids": ["t9"], "retrieved": [{"table_id": "t1"}, {"table_id": "t2"}]},
    ]
    summary = recall_summary(results, [1, 2])
    assert summary[1] == 0.5   # first query hits at 1, second never
    assert summary[2] == 0.5


def test_recall_summary_handles_empty():
    assert recall_summary([], [1, 5]) == {1: 0.0, 5: 0.0}
