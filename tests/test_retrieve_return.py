"""Tests for the optional in-memory return of ranked results from retrieve()."""

from craft_tabqa.retrieval.pipeline import RetrievalResults, _truncate


def _query_result(n):
    return {
        "qid": "q1",
        "question": "Q?",
        "gold_table_ids": ["t0"],
        "retrieved": [{"rank": i + 1, "table_id": f"t{i}", "score": 1.0 - i} for i in range(n)],
    }


def test_truncate_keeps_top_k_per_query():
    results = [_query_result(50)]
    truncated = _truncate(results, 10)
    assert len(truncated[0]["retrieved"]) == 10
    assert truncated[0]["retrieved"][0]["table_id"] == "t0"
    # original list is not mutated
    assert len(results[0]["retrieved"]) == 50


def test_truncate_none_keeps_everything():
    results = [_query_result(50)]
    assert len(_truncate(results, None)[0]["retrieved"]) == 50


def test_retrieval_results_defaults():
    rr = RetrievalResults(recall={"stage1": None, "stage2": None, "stage3": None})
    assert rr.stage1 == [] and rr.stage2 == [] and rr.stage3 == []
