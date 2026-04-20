"""
Unit tests for CRAFT core utilities and evaluation metrics.
Run with:  pytest tests/
"""
import sys
from pathlib import Path

# Ensure scripts/ is on the path when running without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
import pytest

from qa_evaluation import QAEvaluator, _api_call_with_backoff
from craft_core import EvaluationMetrics


# ── _api_call_with_backoff ────────────────────────────────────────────────────

def test_backoff_succeeds_on_first_try():
    result = _api_call_with_backoff(lambda: 42)
    assert result == 42


def test_backoff_retries_and_succeeds():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient error")
        return "ok"

    result = _api_call_with_backoff(flaky, max_retries=5, base_delay=0.0)
    assert result == "ok"
    assert calls["count"] == 3


def test_backoff_raises_after_max_retries():
    def always_fails():
        raise ValueError("permanent error")

    with pytest.raises(ValueError, match="permanent error"):
        _api_call_with_backoff(always_fails, max_retries=3, base_delay=0.0)


# ── QAEvaluator static helpers ────────────────────────────────────────────────

def test_normalize_answer_lowercases_and_strips_punctuation():
    assert QAEvaluator.normalize_answer("Hello, World!") == "hello world"


def test_normalize_answer_collapses_spaces():
    assert QAEvaluator.normalize_answer("  foo   bar  ") == "foo bar"


def test_compute_f1_exact_match():
    f1 = QAEvaluator.compute_f1_score("Brotherly Love", ["Brotherly Love"])
    assert f1 == pytest.approx(1.0)


def test_compute_f1_partial_overlap():
    f1 = QAEvaluator.compute_f1_score("New York City", ["New York"])
    # precision=2/3, recall=2/2 → F1 = 2*(2/3*1)/(2/3+1) = 0.8
    assert f1 == pytest.approx(0.8)


def test_compute_f1_no_overlap():
    f1 = QAEvaluator.compute_f1_score("Paris", ["London"])
    assert f1 == pytest.approx(0.0)


def test_compute_f1_multiple_ground_truths():
    """Should return the maximum F1 across all ground truths."""
    f1 = QAEvaluator.compute_f1_score("Jamie Elliott", ["John Smith", "Jamie Elliott"])
    assert f1 == pytest.approx(1.0)


def test_compute_f1_empty_generated():
    f1 = QAEvaluator.compute_f1_score("", ["answer"])
    assert f1 == pytest.approx(0.0)


# ── EvaluationMetrics ─────────────────────────────────────────────────────────

def test_recall_at_k_gold_at_top():
    rankings = {"q1": ["t1", "t2", "t3"]}
    gold = {"q1": "t1"}
    recalls = EvaluationMetrics.compute_recall_at_k(rankings, gold, k_values=[1, 3])
    assert recalls[1] == pytest.approx(100.0)
    assert recalls[3] == pytest.approx(100.0)


def test_recall_at_k_gold_not_in_top1():
    rankings = {"q1": ["t2", "t1", "t3"]}
    gold = {"q1": "t1"}
    recalls = EvaluationMetrics.compute_recall_at_k(rankings, gold, k_values=[1, 3])
    assert recalls[1] == pytest.approx(0.0)
    assert recalls[3] == pytest.approx(100.0)


def test_recall_at_k_gold_absent():
    rankings = {"q1": ["t2", "t3"]}
    gold = {"q1": "t_missing"}
    recalls = EvaluationMetrics.compute_recall_at_k(rankings, gold, k_values=[1, 10])
    assert recalls[1] == pytest.approx(0.0)
    assert recalls[10] == pytest.approx(0.0)


def test_mrr_gold_at_rank1():
    rankings = {"q1": ["t1", "t2"]}
    gold = {"q1": "t1"}
    mrr = EvaluationMetrics.compute_mrr(rankings, gold)
    assert mrr == pytest.approx(1.0)


def test_mrr_gold_at_rank2():
    rankings = {"q1": ["t2", "t1"]}
    gold = {"q1": "t1"}
    mrr = EvaluationMetrics.compute_mrr(rankings, gold)
    assert mrr == pytest.approx(0.5)


def test_mrr_multiple_queries():
    rankings = {"q1": ["t1", "t2"], "q2": ["t3", "t4"]}
    gold = {"q1": "t1", "q2": "t4"}
    mrr = EvaluationMetrics.compute_mrr(rankings, gold)
    # q1: 1/1=1.0, q2: 1/2=0.5 → avg=0.75
    assert mrr == pytest.approx(0.75)


# ── _cosine_similarity ────────────────────────────────────────────────────────

def test_cosine_similarity_unit_vectors():
    from craft_stages import _cosine_similarity
    import torch
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([1.0, 0.0])
    sims = _cosine_similarity(a, b)
    assert float(sims[0]) == pytest.approx(1.0)
    assert float(sims[1]) == pytest.approx(0.0)


def test_cosine_similarity_scaled_vectors():
    """Cosine similarity should be scale-invariant."""
    from craft_stages import _cosine_similarity
    import torch
    a = torch.tensor([[3.0, 4.0]])
    b = torch.tensor([6.0, 8.0])   # same direction, 2× scale
    sims = _cosine_similarity(a, b)
    assert float(sims[0]) == pytest.approx(1.0)
