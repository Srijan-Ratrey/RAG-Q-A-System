"""Unit tests for the IR metric functions."""

from rag_qa.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    dedupe_preserve_order,
)


def test_dedupe_preserve_order_keeps_ndcg_bounded():
    # File-level relevance: duplicate source files must be collapsed so nDCG
    # never exceeds 1.0 (regression for the >1 nDCG bug).
    raw = ["osha3170.pdf", "osha3170.pdf", "sick.pdf"]
    deduped = dedupe_preserve_order(raw)
    assert deduped == ["osha3170.pdf", "sick.pdf"]
    assert ndcg_at_k(deduped, {"osha3170.pdf"}, k=3) <= 1.0


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"a", "c"}, k=4) == 1.0
    assert recall_at_k(retrieved, {"a", "z"}, k=4) == 0.5
    assert recall_at_k(retrieved, {"d"}, k=2) == 0.0  # d is outside top-2
    assert recall_at_k(retrieved, set(), k=4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_at_k():
    # Relevant item first -> perfect nDCG.
    assert ndcg_at_k(["a", "b"], {"a"}, k=2) == 1.0
    # Relevant item second -> less than perfect but > 0.
    val = ndcg_at_k(["b", "a"], {"a"}, k=2)
    assert 0.0 < val < 1.0
    assert ndcg_at_k(["b", "c"], {"a"}, k=2) == 0.0


def test_ndcg_two_relevant_ordering():
    # Both relevant retrieved in ideal order -> 1.0.
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0
    # One relevant pushed down ranks lower than ideal.
    assert ndcg_at_k(["a", "c", "b"], {"a", "b"}, k=3) < 1.0
