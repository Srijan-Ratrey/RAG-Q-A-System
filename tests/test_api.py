"""Tests for the Flask API: request validation and extractive answering.

Validation tests never reach get_qa_service() (bad input is rejected first),
so they run without loading any model.
"""

import pytest


@pytest.fixture
def client():
    from rag_qa import api
    api.app.config.update(TESTING=True)
    return api.app.test_client()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_ask_requires_body(client):
    assert client.post("/ask", json=None).status_code == 400


def test_ask_requires_query(client):
    assert client.post("/ask", json={"q": "   "}).status_code == 400


@pytest.mark.parametrize("k", [0, 21, 100, "5", 5.5, True])
def test_ask_rejects_bad_k(client, k):
    resp = client.post("/ask", json={"q": "safety", "k": k})
    assert resp.status_code == 400


def test_ask_rejects_bad_mode(client):
    resp = client.post("/ask", json={"q": "safety", "mode": "bogus"})
    assert resp.status_code == 400


# B2 regression: /search must validate k the same way /ask does.
@pytest.mark.parametrize("k", [0, 999, "10", 2.5, True])
def test_search_rejects_bad_k(client, k):
    resp = client.post("/search", json={"q": "safety", "k": k})
    assert resp.status_code == 400


def test_search_rejects_bad_mode(client):
    resp = client.post("/search", json={"q": "safety", "mode": "bogus"})
    assert resp.status_code == 400


# --- Extractive answering (no model needed with embedding_system=None) -------

def _result(text):
    from rag_qa.search_system import SearchResult
    return SearchResult(
        chunk_id="c", text=text, source_title="T", source_url="", source_file="f.pdf",
        page_number=1, chunk_index=0, word_count=len(text.split()),
        vector_score=0.9, confidence=0.9,
    )


def test_answer_uses_nltk_and_does_not_split_on_abbreviations():
    from rag_qa.api import AnswerGenerator
    gen = AnswerGenerator(confidence_threshold=0.5, embedding_system=None)
    text = ("Safety functions must comply with ISO 13849-1. "
            "Use protective measures, e.g. guards and interlocks, where needed. "
            "A risk assessment identifies the required performance level.")
    answer = gen._extract_relevant_sentences("safety", [_result(text)])
    # "ISO 13849-1." and "e.g." must not be treated as sentence breaks.
    assert "ISO 13849-1" in answer
    assert answer.count("13849-1") == 1


def test_answer_abstains_below_threshold():
    from rag_qa.api import AnswerGenerator
    gen = AnswerGenerator(confidence_threshold=0.5, embedding_system=None)
    low = _result("Some marginally relevant machinery text goes here now.")
    low.confidence = 0.2
    answer, conf, reason = gen.extract_answer("safety", [low])
    assert answer is None
    assert "confidence" in reason.lower()
