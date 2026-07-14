"""Tests for the search system: BM25 scoring, hybrid/RRF ranking, confidence."""

from conftest import FakeEmbeddingSystem


def _chunk_info(chunk_id, text, source_file, score):
    return {
        "chunk_id": chunk_id, "text": text, "source_title": source_file,
        "source_url": "", "source_file": source_file, "page_number": 1,
        "chunk_index": 0, "word_count": len(text.split()), "char_count": len(text),
        "similarity_score": score,
    }


def _build(chunk_db, ordered, stale=False):
    from rag_qa.search_system import SearchSystem
    db_path, _ = chunk_db
    fake = FakeEmbeddingSystem(ordered, stale=stale)
    return SearchSystem(db_path=db_path, embedding_system=fake, similarity_threshold=0.0)


def test_bm25_scores_only_matching_candidates(chunk_db):
    _, rows = chunk_db
    ordered = [_chunk_info(r["chunk_id"], r["text"], r["source_file"], 0.9) for r in rows]
    ss = _build(chunk_db, ordered)
    scores = ss._get_bm25_scores_for_chunks("emergency stop", [r["chunk_id"] for r in rows])
    assert len(scores) == len(rows)
    # c1 mentions "emergency stop"; it should outscore the others.
    assert scores[0] > 0
    assert scores[0] >= max(scores[1:])


def test_hybrid_confidence_is_vector_score(chunk_db):
    # C4: confidence must be the raw cosine (comparable across queries), not
    # the per-query-normalized hybrid score.
    _, rows = chunk_db
    ordered = [_chunk_info(r["chunk_id"], r["text"], r["source_file"], 0.75) for r in rows]
    ss = _build(chunk_db, ordered)
    results = ss.hybrid_search("machine guarding", k=3)
    assert results
    for r in results:
        assert r.confidence == r.vector_score


def test_rrf_reranks_by_fused_rank(chunk_db):
    # Vector order puts c3 first, but BM25 strongly favors c1 ("emergency
    # stop"). RRF fusion should pull c1 up.
    _, rows = chunk_db
    by_id = {r["chunk_id"]: r for r in rows}
    ordered = [
        _chunk_info("c3", by_id["c3"]["text"], "osha3170.pdf", 0.80),
        _chunk_info("c1", by_id["c1"]["text"], "osha3170.pdf", 0.79),
        _chunk_info("c2", by_id["c2"]["text"], "sick_guide.pdf", 0.60),
    ]
    ss = _build(chunk_db, ordered)
    results = ss.rrf_search("emergency stop systems", k=3)
    ids = [r.chunk_id for r in results]
    assert set(ids) == {"c1", "c2", "c3"}
    assert ids.index("c1") < ids.index("c2")  # keyword-strong chunk ranks well


def test_tokenize_for_bm25_drops_stopwords_and_short_tokens(chunk_db):
    ordered = []
    ss = _build(chunk_db, ordered)
    tokens = ss._tokenize_for_bm25("The an of machinery is a safety at requirement")
    assert "the" not in tokens and "of" not in tokens and "is" not in tokens
    assert "machinery" in tokens and "safety" in tokens and "requirement" in tokens


def test_unknown_mode_raises(chunk_db):
    import pytest
    ordered = []
    ss = _build(chunk_db, ordered)
    with pytest.raises(ValueError):
        ss.search("q", mode="nonsense")
