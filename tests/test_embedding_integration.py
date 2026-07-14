"""End-to-end test of the FAISS index + corpus-fingerprint staleness guard.

Requires the sentence-transformer model. If it cannot be loaded (e.g. offline
CI with no cache), the whole module is skipped rather than failing.
"""

import sqlite3

import pytest

from conftest import insert_chunks


@pytest.fixture(scope="module")
def embedding_cls():
    from rag_qa.embedding_system import EmbeddingSystem
    try:
        # Probe: construct once so a missing model skips instead of erroring.
        EmbeddingSystem(db_path=":memory:",
                        index_path="/tmp/_probe.bin",
                        chunk_id_map_path="/tmp/_probe.json")
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"embedding model unavailable: {e}")
    return EmbeddingSystem


def _seed(processor, chunk_ids):
    rows = [{"chunk_id": cid, "text": f"Safety requirement number {cid} for industrial machinery.",
             "source_file": "doc.pdf", "chunk_index": i}
            for i, cid in enumerate(chunk_ids)]
    insert_chunks(processor.db_path, rows)


def test_build_index_and_detect_staleness(embedding_cls, tmp_path, processor):
    _seed(processor, ["a", "b", "c"])
    es = embedding_cls(
        db_path=processor.db_path,
        index_path=str(tmp_path / "idx.bin"),
        chunk_id_map_path=str(tmp_path / "map.json"),
    )
    stats = es.build_index(force_rebuild=True)
    assert stats["total_vectors"] == 3
    assert es.is_index_stale() is False

    # Add a chunk without rebuilding -> fingerprint must now report stale.
    with sqlite3.connect(processor.db_path) as conn:
        insert_chunks(processor.db_path, [
            {"chunk_id": "d", "text": "New machine guarding requirement added later.",
             "source_file": "doc.pdf", "chunk_index": 3}])
    assert es.is_index_stale() is True


def test_search_returns_ranked_results(embedding_cls, tmp_path, processor):
    _seed(processor, ["a", "b", "c"])
    es = embedding_cls(
        db_path=processor.db_path,
        index_path=str(tmp_path / "idx2.bin"),
        chunk_id_map_path=str(tmp_path / "map2.json"),
    )
    es.build_index(force_rebuild=True)
    results = es.search_similar("machinery safety requirement", k=2)
    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert "similarity_score" in results[0]
