"""Shared pytest fixtures and path setup for the RAG Q&A System tests."""

import json
import os
import sqlite3
import sys

import pytest

# Make the rag_qa package importable even without `pip install -e .`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def sources_file(tmp_path):
    """A minimal sources.json: one entry with an explicit filename, one without."""
    data = [
        {
            "title": "OSHA 3170 — Safeguarding Equipment and Protecting Employees",
            "url": "https://example.org/osha3170.pdf",
            "filename": "osha3170.pdf",
        },
        {
            "title": "SICK — Guide for Safe Machinery: Six Steps to a Safe Machine",
            "url": "https://example.org/sick_guide.pdf",
        },
    ]
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.fixture
def processor(tmp_path, sources_file):
    """A DocumentProcessor backed by a temp DB and small chunk sizes."""
    from rag_qa.document_processor import DocumentProcessor

    db_path = tmp_path / "data" / "test.db"
    return DocumentProcessor(
        sources_file=sources_file,
        pdf_dir=str(tmp_path),
        db_path=str(db_path),
        chunk_size=40,
        chunk_overlap=10,
    )


def insert_chunks(db_path, rows):
    """Insert chunk rows (list of dicts) into a DB created by DocumentProcessor."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, text, source_title, source_url, source_file,
                 page_number, chunk_index, word_count, char_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["chunk_id"], r["text"], r.get("source_title", "T"),
                    r.get("source_url", ""), r.get("source_file", "f.pdf"),
                    r.get("page_number", 1), r.get("chunk_index", 0),
                    len(r["text"].split()), len(r["text"]),
                ),
            )
        conn.commit()


@pytest.fixture
def chunk_db(tmp_path, processor):
    """A DB pre-populated with a few keyword-distinct chunks."""
    db_path = processor.db_path
    rows = [
        {"chunk_id": "c1", "text": "Emergency stop systems must halt machinery immediately when activated.",
         "source_file": "osha3170.pdf", "chunk_index": 0},
        {"chunk_id": "c2", "text": "Risk assessment follows the procedure defined in ISO 12100 for machinery.",
         "source_file": "sick_guide.pdf", "chunk_index": 0},
        {"chunk_id": "c3", "text": "Machine guarding protects operators at the point of operation from amputations.",
         "source_file": "osha3170.pdf", "chunk_index": 1},
    ]
    insert_chunks(db_path, rows)
    return db_path, rows


class FakeEmbeddingSystem:
    """Stand-in embedding system so search tests avoid loading a real model."""

    def __init__(self, ordered_results, stale=False):
        # ordered_results: list of chunk-info dicts (highest vector score first),
        # each with 'similarity_score' and the standard chunk fields.
        self._ordered = ordered_results
        self._stale = stale

    def search_similar(self, query, k=5, return_scores=True):
        return self._ordered[:k]

    def _compute_corpus_fingerprint(self, chunk_ids=None):
        return "test-fingerprint"

    def is_index_stale(self):
        return self._stale
