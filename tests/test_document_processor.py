"""Tests for document processing: text cleaning, chunking, source matching."""

import sqlite3

from rag_qa.document_processor import DocumentChunk


# --- B1 regression: newline-dependent cleaning rules must actually fire ------

def test_clean_text_removes_standalone_page_numbers(processor):
    text = "First paragraph of content here.\n42\nSecond paragraph continues on."
    cleaned = processor._clean_text(text)
    # The lone page-number line must be gone (the original code collapsed all
    # whitespace first, so this rule never fired).
    assert "42" not in cleaned.split()
    assert "First paragraph" in cleaned
    assert "Second paragraph" in cleaned


def test_clean_text_removes_all_caps_header(processor):
    text = "Body text before.\nMACHINE SAFETY REQUIREMENTS\nBody text after here."
    cleaned = processor._clean_text(text)
    assert "MACHINE SAFETY REQUIREMENTS" not in cleaned
    assert "Body text before" in cleaned
    assert "Body text after" in cleaned


def test_clean_text_joins_broken_sentences(processor):
    text = "The machine stops.\nAnother sentence follows immediately."
    cleaned = processor._clean_text(text)
    # A period+newline+capital should become a single space (no orphan newline).
    assert "stops. Another" in cleaned


def test_clean_text_collapses_horizontal_whitespace(processor):
    assert processor._clean_text("a     b\t\tc") == "a b c"


# --- Chunking ----------------------------------------------------------------

def test_chunking_produces_overlap(processor):
    # 8 short sentences (~6 tokens each); chunk_size=40, overlap=10 forces
    # multiple chunks with a shared overlap sentence between them.
    sentences = [(f"Distinct filler sentence number {i}.", 1) for i in range(8)]
    chunks = processor._chunk_sentences(sentences)
    assert len(chunks) >= 2
    # The tail sentence(s) of chunk 0 must reappear at the start of chunk 1.
    overlapping = [s for s in sentences
                   if s[0] in chunks[0][0] and s[0] in chunks[1][0]]
    assert overlapping, "expected at least one sentence shared across chunks"


def test_chunking_splits_periodless_oversized_sentence(processor):
    # Documents with little sentence punctuation (resumes, tables, slide
    # exports) can tokenize into one enormous "sentence". Without word-level
    # splitting this collapsed into a single giant chunk whose averaged
    # embedding matched no specific query. chunk_size=40, so a ~120-word
    # punctuation-free run must break into multiple chunks.
    big = " ".join(f"word{i}" for i in range(120))  # no sentence terminators
    chunks = processor._chunk_sentences([(big, 1)])
    assert len(chunks) >= 2
    # Every chunk must respect the word-count target (allowing overlap slack).
    assert all(len(text.split()) <= processor.chunk_size * 2 for text, _ in chunks)
    # No content is lost in the split.
    assert "word0" in chunks[0][0] and "word119" in chunks[-1][0]


def test_chunking_spans_page_boundaries(processor):
    # Sentences from two pages; a chunk should be allowed to include both.
    sentences = [("Sentence one on the first page here.", 1),
                 ("Sentence two also on the first page.", 1),
                 ("Sentence three now on the second page.", 2)]
    chunks = processor._chunk_sentences(sentences)
    # With a large chunk size everything lands in one chunk spanning pages 1-2.
    assert len(chunks) == 1
    text, page = chunks[0]
    assert "first page" in text and "second page" in text
    assert page == 1  # page of the chunk's first sentence


# --- C5 source attribution ---------------------------------------------------

def test_source_matching_prefers_exact_filename(processor):
    title, url = processor._find_source_metadata("osha3170.pdf")
    assert title.startswith("OSHA 3170")
    assert url == "https://example.org/osha3170.pdf"


def test_source_matching_falls_back_when_unknown(processor):
    title, url = processor._find_source_metadata("totally_unknown_doc.pdf")
    # No metadata -> filename-derived title, empty url.
    assert "totally unknown doc" in title.lower()
    assert url == ""


def test_generate_chunk_id_is_deterministic(processor):
    a = processor._generate_chunk_id("hello world", "f.pdf", 0)
    b = processor._generate_chunk_id("hello world", "f.pdf", 0)
    c = processor._generate_chunk_id("hello world", "f.pdf", 1)
    assert a == b and a != c


# --- C2 reprocessing must not leave orphan chunks ----------------------------

def test_save_chunks_replaces_existing_source_rows(processor):
    def make(cid, idx):
        return DocumentChunk(
            chunk_id=cid, text=f"chunk text body number {idx} here",
            source_title="T", source_url="", source_file="doc.pdf",
            page_number=1, chunk_index=idx, word_count=6, char_count=25,
        )

    processor.save_chunks([make("old1", 0), make("old2", 1)])
    # Reprocess the same file -> different chunk ids.
    processor.save_chunks([make("new1", 0)])

    with sqlite3.connect(processor.db_path) as conn:
        ids = {row[0] for row in conn.execute(
            "SELECT chunk_id FROM chunks WHERE source_file = 'doc.pdf'")}
    assert ids == {"new1"}  # old rows deleted, no orphans
