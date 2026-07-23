"""
Document processing module for RAG Q&A system.
Handles PDF extraction, text cleaning, and intelligent chunking.
"""

import os
import re
import json
import sqlite3
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass

import pypdf
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

# Download required NLTK data. Newer NLTK (>=3.8.2) splits the tokenizer
# tables into a separate ``punkt_tab`` package, so guard for both.
for _resource in ('punkt', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{_resource}')
    except (LookupError, OSError):
        try:
            nltk.download(_resource, quiet=True)
        except Exception:  # pragma: no cover - network/offline fallback
            pass

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    source_title: str
    source_url: str
    source_file: str
    page_number: Optional[int]
    chunk_index: int
    word_count: int
    char_count: int

class DocumentProcessor:
    """Handles document ingestion, chunking, and storage."""
    
    def __init__(self, 
                 sources_file: str = "sources.json",
                 pdf_dir: str = "industrial-safety-pdfs",
                 db_path: str = "data/rag_database.db",
                 chunk_size: int = 250,  # Target words per chunk
                 chunk_overlap: int = 50):  # Overlap in words
        """
        Initialize the document processor.
        
        Args:
            sources_file: Path to sources.json metadata file
            pdf_dir: Directory containing PDF files
            db_path: Path to SQLite database
            chunk_size: Target number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
        """
        self.sources_file = sources_file
        self.pdf_dir = pdf_dir
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load sources metadata (also populates self.sources_by_filename).
        self.sources_by_filename: Dict[str, Dict] = {}
        self.sources_metadata = self._load_sources_metadata()
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _load_sources_metadata(self) -> Dict[str, Dict]:
        """Load sources metadata from sources.json."""
        try:
            with open(self.sources_file, 'r', encoding='utf-8') as f:
                sources_list = json.load(f)
            
            # Convert to dict for easier lookup by title. Also build an exact
            # filename index from any entries that declare a ``filename`` field
            # (the reliable way to attribute a chunk to its source).
            sources_dict = {}
            self.sources_by_filename = {}
            for source in sources_list:
                title = source['title']
                sources_dict[title] = source
                filename = source.get('filename')
                if filename:
                    self.sources_by_filename[filename.lower()] = source

            self.logger.info(
                f"Loaded metadata for {len(sources_dict)} sources "
                f"({len(self.sources_by_filename)} with explicit filenames)"
            )
            return sources_dict
            
        except Exception as e:
            self.logger.error(f"Error loading sources metadata: {e}")
            return {}
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source_title TEXT NOT NULL,
                    source_url TEXT,
                    source_file TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_index INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create documents table for tracking processed files
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    file_path TEXT PRIMARY KEY,
                    source_title TEXT,
                    source_url TEXT,
                    total_chunks INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_hash TEXT
                )
            """)
            
            # Index for source-scoped lookups (get_chunk_context, deletes).
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file)")
            # A plain B-tree index on the full chunk text can't serve keyword
            # search and only bloats the DB, so drop it if an older run created
            # it. Use SQLite FTS5 if in-DB keyword search is ever needed.
            cursor.execute("DROP INDEX IF EXISTS idx_chunks_text")

            conn.commit()
            self.logger.info("Database initialized successfully")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (text, page_number) tuples
        """
        pages_text = []

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():  # Only add non-empty pages
                            pages_text.append((text, page_num))
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {page_num} from {pdf_path}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to read PDF {pdf_path}: {e}")

        return pages_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.

        Order matters: every newline-dependent rule (page-number removal,
        header stripping, broken-sentence joining) must run *before* runs of
        whitespace are collapsed, otherwise the leading ``\\s+`` collapse would
        delete the newlines those rules rely on and they would never fire.
        """
        # Normalize line endings so the newline rules below behave predictably.
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # --- Newline-dependent rules (run first, while newlines still exist) ---
        # Page numbers sitting alone on a line.
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        # ALL-CAPS header/footer lines (10+ chars of caps and spaces).
        text = re.sub(r'\n[A-Z][A-Z ]{9,}\n', '\n', text)
        # Join sentences broken across a line break.
        text = re.sub(r'(?<=[.!?])\s*\n(?=[A-Z])', ' ', text)
        # Collapse 3+ consecutive newlines into a single paragraph break.
        text = re.sub(r'\n\s*\n(?:\s*\n)+', '\n\n', text)

        # --- Collapse remaining horizontal whitespace (newlines preserved) ---
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)

        # Remove excessive punctuation.
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{3,}', '---', text)

        return text.strip()
    
    def _chunk_sentences(self,
                         sentences_with_pages: List[Tuple[str, int]]
                         ) -> List[Tuple[str, int]]:
        """
        Create overlapping chunks from a page-tagged sentence stream.

        Chunks may span page boundaries (so a paragraph split across a page
        break is kept together and the overlap window applies across pages).
        Each sentence is tokenized exactly once.

        Args:
            sentences_with_pages: List of (sentence, page_number) tuples in
                document order.

        Returns:
            List of (chunk_text, page_number) tuples, where page_number is the
            page the chunk starts on.
        """
        chunks: List[Tuple[str, int]] = []
        # Each item is (sentence, page_number, word_count).
        current: List[Tuple[str, int, int]] = []
        current_word_count = 0

        def finalize(items: List[Tuple[str, int, int]]) -> Tuple[str, int]:
            return ' '.join(item[0] for item in items), items[0][1]

        for sentence, page in self._split_oversized_sentences(sentences_with_pages):
            sentence_words = len(word_tokenize(sentence))

            # If adding this sentence would exceed chunk size, finalize current.
            if current_word_count + sentence_words > self.chunk_size and current:
                chunks.append(finalize(current))

                # Start the next chunk with a trailing overlap window.
                overlap_words = 0
                overlap_items: List[Tuple[str, int, int]] = []
                for item in reversed(current):
                    if overlap_words + item[2] <= self.chunk_overlap:
                        overlap_items.insert(0, item)
                        overlap_words += item[2]
                    else:
                        break

                current = overlap_items + [(sentence, page, sentence_words)]
                current_word_count = overlap_words + sentence_words
            else:
                current.append((sentence, page, sentence_words))
                current_word_count += sentence_words

        if current:
            chunks.append(finalize(current))

        return chunks

    def _split_oversized_sentences(self,
                                   sentences_with_pages: List[Tuple[str, int]]
                                   ) -> Iterator[Tuple[str, int]]:
        """Break any single sentence longer than ``chunk_size`` into word-bounded
        pieces so the chunker still splits it.

        Sentence-based chunking assumes real sentence punctuation. Documents
        with little of it — resumes, tables, slide exports — can tokenize into
        one enormous "sentence" that would otherwise become a single giant
        chunk, whose averaged embedding matches no specific query well. Splitting
        such sentences on word boundaries keeps them as several embeddable units;
        normal prose (sentences <= chunk_size) passes through unchanged.
        """
        for sentence, page in sentences_with_pages:
            words = sentence.split()
            if len(words) <= self.chunk_size:
                yield sentence, page
                continue
            for i in range(0, len(words), self.chunk_size):
                yield ' '.join(words[i:i + self.chunk_size]), page

    def _find_source_metadata(self, filename: str) -> Tuple[str, str]:
        """
        Find source title and URL for a given filename.
        
        Args:
            filename: PDF filename
            
        Returns:
            Tuple of (title, url)
        """
        # 1. Exact filename match — the only unambiguous signal. Requires a
        #    ``filename`` field in sources.json (see README).
        exact = self.sources_by_filename.get(filename.lower())
        if exact:
            return exact['title'], exact.get('url', '')

        filename_clean = filename.lower().replace('.pdf', '')

        # 2. Filename fully contained in a title.
        for title, metadata in self.sources_metadata.items():
            if filename_clean in title.lower():
                return title, metadata.get('url', '')

        # 3. Fuzzy word-overlap fallback. This is unreliable with many
        #    similar-sounding documents, so warn loudly when we rely on it.
        #    Only consider distinctive words (length > 3) to avoid matching on
        #    filler like "a"/"to"/"for", and require exact token overlap.
        filename_words = set(filename_clean.replace('-', ' ').replace('_', ' ').split())
        for title, metadata in self.sources_metadata.items():
            title_words = {w for w in title.lower().split() if len(w) > 3}

            matches = len(title_words & filename_words)
            if matches >= 2:  # At least 2 distinctive word matches
                self.logger.warning(
                    f"Fuzzy-matched {filename!r} to source {title!r} "
                    f"(no explicit filename in sources.json) - citation may be wrong"
                )
                return title, metadata.get('url', '')

        # 4. Give up: use the filename itself as the title.
        self.logger.warning(f"No metadata found for {filename}, using filename as title")
        return filename.replace('.pdf', '').replace('_', ' ').replace('-', ' '), ''
    
    def _generate_chunk_id(self, text: str, source_file: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{source_file}_{chunk_index}_{text[:50]}"
        return hashlib.md5(content.encode()).hexdigest()

    def _compute_file_hash(self, pdf_path: str) -> str:
        """Compute an MD5 hash of the file's bytes (for change detection)."""
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            for block in iter(lambda: f.read(65536), b''):
                hasher.update(block)
        return hasher.hexdigest()

    def _get_stored_file_hash(self, filename: str) -> Optional[str]:
        """Return the file_hash recorded for a document, or None if unknown."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_hash FROM documents WHERE file_path = ?", (filename,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    def process_document(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Process a single PDF document into chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        filename = os.path.basename(pdf_path)
        self.logger.info(f"Processing document: {filename}")
        
        # Extract text from PDF
        pages_text = self._extract_text_from_pdf(pdf_path)
        if not pages_text:
            self.logger.warning(f"No text extracted from {filename}")
            return []
        
        # Get source metadata
        source_title, source_url = self._find_source_metadata(filename)

        # Build a flat, page-tagged sentence stream for the whole document so
        # chunks can span page boundaries (see _chunk_sentences).
        sentences_with_pages: List[Tuple[str, int]] = []
        for page_text, page_num in pages_text:
            cleaned_text = self._clean_text(page_text)
            if not cleaned_text:
                continue
            for sentence in sent_tokenize(cleaned_text):
                sentences_with_pages.append((sentence, page_num))

        # Create overlapping chunks.
        all_chunks = []
        chunk_index = 0
        for chunk_text, page_num in self._chunk_sentences(sentences_with_pages):
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue

            chunk_id = self._generate_chunk_id(chunk_text, filename, chunk_index)
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_title=source_title,
                source_url=source_url,
                source_file=filename,
                page_number=page_num,
                chunk_index=chunk_index,
                word_count=len(word_tokenize(chunk_text)),
                char_count=len(chunk_text)
            )
            all_chunks.append(chunk)
            chunk_index += 1

        self.logger.info(f"Created {len(all_chunks)} chunks from {filename}")
        return all_chunks
    
    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Save chunks to database.

        Deletes any existing chunks for the source files present in this batch
        first, so re-processing a document (whose extraction may yield
        different chunk IDs) never leaves orphan rows behind.
        """
        if not chunks:
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for source_file in {chunk.source_file for chunk in chunks}:
                cursor.execute(
                    "DELETE FROM chunks WHERE source_file = ?", (source_file,)
                )

            for chunk in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, text, source_title, source_url, source_file, 
                     page_number, chunk_index, word_count, char_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id, chunk.text, chunk.source_title, chunk.source_url,
                    chunk.source_file, chunk.page_number, chunk.chunk_index,
                    chunk.word_count, chunk.char_count
                ))
            
            conn.commit()
    
    def process_all_documents(self, force_reprocess: bool = False) -> Dict[str, int]:
        """
        Process all PDF documents in the directory.

        Args:
            force_reprocess: Re-process every file even if its content hash is
                unchanged since the last run.

        Returns:
            Dictionary with processing statistics
        """
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")

        stats = {
            'total_files': len(pdf_files),
            'processed_files': 0,
            'skipped_files': 0,
            'total_chunks': 0,
            'failed_files': []
        }

        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                pdf_path = os.path.join(self.pdf_dir, filename)
                file_hash = self._compute_file_hash(pdf_path)

                # Skip files whose bytes are unchanged since the last run.
                if not force_reprocess and self._get_stored_file_hash(filename) == file_hash:
                    self.logger.info(f"Skipping unchanged file: {filename}")
                    stats['skipped_files'] += 1
                    continue

                chunks = self.process_document(pdf_path)

                if chunks:
                    self.save_chunks(chunks)
                    stats['processed_files'] += 1
                    stats['total_chunks'] += len(chunks)

                    # Update documents table (now recording the content hash).
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        source_title, source_url = self._find_source_metadata(filename)
                        cursor.execute("""
                            INSERT OR REPLACE INTO documents
                            (file_path, source_title, source_url, total_chunks, file_hash)
                            VALUES (?, ?, ?, ?, ?)
                        """, (filename, source_title, source_url, len(chunks), file_hash))
                        conn.commit()

                else:
                    stats['failed_files'].append(filename)

            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
                stats['failed_files'].append(filename)

        self.logger.info(f"Processing complete: {stats}")
        return stats
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the processed documents."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get chunk statistics
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(word_count), AVG(char_count) FROM chunks")
            avg_words, avg_chars = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(DISTINCT source_file) FROM chunks")
            unique_sources = cursor.fetchone()[0]
            
            # Get top sources by chunk count
            cursor.execute("""
                SELECT source_file, COUNT(*) as chunk_count 
                FROM chunks 
                GROUP BY source_file 
                ORDER BY chunk_count DESC 
                LIMIT 5
            """)
            top_sources = cursor.fetchall()
            
            return {
                'total_chunks': total_chunks,
                'unique_sources': unique_sources,
                'avg_words_per_chunk': round(avg_words or 0, 1),
                'avg_chars_per_chunk': round(avg_chars or 0, 1),
                'top_sources': top_sources
            }

def main():
    """Main function for testing document processing."""
    processor = DocumentProcessor()
    
    # Process all documents
    stats = processor.process_all_documents()
    print(f"Processing Statistics: {stats}")
    
    # Show database stats
    db_stats = processor.get_database_stats()
    print(f"Database Statistics: {db_stats}")

if __name__ == "__main__":
    main()
