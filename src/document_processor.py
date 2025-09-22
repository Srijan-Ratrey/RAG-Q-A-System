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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import PyPDF2
import pypdf
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
        
        # Load sources metadata
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
            
            # Convert to dict for easier lookup by title
            sources_dict = {}
            for source in sources_list:
                # Create a simplified filename key for matching
                title = source['title']
                sources_dict[title] = source
            
            self.logger.info(f"Loaded metadata for {len(sources_dict)} sources")
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
            
            # Create indexes for better search performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_text ON chunks(text)")
            
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
        
        # Try PyPDF2 first, fallback to pypdf
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():  # Only add non-empty pages
                            pages_text.append((text, page_num))
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {page_num} from {pdf_path}: {e}")
                        
        except Exception as e:
            self.logger.warning(f"PyPDF2 failed for {pdf_path}, trying pypdf: {e}")
            
            # Fallback to pypdf
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            text = page.extract_text()
                            if text.strip():
                                pages_text.append((text, page_num))
                        except Exception as e:
                            self.logger.warning(f"Error extracting page {page_num} with pypdf: {e}")
                            
            except Exception as e:
                self.logger.error(f"Both PDF extractors failed for {pdf_path}: {e}")
        
        return pages_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns (common in technical docs)
        text = re.sub(r'\n\d+\s*\n', '\n', text)  # Page numbers on separate lines
        text = re.sub(r'\n[A-Z\s]{10,}\n', '\n', text)  # ALL CAPS headers
        
        # Clean up line breaks and spacing
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'(?<=[.!?])\s*\n(?=[A-Z])', ' ', text)  # Join broken sentences
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text.strip()
    
    def _create_chunks(self, text: str, source_file: str, page_number: int = None) -> List[str]:
        """
        Create overlapping chunks from text.
        
        Args:
            text: Input text to chunk
            source_file: Source filename
            page_number: Page number (optional)
            
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(word_tokenize(sentence))
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_words = 0
                overlap_sentences = []
                
                # Add sentences from the end for overlap
                for i in range(len(current_chunk) - 1, -1, -1):
                    sent_words = len(word_tokenize(current_chunk[i]))
                    if overlap_words + sent_words <= self.chunk_overlap:
                        overlap_sentences.insert(0, current_chunk[i])
                        overlap_words += sent_words
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_word_count = overlap_words + sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _find_source_metadata(self, filename: str) -> Tuple[str, str]:
        """
        Find source title and URL for a given filename.
        
        Args:
            filename: PDF filename
            
        Returns:
            Tuple of (title, url)
        """
        filename_clean = filename.lower().replace('.pdf', '')
        
        # Try exact title matching first
        for title, metadata in self.sources_metadata.items():
            if filename_clean in title.lower():
                return title, metadata.get('url', '')
        
        # Try partial matching
        for title, metadata in self.sources_metadata.items():
            title_words = title.lower().split()
            filename_words = filename_clean.replace('-', ' ').replace('_', ' ').split()
            
            # If filename contains key words from title
            matches = sum(1 for word in title_words if any(word in fw for fw in filename_words))
            if matches >= 2:  # At least 2 word matches
                return title, metadata.get('url', '')
        
        # Fallback to filename as title
        self.logger.warning(f"No metadata found for {filename}, using filename as title")
        return filename.replace('.pdf', '').replace('_', ' ').replace('-', ' '), ''
    
    def _generate_chunk_id(self, text: str, source_file: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{source_file}_{chunk_index}_{text[:50]}"
        return hashlib.md5(content.encode()).hexdigest()
    
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
        
        # Process each page and create chunks
        all_chunks = []
        chunk_index = 0
        
        for page_text, page_num in pages_text:
            cleaned_text = self._clean_text(page_text)
            if not cleaned_text:
                continue
            
            # Create chunks for this page
            page_chunks = self._create_chunks(cleaned_text, filename, page_num)
            
            for chunk_text in page_chunks:
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                chunk_id = self._generate_chunk_id(chunk_text, filename, chunk_index)
                word_count = len(word_tokenize(chunk_text))
                char_count = len(chunk_text)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_title=source_title,
                    source_url=source_url,
                    source_file=filename,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    word_count=word_count,
                    char_count=char_count
                )
                
                all_chunks.append(chunk)
                chunk_index += 1
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {filename}")
        return all_chunks
    
    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Save chunks to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
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
    
    def process_all_documents(self) -> Dict[str, int]:
        """
        Process all PDF documents in the directory.
        
        Returns:
            Dictionary with processing statistics
        """
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        stats = {
            'total_files': len(pdf_files),
            'processed_files': 0,
            'total_chunks': 0,
            'failed_files': []
        }
        
        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                pdf_path = os.path.join(self.pdf_dir, filename)
                chunks = self.process_document(pdf_path)
                
                if chunks:
                    self.save_chunks(chunks)
                    stats['processed_files'] += 1
                    stats['total_chunks'] += len(chunks)
                    
                    # Update documents table
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        source_title, source_url = self._find_source_metadata(filename)
                        cursor.execute("""
                            INSERT OR REPLACE INTO documents 
                            (file_path, source_title, source_url, total_chunks)
                            VALUES (?, ?, ?, ?)
                        """, (filename, source_title, source_url, len(chunks)))
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
