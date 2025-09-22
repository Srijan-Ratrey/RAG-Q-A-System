"""
Search system for RAG Q&A with baseline similarity search and hybrid reranker.
Combines vector similarity with BM25 keyword scoring for improved results.
"""

import sqlite3
import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from embedding_system import EmbeddingSystem

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

@dataclass
class SearchResult:
    """Represents a search result with all relevant information."""
    chunk_id: str
    text: str
    source_title: str
    source_url: str
    source_file: str
    page_number: Optional[int]
    chunk_index: int
    word_count: int
    
    # Scores
    vector_score: float
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    final_rank: int = 0
    
    # Metadata
    reranker_used: str = "baseline"
    confidence: float = 0.0

class SearchSystem:
    """Comprehensive search system with baseline and hybrid reranking."""
    
    def __init__(self,
                 db_path: str = "data/rag_database.db",
                 embedding_system: Optional[EmbeddingSystem] = None,
                 hybrid_alpha: float = 0.7,  # Weight for vector vs BM25 scores
                 similarity_threshold: float = 0.3):
        """
        Initialize the search system.
        
        Args:
            db_path: Path to SQLite database
            embedding_system: Pre-initialized embedding system
            hybrid_alpha: Weight for combining vector and BM25 scores (0-1)
            similarity_threshold: Minimum similarity score for results
        """
        self.db_path = db_path
        self.hybrid_alpha = hybrid_alpha
        self.similarity_threshold = similarity_threshold
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get English stopwords first
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
        
        # Initialize embedding system
        if embedding_system is None:
            self.embedding_system = EmbeddingSystem(db_path=db_path)
        else:
            self.embedding_system = embedding_system
        
        # Initialize BM25
        self.bm25 = None
        self.chunk_texts = []
        self.chunk_ids = []
        self._build_bm25_index()
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from all document chunks."""
        self.logger.info("Building BM25 index...")
        
        # Get all chunks from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
            chunks = cursor.fetchall()
        
        if not chunks:
            self.logger.error("No chunks found for BM25 indexing!")
            return
        
        self.chunk_ids = [chunk[0] for chunk in chunks]
        self.chunk_texts = [chunk[1] for chunk in chunks]
        
        # Tokenize texts for BM25
        tokenized_corpus = []
        for text in self.chunk_texts:
            tokens = self._tokenize_for_bm25(text)
            tokenized_corpus.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.logger.info(f"BM25 index built with {len(self.chunk_texts)} documents")
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Clean and normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def _get_bm25_scores(self, query: str) -> List[float]:
        """Get BM25 scores for a query."""
        if self.bm25 is None:
            return [0.0] * len(self.chunk_texts)
        
        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return [0.0] * len(self.chunk_texts)
        
        scores = self.bm25.get_scores(query_tokens)
        return scores.tolist()
    
    def baseline_search(self, 
                       query: str, 
                       k: int = 5) -> List[SearchResult]:
        """
        Perform baseline vector similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Get vector similarity results
        vector_results = self.embedding_system.search_similar(query, k=k)
        
        search_results = []
        for i, result in enumerate(vector_results):
            # Filter by similarity threshold
            if result['similarity_score'] < self.similarity_threshold:
                continue
            
            search_result = SearchResult(
                chunk_id=result['chunk_id'],
                text=result['text'],
                source_title=result['source_title'],
                source_url=result['source_url'],
                source_file=result['source_file'],
                page_number=result['page_number'],
                chunk_index=result['chunk_index'],
                word_count=result['word_count'],
                vector_score=result['similarity_score'],
                final_rank=i + 1,
                reranker_used="baseline",
                confidence=result['similarity_score']
            )
            search_results.append(search_result)
        
        return search_results
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     initial_k: int = None) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity and BM25.
        
        Args:
            query: Search query
            k: Number of final results to return
            initial_k: Number of initial vector results to rerank (default: k*3)
            
        Returns:
            List of SearchResult objects, reranked by hybrid score
        """
        if initial_k is None:
            initial_k = min(k * 3, 50)  # Get more candidates for reranking
        
        # Get initial vector similarity results
        vector_results = self.embedding_system.search_similar(query, k=initial_k)
        
        if not vector_results:
            return []
        
        # Get BM25 scores for all documents
        all_bm25_scores = self._get_bm25_scores(query)
        
        # Normalize scores to 0-1 range
        vector_scores = [r['similarity_score'] for r in vector_results]
        relevant_bm25_scores = []
        
        # Get BM25 scores for the retrieved chunks
        for result in vector_results:
            chunk_id = result['chunk_id']
            if chunk_id in self.chunk_ids:
                idx = self.chunk_ids.index(chunk_id)
                relevant_bm25_scores.append(all_bm25_scores[idx])
            else:
                relevant_bm25_scores.append(0.0)
        
        # Normalize BM25 scores
        if relevant_bm25_scores and max(relevant_bm25_scores) > 0:
            max_bm25 = max(relevant_bm25_scores)
            normalized_bm25_scores = [score / max_bm25 for score in relevant_bm25_scores]
        else:
            normalized_bm25_scores = [0.0] * len(relevant_bm25_scores)
        
        # Calculate hybrid scores
        search_results = []
        for i, (result, bm25_score) in enumerate(zip(vector_results, normalized_bm25_scores)):
            vector_score = result['similarity_score']
            
            # Hybrid score: weighted combination
            hybrid_score = (self.hybrid_alpha * vector_score + 
                           (1 - self.hybrid_alpha) * bm25_score)
            
            # Filter by similarity threshold (applied to vector score)
            if vector_score < self.similarity_threshold:
                continue
            
            search_result = SearchResult(
                chunk_id=result['chunk_id'],
                text=result['text'],
                source_title=result['source_title'],
                source_url=result['source_url'],
                source_file=result['source_file'],
                page_number=result['page_number'],
                chunk_index=result['chunk_index'],
                word_count=result['word_count'],
                vector_score=vector_score,
                bm25_score=bm25_score,
                hybrid_score=hybrid_score,
                reranker_used="hybrid",
                confidence=hybrid_score
            )
            search_results.append(search_result)
        
        # Sort by hybrid score (descending)
        search_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # Update rankings and return top k
        for i, result in enumerate(search_results[:k]):
            result.final_rank = i + 1
        
        return search_results[:k]
    
    def search(self, 
              query: str, 
              k: int = 5, 
              mode: str = "hybrid") -> List[SearchResult]:
        """
        Main search interface.
        
        Args:
            query: Search query
            k: Number of results to return
            mode: Search mode ("baseline" or "hybrid")
            
        Returns:
            List of SearchResult objects
        """
        if mode.lower() == "baseline":
            return self.baseline_search(query, k=k)
        elif mode.lower() == "hybrid":
            return self.hybrid_search(query, k=k)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
    
    def get_chunk_context(self, 
                         chunk_id: str, 
                         context_size: int = 1) -> Optional[str]:
        """
        Get surrounding context for a chunk (previous/next chunks from same document).
        
        Args:
            chunk_id: Target chunk ID
            context_size: Number of chunks before/after to include
            
        Returns:
            Extended context text or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get the target chunk info
            cursor.execute("""
                SELECT source_file, chunk_index, text 
                FROM chunks 
                WHERE chunk_id = ?
            """, (chunk_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            source_file, chunk_index, target_text = result
            
            # Get surrounding chunks
            cursor.execute("""
                SELECT text, chunk_index
                FROM chunks 
                WHERE source_file = ? 
                AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
            """, (source_file, 
                  chunk_index - context_size, 
                  chunk_index + context_size))
            
            context_chunks = cursor.fetchall()
            
            if context_chunks:
                context_texts = [chunk[0] for chunk in context_chunks]
                return " ... ".join(context_texts)
            
            return target_text
    
    def explain_search(self, 
                      query: str, 
                      k: int = 3) -> Dict:
        """
        Explain how search results are ranked (for debugging/analysis).
        
        Args:
            query: Search query
            k: Number of results to analyze
            
        Returns:
            Dictionary with detailed scoring information
        """
        # Get both baseline and hybrid results
        baseline_results = self.baseline_search(query, k=k)
        hybrid_results = self.hybrid_search(query, k=k)
        
        # Analyze score differences
        explanation = {
            'query': query,
            'hybrid_alpha': self.hybrid_alpha,
            'similarity_threshold': self.similarity_threshold,
            'baseline_results': [],
            'hybrid_results': [],
            'ranking_changes': []
        }
        
        # Baseline results details
        for result in baseline_results:
            explanation['baseline_results'].append({
                'rank': result.final_rank,
                'chunk_id': result.chunk_id,
                'vector_score': result.vector_score,
                'source': result.source_title,
                'text_preview': result.text[:100] + "..."
            })
        
        # Hybrid results details
        for result in hybrid_results:
            explanation['hybrid_results'].append({
                'rank': result.final_rank,
                'chunk_id': result.chunk_id,
                'vector_score': result.vector_score,
                'bm25_score': result.bm25_score,
                'hybrid_score': result.hybrid_score,
                'source': result.source_title,
                'text_preview': result.text[:100] + "..."
            })
        
        # Find ranking changes
        baseline_order = [r.chunk_id for r in baseline_results]
        hybrid_order = [r.chunk_id for r in hybrid_results]
        
        for i, chunk_id in enumerate(hybrid_order):
            if chunk_id in baseline_order:
                baseline_rank = baseline_order.index(chunk_id) + 1
                hybrid_rank = i + 1
                if baseline_rank != hybrid_rank:
                    explanation['ranking_changes'].append({
                        'chunk_id': chunk_id,
                        'baseline_rank': baseline_rank,
                        'hybrid_rank': hybrid_rank,
                        'change': baseline_rank - hybrid_rank
                    })
        
        return explanation

def main():
    """Test the search system."""
    search_system = SearchSystem()
    
    # Test queries
    test_queries = [
        "What are the safety requirements for industrial machinery?",
        "risk assessment procedures",
        "machine guarding requirements",
        "emergency stop systems"
    ]
    
    print("🔍 Testing Search System")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 40)
        
        # Compare baseline vs hybrid
        baseline_results = search_system.search(query, k=3, mode="baseline")
        hybrid_results = search_system.search(query, k=3, mode="hybrid")
        
        print("Baseline Results:")
        for result in baseline_results:
            print(f"  {result.final_rank}. Score: {result.vector_score:.3f} | {result.source_title}")
        
        print("\nHybrid Results:")
        for result in hybrid_results:
            print(f"  {result.final_rank}. Vector: {result.vector_score:.3f}, BM25: {result.bm25_score:.3f}, "
                  f"Hybrid: {result.hybrid_score:.3f} | {result.source_title}")
        
        print()

if __name__ == "__main__":
    main()
