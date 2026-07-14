"""
Search system for RAG Q&A with baseline similarity search and hybrid reranker.
Combines vector similarity with BM25 keyword scoring for improved results.
"""

import os
import json
import sqlite3
import logging
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from rag_qa.embedding_system import EmbeddingSystem

# Download required NLTK data. Newer NLTK splits punkt into punkt_tab.
for _resource, _path in (
    ('punkt', 'tokenizers/punkt'),
    ('punkt_tab', 'tokenizers/punkt_tab'),
    ('stopwords', 'corpora/stopwords'),
):
    try:
        nltk.data.find(_path)
    except (LookupError, OSError):
        try:
            nltk.download(_resource, quiet=True)
        except Exception:  # pragma: no cover - offline fallback
            pass

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
                 similarity_threshold: float = 0.3,
                 rrf_k: int = 60,  # Reciprocal Rank Fusion constant
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 bm25_cache_path: Optional[str] = None):
        """
        Initialize the search system.

        Args:
            db_path: Path to SQLite database
            embedding_system: Pre-initialized embedding system
            hybrid_alpha: Weight for combining vector and BM25 scores (0-1)
            similarity_threshold: Minimum similarity score for results
            rrf_k: Rank-fusion constant for the "rrf" mode (higher = flatter)
            cross_encoder_model: HF cross-encoder for the "cross_encoder" mode
            bm25_cache_path: Where to cache the tokenized BM25 corpus (defaults
                to a file alongside the database)
        """
        self.db_path = db_path
        self.hybrid_alpha = hybrid_alpha
        self.similarity_threshold = similarity_threshold
        self.rrf_k = rrf_k
        self.cross_encoder_model = cross_encoder_model
        self.bm25_cache_path = bm25_cache_path or os.path.join(
            os.path.dirname(db_path) or ".", "bm25_corpus.json"
        )
        self._cross_encoder = None  # Lazily loaded on first use

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
        self.chunk_id_to_idx = {}  # chunk_id -> BM25 corpus position (O(1) lookup)
        self._build_bm25_index()

        # Warn loudly if the FAISS index has drifted from the database.
        self._check_consistency()
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from all document chunks.

        The tokenized corpus is cached to disk (keyed by the corpus
        fingerprint) so subsequent startups skip re-tokenization.
        """
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
        # O(1) chunk_id -> corpus position (replaces repeated list.index()).
        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}

        fingerprint = self.embedding_system._compute_corpus_fingerprint(self.chunk_ids)
        tokenized_corpus = self._load_tokenized_corpus(fingerprint)
        if tokenized_corpus is None:
            tokenized_corpus = [self._tokenize_for_bm25(text) for text in self.chunk_texts]
            self._save_tokenized_corpus(fingerprint, tokenized_corpus)

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.logger.info(f"BM25 index built with {len(self.chunk_texts)} documents")

    def _load_tokenized_corpus(self, fingerprint: str) -> Optional[List[List[str]]]:
        """Return the cached tokenized corpus if it matches the fingerprint."""
        if not os.path.exists(self.bm25_cache_path):
            return None
        try:
            with open(self.bm25_cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if cache.get('fingerprint') == fingerprint:
                self.logger.info("Loaded cached BM25 tokenized corpus")
                return cache['corpus']
        except Exception as e:
            self.logger.warning(f"Failed to load BM25 cache: {e}")
        return None

    def _save_tokenized_corpus(self, fingerprint: str, corpus: List[List[str]]) -> None:
        """Persist the tokenized corpus keyed by the corpus fingerprint."""
        try:
            os.makedirs(os.path.dirname(self.bm25_cache_path) or ".", exist_ok=True)
            with open(self.bm25_cache_path, 'w', encoding='utf-8') as f:
                json.dump({'fingerprint': fingerprint, 'corpus': corpus}, f)
        except Exception as e:
            self.logger.warning(f"Failed to write BM25 cache: {e}")

    def _check_consistency(self) -> None:
        """Warn if the FAISS index no longer matches the database corpus."""
        try:
            if self.embedding_system.is_index_stale():
                self.logger.error(
                    "FAISS index is STALE relative to the database. Vector "
                    "search may return wrong results. Rebuild with "
                    "EmbeddingSystem.build_index(force_rebuild=True)."
                )
        except Exception as e:  # pragma: no cover - defensive
            self.logger.warning(f"Could not verify index consistency: {e}")
    
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
        """Get BM25 scores for a query over the whole corpus."""
        if self.bm25 is None:
            return [0.0] * len(self.chunk_texts)

        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return [0.0] * len(self.chunk_texts)

        scores = self.bm25.get_scores(query_tokens)
        return scores.tolist()

    def _get_bm25_scores_for_chunks(self, query: str, chunk_ids: List[str]) -> List[float]:
        """
        BM25 scores for a specific set of chunks only.

        Uses BM25Okapi.get_batch_scores so we score just the ~k*3 candidates
        instead of the entire corpus on every query.
        """
        if self.bm25 is None:
            return [0.0] * len(chunk_ids)

        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return [0.0] * len(chunk_ids)

        # Map chunk_ids to corpus indices; chunks missing from BM25 score 0.
        candidate_indices = [self.chunk_id_to_idx.get(cid) for cid in chunk_ids]
        valid = [idx for idx in candidate_indices if idx is not None]
        batch = self.bm25.get_batch_scores(query_tokens, valid) if valid else []

        scores, it = [], iter(batch)
        for idx in candidate_indices:
            scores.append(float(next(it)) if idx is not None else 0.0)
        return scores
    
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

        # BM25 scores for just the candidate chunks (not the whole corpus).
        candidate_ids = [r['chunk_id'] for r in vector_results]
        relevant_bm25_scores = self._get_bm25_scores_for_chunks(query, candidate_ids)

        # Normalize BM25 scores to 0-1 within this candidate set.
        max_bm25 = max(relevant_bm25_scores) if relevant_bm25_scores else 0.0
        if max_bm25 > 0:
            normalized_bm25_scores = [s / max_bm25 for s in relevant_bm25_scores]
        else:
            normalized_bm25_scores = [0.0] * len(relevant_bm25_scores)

        # Calculate hybrid scores
        search_results = []
        for result, bm25_score in zip(vector_results, normalized_bm25_scores):
            vector_score = result['similarity_score']

            # Hybrid score: weighted combination (used for RANKING only).
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
                # Confidence is the raw cosine similarity (comparable across
                # queries), NOT the hybrid score whose BM25 term is normalized
                # per-query and therefore not comparable. This lets a single
                # abstention threshold behave consistently across modes.
                confidence=vector_score
            )
            search_results.append(search_result)

        # Sort by hybrid score (descending)
        search_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        # Update rankings and return top k
        for i, result in enumerate(search_results[:k]):
            result.final_rank = i + 1

        return search_results[:k]

    def rrf_search(self, query: str, k: int = 5, initial_k: int = None) -> List[SearchResult]:
        """
        Rank-fusion search using Reciprocal Rank Fusion (RRF).

        Fuses the vector ranking and the BM25 ranking by rank position rather
        than by score magnitude, so it needs no score normalization and is
        immune to the per-query BM25 scaling problem that affects hybrid mode.
        RRF score = sum over rankings of 1 / (rrf_k + rank).
        """
        if initial_k is None:
            initial_k = min(k * 3, 50)

        vector_results = self.embedding_system.search_similar(query, k=initial_k)
        if not vector_results:
            return []

        candidate_ids = [r['chunk_id'] for r in vector_results]
        bm25_scores = self._get_bm25_scores_for_chunks(query, candidate_ids)

        # Vector rank: order returned by FAISS. BM25 rank: order by BM25 score.
        vector_rank = {cid: i for i, cid in enumerate(candidate_ids)}
        bm25_order = sorted(range(len(candidate_ids)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank = {candidate_ids[pos]: rank for rank, pos in enumerate(bm25_order)}

        fused = {}
        for cid in candidate_ids:
            score = 1.0 / (self.rrf_k + vector_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (self.rrf_k + bm25_rank[cid])
            fused[cid] = score

        results_by_id = {r['chunk_id']: r for r in vector_results}
        search_results = []
        for cid in candidate_ids:
            result = results_by_id[cid]
            if result['similarity_score'] < self.similarity_threshold:
                continue
            search_results.append(SearchResult(
                chunk_id=cid,
                text=result['text'],
                source_title=result['source_title'],
                source_url=result['source_url'],
                source_file=result['source_file'],
                page_number=result['page_number'],
                chunk_index=result['chunk_index'],
                word_count=result['word_count'],
                vector_score=result['similarity_score'],
                bm25_score=bm25_scores[vector_rank[cid]],
                hybrid_score=fused[cid],
                reranker_used="rrf",
                confidence=result['similarity_score'],
            ))

        search_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        for i, result in enumerate(search_results[:k]):
            result.final_rank = i + 1
        return search_results[:k]

    def _get_cross_encoder(self):
        """Lazily load the cross-encoder model (optional dependency/download)."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self.logger.info(f"Loading cross-encoder: {self.cross_encoder_model}")
            self._cross_encoder = CrossEncoder(self.cross_encoder_model)
        return self._cross_encoder

    def cross_encoder_search(self, query: str, k: int = 5,
                             initial_k: int = None) -> List[SearchResult]:
        """
        Rerank candidates with a cross-encoder for the strongest relevance
        signal. Falls back to hybrid mode if the model cannot be loaded
        (e.g. offline). Confidence is the sigmoid of the cross-encoder score,
        so its abstention threshold differs from the cosine-based modes.
        """
        if initial_k is None:
            initial_k = min(k * 4, 50)

        vector_results = self.embedding_system.search_similar(query, k=initial_k)
        if not vector_results:
            return []

        try:
            model = self._get_cross_encoder()
        except Exception as e:
            self.logger.warning(f"Cross-encoder unavailable ({e}); using hybrid.")
            return self.hybrid_search(query, k=k)

        pairs = [[query, r['text']] for r in vector_results]
        raw_scores = model.predict(pairs)
        # Map logits to (0, 1) so confidence is interpretable.
        confidences = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores)))

        search_results = []
        for result, raw, conf in zip(vector_results, raw_scores, confidences):
            search_results.append(SearchResult(
                chunk_id=result['chunk_id'],
                text=result['text'],
                source_title=result['source_title'],
                source_url=result['source_url'],
                source_file=result['source_file'],
                page_number=result['page_number'],
                chunk_index=result['chunk_index'],
                word_count=result['word_count'],
                vector_score=result['similarity_score'],
                bm25_score=0.0,
                hybrid_score=float(raw),
                reranker_used="cross_encoder",
                confidence=float(conf),
            ))

        search_results.sort(key=lambda x: x.hybrid_score, reverse=True)
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
            mode: Search mode ("baseline", "hybrid", "rrf", or "cross_encoder")

        Returns:
            List of SearchResult objects
        """
        mode = mode.lower()
        if mode == "baseline":
            return self.baseline_search(query, k=k)
        elif mode == "hybrid":
            return self.hybrid_search(query, k=k)
        elif mode == "rrf":
            return self.rrf_search(query, k=k)
        elif mode in ("cross_encoder", "cross-encoder"):
            return self.cross_encoder_search(query, k=k)
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
