"""
Embedding system for RAG Q&A using sentence transformers and FAISS.
Handles vector generation, storage, and similarity search.
"""

import os
import sqlite3
import pickle
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class EmbeddingSystem:
    """Manages document embeddings and vector similarity search."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 db_path: str = "data/rag_database.db",
                 index_path: str = "data/faiss_index.bin",
                 chunk_id_map_path: str = "data/chunk_id_map.pkl",
                 batch_size: int = 32):
        """
        Initialize the embedding system.
        
        Args:
            model_name: Name of the sentence transformer model
            db_path: Path to SQLite database with chunks
            index_path: Path to save/load FAISS index
            chunk_id_map_path: Path to save/load chunk ID mapping
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.db_path = db_path
        self.index_path = index_path
        self.chunk_id_map_path = chunk_id_map_path
        self.batch_size = batch_size
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Initialize FAISS index and chunk mapping
        self.index = None
        self.chunk_id_map = {}  # Maps index position to chunk_id
        self.reverse_chunk_map = {}  # Maps chunk_id to index position
        
        # Try to load existing index
        self._load_existing_index()
    
    def _load_existing_index(self) -> bool:
        """Load existing FAISS index and chunk mapping if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.chunk_id_map_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load chunk mapping
                with open(self.chunk_id_map_path, 'rb') as f:
                    self.chunk_id_map = pickle.load(f)
                
                # Create reverse mapping
                self.reverse_chunk_map = {chunk_id: idx for idx, chunk_id in self.chunk_id_map.items()}
                
                self.logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
                return True
                
            except Exception as e:
                self.logger.warning(f"Failed to load existing index: {e}")
                return False
        
        return False
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Use Inner Product (IP) similarity for cosine similarity
        # Note: We'll normalize vectors so IP = cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunk_id_map = {}
        self.reverse_chunk_map = {}
        self.logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def _get_chunks_from_db(self) -> List[Tuple[str, str]]:
        """
        Retrieve all chunks from database.
        
        Returns:
            List of (chunk_id, text) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
            return cursor.fetchall()
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity using inner product."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Normalized embeddings array
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize manually
        )
        
        return self._normalize_embeddings(embeddings)
    
    def build_index(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Build or rebuild the FAISS index from all chunks in database.
        
        Args:
            force_rebuild: Whether to rebuild even if index exists
            
        Returns:
            Dictionary with build statistics
        """
        if self.index is not None and not force_rebuild:
            self.logger.info("Index already exists. Use force_rebuild=True to rebuild.")
            return {
                'total_vectors': self.index.ntotal,
                'embedding_dim': self.embedding_dim,
                'status': 'existing'
            }
        
        # Get all chunks from database
        self.logger.info("Retrieving chunks from database...")
        chunks = self._get_chunks_from_db()
        
        if not chunks:
            self.logger.error("No chunks found in database!")
            return {'error': 'No chunks found'}
        
        self.logger.info(f"Found {len(chunks)} chunks to embed")
        
        # Create new index
        self._create_new_index()
        
        # Extract texts and chunk IDs
        chunk_ids = [chunk[0] for chunk in chunks]
        texts = [chunk[1] for chunk in chunks]
        
        # Generate embeddings in batches
        self.logger.info("Generating embeddings...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.generate_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings_matrix = np.vstack(all_embeddings)
        self.logger.info(f"Generated embeddings shape: {embeddings_matrix.shape}")
        
        # Add to FAISS index
        self.logger.info("Adding embeddings to FAISS index...")
        self.index.add(embeddings_matrix.astype('float32'))
        
        # Update chunk mapping
        for i, chunk_id in enumerate(chunk_ids):
            self.chunk_id_map[i] = chunk_id
            self.reverse_chunk_map[chunk_id] = i
        
        # Save index and mapping
        self._save_index()
        
        stats = {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'status': 'rebuilt'
        }
        
        self.logger.info(f"Index build complete: {stats}")
        return stats
    
    def _save_index(self) -> None:
        """Save FAISS index and chunk mapping to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save chunk mapping
        with open(self.chunk_id_map_path, 'wb') as f:
            pickle.dump(self.chunk_id_map, f)
        
        self.logger.info(f"Saved index with {self.index.ntotal} vectors")
    
    def search_similar(self, 
                      query: str, 
                      k: int = 5, 
                      return_scores: bool = True) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Query string
            k: Number of results to return
            return_scores: Whether to include similarity scores
            
        Returns:
            List of result dictionaries with chunk information
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if self.index.ntotal == 0:
            self.logger.warning("Index is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            chunk_id = self.chunk_id_map.get(idx)
            if chunk_id is None:
                self.logger.warning(f"No chunk ID found for index {idx}")
                continue
            
            # Get chunk details from database
            chunk_info = self._get_chunk_details(chunk_id)
            if chunk_info:
                result = {
                    'chunk_id': chunk_id,
                    'similarity_score': float(score),
                    'rank': i + 1,
                    **chunk_info
                }
                results.append(result)
        
        return results
    
    def _get_chunk_details(self, chunk_id: str) -> Optional[Dict]:
        """Get full chunk details from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_id, text, source_title, source_url, source_file, 
                       page_number, chunk_index, word_count, char_count
                FROM chunks 
                WHERE chunk_id = ?
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'chunk_id': row[0],
                    'text': row[1],
                    'source_title': row[2],
                    'source_url': row[3],
                    'source_file': row[4],
                    'page_number': row[5],
                    'chunk_index': row[6],
                    'word_count': row[7],
                    'char_count': row[8]
                }
        return None
    
    def search_by_chunk_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """
        Retrieve full information for specific chunk IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of chunk information dictionaries
        """
        results = []
        for chunk_id in chunk_ids:
            chunk_info = self._get_chunk_details(chunk_id)
            if chunk_info:
                results.append(chunk_info)
        return results
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index."""
        if self.index is None:
            return {'status': 'not_built'}
        
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'index_type': 'FlatIP',
            'status': 'ready'
        }
    
    def test_search(self, query: str = "machine safety requirements") -> None:
        """Test the search functionality with a sample query."""
        self.logger.info(f"Testing search with query: '{query}'")
        
        results = self.search_similar(query, k=3)
        
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 60)
        
        for result in results:
            print(f"\n📄 Rank {result['rank']} (Score: {result['similarity_score']:.3f})")
            print(f"Source: {result['source_title']}")
            print(f"File: {result['source_file']}")
            if result['page_number']:
                print(f"Page: {result['page_number']}")
            print(f"Text: {result['text'][:200]}...")
            print("-" * 40)

def main():
    """Main function for testing embedding system."""
    embedding_system = EmbeddingSystem()
    
    # Build index
    print("🚀 Building embedding index...")
    stats = embedding_system.build_index()
    print(f"Build stats: {stats}")
    
    # Test search
    embedding_system.test_search("What are the safety requirements for industrial machinery?")
    embedding_system.test_search("risk assessment procedures")
    embedding_system.test_search("machine guarding")

if __name__ == "__main__":
    main()
