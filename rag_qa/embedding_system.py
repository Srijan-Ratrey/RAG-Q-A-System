"""
Embedding system for RAG Q&A using sentence transformers and FAISS.
Handles vector generation, storage, and similarity search.
"""

import os
import json
import hashlib
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

import faiss
from sentence_transformers import SentenceTransformer


class EmbeddingSystem:
    """Manages document embeddings and vector similarity search."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        db_path: str = "data/rag_database.db",
        index_path: str = "data/faiss_index.bin",
        chunk_id_map_path: str = "data/chunk_id_map.json",
        batch_size: int = 32,
        model=None,
    ):
        """
        Initialize the embedding system.

        Args:
            model_name: Name of the sentence transformer model
            db_path: Path to SQLite database with chunks
            index_path: Path to save/load FAISS index
            chunk_id_map_path: Path to save/load chunk ID mapping
            batch_size: Batch size for embedding generation
            model: A preloaded SentenceTransformer to reuse instead of loading a
                fresh one. Lets callers that build many indices (e.g. the upload
                UI) share a single model instead of reloading it each time.
        """
        self.model_name = model_name
        self.db_path = db_path
        self.index_path = index_path
        self.chunk_id_map_path = chunk_id_map_path
        self.batch_size = batch_size

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize model (reuse a preloaded one when provided).
        if model is not None:
            self.model = model
        else:
            self.logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Model ready. Embedding dimension: {self.embedding_dim}")

        # Initialize FAISS index and chunk mapping
        self.index = None
        self.chunk_id_map = {}  # Maps index position to chunk_id
        self.reverse_chunk_map = {}  # Maps chunk_id to index position
        self.corpus_fingerprint = None  # Fingerprint of the indexed corpus

        # Try to load existing index
        self._load_existing_index()

    def _load_existing_index(self) -> bool:
        """Load existing FAISS index and chunk mapping if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.chunk_id_map_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)

                # Load chunk mapping (JSON: safe, human-readable). JSON object
                # keys are strings, so cast positions back to int.
                with open(self.chunk_id_map_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                raw_map = payload.get("chunk_id_map", {})
                self.chunk_id_map = {
                    int(idx): chunk_id for idx, chunk_id in raw_map.items()
                }
                self.corpus_fingerprint = payload.get("fingerprint")

                # Create reverse mapping
                self.reverse_chunk_map = {
                    chunk_id: idx for idx, chunk_id in self.chunk_id_map.items()
                }

                self.logger.info(
                    f"Loaded existing index with {self.index.ntotal} vectors"
                )
                return True

            except Exception as e:
                self.logger.warning(f"Failed to load existing index: {e}")
                return False

        return False

    def _compute_corpus_fingerprint(self, chunk_ids: Optional[List[str]] = None) -> str:
        """
        Compute a fingerprint of the corpus from its chunk IDs.

        Because chunk IDs are content-derived (see DocumentProcessor), this
        changes whenever any chunk's text, ordering, or membership changes.
        Used to detect when the FAISS index has drifted from the database.
        """
        if chunk_ids is None:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id")
                chunk_ids = [row[0] for row in cursor.fetchall()]

        hasher = hashlib.sha256()
        hasher.update(str(len(chunk_ids)).encode())
        for chunk_id in sorted(chunk_ids):
            hasher.update(chunk_id.encode())
        return hasher.hexdigest()

    def is_index_stale(self) -> bool:
        """Return True if the loaded index no longer matches the database."""
        if self.index is None or self.corpus_fingerprint is None:
            return True
        return self.corpus_fingerprint != self._compute_corpus_fingerprint()

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

    def generate_embeddings(
        self, texts: List[str], show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a list of texts.

        Normalization is delegated to SentenceTransformer
        (``normalize_embeddings=True``) so inner product equals cosine
        similarity, matching the FAISS ``IndexFlatIP``.

        Args:
            texts: List of text strings
            show_progress_bar: Show a progress bar (off for single queries)

        Returns:
            Normalized float32 embeddings array
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def build_index(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Build or rebuild the FAISS index from all chunks in database.

        Args:
            force_rebuild: Whether to rebuild even if index exists

        Returns:
            Dictionary with build statistics
        """
        if self.index is not None and not force_rebuild:
            # An index exists — but only reuse it if it still matches the DB.
            if self.is_index_stale():
                self.logger.warning(
                    "Existing index is stale (corpus fingerprint mismatch); "
                    "rebuilding from the database."
                )
            else:
                self.logger.info("Index already exists and matches the database.")
                return {
                    "total_vectors": self.index.ntotal,
                    "embedding_dim": self.embedding_dim,
                    "status": "existing",
                }

        # Get all chunks from database
        self.logger.info("Retrieving chunks from database...")
        chunks = self._get_chunks_from_db()

        if not chunks:
            self.logger.error("No chunks found in database!")
            return {"error": "No chunks found"}

        self.logger.info(f"Found {len(chunks)} chunks to embed")

        # Create new index
        self._create_new_index()

        # Extract texts and chunk IDs
        chunk_ids = [chunk[0] for chunk in chunks]
        texts = [chunk[1] for chunk in chunks]

        # Generate all embeddings in one call — SentenceTransformer batches
        # internally (batch_size), so there is no need to batch again here.
        self.logger.info("Generating embeddings...")
        embeddings_matrix = self.generate_embeddings(texts, show_progress_bar=True)
        self.logger.info(f"Generated embeddings shape: {embeddings_matrix.shape}")

        # Add to FAISS index
        self.logger.info("Adding embeddings to FAISS index...")
        self.index.add(embeddings_matrix)

        # Update chunk mapping
        for i, chunk_id in enumerate(chunk_ids):
            self.chunk_id_map[i] = chunk_id
            self.reverse_chunk_map[chunk_id] = i

        # Record the fingerprint of exactly the corpus we just indexed.
        self.corpus_fingerprint = self._compute_corpus_fingerprint(chunk_ids)

        # Save index and mapping
        self._save_index()

        stats = {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "status": "rebuilt",
        }

        self.logger.info(f"Index build complete: {stats}")
        return stats

    def _save_index(self) -> None:
        """Save FAISS index and chunk mapping to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

        # Save chunk mapping + corpus fingerprint as JSON (no pickle: the map
        # is just {position: chunk_id} and JSON carries no code-execution risk).
        payload = {
            "version": 1,
            "fingerprint": self.corpus_fingerprint,
            "chunk_id_map": {
                str(idx): chunk_id for idx, chunk_id in self.chunk_id_map.items()
            },
        }
        with open(self.chunk_id_map_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        self.logger.info(f"Saved index with {self.index.ntotal} vectors")

    def search_similar(
        self, query: str, k: int = 5, return_scores: bool = True
    ) -> List[Dict]:
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
        scores, indices = self.index.search(query_embedding, k)

        # Resolve FAISS positions to chunk IDs, preserving rank order.
        ranked = []  # (chunk_id, score)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            chunk_id = self.chunk_id_map.get(int(idx))
            if chunk_id is None:
                self.logger.warning(f"No chunk ID found for index {idx}")
                continue
            ranked.append((chunk_id, float(score)))

        if not ranked:
            return []

        # Fetch all chunk details in a single query (avoids per-chunk N+1).
        details = self._get_chunk_details_batch([chunk_id for chunk_id, _ in ranked])

        results = []
        for rank, (chunk_id, score) in enumerate(ranked, 1):
            chunk_info = details.get(chunk_id)
            if chunk_info:
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "similarity_score": score,
                        "rank": rank,
                        **chunk_info,
                    }
                )

        return results

    def _get_chunk_details(self, chunk_id: str) -> Optional[Dict]:
        """Get full chunk details from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_id, text, source_title, source_url, source_file, 
                       page_number, chunk_index, word_count, char_count
                FROM chunks 
                WHERE chunk_id = ?
            """,
                (chunk_id,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "chunk_id": row[0],
                    "text": row[1],
                    "source_title": row[2],
                    "source_url": row[3],
                    "source_file": row[4],
                    "page_number": row[5],
                    "chunk_index": row[6],
                    "word_count": row[7],
                    "char_count": row[8],
                }
        return None

    def _get_chunk_details_batch(self, chunk_ids: List[str]) -> Dict[str, Dict]:
        """Get full chunk details for many chunk IDs in a single query."""
        if not chunk_ids:
            return {}

        placeholders = ",".join("?" * len(chunk_ids))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT chunk_id, text, source_title, source_url, source_file,
                       page_number, chunk_index, word_count, char_count
                FROM chunks
                WHERE chunk_id IN ({placeholders})
            """,
                chunk_ids,
            )
            rows = cursor.fetchall()

        return {
            row[0]: {
                "chunk_id": row[0],
                "text": row[1],
                "source_title": row[2],
                "source_url": row[3],
                "source_file": row[4],
                "page_number": row[5],
                "chunk_index": row[6],
                "word_count": row[7],
                "char_count": row[8],
            }
            for row in rows
        }

    def search_by_chunk_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """
        Retrieve full information for specific chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of chunk information dictionaries (order follows chunk_ids)
        """
        details = self._get_chunk_details_batch(chunk_ids)
        return [details[cid] for cid in chunk_ids if cid in details]

    def get_index_stats(self) -> Dict:
        """Get statistics about the index."""
        if self.index is None:
            return {"status": "not_built"}

        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "index_type": "FlatIP",
            "status": "ready",
        }

    def test_search(self, query: str = "machine safety requirements") -> None:
        """Test the search functionality with a sample query."""
        self.logger.info(f"Testing search with query: '{query}'")

        results = self.search_similar(query, k=3)

        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 60)

        for result in results:
            print(
                f"\n📄 Rank {result['rank']} (Score: {result['similarity_score']:.3f})"
            )
            print(f"Source: {result['source_title']}")
            print(f"File: {result['source_file']}")
            if result["page_number"]:
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
    embedding_system.test_search(
        "What are the safety requirements for industrial machinery?"
    )
    embedding_system.test_search("risk assessment procedures")
    embedding_system.test_search("machine guarding")


if __name__ == "__main__":
    main()
