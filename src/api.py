"""
Flask API for RAG Q&A system.
Provides endpoint for question answering with grounded answers and citations.
"""

import os
import logging
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search_system import SearchSystem, SearchResult

class AnswerGenerator:
    """Generates extractive answers from search results with citations."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize answer generator.
        
        Args:
            confidence_threshold: Minimum confidence for providing an answer
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def extract_answer(self, 
                      query: str, 
                      search_results: List[SearchResult]) -> Tuple[Optional[str], float, str]:
        """
        Extract an answer from search results.
        
        Args:
            query: Original query
            search_results: List of search results
            
        Returns:
            Tuple of (answer_text, confidence, reason)
        """
        if not search_results:
            return None, 0.0, "No relevant documents found"
        
        # Get top result for answer extraction
        top_result = search_results[0]
        
        # Check confidence threshold
        if top_result.confidence < self.confidence_threshold:
            return None, top_result.confidence, f"Low confidence score: {top_result.confidence:.3f}"
        
        # For extractive answering, we'll use the most relevant chunk
        # In a more sophisticated system, we might use NLG models
        answer_text = self._extract_relevant_sentences(query, top_result.text)
        
        return answer_text, top_result.confidence, "Answer extracted from top result"
    
    def _extract_relevant_sentences(self, query: str, text: str) -> str:
        """
        Extract most relevant sentences from a chunk of text.
        
        Args:
            query: Original query
            text: Source text
            
        Returns:
            Extracted answer text
        """
        # Simple extractive approach: return first few sentences or paragraph
        sentences = text.split('. ')
        
        # Return first 2-3 sentences or up to 200 characters
        if len(sentences) >= 2:
            answer = '. '.join(sentences[:2]) + '.'
        else:
            answer = text
        
        # Truncate if too long
        if len(answer) > 300:
            answer = answer[:297] + "..."
        
        return answer.strip()

class QAService:
    """Main Q&A service combining search and answer generation."""
    
    def __init__(self, 
                 db_path: str = "data/rag_database.db",
                 confidence_threshold: float = 0.5):
        """
        Initialize Q&A service.
        
        Args:
            db_path: Path to database
            confidence_threshold: Minimum confidence for answers
        """
        self.search_system = SearchSystem(db_path=db_path)
        self.answer_generator = AnswerGenerator(confidence_threshold=confidence_threshold)
        self.logger = logging.getLogger(__name__)
    
    def ask(self, 
           query: str, 
           k: int = 5, 
           mode: str = "hybrid") -> Dict:
        """
        Answer a question using the Q&A system.
        
        Args:
            query: Question to answer
            k: Number of context chunks to return
            mode: Search mode ("baseline" or "hybrid")
            
        Returns:
            Dictionary with answer, contexts, and metadata
        """
        try:
            # Search for relevant documents
            search_results = self.search_system.search(query, k=k, mode=mode)
            
            # Generate answer
            answer, confidence, reason = self.answer_generator.extract_answer(query, search_results)
            
            # Format contexts for response
            contexts = []
            for result in search_results:
                context = {
                    "text": result.text,
                    "score": result.hybrid_score if mode == "hybrid" else result.vector_score,
                    "source": result.source_title,
                    "url": result.source_url if result.source_url else "",
                    "file": result.source_file,
                    "page": result.page_number,
                    "chunk_id": result.chunk_id,
                    "vector_score": result.vector_score,
                    "bm25_score": result.bm25_score if hasattr(result, 'bm25_score') else 0.0
                }
                contexts.append(context)
            
            response = {
                "answer": answer,
                "contexts": contexts,
                "reranker_used": mode,
                "confidence": confidence,
                "abstain_reason": reason if answer is None else None,
                "query": query,
                "total_results": len(search_results)
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}")
            return {
                "answer": None,
                "contexts": [],
                "reranker_used": mode,
                "confidence": 0.0,
                "abstain_reason": f"System error: {str(e)}",
                "query": query,
                "total_results": 0
            }

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize QA service (will be done on first request to handle startup time)
qa_service = None

def get_qa_service():
    """Get or initialize QA service."""
    global qa_service
    if qa_service is None:
        # Get database path relative to project root
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rag_database.db")
        qa_service = QAService(db_path=db_path)
    return qa_service

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "RAG Q&A API",
        "version": "1.0.0"
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Main Q&A endpoint.
    
    Request format:
    {
        "q": "What are the safety requirements?",
        "k": 5,
        "mode": "hybrid"
    }
    
    Response format:
    {
        "answer": "Safety requirements include...",
        "contexts": [...],
        "reranker_used": "hybrid",
        "confidence": 0.85
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Invalid JSON in request body"
            }), 400
        
        # Extract parameters
        query = data.get('q', '').strip()
        k = data.get('k', 5)
        mode = data.get('mode', 'hybrid').lower()
        
        # Validate parameters
        if not query:
            return jsonify({
                "error": "Query parameter 'q' is required and cannot be empty"
            }), 400
        
        if not isinstance(k, int) or k < 1 or k > 20:
            return jsonify({
                "error": "Parameter 'k' must be an integer between 1 and 20"
            }), 400
        
        if mode not in ['baseline', 'hybrid']:
            return jsonify({
                "error": "Parameter 'mode' must be 'baseline' or 'hybrid'"
            }), 400
        
        # Process query
        qa_service = get_qa_service()
        response = qa_service.ask(query, k=k, mode=mode)
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error in /ask endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/search', methods=['POST'])
def search_documents():
    """
    Document search endpoint (no answer generation).
    
    Request format:
    {
        "q": "search query",
        "k": 10,
        "mode": "hybrid"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Invalid JSON in request body"}), 400
        
        query = data.get('q', '').strip()
        k = data.get('k', 10)
        mode = data.get('mode', 'hybrid').lower()
        
        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        
        qa_service = get_qa_service()
        search_results = qa_service.search_system.search(query, k=k, mode=mode)
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "chunk_id": result.chunk_id,
                "text": result.text,
                "source": result.source_title,
                "url": result.source_url,
                "file": result.source_file,
                "page": result.page_number,
                "rank": result.final_rank,
                "vector_score": result.vector_score,
                "bm25_score": getattr(result, 'bm25_score', 0.0),
                "hybrid_score": getattr(result, 'hybrid_score', result.vector_score),
                "confidence": result.confidence
            })
        
        return jsonify({
            "query": query,
            "results": results,
            "total_results": len(results),
            "mode": mode
        })
        
    except Exception as e:
        app.logger.error(f"Error in /search endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        qa_service = get_qa_service()
        
        # Get embedding system stats
        embedding_stats = qa_service.search_system.embedding_system.get_index_stats()
        
        # Get database stats (simplified)
        import sqlite3
        db_path = qa_service.search_system.db_path
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT source_file) FROM chunks")
            total_documents = cursor.fetchone()[0]
        
        return jsonify({
            "embedding_stats": embedding_stats,
            "database_stats": {
                "total_chunks": total_chunks,
                "total_documents": total_documents
            },
            "bm25_stats": {
                "index_size": len(qa_service.search_system.chunk_texts),
                "status": "ready" if qa_service.search_system.bm25 else "not_ready"
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error in /stats endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation."""
    return jsonify({
        "service": "RAG Q&A API",
        "version": "1.0.0",
        "description": "Question answering system for industrial safety documents",
        "endpoints": {
            "/health": "GET - Health check",
            "/ask": "POST - Ask a question",
            "/search": "POST - Search documents",
            "/stats": "GET - System statistics"
        },
        "example_request": {
            "url": "/ask",
            "method": "POST",
            "body": {
                "q": "What are the safety requirements for industrial machinery?",
                "k": 5,
                "mode": "hybrid"
            }
        }
    })

def main():
    """Run the Flask application."""
    # Configuration
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 9000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"🚀 Starting RAG Q&A API on {host}:{port}")
    print(f"📚 Using database: {os.path.join('data', 'rag_database.db')}")
    print(f"🔍 Available endpoints: /health, /ask, /search, /stats")
    
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
