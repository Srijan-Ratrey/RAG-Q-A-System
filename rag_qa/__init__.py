"""RAG Q&A System — retrieval-augmented Q&A over industrial safety documents.

Public modules:
    document_processor  PDF extraction, cleaning, and chunking
    embedding_system    Sentence embeddings + FAISS index (with consistency fingerprint)
    search_system       baseline / hybrid / rrf / cross_encoder retrieval
    api                 Flask API and extractive answer generation
    evaluation          IR metrics + evaluation drivers
"""

__version__ = "1.0.0"
