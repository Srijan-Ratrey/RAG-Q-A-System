#!/usr/bin/env python3
"""
Example usage of the RAG Q&A API.
Shows how to interact with the system programmatically.
"""

import requests
import json

# API Configuration
API_BASE = "http://localhost:8080"

def ask_question(question, k=5, mode="hybrid"):
    """Ask a question to the RAG system."""
    url = f"{API_BASE}/ask"
    payload = {
        "q": question,
        "k": k,
        "mode": mode
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def search_documents(query, k=10, mode="hybrid"):
    """Search documents without answer generation."""
    url = f"{API_BASE}/search"
    payload = {
        "q": query,
        "k": k,
        "mode": mode
    }
    
    response = requests.post(url, json=payload)
    return response.json() if response.status_code == 200 else None

def get_system_stats():
    """Get system statistics."""
    url = f"{API_BASE}/stats"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def main():
    """Example usage scenarios."""
    
    print("🤖 RAG Q&A System - Usage Examples")
    print("=" * 50)
    
    # Example 1: Basic safety question
    print("\n📝 Example 1: Basic Safety Question")
    result = ask_question("What are the basic safety requirements for machinery?")
    
    if result and result.get('answer'):
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Sources: {len(result['contexts'])} documents")
        print(f"Top source: {result['contexts'][0]['source']}")
    else:
        print("No answer provided")
    
    # Example 2: Technical question with baseline
    print("\n📝 Example 2: Technical Question (Baseline)")
    result = ask_question(
        "How do you perform risk assessment according to EN ISO 12100?", 
        k=3, 
        mode="baseline"
    )
    
    if result and result.get('answer'):
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Mode: {result['reranker_used']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    # Example 3: Document search
    print("\n🔍 Example 3: Document Search")
    search_result = search_documents("emergency stop systems", k=5)
    
    if search_result:
        print(f"Found {len(search_result['results'])} results")
        for i, doc in enumerate(search_result['results'][:3], 1):
            print(f"  {i}. {doc['source'][:50]}... (Score: {doc['hybrid_score']:.3f})")
    
    # Example 4: System stats
    print("\n📊 Example 4: System Statistics")
    stats = get_system_stats()
    
    if stats:
        print(f"Total vectors: {stats['embedding_stats']['total_vectors']:,}")
        print(f"Total chunks: {stats['database_stats']['total_chunks']:,}")
        print(f"Documents: {stats['database_stats']['total_documents']}")
        print(f"Index status: {stats['embedding_stats']['status']}")

if __name__ == "__main__":
    main()
