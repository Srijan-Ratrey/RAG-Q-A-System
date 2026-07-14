#!/bin/bash
# Script to rebuild the RAG database and indices from PDFs

set -e  # Exit on error

echo "🚀 Rebuilding RAG Q&A System Data"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found. Please run 'python3 -m venv venv' first"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if industrial-safety-pdfs directory exists
if [ ! -d "industrial-safety-pdfs" ]; then
    echo "❌ Error: PDF directory 'industrial-safety-pdfs' not found"
    echo "   Please ensure PDF files are in the correct directory"
    exit 1
fi

# Count PDF files
pdf_count=$(find industrial-safety-pdfs -name "*.pdf" | wc -l)
echo "📄 Found $pdf_count PDF files"

if [ $pdf_count -eq 0 ]; then
    echo "❌ Error: No PDF files found in industrial-safety-pdfs/"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Remove existing data files
echo "🧹 Cleaning existing data files..."
rm -f data/rag_database.db data/faiss_index.bin data/chunk_id_map.json \
      data/chunk_id_map.pkl data/bm25_corpus.json

# Step 1: Process documents and create database
echo "📚 Step 1: Processing PDF documents..."
python -m rag_qa.document_processor

# Check if database was created
if [ ! -f "data/rag_database.db" ]; then
    echo "❌ Error: Database creation failed"
    exit 1
fi

# Step 2: Build embeddings and FAISS index
echo "🧠 Step 2: Building embeddings and search index..."
python -m rag_qa.embedding_system

# Check if index was created
if [ ! -f "data/faiss_index.bin" ]; then
    echo "❌ Error: FAISS index creation failed"
    exit 1
fi

# Get final statistics
echo ""
echo "📊 Final Statistics:"
echo "==================="

# Database stats
chunks=$(sqlite3 data/rag_database.db "SELECT COUNT(*) FROM chunks;" 2>/dev/null || echo "0")
echo "  📄 Total chunks: $chunks"

sources=$(sqlite3 data/rag_database.db "SELECT COUNT(DISTINCT source_file) FROM chunks;" 2>/dev/null || echo "0")
echo "  📁 Source files: $sources"

# File sizes
echo "  💾 Database size: $(du -sh data/rag_database.db | cut -f1)"
echo "  🔍 Index size: $(du -sh data/faiss_index.bin | cut -f1)"
echo "  🗂️  Mapping size: $(du -sh data/chunk_id_map.json | cut -f1)"

echo ""
echo "✅ Data rebuild complete!"
echo "🚀 You can now start the API with: python -m rag_qa.api"
