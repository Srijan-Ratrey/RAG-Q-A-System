# Setup Instructions

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd RAG
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .          # installs the rag_qa package + dependencies
```

### 2. Download NLTK Data
```bash
python -c "import nltk; [nltk.download(r) for r in ('punkt','punkt_tab','stopwords')]"
```

### 3. Process Documents
```bash
python -m rag_qa.document_processor
```

### 4. Build Embeddings
```bash
python -m rag_qa.embedding_system
```

### 5. Start API
```bash
python -m rag_qa.api
```

### 6. Test System
```bash
# Health check
curl http://localhost:9000/health

# Ask a question
curl -X POST http://localhost:9000/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "What are machine safety requirements?", "k": 5, "mode": "hybrid"}'
```

### 7. Run Evaluation
```bash
python -m rag_qa.evaluation.heuristic
```

## Port Conflicts
If port 9000 is in use (macOS AirPlay):
```bash
FLASK_PORT=8080 python -m rag_qa.api
```

## Troubleshooting

### Missing PDF Files
The system expects PDFs in `industrial-safety-pdfs/` (not committed). Either:
1. Download the listed sources: `python scripts/download_pdfs.py`, or
2. Add your own PDFs to this directory, update `sources.json` (include a
   `filename` field for exact citation matching), then rerun processing.

### Database Issues
If database is corrupted:
```bash
rm data/rag_database.db data/faiss_index.bin data/chunk_id_map.json data/bm25_corpus.json
python -m rag_qa.document_processor
python -m rag_qa.embedding_system
```

### Memory Issues
For large document sets, reduce batch size in embedding_system.py:
```python
batch_size=16  # Default is 32
```
