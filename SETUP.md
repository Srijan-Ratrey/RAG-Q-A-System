# Setup Instructions

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd RAG
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Process Documents
```bash
python src/document_processor.py
```

### 4. Build Embeddings
```bash
python src/embedding_system.py
```

### 5. Start API
```bash
python src/api.py
```

### 6. Test System
```bash
# Health check
curl http://localhost:5000/health

# Ask a question
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "What are machine safety requirements?", "k": 5, "mode": "hybrid"}'
```

### 7. Run Evaluation
```bash
python evaluation.py
```

## Port Conflicts
If port 5000 is in use (macOS AirPlay):
```bash
FLASK_PORT=8080 python src/api.py
```

## Troubleshooting

### Missing PDF Files
The system expects PDFs in `industrial-safety-pdfs/`. If not available:
1. Add your own PDF documents to this directory
2. Update `sources.json` with metadata
3. Rerun document processing

### Database Issues
If database is corrupted:
```bash
rm data/rag_database.db data/faiss_index.bin data/chunk_id_map.pkl
python src/document_processor.py
python src/embedding_system.py
```

### Memory Issues
For large document sets, reduce batch size in embedding_system.py:
```python
batch_size=16  # Default is 32
```
