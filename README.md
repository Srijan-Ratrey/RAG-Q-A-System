# RAG Q&A System for Industrial Safety Documents

A small question-answering service over industrial & machine safety PDFs with baseline similarity search and reranker enhancement.

## 🎯 Project Overview

This project implements a retrieval-augmented generation (RAG) system specifically designed for industrial safety documentation. It features:

- **Baseline**: Cosine similarity search using sentence embeddings
- **Enhancement**: Hybrid reranker combining vector similarity with keyword matching (BM25)
- **Data**: 20 industrial safety PDFs with proper citations
- **API**: Simple REST endpoint for question answering

## 📋 Requirements Analysis

### Core Functionality
- ✅ Document ingestion and chunking (paragraph-sized pieces)
- ✅ Embedding generation using `all-MiniLM-L6-v2`
- ✅ Vector storage with FAISS
- ✅ Baseline cosine similarity search
- ✅ Hybrid reranker (BM25 + vector scores)
- ✅ Extractive answers with citations
- ✅ Abstention mechanism for low-confidence queries
- ✅ Single API endpoint: `POST /ask`

### Technical Constraints
- ✅ **No paid APIs** - Using free local models only
- ✅ **CPU only** - No GPU dependencies
- ✅ **SQLite storage** - Lightweight local database
- ✅ **Reproducible** - Seeded random operations
- ✅ **Extractive answers** - Grounded in source text

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment recommended

### Installation

1. **Clone and navigate to project**:
   ```bash
   cd /path/to/RAG
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install the package** (editable, pulls in dependencies):
   ```bash
   pip install -e .
   # or: pip install -r requirements.txt
   ```

4. **Verify setup**:
   ```bash
   python scripts/verify_setup.py
   ```

## 📁 Project Structure

```
RAG-Q-A-System/
├── rag_qa/                       # Importable package
│   ├── document_processor.py     # PDF extraction, cleaning, chunking
│   ├── embedding_system.py       # Embeddings + FAISS index (+ fingerprint)
│   ├── search_system.py          # baseline / hybrid / rrf / cross_encoder
│   ├── api.py                    # Flask API + extractive answering
│   └── evaluation/
│       ├── metrics.py            # Recall@k / MRR / nDCG (pure functions)
│       ├── ir.py                 # IR evaluation driver (labeled qrels)
│       └── heuristic.py          # Baseline↔hybrid heuristic comparison
├── tests/                        # pytest suite (unit + one integration test)
├── scripts/
│   ├── rebuild_data.sh           # Rebuild DB + indices from PDFs
│   ├── download_pdfs.py          # Fetch source PDFs; records filenames
│   └── verify_setup.py           # Setup verification script
├── examples/
│   ├── example_usage.py          # Sample API client
│   └── qrels.example.json        # Relevance labels template for IR eval
├── data/                         # Built artifacts (gitignored): DB, index, maps
├── industrial-safety-pdfs/       # Source PDFs (not committed; see download_pdfs.py)
├── sources.json                  # Citation metadata (title, url, filename)
├── pyproject.toml                # Package metadata, console scripts, pytest config
├── requirements.txt
├── Dockerfile                    # Production image (gunicorn)
└── README.md
```

Installed as a package (`pip install -e .`), so modules import as
`rag_qa.search_system` etc. — no `sys.path` hacks. Console scripts:
`rag-api`, `rag-eval`, `rag-ir-eval`.

## 🔧 Implementation Status

### Phase 1: Document Processing ✅ 
- ✅ PDF text extraction with chunking
- ✅ SQLite database schema design
- ✅ Chunk storage with metadata
- **Result**: 3,084 chunks from 20 PDFs

### Phase 2: Embedding & Search ✅
- ✅ Embedding generation pipeline
- ✅ FAISS index creation and management
- ✅ Baseline similarity search implementation
- **Model**: all-MiniLM-L6-v2 (384 dimensions)

### Phase 3: Reranker ✅
- ✅ BM25 keyword scoring
- ✅ Hybrid score combination
- ✅ Confidence thresholding
- **Method**: Hybrid (70% vector + 30% BM25)

### Phase 4: API & Evaluation ✅
- ✅ Flask API endpoint
- ✅ Answer generation with citations
- ✅ Test questions and evaluation
- ✅ Performance comparison

## 🎯 API Specification

### Endpoint: `POST /ask`

**Request**:
```json
{
  "q": "What are the safety requirements for industrial machinery?",
  "k": 5,
  "mode": "hybrid"
}
```

**Response**:
```json
{
  "answer": "Industrial machinery must meet safety requirements including...",
  "contexts": [
    {
      "text": "Relevant chunk text...",
      "score": 0.85,
      "source": "OSHA 3170 — Safeguarding Equipment...",
      "url": "https://www.osha.gov/sites/default/files/publications/osha3170.pdf",
      "chunk_id": "chunk_123"
    }
  ],
  "reranker_used": "hybrid",
  "confidence": 0.82
}
```

## 📊 Data Sources

Working with 20 industrial safety PDFs covering:
- EU Machinery Regulation 2023/1230
- OSHA safety guidelines
- ISO 13849-1 functional safety standards
- Machine guarding best practices
- Risk assessment methodologies

All sources are properly attributed in `sources.json` with titles and URLs.

## ⚙️ Configuration

Key configuration options:
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Storage**: FAISS with inner product similarity
- **Reranker**: Hybrid BM25 + vector scores
- **Chunk Size**: ~paragraph-sized (target: 200-300 words)
- **Overlap**: 50 tokens between chunks
- **Top-K**: 5 results by default

## 🧪 Testing & Evaluation

The system will be evaluated on 8 test questions covering:
- Basic safety concepts
- Specific regulatory requirements
- Technical implementation details
- Complex multi-step procedures

Metrics:
- Answer relevance (human evaluation)
- Citation accuracy
- Response time
- Before/after reranker performance

## 📚 Dependencies

### Core ML Libraries
- `sentence-transformers>=2.3.1` - Embeddings + optional cross-encoder reranker
- `faiss-cpu>=1.7.4` - Vector similarity search
- `transformers>=4.30.2` - HuggingFace ecosystem (pulled in transitively)
- `rank-bm25>=0.2.2` - BM25 keyword scoring

### Document Processing
- `pypdf>=3.14.0` - PDF text extraction (replaces the deprecated PyPDF2)
- `nltk>=3.8.1` - Sentence/word tokenization (needs `punkt`, `punkt_tab`, `stopwords`)

### Web Framework
- `flask>=3.0.0` / `flask-cors>=4.0.0` - API + CORS
- `gunicorn>=21.2.0` - Production WSGI server

### Database & Utilities
- `sqlite3` (built-in) - Local database
- `numpy` - Vector math
- `pytest` - Tests

Pins are loosened to ranges so the project installs on modern Python
(`numpy>=1.26` is required for Python 3.12+).

## 📊 Evaluation

Two evaluation entry points:

- **`rag-ir-eval` (recommended)** — proper retrieval metrics
  (**Recall@k, MRR, nDCG@k**) against labeled query→relevant-file judgements,
  with **warmup-corrected latency** per mode. This is the honest way to compare
  baseline vs hybrid vs RRF vs cross-encoder.
  ```bash
  cp examples/qrels.example.json qrels.json          # then fill in labels
  python -m rag_qa.evaluation.ir --k 5               # or: rag-ir-eval --k 5
  #                                                  # add --cross-encoder to include it
  ```
- **`rag-eval` (`rag_qa.evaluation.heuristic`)** — the original heuristic comparison (answer rate,
  confidence deltas, ranking churn). Useful as a smoke test, but its
  "confidence improvement" is **not** a retrieval-quality metric: hybrid mode's
  score mixes a per-query-normalized BM25 term, so absolute confidence values
  aren't comparable across queries. Prefer `rag-ir-eval` for real numbers.

> ⚠️ Earlier revisions of this README quoted specific gains (e.g. "+5.9%
> confidence", "~9x faster"). Those came from the heuristic script and, for
> latency, did not exclude first-call warmup — hybrid does strictly more work
> than baseline, so it is at best comparable, not faster. Re-run
> `rag-ir-eval` on your own corpus and labels for trustworthy figures.

## 🎓 Learning Objectives

This project demonstrates:
- **RAG Architecture**: End-to-end retrieval-augmented generation
- **Vector Search**: Semantic similarity with embeddings
- **Hybrid Ranking**: Combining multiple relevance signals
- **Production Patterns**: API design, error handling, configuration
- **Evaluation**: Systematic before/after performance analysis

## 🚀 Quick Start Examples

### 1. Clone and Setup
```bash
git clone https://github.com/Srijan-Ratrey/RAG-Q-A-System.git
cd RAG-Q-A-System
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build Data (First Time Setup)
```bash
# Download NLTK data (punkt_tab is required on NLTK >= 3.8.2)
python -c "import nltk; [nltk.download(r) for r in ('punkt','punkt_tab','stopwords')]"

# Fetch the source PDFs (writes into industrial-safety-pdfs/ and records each
# file's name back into sources.json for exact citation matching)
python scripts/download_pdfs.py

# Option A: Use rebuild script (recommended)
./scripts/rebuild_data.sh

# Option B: Manual setup
python -m rag_qa.document_processor
python -m rag_qa.embedding_system
```

> The PDFs and built artifacts (`data/*.db`, `*.bin`, `*.json`) are gitignored,
> so this step is required before the API can answer questions.

### 3. Start the API Server
```bash
python -m rag_qa.api          # or: rag-api
# If port 9000 is in use: FLASK_PORT=8080 python -m rag_qa.api
```

### 4. Test with Working cURL Examples

**Easy Question Example (General Safety):**
```bash
curl -X POST http://localhost:9000/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "What are machine safety requirements?", "k": 5, "mode": "hybrid"}'
```

**Tricky Question Example (Technical Calculation):**
```bash
curl -X POST http://localhost:9000/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "How do you calculate Performance Level PLr for safety functions?", "k": 3, "mode": "baseline"}'
```

Both examples return JSON responses with:
- `answer`: Extracted text or null if abstaining
- `contexts`: Relevant document chunks with scores
- `confidence`: Answer confidence score
- `reranker_used`: Search mode employed

### 5. Run Full Evaluation
```bash
python -m rag_qa.evaluation.ir --k 5     # rank metrics (needs qrels.json)
python -m rag_qa.evaluation.heuristic    # heuristic smoke test
```

### 6. Run the Tests
```bash
pytest                                              # full suite
pytest --ignore=tests/test_embedding_integration.py # skip the model-download test
```

### 7. Run with Docker (production-style)
```bash
docker build -t rag-qa .
# Mount a prebuilt data/ directory so the container has the DB + index:
docker run --rm -p 9000:9000 -v "$PWD/data:/app/data" rag-qa
```

## 🔎 Search Modes

`mode` accepts:

| mode | what it does |
|------|--------------|
| `baseline` | Pure vector cosine similarity |
| `hybrid` | 70% vector + 30% BM25 (default) |
| `rrf` | Reciprocal Rank Fusion of vector + BM25 ranks (no score normalization) |
| `cross_encoder` | Reranks candidates with a cross-encoder (best quality; loads an extra model, falls back to hybrid if unavailable) |

`confidence` is the top result's cosine similarity for all cosine-based modes
(comparable across queries); for `cross_encoder` it is the sigmoid of the
cross-encoder score.

## 🎓 What I Learned

### **Technical Insights**
1. **Hybrid reranking works**: BM25 + vector scoring improved confidence by 5.9%
2. **Keyword matching adds value**: 87.5% reranking rate shows lexical relevance helps
3. **Local models are viable**: all-MiniLM-L6-v2 performs well on technical documents
4. **Chunking strategy matters**: ~191 words per chunk with overlap worked well

### **System Design Lessons**
1. **Threshold tuning is critical**: 0.5 confidence threshold balanced coverage vs quality
2. **Citation tracking works**: Every answer properly attributed to source documents
3. **Abstention prevents hallucination**: System refuses low-confidence answers
4. **Performance scales well**: Sub-second response times with 3k+ document chunks

### **Domain-Specific Findings**
1. **Technical queries benefit most**: Complex safety calculations saw biggest improvements
2. **Standard references work well**: EN ISO, OSHA standards retrieved accurately  
3. **Procedural knowledge gaps**: LOTO procedures had lower baseline confidence
4. **Regulatory content is rich**: EU Machinery Directive provided high-quality matches

---

**Status**: ✅ Working — pipeline, hybrid/RRF/cross-encoder retrieval, API, tests, and IR evaluation all in place.
**Corpus**: 20 source documents (chunk count depends on your extraction run).
**Metrics**: run `rag-ir-eval` against your own `qrels.json` for trustworthy Recall@k / MRR / nDCG figures.
