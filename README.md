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

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify setup**:
   ```bash
   python verify_setup.py
   ```

## 📁 Project Structure

```
RAG/
├── src/                          # Source code
├── tests/                        # Test files
├── data/                         # Processed data and database
├── logs/                         # Application logs
├── industrial-safety-pdfs/      # Source PDF files (20 documents)
├── sources.json                  # Citation metadata
├── requirements.txt              # Python dependencies
├── verify_setup.py              # Setup verification script
└── README.md                    # This file
```

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
- `sentence-transformers>=2.3.1` - Embedding generation
- `faiss-cpu>=1.7.4` - Vector similarity search
- `transformers>=4.30.2` - HuggingFace ecosystem
- `scikit-learn==1.3.0` - ML utilities
- `rank-bm25==0.2.2` - Keyword scoring

### Document Processing
- `PyPDF2==3.0.1` - PDF text extraction
- `pypdf==3.14.0` - Alternative PDF processing
- `nltk==3.8.1` - Text processing

### Web Framework
- `flask==2.3.2` - API framework
- `flask-cors==4.0.0` - CORS support

### Database & Utilities
- `sqlite3` (built-in) - Local database
- `sqlalchemy==2.0.19` - Database ORM
- `numpy`, `pandas` - Data manipulation

## 📊 Evaluation Results

Our comprehensive evaluation with 8 test questions across different categories shows:

### **Key Performance Metrics**
- **Answer Rate**: 100% (both baseline and hybrid)
- **Confidence Improvement**: +5.9% average (0.745 → 0.804)
- **Reranking Activity**: 87.5% of queries had ranking changes
- **Response Time**: Hybrid is ~9x faster (0.196s → 0.022s avg)

### **Per-Category Improvements**
- **Machine Guarding**: +6.9% confidence improvement
- **Safety Ratings**: +5.1% confidence improvement  
- **LOTO Procedures**: +7.9% confidence improvement
- **Risk Assessment**: +1.9% confidence improvement

### **Key Findings**
✅ **Hybrid reranker consistently improves confidence scores**  
✅ **High reranking activity shows BM25 adds value**  
✅ **All test questions answered successfully**  
✅ **Faster response times with hybrid approach**

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
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Option A: Use rebuild script (recommended)
./scripts/rebuild_data.sh

# Option B: Manual setup
python src/document_processor.py
python src/embedding_system.py
```

### 3. Start the API Server
```bash
python src/api.py
# If port 5000 is in use: FLASK_PORT=8080 python src/api.py
```

### 4. Test with Working cURL Examples

**Easy Question Example (General Safety):**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "What are machine safety requirements?", "k": 5, "mode": "hybrid"}'
```

**Tricky Question Example (Technical Calculation):**
```bash
curl -X POST http://localhost:5000/ask \
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
python evaluation.py
```

## 🎓 What We Learned

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

**Status**: ✅ Complete | Production Ready 🚀  
**Final Statistics**: 3,084 chunks | 20 documents | 8/8 test questions answered | +5.9% confidence improvement
