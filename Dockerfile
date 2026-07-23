FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; faiss-cpu ships manylinux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the NLTK data the app needs at runtime.
RUN python -c "import nltk; [nltk.download(r) for r in ('punkt','punkt_tab','stopwords')]"

COPY . .

# Install the package itself so `rag_qa` is importable (deps already present).
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 9000

# Production WSGI server (never the Flask dev server).
CMD ["gunicorn", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:9000", "rag_qa.api:app"]
