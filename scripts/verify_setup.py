#!/usr/bin/env python3
"""
Verify that the RAG Q&A System is set up correctly.

Checks Python version, required packages, NLTK data, and whether the data
artifacts (SQLite DB, FAISS index, chunk map) have been built. Exits non-zero
if anything required is missing so it can be used in CI or a setup script.

Usage:
    python verify_setup.py
"""

import importlib
import os
import sys

# scripts/verify_setup.py -> repo root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_PACKAGES = [
    "numpy", "faiss", "sentence_transformers", "transformers",
    "pypdf", "nltk", "flask", "flask_cors", "rank_bm25", "requests",
]

DATA_FILES = [
    os.path.join("data", "rag_database.db"),
    os.path.join("data", "faiss_index.bin"),
    os.path.join("data", "chunk_id_map.json"),
]


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✅" if ok else "❌"
    print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("🔎 Verifying RAG Q&A System setup\n")
    all_ok = True

    print("Python:")
    py_ok = sys.version_info >= (3, 9)
    all_ok &= check(f"Python {sys.version_info.major}.{sys.version_info.minor}",
                    py_ok, "3.9+ required")

    print("\nPackages:")
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            check(pkg, True)
        except Exception as e:
            all_ok = False
            check(pkg, False, str(e))

    print("\nNLTK data:")
    try:
        import nltk
        for resource, path in (("punkt", "tokenizers/punkt"),
                               ("stopwords", "corpora/stopwords")):
            try:
                nltk.data.find(path)
                check(resource, True)
            except (LookupError, OSError):
                all_ok = False
                check(resource, False, f"run: python -c \"import nltk; nltk.download('{resource}')\"")
    except Exception as e:
        all_ok = False
        check("nltk", False, str(e))

    print("\nData artifacts (optional until you build the corpus):")
    data_present = True
    for rel in DATA_FILES:
        exists = os.path.exists(os.path.join(PROJECT_ROOT, rel))
        data_present &= exists
        check(rel, exists)
    if not data_present:
        print("     ℹ️  Build them with: bash scripts/rebuild_data.sh")
        print("     ℹ️  (needs PDFs — fetch with: python scripts/download_pdfs.py)")

    print()
    if all_ok:
        print("✅ Environment looks good.")
        return 0
    print("❌ Setup incomplete — see the ❌ items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
