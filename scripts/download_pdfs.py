#!/usr/bin/env python3
"""
Download the source PDFs listed in sources.json into industrial-safety-pdfs/.

For each entry it derives a stable local filename (the URL basename when it
ends in .pdf, otherwise a slug of the title), downloads the file, and writes
the resolved ``filename`` back into sources.json. That ``filename`` field is
what DocumentProcessor uses for exact, unambiguous source attribution.

Usage:
    python scripts/download_pdfs.py            # download missing files
    python scripts/download_pdfs.py --force    # re-download everything
"""

import argparse
import json
import os
import re
import sys
from urllib.parse import urlparse, unquote

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES_FILE = os.path.join(PROJECT_ROOT, "sources.json")
PDF_DIR = os.path.join(PROJECT_ROOT, "industrial-safety-pdfs")

USER_AGENT = "Mozilla/5.0 (compatible; RAG-QA-System/1.0; +https://github.com/)"


def _slugify(title: str) -> str:
    """Turn a title into a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:80] or "source"


def derive_filename(source: dict, used: set) -> str:
    """Choose a stable local filename for a source, avoiding collisions."""
    if source.get("filename"):
        return source["filename"]

    basename = os.path.basename(urlparse(source["url"]).path)
    basename = unquote(basename)
    if basename.lower().endswith(".pdf"):
        candidate = re.sub(r"[^\w.-]", "_", basename)
    else:
        candidate = f"{_slugify(source['title'])}.pdf"

    # De-duplicate if two sources map to the same name.
    name = candidate
    i = 2
    while name.lower() in used:
        stem, ext = os.path.splitext(candidate)
        name = f"{stem}-{i}{ext}"
        i += 1
    used.add(name.lower())
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description="Download source PDFs.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist"
    )
    args = parser.parse_args()

    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        sources = json.load(f)

    os.makedirs(PDF_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    used_names = {s["filename"].lower() for s in sources if s.get("filename")}
    ok, failed = 0, []

    for source in sources:
        filename = derive_filename(source, used_names)
        source["filename"] = filename  # persist for exact source matching
        dest = os.path.join(PDF_DIR, filename)

        if os.path.exists(dest) and not args.force:
            print(f"✓ exists   {filename}")
            ok += 1
            continue

        try:
            print(f"↓ download {filename}  <- {source['url']}")
            resp = session.get(source["url"], timeout=60, allow_redirects=True)
            resp.raise_for_status()
            with open(dest, "wb") as out:
                out.write(resp.content)
            ok += 1
        except Exception as e:
            print(f"✗ FAILED   {filename}: {e}", file=sys.stderr)
            failed.append((filename, str(e)))

    # Write resolved filenames back so sources.json stays the source of truth.
    with open(SOURCES_FILE, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDone: {ok}/{len(sources)} available, {len(failed)} failed.")
    if failed:
        print("Failed downloads (fetch manually into industrial-safety-pdfs/):")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
