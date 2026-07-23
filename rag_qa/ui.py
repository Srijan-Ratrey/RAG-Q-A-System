"""Gradio UI for the RAG Q&A System — bring your own PDFs.

Upload (or drag-and-drop) any PDFs, build a fresh index from just those files,
and ask questions about them. Each build is isolated in its own temp directory
and lives only for the session; nothing is downloaded or persisted.

Run:
    python -m rag_qa.ui        # or: rag-ui
    #  http://127.0.0.1:7860
"""

import os

# Load the (cached) embedding model from disk instead of pinging HuggingFace at
# startup — a flaky network otherwise stalls model load ~90s on HEAD timeouts.
# Override by exporting HF_HUB_OFFLINE=0 before launch to allow update checks.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import shutil
import tempfile

import gradio as gr
from sentence_transformers import SentenceTransformer


def _patch_gradio_schema_bug():
    """Work around a Gradio 4.44.1 crash in API-info generation.

    `gradio_client.utils` chokes ("argument of type 'bool' is not iterable")
    when a component's JSON schema contains a boolean node (e.g. from gr.State /
    gr.Files). Gradio 4.44.1 is the last 4.x and Gradio 5 needs Python >=3.10,
    so we defensively tolerate boolean schema nodes.
    """
    try:
        import gradio_client.utils as gcu
    except Exception:  # pragma: no cover
        return
    _orig = gcu._json_schema_to_python_type

    def _safe(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig(schema, defs)

    gcu._json_schema_to_python_type = _safe
    _orig_get_type = gcu.get_type
    gcu.get_type = lambda schema: (
        "Any" if isinstance(schema, bool) else _orig_get_type(schema)
    )


_patch_gradio_schema_bug()

from rag_qa.api import QAService
from rag_qa.document_processor import DocumentProcessor
from rag_qa.embedding_system import EmbeddingSystem
from rag_qa.search_system import SearchSystem

MODEL_NAME = "all-MiniLM-L6-v2"
MODES = ["hybrid", "baseline", "rrf", "cross_encoder"]

# The embedding model is loaded once and shared across every per-session build.
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _render_contexts(contexts) -> str:
    if not contexts:
        return "_No source passages retrieved._"
    lines = ["### Sources"]
    for i, c in enumerate(contexts, 1):
        source = c.get("source") or c.get("file") or "unknown"
        page = c.get("page")
        page_str = f", p.{page}" if page else ""
        header = f"**{i}. {source}**{page_str}"
        scores = (
            f"score {c.get('score', 0):.3f}  |  "
            f"vector {c.get('vector_score', 0):.3f}  |  "
            f"bm25 {c.get('bm25_score', 0):.3f}"
        )
        text = (c.get("text") or "").strip()
        if len(text) > 600:
            text = text[:600].rstrip() + "…"
        lines.append(f"{header}\n\n_{scores}_\n\n> {text}\n")
    return "\n".join(lines)


def build_index(files, state, progress=gr.Progress()):
    """Ingest uploaded PDFs into a fresh per-session corpus.

    Returns (status_markdown, new_state). new_state is None on failure.
    """
    # Clean up any previous session's temp corpus first.
    _cleanup_state(state)

    if not files:
        return "⚠️ Please add one or more PDF files first.", None

    paths = [getattr(f, "name", f) for f in files]
    pdfs = [p for p in paths if str(p).lower().endswith(".pdf")]
    if not pdfs:
        return "⚠️ Only PDF files are supported.", None

    tmp = tempfile.mkdtemp(prefix="rag_ui_")
    db_path = os.path.join(tmp, "rag.db")
    index_path = os.path.join(tmp, "faiss.bin")
    map_path = os.path.join(tmp, "chunk_id_map.json")
    bm25_path = os.path.join(tmp, "bm25.json")

    try:
        # Empty sources metadata -> titles fall back to the uploaded filename.
        sources_path = os.path.join(tmp, "sources.json")
        with open(sources_path, "w", encoding="utf-8") as f:
            f.write("[]")
        proc = DocumentProcessor(
            sources_file=sources_path,
            pdf_dir=tmp,
            db_path=db_path,
        )

        total_chunks, per_file = 0, []
        for i, pdf in enumerate(pdfs):
            name = os.path.basename(pdf)
            progress((i) / len(pdfs), desc=f"Processing {name}")
            chunks = proc.process_document(pdf)
            proc.save_chunks(chunks)
            total_chunks += len(chunks)
            per_file.append((name, len(chunks)))

        if total_chunks == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            return (
                "⚠️ No extractable text found. Scanned/image-only PDFs "
                "aren't supported (no OCR)."
            ), None

        progress(0.8, desc="Building search index")
        es = EmbeddingSystem(
            model_name=MODEL_NAME,
            db_path=db_path,
            index_path=index_path,
            chunk_id_map_path=map_path,
            model=get_model(),
        )
        es.build_index(force_rebuild=True)

        search_system = SearchSystem(
            db_path=db_path,
            embedding_system=es,
            bm25_cache_path=bm25_path,
        )
        service = QAService(search_system=search_system)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return f"❌ Failed to build index: `{e}`", None

    file_list = "\n".join(f"- {n} ({c} chunks)" for n, c in per_file)
    status = (
        f"✅ Indexed **{total_chunks} chunks** from **{len(pdfs)} file(s)**. "
        f"Ask away below.\n\n{file_list}"
    )
    return status, {"service": service, "tmp": tmp}


def _cleanup_state(state):
    if state and state.get("tmp"):
        shutil.rmtree(state["tmp"], ignore_errors=True)


def answer_query(question, mode, k, state):
    """Query the current session's corpus. Returns (answer, meta, sources)."""
    if not question or not question.strip():
        return "Please enter a question.", "", ""
    if not state or not state.get("service"):
        return (
            (
                "### ⏳ No index yet\nUpload PDF(s) and click **Build index** "
                "first, then ask your question."
            ),
            "",
            "",
        )

    try:
        resp = state["service"].ask(question.strip(), k=int(k), mode=mode)
    except Exception as e:
        return f"### Error\n`{e}`", "", ""

    confidence = resp.get("confidence", 0.0)
    if resp.get("answer"):
        answer_md = f"### Answer\n{resp['answer']}"
    else:
        answer_md = (
            f"### No answer — the system abstained\n"
            f"_{resp.get('abstain_reason', 'Low confidence')}_"
        )
    meta_md = (
        f"**Confidence:** {confidence:.3f}  ·  "
        f"**Mode:** {resp.get('reranker_used', mode)}  ·  "
        f"**Passages:** {resp.get('total_results', 0)}"
    )
    return answer_md, meta_md, _render_contexts(resp.get("contexts", []))


def build_demo() -> "gr.Blocks":
    with gr.Blocks(title="RAG Q&A — Chat with your PDFs") as demo:
        gr.Markdown(
            "# 🔍 Chat with your PDFs\n"
            "Upload one or more PDFs, build an index, then ask questions about "
            "**your** documents. Answers are **extractive** (sentences pulled "
            "from the sources, ranked by relevance) with citations — no text is "
            "generated by an LLM."
        )

        state = gr.State()

        with gr.Row():
            files = gr.Files(
                label="Drop PDFs here or click to browse",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath",
            )
        with gr.Row():
            build_btn = gr.Button("Build index", variant="primary")
        status = gr.Markdown()

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question",
                    lines=2,
                    placeholder="Ask something about the PDFs you uploaded…",
                )
            with gr.Column(scale=1):
                mode = gr.Radio(MODES, value="hybrid", label="Retrieval mode")
                k = gr.Slider(1, 20, value=5, step=1, label="Passages (k)")
        ask_btn = gr.Button("Ask", variant="primary")

        answer_out = gr.Markdown()
        meta_out = gr.Markdown()
        sources_out = gr.Markdown()

        build_btn.click(build_index, [files, state], [status, state])
        ask_btn.click(
            answer_query,
            [question, mode, k, state],
            [answer_out, meta_out, sources_out],
        )
        question.submit(
            answer_query,
            [question, mode, k, state],
            [answer_out, meta_out, sources_out],
        )
    return demo


def main():
    host = os.environ.get("GRADIO_HOST", "127.0.0.1")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    build_demo().launch(server_name=host, server_port=port, show_api=False)


if __name__ == "__main__":
    main()
