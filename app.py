"""
Streamlit user interface for AskMyPDF.

Run with:   streamlit run app.py

This file only handles the UI. All NLP logic lives in the `app/` package:
    app/pdf_extractor.py -> app/chunker.py -> app/qa_engine.py

(Note: `import app...` below refers to the app/ *folder* (a package), not to
this file. Python gives a package folder priority over a same-named .py file.)
"""

import csv
import hashlib
import html
import io
import json
from pathlib import Path

import streamlit as st

from app.chunker import TextTooShortError, chunk_text, suggest_chunk_size
from app.config import (
    AVAILABLE_MODELS,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNK_SIZE_WORDS_NON_LATIN,
    ENGLISH_MODEL,
    LOW_CONFIDENCE_THRESHOLD,
    OCR_MAX_PAGES,
    SAMPLES_DIR,
    UPLOADS_DIR,
)
from app.hf_inference_api import InferenceAPIError
from app.hub_info import get_model_info
from app.ocr import OCR_CHOICES, OCRError, gpu_available, ocr_available
from app.pdf_extractor import EmptyPDFError, PDFReadError, extract_text, extract_with_ocr
from app.qa_engine import QAEngine

st.set_page_config(page_title="AskMyPDF", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Cached helpers: Streamlit re-runs this whole script on every click, so
# anything slow (loading a model, parsing a PDF, calling the Hub API) is cached.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_engine(model_name: str, backend: str) -> QAEngine:
    """Load the QA model once and keep it in memory across re-runs."""
    return QAEngine(model_name, backend=backend)


@st.cache_data(show_spinner=False)
def cached_extract(pdf_bytes: bytes):
    """Extract text once per distinct PDF (cache key = the file's bytes)."""
    return extract_text(pdf_bytes)


def ocr_once(pdf_bytes: bytes, languages, first_page: int, max_pages: int):
    """
    Run OCR and remember the result for this session.

    Two levels of reuse, because OCR costs ~20 s per page while Streamlit
    re-runs the whole script on every click:

    * the finished document is kept for the exact (file, languages, range);
    * individual pages are kept in a per-page cache, so widening the range
      only OCRs the pages that are genuinely new.

    (st.cache_data is deliberately NOT used here: this function draws a live
    progress bar, and Streamlit cannot replay elements drawn inside a cached
    function -- that raises CacheReplayClosureError.)
    """
    digest = hashlib.sha1(pdf_bytes).hexdigest()
    key = (digest, tuple(languages), int(first_page), int(max_pages))
    if st.session_state.get("ocr_key") != key:
        # One page cache per (file, languages); pages survive range changes.
        caches = st.session_state.setdefault("ocr_page_cache", {})
        page_cache = caches.setdefault((digest, tuple(languages)), {})
        bar = st.progress(0.0, text="Running OCR...")
        try:
            st.session_state["ocr_doc"] = extract_with_ocr(
                pdf_bytes, languages, int(first_page), int(max_pages),
                progress=lambda done, total: bar.progress(
                    done / total, text=f"OCR page {done}/{total} (~20 s each)"),
                cache=page_cache,
            )
            st.session_state["ocr_key"] = key
        finally:
            bar.empty()
    return st.session_state["ocr_doc"]


@st.cache_data(ttl=3600, show_spinner=False)
def cached_model_info(model_id: str):
    return get_model_info(model_id)


def save_upload(name: str, data: bytes) -> Path:
    """Keep a copy of each uploaded PDF in /uploads (name + short hash avoids clashes)."""
    UPLOADS_DIR.mkdir(exist_ok=True)
    digest = hashlib.sha1(data).hexdigest()[:8]
    path = UPLOADS_DIR / f"{Path(name).stem}_{digest}.pdf"
    if not path.exists():
        path.write_bytes(data)
    return path


def is_non_latin(text: str) -> bool:
    """True if the document is written in a non-Latin script (Telugu, Hindi...)."""
    return suggest_chunk_size(text)[0] == CHUNK_SIZE_WORDS_NON_LATIN


@st.cache_data(show_spinner=False)
def example_questions(pdf_file_name: str):
    """Example questions for a bundled sample PDF, from samples/expected_answers.json."""
    path = SAMPLES_DIR / "expected_answers.json"
    if not path.exists():
        return []
    try:
        examples = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [e["question"] for e in examples if e.get("pdf") == pdf_file_name]


def highlight(context: str, start: int, end: int) -> str:
    """Return HTML for the context with the answer span highlighted."""
    return (html.escape(context[:start])
            + "<mark style='background:#ffe066;padding:0 2px;border-radius:3px'><b>"
            + html.escape(context[start:end]) + "</b></mark>"
            + html.escape(context[end:]))


# ---------------------------------------------------------------------------
# Sidebar: model settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    model_label = st.selectbox(
        "Model",
        list(AVAILABLE_MODELS),
        help="Use the multilingual model for non-English PDFs. It is ~4x larger "
             "(slower, ~2.2 GB download) but understands ~100 languages.",
    )
    model_name = AVAILABLE_MODELS[model_label]

    backend_label = st.radio(
        "Run the model",
        ["Locally (transformers)", "Hugging Face Inference API"],
        help="The Inference API runs the model on Hugging Face's servers. "
             "It needs the HF_TOKEN environment variable to be set.",
    )
    backend = "local" if backend_label.startswith("Locally") else "api"

    with st.expander("Chunking settings"):
        auto_chunk = st.checkbox(
            "Auto chunk size (by script)", value=True,
            help="Telugu, Hindi and other non-Latin scripts produce roughly "
                 "2.7 sub-word tokens per word against 1.2 for English, so they "
                 "need smaller chunks to fit the model's 512-token window.")
        chunk_size = st.slider("Chunk size (words)", 50, 500, CHUNK_SIZE_WORDS,
                               step=25, disabled=auto_chunk)
        overlap = st.slider("Overlap (words)", 0, 150, CHUNK_OVERLAP_WORDS,
                            step=10, disabled=auto_chunk)
        if not auto_chunk and overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")

    with st.expander("OCR (scanned PDFs)"):
        if ocr_available():
            ocr_lang = st.selectbox(
                "OCR language", list(OCR_CHOICES),
                help="EasyOCR reads one Indic script at a time, plus English.")
            ocr_first_page = st.number_input(
                "Start at page", 1, 5000, 1,
                help="Skip covers and contents pages -- OCR costs ~20 s per page, "
                     "so reading only the pages you need is the best speed-up.")
            ocr_max_pages = st.number_input(
                "How many pages", 1, 100, OCR_MAX_PAGES,
                help="OCR takes roughly 20 seconds per page on a CPU. Pages "
                     "already read are reused when you change this range.")
        else:
            ocr_lang, ocr_first_page, ocr_max_pages = None, 1, 0
            st.info("OCR is off. Install it with:  pip install easyocr")

    # Which device the neural networks run on -- a GPU makes OCR several
    # times faster and speeds up answering too.
    st.caption(f"⚡ Running on **{'GPU (CUDA)' if gpu_available() else 'CPU'}**")

    # Model metadata straight from the Hugging Face Hub API.
    st.subheader("🤗 Model info (Hub API)")
    info = cached_model_info(model_name)
    if info.exists:
        # The Hub may leave any of these fields out, so never assume a number.
        fmt = lambda n: f"{n:,}" if isinstance(n, int) else "-"   # noqa: E731
        st.markdown(
            f"[`{info.model_id}`](https://huggingface.co/{info.model_id})  \n"
            f"**Task:** {info.task or '-'}  \n"
            f"**Downloads (30 days):** {fmt(info.downloads)}  \n"
            f"**Likes:** {fmt(info.likes)}  \n"
            f"**Language:** {', '.join(info.languages or ['-'])}  \n"
            f"**License:** {info.license}"
        )
    else:
        st.warning(f"{model_name}: {info.error}")

# ---------------------------------------------------------------------------
# Main page: 1) choose a PDF
# ---------------------------------------------------------------------------
st.title("📄 AskMyPDF")
st.caption("Upload a PDF, ask a question, and an extractive QA model finds the "
           "answer span in the document.")

col_upload, col_sample = st.columns([3, 2])
with col_upload:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
with col_sample:
    sample_files = sorted(p.name for p in SAMPLES_DIR.glob("*.pdf"))
    sample = st.selectbox("...or try a sample PDF", ["(none)"] + sample_files,
                          disabled=uploaded is not None)

if uploaded is not None:
    pdf_name, pdf_bytes = uploaded.name, uploaded.getvalue()
    save_upload(pdf_name, pdf_bytes)
elif sample != "(none)":
    pdf_name, pdf_bytes = sample, (SAMPLES_DIR / sample).read_bytes()
else:
    st.info("👆 Upload a PDF or pick a sample to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# 2) extract text (with error handling for bad / empty PDFs)
# ---------------------------------------------------------------------------
doc = None
try:
    with st.spinner("Extracting text from PDF..."):
        doc = cached_extract(pdf_bytes)
except EmptyPDFError as exc:
    # No text layer at all -- a scan. OCR is the only way to read it.
    st.error(f"**No text found:** {exc}")
except PDFReadError as exc:
    st.error(f"**Could not read PDF:** {exc}")
    st.stop()

# A scanned PDF either has no text at all, or a few stray words (a watermark).
# Offer OCR in both cases instead of letting the user query an empty document.
if doc is None or doc.looks_scanned:
    if doc is not None:
        st.warning(
            f"**This PDF looks scanned.** Only {doc.word_count} words were found "
            f"across {doc.num_pages} pages, which usually means the pages are "
            "images of text. Switch on OCR below to read them."
        )
    if not ocr_available():
        st.info("To read scanned PDFs, install OCR:  `pip install easyocr`")
        st.stop()
    if not st.checkbox("🔍 Read this PDF with OCR (slow: ~20 s per page)"):
        st.stop()
    try:
        doc = ocr_once(pdf_bytes, OCR_CHOICES[ocr_lang], int(ocr_first_page),
                       int(ocr_max_pages))
    except (OCRError, EmptyPDFError) as exc:
        st.error(f"**OCR failed:** {exc}")
        st.stop()
    read_pages = sum(1 for p in doc.pages if p.strip())
    st.success(f"OCR read {doc.word_count:,} words from {read_pages} page(s).")

for w in doc.warnings:
    st.warning(w)

# Non-Latin scripts need smaller chunks (see chunker.suggest_chunk_size).
if auto_chunk:
    chunk_size, overlap = suggest_chunk_size(doc.text)
# Clamp once, here, so the chunk count shown and the answering call agree.
overlap = min(overlap, chunk_size - 1)

try:
    n_chunks = len(chunk_text(pages=doc.pages, chunk_size=chunk_size, overlap=overlap))
except TextTooShortError as exc:
    st.error(f"**Not enough text:** {exc}")
    st.stop()

# The English model can't read other scripts: warn instead of returning nonsense.
if is_non_latin(doc.text) and model_name == ENGLISH_MODEL:
    st.warning(
        "**This document isn't in the Latin alphabet, but the English model is "
        "selected.** It will mostly fail to answer. Switch **Model** to "
        "*Multilingual (XLM-RoBERTa-large)* in the sidebar."
    )

mode = "direct (whole text)" if n_chunks == 1 else f"chunked ({n_chunks} chunks)"
c1, c2, c3, c4 = st.columns(4)
read = [i + 1 for i, page in enumerate(doc.pages) if page.strip()]   # real page numbers
if doc.total_pages and read and len(read) < doc.total_pages:
    span = f"{read[0]}" if len(read) == 1 else f"{read[0]}-{read[-1]}"
    pages_label = f"{span} of {doc.total_pages}"
else:
    pages_label = str(doc.num_pages)
c1.metric("Pages", pages_label)
c2.metric("Words", f"{doc.word_count:,}")
c3.metric("Extracted with", doc.method)
c4.metric("QA mode", mode)

with st.expander("Preview extracted text"):
    st.text(doc.text[:5000] + ("\n\n... (truncated)" if len(doc.text) > 5000 else ""))
    # Handy for checking what OCR actually read.
    st.download_button("Download extracted text", doc.text,
                       file_name=f"{Path(pdf_name).stem}_extracted.txt", mime="text/plain")

# ---------------------------------------------------------------------------
# 3) ask a question
# ---------------------------------------------------------------------------
# For the bundled sample PDFs, offer the example questions as a starting point.
suggestions = example_questions(pdf_name)
if suggestions:
    picked = st.selectbox("Example questions for this sample", ["(type my own)"] + suggestions)
    # Only fill the box when the *selection changes*, otherwise a question the
    # user typed after picking an example would be overwritten on every rerun.
    if picked != "(type my own)" and st.session_state.get("last_picked") != picked:
        st.session_state["last_picked"] = picked
        st.session_state["question_box"] = picked

with st.form("qa_form"):
    question = st.text_input("Your question", key="question_box",
                             placeholder="e.g. When was the tower completed?")
    submitted = st.form_submit_button("Get answer", type="primary")

if not submitted:
    st.stop()
if not question.strip():
    st.warning("Please type a question first.")
    st.stop()

bar = None
try:
    with st.spinner(f"Loading {model_name} (first time downloads the model)..."):
        engine = load_engine(model_name, backend)
    bar = st.progress(0.0, text="Reading chunks...")
    result = engine.answer(
        question, pages=doc.pages, chunk_size=chunk_size, overlap=overlap,
        progress=lambda done, total: bar.progress(done / total, text=f"Chunk {done}/{total}"),
    )
except InferenceAPIError as exc:
    st.error(f"**Inference API error:** {exc}")
    st.stop()
except Exception as exc:  # e.g. network failure while downloading the model
    st.error(f"**Something went wrong while answering:** {exc}")
    st.stop()
finally:
    if bar is not None:      # never leave a stalled progress bar on screen
        bar.empty()

# ---------------------------------------------------------------------------
# 4) show the result
# ---------------------------------------------------------------------------
st.divider()
if not result.found:
    st.error("**No answer found.** The model thinks this document doesn't contain "
             "the answer. Try rephrasing the question, or check the PDF text above.")
    st.stop()

st.subheader("Answer")
st.markdown(f"<div style='font-size:1.6rem;font-weight:600'>{html.escape(result.answer)}</div>",
            unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Confidence", f"{result.score:.1%}")
m2.metric("Source", f"Chunk {result.best.chunk.index + 1} of {result.num_chunks}")
m3.metric("Location", result.best.chunk.page_label)
st.progress(result.score)

if result.low_confidence:
    st.warning(f"Low confidence (< {LOW_CONFIDENCE_THRESHOLD:.0%}). The answer may be "
               "wrong; try rephrasing the question.")

st.markdown("**Source context** (answer highlighted):")
with st.container(height=260, border=True):
    st.markdown(highlight(result.best.chunk.text, result.best.start, result.best.end),
                unsafe_allow_html=True)

if len(result.candidates) > 1:
    with st.expander(f"Answers from each chunk ({len(result.candidates)})"):
        st.caption("Each chunk is scored separately and the highest-confidence answer "
                   "wins. Chunks that found no answer are not listed.")
        st.table([{"Chunk": c.chunk.index + 1, "Pages": c.chunk.page_label,
                   "Answer": c.answer, "Confidence": f"{c.score:.1%}"}
                  for c in result.candidates])

# ---------------------------------------------------------------------------
# 5) session history -- every question asked about this PDF, exportable
# ---------------------------------------------------------------------------
history = st.session_state.setdefault("history", [])
row = {
    "PDF": pdf_name,
    "Question": result.question,
    "Answer": result.answer,
    "Confidence": round(result.score, 3),
    "Source": result.best.chunk.page_label,
    "Model": model_name.split("/")[-1],
}
if row not in history:
    history.append(row)

if len(history) > 1:
    with st.expander(f"Question history ({len(history)})"):
        st.table(history)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        st.download_button(
            "Download history as CSV",
            # utf-8-sig so Excel opens Telugu/Hindi answers correctly.
            buffer.getvalue().encode("utf-8-sig"),
            file_name="qa_history.csv", mime="text/csv")
