"""
OPTIONAL step 0: OCR for scanned PDFs.

A normal PDF stores text. A *scanned* PDF stores a photograph of each page, so
pdfplumber and PyPDF2 find nothing to extract -- very common for Indian-language
books. OCR (Optical Character Recognition) looks at the page image and works out
which characters are in it.

How it works here:
    1. pypdfium2 renders each PDF page into an image (it ships with pdfplumber,
       so no extra dependency is needed for this part).
    2. EasyOCR runs a text-detection + text-recognition neural network on that
       image and returns the lines it read.

EasyOCR is an optional dependency: `pip install easyocr`. Everything else in
the project works without it.

Caveats worth knowing:
    * Slow: roughly 20 seconds per page on a CPU, so we limit how many pages run.
    * Imperfect: expect wrong characters, especially for similar-looking Telugu
      letters (ప/ట). QA still works, but confidence scores drop.
    * Reading order on two-column book scans can be jumbled.
"""

import logging
from typing import Callable, Dict, Optional, Sequence

from app.config import OCR_DPI, OCR_LANGUAGES, OCR_MAX_PAGES

logger = logging.getLogger(__name__)

# The language choices offered in the UI. EasyOCR restricts which languages can
# share one model: each Indic script may be combined with English, but not with
# another Indic script. Tamil and Malayalam are left out on purpose -- Tamil
# fails to load in EasyOCR 1.7.2 and Malayalam is not supported at all.
OCR_CHOICES = {
    "Telugu + English": ("te", "en"),
    "Hindi/Marathi + English": ("hi", "en"),
    "Kannada + English": ("kn", "en"),
    "Bengali + English": ("bn", "en"),
    "Assamese": ("as",),
    "Nepali + English": ("ne", "en"),
    "Urdu + English": ("ur", "en"),
    "English only": ("en",),
}


class OCRError(Exception):
    """OCR could not be run (not installed, unsupported language, ...)."""


def ocr_available() -> bool:
    """
    True if EasyOCR is installed and importable.

    Catches every exception, not just ImportError: importing easyocr pulls in
    torch, OpenCV and Pillow, and a broken install of any of those raises
    something else (OSError, RuntimeError...). This runs while the sidebar is
    being drawn, so an escaping error would take down the whole app.
    """
    try:
        import easyocr  # noqa: F401
        return True
    except Exception as exc:
        logger.warning("EasyOCR unavailable: %s", exc)
        return False


# Building a Reader loads neural-network weights (~100 MB on first use, then
# cached on disk), so keep one per language combination.
_readers = {}


def gpu_available() -> bool:
    """
    True if PyTorch can see a CUDA GPU.

    OCR is ~20 s per page on a CPU because it runs two neural networks over a
    large image. On a GPU it is several times faster. A CPU-only PyTorch build
    returns False here, and everything still works -- just slower.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_reader(languages: Sequence[str], gpu: Optional[bool] = None):
    if not ocr_available():
        raise OCRError(
            "OCR needs the EasyOCR package, which isn't installed. "
            "Run:  pip install easyocr"
        )
    import easyocr

    use_gpu = gpu_available() if gpu is None else gpu
    key = (tuple(languages), use_gpu)
    if key not in _readers:
        try:
            _readers[key] = easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)
        except Exception as exc:
            # Out of GPU memory (a 4 GB card is easily filled by a QA model
            # loaded alongside): fall back to the CPU rather than failing.
            if use_gpu:
                logger.warning("OCR on GPU failed (%s); falling back to CPU.", exc)
                return _get_reader(languages, gpu=False)
            raise OCRError(
                f"Could not load OCR for {', '.join(languages)}: {exc}. "
                "Note that EasyOCR allows only one Indic language at a time, "
                "optionally together with English."
            ) from exc
    return _readers[key]


def ocr_pdf(
    source,
    languages: Sequence[str] = OCR_LANGUAGES,
    pages: Optional[Sequence[int]] = None,
    dpi: int = OCR_DPI,
    progress: Optional[Callable[[int, int], None]] = None,
    cache: Optional[Dict[int, str]] = None,
    gpu: Optional[bool] = None,
) -> Dict[int, str]:
    """
    Read chosen pages of a scanned PDF with OCR.

    Args:
        source:    path to a PDF, or its raw bytes.
        languages: EasyOCR language codes, e.g. ("te", "en") for Telugu.
        pages:     1-based page numbers to read. Default: the first
                   OCR_MAX_PAGES pages. Reading only the pages you care about
                   is the cheapest way to make OCR faster -- there is no point
                   spending 20 seconds each on cover and contents pages.
        dpi:       rendering resolution. 300 is a good balance; lower is faster
                   but loses small characters (200 dpi is ~1.5x faster and
                   noticeably less accurate on Indic scripts).
        progress:  optional callback(done, total) for a progress bar.
        cache:     optional {page_number: text} dict of pages already read. It
                   is reused and updated, so widening the page range only
                   OCRs the pages that are actually new.
        gpu:       None (default) uses a CUDA GPU when PyTorch finds one.

    Returns:
        {page_number: text} for the requested pages.

    Raises:
        OCRError: EasyOCR missing, bad language combination, or unreadable PDF.
    """
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - both ship with the project
        raise OCRError(f"Missing image dependency for OCR: {exc}") from exc

    reader = _get_reader(languages, gpu=gpu)

    try:
        pdf = pdfium.PdfDocument(source)
    except Exception as exc:
        raise OCRError(f"Could not open the PDF for OCR: {exc}") from exc

    if pages is None:
        pages = range(1, min(len(pdf), OCR_MAX_PAGES) + 1)
    wanted = [p for p in pages if 1 <= p <= len(pdf)]
    if not wanted:
        raise OCRError(f"No such pages to read: this PDF has {len(pdf)} page(s).")

    cache = {} if cache is None else cache
    result: Dict[int, str] = {}
    for done, page_no in enumerate(wanted, start=1):
        if page_no in cache:                       # already read earlier
            result[page_no] = cache[page_no]
            if progress:
                progress(done, len(wanted))
            continue
        try:
            # scale = dpi / 72 because PDF coordinates are in 72-dpi points.
            image = pdf[page_no - 1].render(scale=dpi / 72).to_pil()
            # paragraph=True merges detected line boxes into readable blocks.
            lines = reader.readtext(np.array(image), detail=0, paragraph=True)
        except MemoryError as exc:
            # A 300-dpi A4 page is a ~2550x3300x3 array; huge pages can run out.
            raise OCRError(
                f"Ran out of memory while reading page {page_no}. "
                f"Try fewer pages, or a lower OCR_DPI in app/config.py."
            ) from exc
        except Exception as exc:
            raise OCRError(f"OCR failed on page {page_no}: {exc}") from exc
        text = "\n".join(lines).strip()
        result[page_no] = cache[page_no] = text
        logger.info("OCR page %d (%d/%d): %d words", page_no, done, len(wanted),
                    len(text.split()))
        if progress:
            progress(done, len(wanted))

    return result
