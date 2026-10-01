"""
Step 1 of the pipeline: PDF -> plain text.

A PDF does not store "paragraphs"; it stores instructions like "draw glyph 'H'
at (x, y)". Text extraction libraries rebuild words and lines from those
positions. We use two libraries:

* pdfplumber (primary)  - layout-aware, usually gives the cleanest text.
* PyPDF2     (fallback) - simpler, but can open some PDFs pdfplumber chokes on.

If both return no text, the PDF is probably a scanned image (a picture of the
page). That case is handled by `extract_with_ocr()` below, which uses app/ocr.py.
"""

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pdfplumber
from PyPDF2 import PdfReader

from app.config import OCR_LANGUAGES, OCR_MAX_PAGES, SCANNED_WORDS_PER_PAGE

logger = logging.getLogger(__name__)

# A PDF can be given as a file path or as raw bytes (e.g. from a web upload).
PDFSource = Union[str, Path, bytes]


class PDFReadError(Exception):
    """The file could not be opened/parsed as a PDF by either library."""


class EmptyPDFError(Exception):
    """The PDF opened fine but contains no extractable text."""


@dataclass
class ExtractionResult:
    """Text extracted from a PDF, kept page by page."""

    pages: List[str]                 # cleaned text of each page (index 0 = page 1)
    method: str                      # which library produced the text
    warnings: List[str] = field(default_factory=list)
    total_pages: Optional[int] = None  # pages in the file (OCR may read only some)

    @property
    def text(self) -> str:
        """All pages joined into one string."""
        return "\n\n".join(p for p in self.pages if p)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def looks_scanned(self) -> bool:
        """
        True if this PDF is probably a scan (a picture of text).

        A text PDF holds hundreds of words per page. A scanned one yields
        almost nothing -- often just a watermark. Such a file needs OCR;
        see app/ocr.py.
        """
        if self.method == "OCR" or not self.pages:
            return False
        return self.word_count / self.num_pages < SCANNED_WORDS_PER_PAGE


def clean_text(raw: str) -> str:
    """
    Normalise extracted text so the tokenizer sees clean input.

    * Re-join words hyphenated across a line break ("infor-\\nmation" -> "information").
    * Collapse runs of spaces/tabs; drop blank-line noise.
    """
    if not raw:
        return ""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)   # undo end-of-line hyphenation
    text = re.sub(r"[ \t]+", " ", text)            # squeeze horizontal whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)       # squeeze blank lines
    return text.strip()


def _open_stream(source: PDFSource):
    """Both libraries accept a file-like object, so normalise input to one."""
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    path = Path(source)
    if not path.exists():
        raise PDFReadError(f"File not found: {path}")
    return open(path, "rb")


def _extract_with_pdfplumber(source: PDFSource) -> List[str]:
    with _open_stream(source) as stream, pdfplumber.open(stream) as pdf:
        return [clean_text(page.extract_text() or "") for page in pdf.pages]


def _extract_with_pypdf2(source: PDFSource) -> List[str]:
    with _open_stream(source) as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            # Many "encrypted" PDFs only have an empty owner password.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise PDFReadError("PDF is password-protected.") from exc
        return [clean_text(page.extract_text() or "") for page in reader.pages]


def extract_with_ocr(
    source: PDFSource,
    languages: Sequence[str] = OCR_LANGUAGES,
    first_page: int = 1,
    max_pages: int = OCR_MAX_PAGES,
    progress: Optional[Callable[[int, int], None]] = None,
    cache: Optional[dict] = None,
) -> ExtractionResult:
    """
    Read a scanned PDF using OCR instead of the PDF's (missing) text layer.

    OCR costs about 20 seconds per page on a CPU, so only pages
    `first_page` .. `first_page + max_pages - 1` are read. Starting at the
    page you actually care about is the simplest speed-up there is.

    Pages before `first_page` are kept as empty strings, so page numbers shown
    in the UI stay the real page numbers of the PDF.

    Raises:
        OCRError:      EasyOCR missing or the OCR run failed.
        EmptyPDFError: OCR found no readable text in the chosen pages.
    """
    from app.ocr import ocr_pdf   # imported lazily: OCR is an optional extra

    wanted = range(max(first_page, 1), max(first_page, 1) + max(max_pages, 1))
    by_number = ocr_pdf(source, languages, pages=wanted, progress=progress, cache=cache)
    if not any(text.strip() for text in by_number.values()):
        raise EmptyPDFError("OCR ran but found no readable text in these pages.")

    # Keep the list index aligned with the real page number (index 0 = page 1).
    last = max(by_number)
    pages = [clean_text(by_number.get(n, "")) for n in range(1, last + 1)]

    total = None
    try:  # how many pages the file really has (we may have read only some)
        with _open_stream(source) as stream:
            total = len(PdfReader(stream).pages)
    except Exception:
        pass

    warnings = []
    read = sorted(by_number)
    if total and len(read) < total:
        span = f"page {read[0]}" if len(read) == 1 else f"pages {read[0]}-{read[-1]}"
        warnings.append(f"OCR read {span} of {total} (OCR is slow). Change the page "
                        f"range in the sidebar to read different pages.")
    return ExtractionResult(pages=pages, method="OCR", warnings=warnings, total_pages=total)


def extract_text(source: PDFSource) -> ExtractionResult:
    """
    Extract text from a PDF, trying pdfplumber first and PyPDF2 second.

    Args:
        source: path to a PDF file, or the PDF's raw bytes.

    Returns:
        ExtractionResult with per-page text and the library that was used.
        Check `.looks_scanned` to see whether OCR is needed instead.

    Raises:
        PDFReadError:  neither library could read the file (corrupt / not a PDF).
        EmptyPDFError: the PDF was read but has no text (e.g. scanned images).
    """
    if isinstance(source, (bytes, bytearray)) and len(source) == 0:
        raise PDFReadError("The uploaded file is empty (0 bytes).")

    warnings: List[str] = []
    errors: List[str] = []

    for name, extractor in (("pdfplumber", _extract_with_pdfplumber),
                            ("PyPDF2", _extract_with_pypdf2)):
        try:
            pages = extractor(source)
        except PDFReadError:
            raise
        except Exception as exc:  # library-specific parse errors vary widely
            logger.warning("%s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
            continue

        if any(pages):
            if errors:
                warnings.append(f"Used {name} fallback ({'; '.join(errors)})")
            return ExtractionResult(pages=pages, method=name, warnings=warnings)

        # Opened fine but found no text -- give the other library a chance.
        errors.append(f"{name}: no text found")

    # If EITHER library opened the file but found no text, this is a scan rather
    # than a broken file -- say so, because that message is what offers OCR.
    # (Using all() here would hide a scan whenever the other library also crashed.)
    if any("no text found" in e for e in errors):
        raise EmptyPDFError(
            "No extractable text found in this PDF. It is most likely a scan "
            "(a picture of text), which needs OCR -- switch on 'Read with OCR'."
        )
    raise PDFReadError("Could not read this PDF. Details: " + " | ".join(errors))
