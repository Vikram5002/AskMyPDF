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
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pdfplumber
from PyPDF2 import PdfReader

from app.config import (
    OCR_LANGUAGES,
    OCR_MAX_PAGES,
    SCANNED_PAGE_WORDS,
    SCANNED_WORDS_PER_PAGE,
    X_TOLERANCE,
)

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
    def empty_pages(self) -> int:
        """Pages that yielded essentially no text (images of text, or blank)."""
        return sum(1 for page in self.pages if len(page.split()) < SCANNED_PAGE_WORDS)

    @property
    def looks_scanned(self) -> bool:
        """
        True if this PDF is probably a scan (a picture of text).

        Two cases, because a book can be scanned in part:

        * the whole file averages almost no text per page, or
        * most individual pages are empty while a few carry text -- e.g. a
          35-page book with 5 typed pages and 30 scanned ones. Averaging alone
          would call that a normal PDF and silently skip the scanned pages.

        Such a file needs OCR; see app/ocr.py.
        """
        if self.method == "OCR" or not self.pages:
            return False
        if self.word_count / self.num_pages < SCANNED_WORDS_PER_PAGE:
            return True
        return self.empty_pages > self.num_pages / 2


def _join_hyphenated(text: str) -> str:
    """
    Repair words split across a line break, without destroying real hyphens.

    Two different things look identical in a PDF:

        "infor-\\nmation"      -> one word broken by the line break -> "information"
        "self-\\nattention"    -> a genuinely hyphenated word       -> "self-attention"

    Blindly deleting the hyphen turns "self-attention" into "selfattention" and
    "state-of-the-art" into "stateoftheart", which the QA model then cannot match.

    The rule used here: if the part before the hyphen also appears as a word on
    its own somewhere in the document ("self", "state"), the hyphen is real and
    is kept. Otherwise ("infor") the two halves are a broken word and are joined.
    """
    # Build the vocabulary from text with the line-break hyphenations removed,
    # so that the broken halves ("infor", "mation") cannot vouch for themselves.
    without_breaks = re.sub(r"([A-Za-z]+)-\n([A-Za-z]+)", " ", text)
    vocabulary = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", without_breaks)}

    def repair(match: "re.Match") -> str:
        left, right = match.group(1), match.group(2)
        if left.lower() in vocabulary:
            return f"{left}-{right}"      # real hyphen, e.g. self-attention
        return f"{left}{right}"           # broken word, e.g. infor-mation

    return re.sub(r"([A-Za-z]+)-\n([A-Za-z]+)", repair, text)


#   Arabic / Hebrew / Urdu / Sindhi letters, plus the "presentation forms"
#   blocks that PDF producers use for the joined shapes of Arabic letters.
#   Everything unprintable except tab and newline, plus the replacement
#   character a failed glyph lookup leaves behind.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]")

_RTL_CHARACTERS = re.compile(r"[֐-ࣿﭐ-﷿ﹰ-﻿]")
_RTL_PRESENTATION_FORMS = re.compile(r"[ﭐ-﷿ﹰ-﻿]")


def _repair_rtl(text: str) -> str:
    """
    Restore right-to-left text (Arabic, Urdu, Sindhi, Hebrew) to reading order.

    PDFs store glyphs by their position on the page, so a right-to-left line is
    recorded left-to-right -- i.e. backwards -- and usually in "presentation
    form" codepoints, the joined shapes of each letter. Extracted raw it looks
    like "ﻞﺤﻣ جﺎﺗ" instead of "تاج محل": reversed, and spelled with characters
    that never match a search or a model's vocabulary.

    Two steps fix it:
      1. NFKC normalisation turns presentation forms back into normal letters.
      2. Each line's runs are put back in logical order. Numbers and Latin words
         embedded in the line keep their own left-to-right order, so "1653"
         doesn't become "3561".

    Text with no RTL presentation forms is returned untouched.
    """
    if not _RTL_PRESENTATION_FORMS.search(text):
        return text

    repaired_lines = []
    for line in unicodedata.normalize("NFKC", text).splitlines():
        runs, current, current_is_rtl = [], "", None
        for character in line:
            is_rtl = bool(_RTL_CHARACTERS.match(character))
            if current_is_rtl is None or is_rtl == current_is_rtl:
                current += character
                current_is_rtl = is_rtl
            else:
                runs.append((current, current_is_rtl))
                current, current_is_rtl = character, is_rtl
        if current:
            runs.append((current, current_is_rtl))
        # The line was laid out right-to-left, so the run order reverses; only
        # the right-to-left runs have their characters reversed as well.
        repaired_lines.append(
            "".join(run[::-1] if is_rtl else run for run, is_rtl in reversed(runs)))
    return "\n".join(repaired_lines)


def clean_text(raw: str) -> str:
    """
    Normalise extracted text so the tokenizer sees clean input.

    * Put right-to-left scripts back into reading order.
    * Re-join words broken across a line break, keeping real hyphens.
    * Collapse runs of spaces/tabs; drop blank-line noise.
    """
    if not raw:
        return ""
    text = _join_hyphenated(_repair_rtl(raw))
    # Drop control characters and U+FFFD. A PDF whose font can't be mapped back
    # to Unicode yields NUL bytes where letters should be (Telugu "ఆగ్రా" comes
    # out as "ఆ\x00\x00"), and those would otherwise end up inside answers.
    text = _CONTROL_CHARACTERS.sub("", text)
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


def _column_boundary(words) -> Optional[float]:
    """
    Find the gutter of a two-column page, or None if the page is one column.

    Research papers, newspapers and many textbooks print two columns. Read
    straight across, line by line, the text becomes alternating halves of two
    unrelated sentences, which destroys the meaning the QA model relies on.

    Detection looks for a narrow vertical strip near the middle that almost no
    word crosses. "Almost" matters: a centred title or a wide table can put a
    word or two across the gutter, and demanding a perfectly clear strip would
    miss a paper's first page -- the page with the abstract on it.
    """
    if len(words) < 60:                      # too little text to judge
        return None
    left_edge = min(w["x0"] for w in words)
    right_edge = max(w["x1"] for w in words)
    span = right_edge - left_edge
    if span <= 0:
        return None

    tolerated = max(3, int(0.02 * len(words)))   # title / table words
    best = None
    for fraction in [0.40 + i * 0.01 for i in range(21)]:
        split = left_edge + span * fraction
        crossing = sum(1 for w in words if w["x0"] < split < w["x1"])
        if crossing > tolerated:
            continue
        left_count = sum(1 for w in words if w["x1"] <= split)
        right_count = sum(1 for w in words if w["x0"] >= split)
        # Both columns must hold a reasonable share of the page's words.
        if min(left_count, right_count) < 0.25 * len(words):
            continue
        key = (crossing, abs(left_count - right_count))
        if best is None or key < best[1]:
            best = (split, key)
    return None if best is None else best[0]


def _page_text(page) -> str:
    """
    Text of a single page, handling word spacing and two-column layouts.

    x_tolerance: pdfplumber decides where one word ends and the next begins by
    the horizontal gap between characters. Its default (3 points) is too wide
    for the tight spacing LaTeX produces, so whole lines come out as
    "WeusedtheAdamoptimizer". 1.5 points splits those correctly.

    On a two-column page the left column is read in full, then the right one.
    """
    words = page.extract_words(x_tolerance=X_TOLERANCE)
    if not words:
        return page.extract_text(x_tolerance=X_TOLERANCE) or ""

    boundary = _column_boundary(words)
    if boundary is None:
        return page.extract_text(x_tolerance=X_TOLERANCE) or ""

    parts = []
    for x0, x1 in ((0, boundary), (boundary, page.width)):
        column = page.crop((x0, 0, x1, page.height), strict=False)
        parts.append(column.extract_text(x_tolerance=X_TOLERANCE) or "")
    return "\n".join(p for p in parts if p.strip())


def _extract_with_pdfplumber(source: PDFSource) -> List[str]:
    with _open_stream(source) as stream, pdfplumber.open(stream) as pdf:
        return [clean_text(_page_text(page)) for page in pdf.pages]


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
