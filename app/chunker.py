"""
Step 2 of the pipeline: split long text into overlapping chunks.

WHY CHUNK?
    A transformer reads a fixed-size window of tokens (RoBERTa: max 512;
    the QA pipeline uses 384 per window). A 20-page PDF has ~10,000 words, so
    it can't be read in one go. We cut the document into ~400-word chunks, ask
    the model the question on every chunk, and keep the best answer.

WHY OVERLAP?
    If chunks were cut back-to-back, an answer sitting right on a boundary
    ("...founded in | 1998...") would be split in two and neither chunk would
    contain it whole. With 50 words of overlap, each chunk repeats the last 50
    words of the previous one, so any answer shorter than 50 words appears
    complete in at least one chunk.

    Example with size=6, overlap=2 (step = 6 - 2 = 4):
        words : w0 w1 w2 w3 w4 w5 w6 w7 w8 w9
        chunk0: w0 w1 w2 w3 w4 w5
        chunk1:             w4 w5 w6 w7 w8 w9
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_OVERLAP_WORDS_NON_LATIN,
    CHUNK_SIZE_WORDS,
    CHUNK_SIZE_WORDS_NON_LATIN,
    MIN_CHARS_FOR_QA,
    MIN_CHUNK_SIZE_WORDS,
    MIN_WORDS_FOR_QA,
    TOKEN_BUDGET_PER_CHUNK,
)


class TextTooShortError(Exception):
    """Too little text to answer questions from."""


@dataclass
class Chunk:
    """One piece of the document that will be sent to the QA model."""

    index: int          # position of the chunk in the document (0-based)
    text: str
    start_word: int     # index of the first word in the whole document
    end_word: int       # index one past the last word
    pages: List[int]    # 1-based page numbers this chunk's words come from

    @property
    def page_label(self) -> str:
        """Human-readable page range, e.g. 'page 3' or 'pages 3-4'."""
        if not self.pages:
            return "unknown page"
        lo, hi = min(self.pages), max(self.pages)
        return f"page {lo}" if lo == hi else f"pages {lo}-{hi}"


def _split_characters(pages, group: int = 1):
    """
    Split space-less scripts (Chinese, Japanese, Thai) into character groups.

    These languages don't separate words with spaces, so `str.split()` returns
    one enormous "word". Treating each character as a unit lets the normal
    chunking arithmetic work unchanged.
    """
    units, unit_pages = [], []
    for page_no, page_text in enumerate(pages, start=1):
        characters = [c for c in page_text if not c.isspace()]
        for i in range(0, len(characters), group):
            units.append("".join(characters[i:i + group]))
            unit_pages.append(page_no)
    return units, unit_pages


def suggest_chunk_size(text: str, tokenizer=None):
    """
    Pick a chunk size whose chunks fit the model's 512-token window.

    Models count sub-word tokens, not words, and the ratio varies enormously
    with the writing system AND the subject matter:

        plain English prose          ~1.2 tokens/word
        Telugu                       ~2.7 tokens/word
        a LaTeX research paper       2-6 tokens/word (formulas, citations,
                                     numbers and rare words all split up)

    So a fixed 400 words is safe for a story but produces 900-2,400 token
    chunks for a research paper, far past the 512-token limit.

    If a `tokenizer` is given, this measures the document's real ratio and
    sizes the chunk from it -- the accurate route. Without one (e.g. the hosted
    API backend) it falls back to a script-based guess.

    Returns:
        (chunk_size, overlap) in words.
    """
    if tokenizer is not None:
        measured = _measure_chunk_size(text, tokenizer)
        if measured is not None:
            return measured

    letters = [c for c in text[:5000] if c.isalpha()]
    if not letters:
        return CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
    non_latin_share = sum(1 for c in letters if ord(c) > 0x24F) / len(letters)
    if non_latin_share > 0.3:
        return CHUNK_SIZE_WORDS_NON_LATIN, CHUNK_OVERLAP_WORDS_NON_LATIN
    return CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def _measure_chunk_size(text: str, tokenizer):
    """
    Measure how many tokens this document uses per word, and size chunks to fit.

    Samples words from the start, middle and end so that a title page or
    reference list doesn't skew the estimate.
    """
    words = text.split()
    if len(words) < 50:
        return None
    sample_size = min(len(words), 900)
    third = sample_size // 3
    middle = max((len(words) - third) // 2, 0)
    sample = (words[:third] + words[middle:middle + third] + words[-third:]) or words
    try:
        token_count = len(tokenizer(" ".join(sample), add_special_tokens=False)["input_ids"])
    except Exception:
        return None
    if not token_count:
        return None

    tokens_per_word = token_count / len(sample)
    # Budget: the 512-token window minus room for the question and the
    # [CLS]/[SEP] markers, with a safety margin.
    chunk_size = int(TOKEN_BUDGET_PER_CHUNK / tokens_per_word)
    chunk_size = max(MIN_CHUNK_SIZE_WORDS, min(chunk_size, CHUNK_SIZE_WORDS))
    return chunk_size, max(10, chunk_size // 8)


def chunk_text(
    text: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
    pages: Optional[Sequence[str]] = None,
) -> List[Chunk]:
    """
    Split text into word-based chunks of `chunk_size` words, where each chunk
    shares `overlap` words with the one before it.

    Pass either `text` (a single string) or `pages` (a list of page strings;
    this lets every chunk remember which pages it came from).

    If the text is shorter than one chunk, a single chunk holding the whole
    text is returned -- there is nothing to split.

    Raises:
        ValueError:        invalid chunk_size / overlap combination.
        TextTooShortError: fewer than MIN_WORDS_FOR_QA words in total.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= overlap < chunk_size:
        # overlap >= chunk_size would make the window never move forward.
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    # Build a flat list of words, remembering the page each word came from.
    if pages is None:
        pages = [text or ""]
    words: List[str] = []
    word_pages: List[int] = []
    for page_no, page_text in enumerate(pages, start=1):
        page_words = page_text.split()
        words.extend(page_words)
        word_pages.extend([page_no] * len(page_words))

    separator = " "
    if len(words) < MIN_WORDS_FOR_QA:
        # Chinese, Japanese and Thai write without spaces between words, so a
        # whole page counts as one "word". Fall back to splitting on characters
        # (each character carries far more meaning than a Latin letter), which
        # keeps those languages usable instead of rejecting them as too short.
        if len(" ".join(words)) >= MIN_CHARS_FOR_QA:
            words, word_pages = _split_characters(pages)
            separator = ""      # these scripts are written without spaces
        else:
            raise TextTooShortError(
                f"Only {len(words)} word(s) of text were found; at least "
                f"{MIN_WORDS_FOR_QA} are needed to answer questions."
            )

    step = chunk_size - overlap   # how far the window slides each time
    chunks: List[Chunk] = []
    start = 0
    while True:
        end = min(start + chunk_size, len(words))
        chunks.append(Chunk(
            index=len(chunks),
            text=separator.join(words[start:end]),
            start_word=start,
            end_word=end,
            pages=sorted(set(word_pages[start:end])),
        ))
        if end == len(words):   # reached the end of the document
            break
        start += step

    return chunks
