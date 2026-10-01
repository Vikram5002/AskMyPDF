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
    MIN_WORDS_FOR_QA,
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


def suggest_chunk_size(text: str):
    """
    Pick a chunk size that fits the model's 512-token window for this script.

    Models count sub-word tokens, and the number of tokens per word depends
    heavily on the writing system:

        English  "photosynthesis"   -> 2 tokens     (~1.2 tokens/word)
        Telugu   "మామిడిపండు"        -> many tokens  (~2.7 tokens/word)

    So 400 English words (~490 tokens) fit in one pass, while 400 Telugu words
    (~1,030 tokens) do not. If most letters in the text are non-Latin we
    therefore return the smaller size.

    Returns:
        (chunk_size, overlap) in words.
    """
    letters = [c for c in text[:5000] if c.isalpha()]
    if not letters:
        return CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
    non_latin_share = sum(1 for c in letters if ord(c) > 0x24F) / len(letters)
    if non_latin_share > 0.3:
        return CHUNK_SIZE_WORDS_NON_LATIN, CHUNK_OVERLAP_WORDS_NON_LATIN
    return CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


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

    if len(words) < MIN_WORDS_FOR_QA:
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
            text=" ".join(words[start:end]),
            start_word=start,
            end_word=end,
            pages=sorted(set(word_pages[start:end])),
        ))
        if end == len(words):   # reached the end of the document
            break
        start += step

    return chunks
