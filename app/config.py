"""
Central configuration: every tunable number lives here, so it can be changed
(and explained) in one place.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"   # PDFs uploaded through the UI
MODELS_DIR = PROJECT_ROOT / "models"     # local Hugging Face model cache
SAMPLES_DIR = PROJECT_ROOT / "samples"   # demo PDFs used by tests / README

# --------------------------------------------------------------------------
# Models (both are fine-tuned on SQuAD 2.0, an extractive QA dataset that
# also contains *unanswerable* questions -- so the model can say "no answer")
# --------------------------------------------------------------------------
ENGLISH_MODEL = "deepset/roberta-base-squad2"            # ~124M params, fast
MULTILINGUAL_MODEL = "deepset/xlm-roberta-large-squad2"  # ~560M params, 100 languages

AVAILABLE_MODELS = {
    "English (RoBERTa-base)": ENGLISH_MODEL,
    "Multilingual (XLM-RoBERTa-large)": MULTILINGUAL_MODEL,
}

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# Transformer models can only read a limited number of tokens at once
# (RoBERTa: 512 tokens max; the QA pipeline uses windows of 384 by default).
# A whole PDF is far longer, so we cut it into chunks of ~400 words.
CHUNK_SIZE_WORDS = 400

# Consecutive chunks share 50 words, so an answer that falls on a chunk
# boundary still appears *complete* in at least one chunk.
CHUNK_OVERLAP_WORDS = 50

# Non-Latin scripts (Telugu, Hindi, Tamil, Chinese...) are split into far more
# sub-word tokens than English: about 2.7 tokens per word instead of 1.2. At
# 400 words such a chunk becomes ~1,000 tokens, double the 512-token limit, so
# for those scripts we use a smaller chunk. See chunker.suggest_chunk_size().
CHUNK_SIZE_WORDS_NON_LATIN = 150
CHUNK_OVERLAP_WORDS_NON_LATIN = 40

# When the model's tokenizer is available, chunk size is measured from the
# document instead of guessed (see chunker.suggest_chunk_size). This is how
# many tokens of the 512-token window are left for the context itself, after
# allowing for the question and the [CLS]/[SEP] markers.
TOKEN_BUDGET_PER_CHUNK = 440

# Never go below this many words per chunk, or answers lose their context.
MIN_CHUNK_SIZE_WORDS = 60

# Fewer words than this and there's not enough text to answer anything
# (e.g. a scanned PDF that only yielded a page number).
MIN_WORDS_FOR_QA = 5

# Chinese, Japanese and Thai don't put spaces between words, so counting words
# fails for them. If a document has fewer words than above but at least this
# many characters, it is chunked by character instead (see chunker.py).
MIN_CHARS_FOR_QA = 40

# --------------------------------------------------------------------------
# OCR (optional, for scanned PDFs -- see app/ocr.py)
# --------------------------------------------------------------------------
# A real text PDF has hundreds of words per page. If a PDF averages fewer
# words per page than this, it is almost certainly a scan (a picture of text)
# and needs OCR.
SCANNED_WORDS_PER_PAGE = 20

# A single page with fewer words than this counts as "empty". If more than half
# the pages are empty, the file is treated as (partly) scanned even when its
# average looks fine -- e.g. 5 text pages plus 30 scanned ones.
SCANNED_PAGE_WORDS = 10

# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
# pdfplumber decides where one word ends and the next begins from the gap
# between characters. Its default of 3 points is too wide for the tight
# spacing LaTeX produces, which glues whole lines into
# "WeusedtheAdamoptimizer". 1.5 points splits them correctly.
X_TOLERANCE = 1.5

# EasyOCR language codes. One Indic language at a time, optionally with English.
OCR_LANGUAGES = ("te", "en")   # Telugu + English

OCR_DPI = 300          # page rendering resolution for OCR
OCR_MAX_PAGES = 10     # safety limit: OCR takes ~20 s per page on a CPU

# --------------------------------------------------------------------------
# Answer confidence
# --------------------------------------------------------------------------
# Answers scoring below this are still shown, but flagged as low-confidence.
LOW_CONFIDENCE_THRESHOLD = 0.10

# Longest answer span (in tokens) the model is allowed to return.
MAX_ANSWER_TOKENS = 30

# --------------------------------------------------------------------------
# Model input window (tokens, not words)
# --------------------------------------------------------------------------
# RoBERTa / XLM-RoBERTa accept at most 512 tokens (question + context).
# The pipeline's default is only 384, which would split almost every
# 400-word chunk (~480-500 tokens of English) into two windows. Using the
# full 512 lets a normal chunk be read in ONE pass.
MAX_SEQ_LEN_TOKENS = 512

# If a chunk still doesn't fit, consecutive windows overlap by this many tokens.
DOC_STRIDE_TOKENS = 128
