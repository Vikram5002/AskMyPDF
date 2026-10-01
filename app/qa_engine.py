"""
Step 3 of the pipeline: extractive question answering with Hugging Face.

EXTRACTIVE QA IN ONE PARAGRAPH
    The model does not *write* an answer. It receives
        [CLS] question [SEP] context [SEP]
    and for every token in the context predicts two numbers: how likely that
    token is the START of the answer and how likely it is the END. The answer
    is the span context[start:end] with the best start*end probability. That
    probability is the "confidence score" (between 0 and 1).

"NO ANSWER"
    Our models were trained on SQuAD 2.0, where 1/3 of the questions have no
    answer in the text. For those, the model learned to point at the [CLS]
    token. With `handle_impossible_answer=True` the pipeline returns an empty
    answer when that "null" choice beats every real span.

SHORT vs LONG DOCUMENTS
    * Short (fits in one 400-word chunk): the whole text is one context.
    * Long: split into overlapping 400-word chunks (see chunker.py), ask the
      model once per chunk, and keep the answer with the highest score.

TWO LEVELS OF "CONTEXT LENGTH" HANDLING
    Models count *tokens* (sub-word pieces), not words: 400 English words are
    roughly 480-500 tokens. RoBERTa can read at most 512 tokens, so we ask
    the pipeline to use that full window (`max_seq_len=512`) and a normal
    chunk fits in one pass. If a chunk is unusually token-dense and still
    overflows, the pipeline itself slides a 512-token window over it with a
    128-token overlap (`doc_stride`). So:
      * our word-chunking = coarse split; also tells the user WHERE the answer is
      * tokenizer windowing = safety net; the model never sees > 512 tokens
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import app  # noqa: F401  (sets HF_HUB_CACHE / USE_TF before transformers loads)
from app.chunker import Chunk, chunk_text, suggest_chunk_size
from app.config import (
    DOC_STRIDE_TOKENS,
    ENGLISH_MODEL,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_ANSWER_TOKENS,
    MAX_SEQ_LEN_TOKENS,
)

logger = logging.getLogger(__name__)

# Signature every backend follows: (question, context) -> {"answer", "score", "start", "end"}
QAFunction = Callable[[str, str], Dict]


@dataclass
class Candidate:
    """The best answer found inside one chunk."""

    answer: str
    score: float
    start: int          # character offsets of the answer inside chunk.text
    end: int
    chunk: Chunk


@dataclass
class QAResult:
    """Everything the UI needs to show about an answer."""

    question: str
    answer: str                         # "" when no answer was found
    score: float                        # model confidence, 0..1
    model: str
    mode: str                           # "direct" (short doc) or "chunked" (long doc)
    num_chunks: int
    best: Optional[Candidate] = None    # where the answer came from
    candidates: List[Candidate] = field(default_factory=list)  # all chunks' answers, best first

    @property
    def found(self) -> bool:
        return bool(self.answer)

    @property
    def low_confidence(self) -> bool:
        return self.found and self.score < LOW_CONFIDENCE_THRESHOLD


def _trim_span(context: str, start: int, end: int):
    """
    Shrink an answer span so it doesn't start/end with spaces or stray
    punctuation (sub-word tokenizers sometimes return e.g. "1889,").
    Returns the new (start, end) character offsets into `context`.
    """
    junk = " \t\n,;:"
    while start < end and context[start] in junk:
        start += 1
    while end > start and context[end - 1] in junk:
        end -= 1
    return start, end


def load_local_pipeline(model_name: str = ENGLISH_MODEL) -> QAFunction:
    """
    Load a QA model with `transformers.pipeline` and wrap it as a QAFunction.

    The first call downloads the weights into /models (~500 MB for the English
    model, ~2.2 GB for the multilingual one); later calls load from disk.
    """
    import torch
    from transformers import pipeline

    def build(device: int):
        return pipeline("question-answering", model=model_name,
                        tokenizer=model_name, device=device)

    if torch.cuda.is_available():
        try:
            qa_pipeline = build(0)          # GPU
        except Exception as exc:
            # A 4 GB card can run out of memory with the large multilingual
            # model, especially while OCR also holds GPU memory. Keep working.
            logger.warning("Loading %s on the GPU failed (%s); using the CPU.",
                           model_name, exc)
            torch.cuda.empty_cache()
            qa_pipeline = build(-1)
    else:
        qa_pipeline = build(-1)             # CPU

    def ask(question: str, context: str) -> Dict:
        return qa_pipeline(
            question=question,
            context=context,
            handle_impossible_answer=True,   # allow an empty "no answer" result
            max_answer_len=MAX_ANSWER_TOKENS,
            max_seq_len=MAX_SEQ_LEN_TOKENS,  # use the model's full 512-token window
            doc_stride=DOC_STRIDE_TOKENS,
        )

    return ask


class QAEngine:
    """
    Answers questions about a document using a Hugging Face QA model.

    Example:
        engine = QAEngine()                           # English model, run locally
        result = engine.answer("Who built it?", text="...")
        print(result.answer, result.score)
    """

    def __init__(self, model_name: str = ENGLISH_MODEL, backend: str = "local"):
        """
        Args:
            model_name: any Hugging Face Hub model fine-tuned for QA.
            backend:    "local" -> download and run the model on this machine.
                        "api"   -> call Hugging Face's hosted Inference API
                                   (needs the HF_TOKEN environment variable).
        """
        self.model_name = model_name
        self.backend = backend
        if backend == "local":
            self._ask = load_local_pipeline(model_name)
        elif backend == "api":
            from app.hf_inference_api import make_api_qa_function
            self._ask = make_api_qa_function(model_name)
        else:
            raise ValueError(f"Unknown backend {backend!r}; use 'local' or 'api'.")

    def answer(
        self,
        question: str,
        text: Optional[str] = None,
        pages: Optional[Sequence[str]] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> QAResult:
        """
        Answer `question` from a document given as `text` or as a list of `pages`.

        Args:
            chunk_size/overlap: in words. Left as None, they are chosen from the
                text's script (smaller for Telugu, Hindi, ... -- see
                chunker.suggest_chunk_size).
            progress: optional callback(done, total), e.g. to drive a progress bar.

        Raises:
            ValueError:        empty question.
            TextTooShortError: document has too little text (from chunker).
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")

        # A short document (<= chunk_size words) comes back as exactly one chunk
        # holding the whole text -> "direct" mode, one model call. A long one
        # becomes several overlapping chunks -> "chunked" mode.
        if chunk_size is None or overlap is None:
            suggested_size, suggested_overlap = suggest_chunk_size(text or " ".join(pages or []))
            # `is None` rather than `or`, so an explicit 0 still reaches the
            # validation in chunk_text() instead of being silently replaced.
            if chunk_size is None:
                chunk_size = suggested_size
            if overlap is None:
                # Keep the suggested overlap valid for a caller-supplied
                # chunk_size that is smaller than it.
                overlap = min(suggested_overlap, max(chunk_size - 1, 0))

        chunks = chunk_text(text=text, pages=pages, chunk_size=chunk_size, overlap=overlap)
        mode = "direct" if len(chunks) == 1 else "chunked"

        candidates: List[Candidate] = []
        for i, chunk in enumerate(chunks):
            out = self._ask(question, chunk.text)
            start, end = _trim_span(chunk.text, int(out["start"]), int(out["end"]))
            if out["answer"].strip() and end > start:   # skip the "no answer" result
                candidates.append(Candidate(
                    answer=chunk.text[start:end],
                    # When a chunk overflows the window and the same answer is
                    # found in two overlapping windows, the pipeline ADDS the
                    # two scores, which can exceed 1. Cap it so it stays a probability.
                    score=min(float(out["score"]), 1.0),
                    start=start,
                    end=end,
                    chunk=chunk,
                ))
            if progress:
                progress(i + 1, len(chunks))

        # CONFIDENCE-BASED SELECTION: keep the answer the model was most sure
        # about. Caveat: each chunk's score is normalised *within that chunk*,
        # so scores from different chunks are only roughly comparable -- a
        # simple heuristic, but it works well in practice.
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0] if candidates else None

        return QAResult(
            question=question,
            answer=best.answer if best else "",
            score=best.score if best else 0.0,
            model=self.model_name,
            mode=mode,
            num_chunks=len(chunks),
            best=best,
            candidates=candidates,
        )
