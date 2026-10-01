"""
OPTIONAL: run the QA model on Hugging Face's servers instead of locally.

Local mode downloads ~500 MB-2.2 GB of weights and uses your CPU/GPU. The
hosted Inference API sends (question, context) over HTTPS to Hugging Face,
which runs the same model and returns the same {answer, score, start, end}
result -- so the rest of the app doesn't change.

Setup:
    1. Create a free "Read" token at https://huggingface.co/settings/tokens
    2. Copy .env.example to .env and put the token in it:
           HF_TOKEN=hf_...
       (.env is git-ignored. An environment variable or `huggingface-cli login`
       works too -- see app/__init__.py:load_dotenv.)
    3. Pick "Hugging Face Inference API" as the backend in the sidebar
       (or use QAEngine(backend="api") in code).

Trade-offs: no download and no local compute, but it needs internet, is
rate-limited on the free tier, and your document text leaves your machine.

`huggingface_hub` is already installed as a dependency of `transformers`,
so this adds no new requirement.
"""

import os
from typing import Dict

from huggingface_hub import InferenceClient, get_token

from app.config import DOC_STRIDE_TOKENS, MAX_ANSWER_TOKENS, MAX_SEQ_LEN_TOKENS


class InferenceAPIError(Exception):
    """The hosted API could not be used (no token, network error, etc.)."""


def make_api_qa_function(model_name: str):
    """
    Build a function (question, context) -> {"answer","score","start","end"}
    that calls the hosted Inference API for `model_name`.
    """
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise InferenceAPIError(
            "No Hugging Face token found. Create one at "
            "https://huggingface.co/settings/tokens and set the HF_TOKEN "
            "environment variable, then restart the app."
        )

    client = InferenceClient(model=model_name, token=token, timeout=60)

    def ask(question: str, context: str) -> Dict:
        try:
            out = client.question_answering(
                question=question,
                context=context,
                handle_impossible_answer=True,
                max_answer_len=MAX_ANSWER_TOKENS,
                max_seq_len=MAX_SEQ_LEN_TOKENS,
                doc_stride=DOC_STRIDE_TOKENS,
            )
        except Exception as exc:
            message = str(exc)
            if "401" in message or "Unauthorized" in message:
                raise InferenceAPIError(
                    "Hugging Face rejected the token (401 Unauthorized). Check that "
                    "HF_TOKEN in your .env file is correct and still valid."
                ) from exc
            if "403" in message or "sufficient permissions" in message:
                raise InferenceAPIError(
                    "The token is valid but lacks inference permission (403 Forbidden). "
                    "At https://huggingface.co/settings/tokens create a token of type "
                    "'Read', or tick 'Make calls to Inference Providers' on your "
                    "fine-grained token, then update HF_TOKEN in your .env file."
                ) from exc
            if "429" in message or "rate limit" in message.lower():
                raise InferenceAPIError(
                    "Rate limit reached on the free Inference API tier. Wait a "
                    "moment, or switch the sidebar back to running the model locally."
                ) from exc
            raise InferenceAPIError(f"Inference API call failed: {exc}") from exc
        # The API may return a single result or a list (when top_k > 1).
        if isinstance(out, list):
            out = out[0]
        # A "no answer" result may come back with answer/start/end = None.
        return {"answer": out.answer or "", "score": out.score or 0.0,
                "start": out.start or 0, "end": out.end or 0}

    return ask
