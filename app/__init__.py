"""
PDF Question-Answering System.

Core NLP package. Each module is one step of the pipeline:

    pdf_extractor.py  ->  1. get raw text out of the PDF
    chunker.py        ->  2. split long text into overlapping chunks
    qa_engine.py      ->  3. run the Hugging Face QA model and pick the best answer
    hf_inference_api.py   (optional) same QA step, but on Hugging Face's servers
    hub_info.py           fetch model metadata from the Hugging Face Hub API

Importing this package first loads the project's .env file (if present) and
sets a few environment variables, *before* `transformers` is imported anywhere:

* HF_TOKEN     -> read from .env; only needed for the hosted Inference API.
* HF_HUB_CACHE -> downloaded model weights go into the project's /models folder
  instead of the global ~/.cache/huggingface directory.
* USE_TF=0     -> tells transformers to use PyTorch only and never try to import
  TensorFlow (a broken TensorFlow install elsewhere on a machine can otherwise
  crash the import).
"""

import os
from pathlib import Path

from app.config import MODELS_DIR, PROJECT_ROOT


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """
    Read KEY=value lines from a .env file into environment variables.

    Keeps secrets (like HF_TOKEN) out of the source code, so the file can be
    git-ignored. Blank lines and #comments are skipped, and surrounding quotes
    are removed. A variable that is already set in the real environment wins,
    which lets you override the file for a single run.

    (This is a 10-line version of the popular `python-dotenv` package, so the
    project doesn't need an extra dependency.)
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv()

os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("USE_TF", "0")
# Harmless on Windows (the cache just copies files instead of symlinking them).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
