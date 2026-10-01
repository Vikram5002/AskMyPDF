"""
Look up model metadata on the Hugging Face Hub.

Before loading a model we can ask the Hub API whether it exists and what it
is: its task ("pipeline_tag"), download count, likes, languages, etc. The UI
shows this in the sidebar, and running this file prints it:

    python -m app.hub_info
"""

from dataclasses import dataclass
from typing import List, Optional

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from app.config import AVAILABLE_MODELS


@dataclass
class ModelInfo:
    model_id: str
    exists: bool
    task: Optional[str] = None        # e.g. "question-answering"
    downloads: Optional[int] = None   # downloads over the last 30 days
    likes: Optional[int] = None
    languages: Optional[List[str]] = None
    license: Optional[str] = None
    error: Optional[str] = None


def get_model_info(model_id: str) -> ModelInfo:
    """Fetch metadata for `model_id` from the Hub. Never raises."""
    try:
        info = HfApi().model_info(model_id)
        # Reading the model card must stay inside the try: a malformed card
        # would otherwise raise out of this function, which promises not to.
        card = info.card_data.to_dict() if info.card_data else {}
        languages = card.get("language")
        if isinstance(languages, str):
            languages = [languages]
    except RepositoryNotFoundError:
        return ModelInfo(model_id, exists=False, error="Model not found on the Hub.")
    except Exception as exc:  # offline, rate-limited, malformed card, ...
        return ModelInfo(model_id, exists=False, error=f"Hub API unavailable: {exc}")
    return ModelInfo(
        model_id=model_id,
        exists=True,
        task=info.pipeline_tag,
        downloads=info.downloads,
        likes=info.likes,
        languages=languages,
        license=card.get("license"),
    )


if __name__ == "__main__":
    for label, model_id in AVAILABLE_MODELS.items():
        m = get_model_info(model_id)
        print(f"\n{label}: {model_id}")
        if not m.exists:
            print(f"  ERROR: {m.error}")
            continue
        fmt = lambda n: f"{n:,}" if isinstance(n, int) else "-"   # noqa: E731
        print(f"  task      : {m.task}")
        print(f"  downloads : {fmt(m.downloads)} (last 30 days)")
        print(f"  likes     : {fmt(m.likes)}")
        print(f"  languages : {', '.join(m.languages or ['-'])}")
        print(f"  license   : {m.license}")
