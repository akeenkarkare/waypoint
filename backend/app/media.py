"""Media pipeline: turn photos and voice notes into text for the memory graph.

Neither the image nor the audio is what Cognee reasons over — the *text* is.
So this module's whole job is: photo -> caption text (Claude Haiku vision),
voice -> transcript text (local Whisper, free). That text is folded into the
memory entry and handed to cognee_service.add_entry().
"""

import base64
from pathlib import Path

from .config import settings

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_CAPTION_PROMPT = (
    "You are labeling a travel photo for a personal memory journal. "
    "In 1-3 sentences, describe what is actually visible: the place or setting, "
    "any people, the activity, and the mood or atmosphere. Name concrete things "
    "(landmarks, objects, food) when you can see them. Do not guess names of people "
    "or invent details. Write plainly, no preamble."
)

_whisper_model = None


def caption_photo(image_path: Path) -> str:
    """Describe a photo as memory-graph-friendly text using Claude Haiku vision."""
    suffix = image_path.suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image type: {suffix}")

    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    },
                    {"type": "text", "text": _CAPTION_PROMPT},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _get_whisper():
    """Load the local Whisper model once. Small model = fast, free, good enough."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        # int8 on CPU keeps it light; "base" balances speed and accuracy for notes.
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_voice(audio_path: Path) -> str:
    """Transcribe a voice note to text locally (no API, no cost)."""
    model = _get_whisper()
    segments, _ = model.transcribe(str(audio_path))
    return " ".join(segment.text.strip() for segment in segments).strip()
