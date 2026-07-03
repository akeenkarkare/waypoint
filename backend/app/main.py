"""Waypoint backend — FastAPI over the Cognee memory graph.

Endpoints (v0):
    GET  /health        liveness + config sanity
    POST /entry         add one memory (JSON), then cognify
    POST /ask           natural-language question over the memory graph
    GET  /graph         nodes/edges for the frontend visualization
    POST /entry/media   add one memory with a photo and/or voice note (multipart)
"""

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import cognee_service, media
from .config import settings

app = FastAPI(title="Waypoint API", version="0.1.0")

# Open CORS for hackathon dev (Expo tunnel / web preview hit this from anywhere).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MemoryEntry(BaseModel):
    trip: str | None = None
    date: str | None = None
    place: str | None = None
    people: list[str] = []
    activity: str | None = None
    emotion: str | None = None
    caption: str | None = None
    thoughts: str | None = None


class Question(BaseModel):
    question: str


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "llm_key_set": bool(settings.cognee_llm_key),
        "llm_model": settings.llm_model,
        "dataset": settings.dataset,
    }


@app.post("/entry")
async def create_entry(entry: MemoryEntry) -> dict:
    try:
        await cognee_service.add_entry(entry.model_dump())
        await cognee_service.cognify_all()
    except Exception as exc:  # surface config/backend errors clearly for now
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "stored"}


async def _save_upload(upload: UploadFile) -> Path:
    """Persist an uploaded file under the media dir with a unique name."""
    suffix = Path(upload.filename or "").suffix.lower()
    dest = settings.media_dir / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(await upload.read())
    return dest


@app.post("/entry/media")
async def create_entry_media(
    trip: str | None = Form(None),
    date: str | None = Form(None),
    place: str | None = Form(None),
    people: str | None = Form(None),  # comma-separated
    activity: str | None = Form(None),
    emotion: str | None = Form(None),
    thoughts: str | None = Form(None),
    caption: str | None = Form(None),
    photo: UploadFile | None = File(None),
    voice: UploadFile | None = File(None),
) -> dict:
    """Add one memory from a photo and/or voice note plus optional typed fields.

    Photo -> Claude Haiku vision caption. Voice -> local Whisper transcript.
    Both become text that Cognee extracts into the memory graph.
    """
    photo_caption = None
    transcript = None
    photo_file = None
    try:
        if photo is not None:
            photo_path = await _save_upload(photo)
            photo_file = photo_path.name
            # Blocking (network / CPU) — keep the event loop free.
            photo_caption = await asyncio.to_thread(media.caption_photo, photo_path)

        if voice is not None:
            voice_path = await _save_upload(voice)
            transcript = await asyncio.to_thread(media.transcribe_voice, voice_path)

        # Merge auto-derived text with anything the user typed.
        merged_caption = " ".join(c for c in (caption, photo_caption) if c) or None
        merged_thoughts = "\n".join(t for t in (thoughts, transcript) if t) or None

        entry = {
            "trip": trip,
            "date": date,
            "place": place,
            "people": [p.strip() for p in people.split(",")] if people else [],
            "activity": activity,
            "emotion": emotion,
            "caption": merged_caption,
            "thoughts": merged_thoughts,
        }
        await cognee_service.add_entry(entry)
        await cognee_service.cognify_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "stored",
        "photo_file": photo_file,
        "photo_caption": photo_caption,
        "transcript": transcript,
    }


@app.post("/ask")
async def ask(q: Question) -> dict:
    try:
        answer = await cognee_service.ask(q.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"question": q.question, "answer": answer}


@app.get("/graph")
async def graph() -> dict:
    try:
        return await cognee_service.get_graph()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
