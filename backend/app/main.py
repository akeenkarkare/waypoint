"""Waypoint backend — FastAPI over the Cognee memory graph.

Endpoints (v0):
    GET  /health        liveness + config sanity
    POST /entry         add one memory (JSON), then cognify
    POST /ask           natural-language question over the memory graph
    GET  /graph         nodes/edges for the frontend visualization

Media pipeline (photo caption / voice transcribe) lands next — kept out of v0
so the memory graph, which is the real differentiator, works end to end first.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import cognee_service
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
