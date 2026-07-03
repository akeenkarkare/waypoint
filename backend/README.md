# Waypoint — Backend

FastAPI over a [Cognee](https://github.com/topoteretes/cognee) memory graph. Turns
travel memories (photos + captions + notes + voice) into a connected knowledge
graph you can ask questions about.

## Setup

Requires Python 3.10–3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
cd backend
cp .env.example .env        # then paste your OpenAI key into .env
uv sync                     # creates .venv and installs deps
uv run uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API.

## What you need

- **An OpenAI API key** in `.env` (`OPENAI_API_KEY`). One key covers Cognee's
  graph extraction + embeddings, and later the Whisper/vision pipeline.
- (Optional) `ANTHROPIC_API_KEY` to use Claude for story generation.

## Endpoints (v0)

| Method | Path      | Purpose                                    |
|--------|-----------|--------------------------------------------|
| GET    | `/health` | liveness + whether the LLM key is set      |
| POST   | `/entry`  | add one memory, then rebuild the graph     |
| POST   | `/ask`    | natural-language question over the graph   |
| GET    | `/graph`  | nodes/edges for the frontend visualization |

## Layout

```
app/
  main.py            FastAPI app + routes
  cognee_service.py  the add / cognify / search wrapper (the core)
  config.py          env-based settings
data/                local media + Cognee stores (gitignored)
```

## Roadmap

- Media pipeline: photo -> GPT-4o caption, voice -> Whisper, feed text to Cognee
- Story generation over `search` results
- Cross-trip queries (neighborhood / connection-path) once those land upstream
