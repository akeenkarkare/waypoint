"""Thin wrapper around Cognee — the memory graph that powers Waypoint.

The whole product rides on three stable calls:
    add()      -> ingest a memory entry as text
    cognify()  -> extract entities/relationships into the knowledge graph
    search()   -> answer natural-language questions over that graph

Everything Waypoint-specific (captions, emotions, people, places) is folded
into the text we hand to add(); Cognee's cognify step turns it into the graph.
"""

import os

from .config import settings

_configured = False


def _ensure_configured() -> None:
    """Point Cognee at our LLM + embedding provider. Idempotent."""
    global _configured
    if _configured:
        return

    key = settings.cognee_llm_key
    if not key:
        raise RuntimeError(
            "No LLM key configured. Set ANTHROPIC_API_KEY in backend/.env"
        )

    # Single-user demo: turn off multi-tenant access control so add/cognify/
    # search don't need an authenticated principal. Must be set before cognee
    # resolves its auth posture.
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    import cognee

    # LLM: Claude Haiku via Anthropic — cheapest capable model for cognify + Q&A.
    cognee.config.set_llm_provider("anthropic")
    cognee.config.set_llm_model(settings.llm_model)
    cognee.config.set_llm_api_key(key)

    # Embeddings: local + free via fastembed. No API key, no per-call cost.
    cognee.config.set_embedding_provider(settings.embedding_provider)
    cognee.config.set_embedding_model(settings.embedding_model)
    cognee.config.set_embedding_dimensions(settings.embedding_dimensions)

    # Keep Cognee's graph/vector/relational stores under backend/data/ instead
    # of the default location inside site-packages (wiped on every reinstall).
    cognee.config.system_root_directory(str(settings.cognee_root / "system"))
    cognee.config.data_root_directory(str(settings.cognee_root / "data"))

    _configured = True


def _memory_to_text(entry: dict) -> str:
    """Flatten a Waypoint entry into the narrative text Cognee ingests.

    We name people / place / activity / emotion explicitly so cognify lifts
    them into the graph as connectable entities rather than loose prose.
    """
    lines: list[str] = []
    if entry.get("trip"):
        lines.append(f"Trip: {entry['trip']}")
    if entry.get("date"):
        lines.append(f"Date: {entry['date']}")
    if entry.get("place"):
        lines.append(f"Place: {entry['place']}")
    if entry.get("people"):
        lines.append(f"People: {', '.join(entry['people'])}")
    if entry.get("activity"):
        lines.append(f"Activity: {entry['activity']}")
    if entry.get("emotion"):
        lines.append(f"Emotion: {entry['emotion']}")
    if entry.get("caption"):
        lines.append(f"Caption: {entry['caption']}")
    if entry.get("thoughts"):
        lines.append(f"Notes: {entry['thoughts']}")
    return "\n".join(lines)


async def add_entry(entry: dict) -> None:
    """Ingest one memory entry. Call cognify_all() afterward to build the graph."""
    _ensure_configured()
    import cognee

    text = _memory_to_text(entry)
    await cognee.add(text, dataset_name=settings.dataset)


async def cognify_all() -> None:
    """Build/update the knowledge graph from everything added so far."""
    _ensure_configured()
    import cognee

    await cognee.cognify(datasets=[settings.dataset])


async def ask(question: str) -> str:
    """Answer a natural-language question over the memory graph."""
    _ensure_configured()
    import cognee
    from cognee import SearchType

    results = await cognee.search(
        query_text=question,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[settings.dataset],
    )
    if isinstance(results, list):
        return "\n".join(str(r) for r in results)
    return str(results)


async def get_graph() -> dict:
    """Return nodes/edges for the frontend visualization (best-effort).

    Uses Cognee's graph engine directly. Shape may need tweaking once we see
    the real payload — kept isolated here so the viz contract is one function.
    """
    _ensure_configured()
    from cognee.infrastructure.databases.graph import get_graph_engine

    engine = await get_graph_engine()
    nodes, edges = await engine.get_graph_data()
    return {
        "nodes": [{"id": str(n[0]), **(n[1] or {})} for n in nodes],
        "edges": [{"source": str(e[0]), "target": str(e[1]), "label": e[2]} for e in edges],
    }
