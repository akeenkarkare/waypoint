"""Runtime configuration, loaded from environment / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM: Claude Haiku via Anthropic (cheapest capable model).
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5"

    # Embeddings: local + free via fastembed. No API key, no cost.
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    media_dir: Path = _BACKEND_ROOT / "data" / "media"
    cognee_root: Path = _BACKEND_ROOT / "data" / "cognee"

    # Single-user demo: everything lands in one dataset.
    dataset: str = "waypoint_memories"

    @property
    def cognee_llm_key(self) -> str:
        return self.anthropic_api_key


settings = Settings()
settings.media_dir.mkdir(parents=True, exist_ok=True)
