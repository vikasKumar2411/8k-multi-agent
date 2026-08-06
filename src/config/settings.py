from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"

    ollama_embedding_model: str = "nomic-embed-text"
    ollama_chat_model: str = "qwen2.5:7b-instruct"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sec_8k_chunks_v1"

    query_planner_temperature: float = 0.0
    ollama_timeout_seconds: int = 180

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    langgraph_database_url: str = (
    "postgresql://sec_copilot:sec_copilot"
    "@localhost:5432/sec_copilot"
    "?sslmode=disable"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()