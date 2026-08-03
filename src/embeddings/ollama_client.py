from typing import Sequence

import requests

from src.config.settings import get_settings


class OllamaEmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_embedding_model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": list(texts),
            },
            timeout=120,
        )
        response.raise_for_status()

        payload = response.json()
        embeddings = payload.get("embeddings")

        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings.")

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} embeddings, "
                f"received {len(embeddings)}."
            )

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0]