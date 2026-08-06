from typing import Sequence

import requests

from src.config.settings import get_settings
from src.observability import (
    OperationType,
    observe_operation,
)


class OllamaEmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()

        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_embedding_model

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        normalized_texts = [
            str(text)
            for text in texts
        ]

        # Used only for character/token estimation.
        # Prompt text itself is not logged.
        combined_input = "\n".join(
            normalized_texts
        )

        with observe_operation(
            operation_type=OperationType.EMBEDDING,
            operation_name="ollama_embed_texts",
            provider="ollama",
            model_name=self.model,
            input_text=combined_input,
            retry_count=0,
            attributes={
                "batch_size": len(normalized_texts),
                "endpoint": "/api/embed",
            },
        ) as observation:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": normalized_texts,
                },
                timeout=120,
            )

            observation.set_attribute(
                "http_status_code",
                response.status_code,
            )

            response.raise_for_status()

            payload = response.json()
            embeddings = payload.get("embeddings")

            if not embeddings:
                raise RuntimeError(
                    "Ollama returned no embeddings."
                )

            if len(embeddings) != len(normalized_texts):
                raise RuntimeError(
                    f"Expected {len(normalized_texts)} embeddings, "
                    f"received {len(embeddings)}."
                )

            vector_dimensions = {
                len(vector)
                for vector in embeddings
            }

            if len(vector_dimensions) != 1:
                raise RuntimeError(
                    "Ollama returned embeddings with "
                    "inconsistent dimensions."
                )

            vector_dimension = next(
                iter(vector_dimensions)
            )

            observation.update_attributes(
                {
                    "embedding_count": len(embeddings),
                    "vector_dimension": vector_dimension,
                    "response_validation_success": True,
                }
            )

            return embeddings

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise ValueError(
                "Embedding query cannot be empty."
            )

        embeddings = self.embed_texts(
            [text]
        )

        return embeddings[0]