import json
from typing import Any

import requests
from pydantic import BaseModel

from src.config.settings import get_settings


class OllamaChatClient:
    def __init__(self) -> None:
        settings = get_settings()

        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_chat_model
        self.timeout_seconds = settings.ollama_timeout_seconds

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        schema = response_model.model_json_schema()

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": schema,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=self.timeout_seconds,
        )

        response.raise_for_status()

        payload = response.json()

        message = payload.get("message") or {}
        content = message.get("content")

        if not content:
            raise RuntimeError(
                "Ollama returned an empty chat response"
            )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama response was not valid JSON. "
                f"Response: {content[:500]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(
                "Expected Ollama to return a JSON object"
            )

        return parsed