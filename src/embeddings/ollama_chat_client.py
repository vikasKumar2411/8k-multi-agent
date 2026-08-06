import json
from typing import Any

import requests
from pydantic import BaseModel

from src.config.settings import get_settings
from src.observability import (
    OperationType,
    observe_operation,
)


class OllamaChatClient:
    def __init__(self) -> None:
        settings = get_settings()

        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_chat_model
        self.timeout_seconds = (
            settings.ollama_timeout_seconds
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        schema = response_model.model_json_schema()

        # This is used only to calculate prompt-size telemetry.
        # Prompt content is not written into logs.
        combined_prompt = (
            f"{system_prompt}\n\n{user_prompt}"
        )

        with observe_operation(
            operation_type=OperationType.LLM,
            operation_name=(
                "ollama_structured_generation"
            ),
            provider="ollama",
            model_name=self.model,
            input_text=combined_prompt,
            retry_count=0,
            attributes={
                "endpoint": "/api/chat",
                "temperature": temperature,
                "stream": False,
                "structured_output": True,
                "response_schema": (
                    response_model.__name__
                ),
                "system_prompt_character_count": (
                    len(system_prompt)
                ),
                "user_prompt_character_count": (
                    len(user_prompt)
                ),
            },
        ) as observation:
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

            observation.set_attribute(
                "http_status_code",
                response.status_code,
            )

            response.raise_for_status()

            payload = response.json()

            message = payload.get("message") or {}
            content = message.get("content")

            if not content:
                observation.set_attribute(
                    "empty_response",
                    True,
                )

                raise RuntimeError(
                    "Ollama returned an empty chat response"
                )

            # Capture output size before attempting JSON parsing.
            # This ensures malformed responses still have useful
            # failure telemetry.
            observation.set_output_text(
                content
            )

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                observation.set_attribute(
                    "json_parse_success",
                    False,
                )

                raise RuntimeError(
                    "Ollama response was not valid JSON. "
                    f"Response: {content[:500]}"
                ) from exc

            observation.set_attribute(
                "json_parse_success",
                True,
            )

            if not isinstance(parsed, dict):
                observation.set_attribute(
                    "response_is_object",
                    False,
                )

                raise RuntimeError(
                    "Expected Ollama to return a JSON object"
                )

            observation.update_attributes(
                {
                    "response_is_object": True,
                    "response_field_count": len(parsed),
                    "response_received": True,
                }
            )

            # Ollama may return these fields depending on version.
            # Record them when available without depending on them.
            for source_key, telemetry_key in (
                (
                    "prompt_eval_count",
                    "provider_input_tokens",
                ),
                (
                    "eval_count",
                    "provider_output_tokens",
                ),
                (
                    "total_duration",
                    "provider_total_duration_ns",
                ),
                (
                    "load_duration",
                    "provider_load_duration_ns",
                ),
                (
                    "prompt_eval_duration",
                    "provider_prompt_eval_duration_ns",
                ),
                (
                    "eval_duration",
                    "provider_generation_duration_ns",
                ),
            ):
                value = payload.get(source_key)

                if value is not None:
                    observation.set_attribute(
                        telemetry_key,
                        value,
                    )

            return parsed