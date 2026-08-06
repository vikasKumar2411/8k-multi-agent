# src/observability/logging.py

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any


STANDARD_LOG_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """
    Format application logs as one JSON object per line.

    This format is suitable for:
    - local inspection
    - Docker logs
    - CloudWatch Logs
    - OpenTelemetry log pipelines
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in STANDARD_LOG_RECORD_FIELDS:
                continue

            if key.startswith("_"):
                continue

            payload[key] = self._make_json_safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, dict):
            return {
                str(key): JsonFormatter._make_json_safe(
                    nested_value
                )
                for key, nested_value in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                JsonFormatter._make_json_safe(item)
                for item in value
            ]

        return str(value)


def configure_observability_logging(
    *,
    level: str | None = None,
) -> None:
    """
    Configure root logging once for the application.

    LOG_LEVEL can be set to DEBUG, INFO, WARNING, or ERROR.
    """

    configured_level = (
        level
        or os.getenv("LOG_LEVEL", "INFO")
    ).upper()

    numeric_level = getattr(
        logging,
        configured_level,
        logging.INFO,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for existing_handler in list(
        root_logger.handlers
    ):
        root_logger.removeHandler(
            existing_handler
        )

    handler = logging.StreamHandler(
        sys.stdout
    )
    handler.setFormatter(
        JsonFormatter()
    )

    root_logger.addHandler(handler)