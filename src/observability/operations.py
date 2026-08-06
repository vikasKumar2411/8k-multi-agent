# src/observability/operations.py

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from src.observability.context import (
    get_node_name,
    get_run_id,
    get_thread_id,
    get_workflow_name,
)
from src.observability.operation_events import (
    OperationEvent,
    OperationEventType,
    OperationType,
)


LOGGER = logging.getLogger(
    "sec_copilot.operation"
)


def estimate_tokens_from_characters(
    character_count: int,
) -> int:
    """
    Lightweight provider-independent token estimate.

    This is intentionally approximate. Provider-reported usage will
    replace it when available.
    """

    if character_count <= 0:
        return 0

    return max(
        1,
        round(character_count / 4),
    )


def emit_operation_event(
    event: OperationEvent,
) -> None:
    LOGGER.info(
        event.event_type.value,
        extra={
            "event": event.model_dump(
                mode="json",
                exclude_none=True,
            )
        },
    )


class OperationObservation:
    """
    Mutable metadata container used inside an observed operation.
    """

    def __init__(self) -> None:
        self.output_character_count: int | None = None
        self.estimated_output_tokens: int | None = None
        self.retry_count: int | None = None
        self.attributes: dict[str, Any] = {}

    def set_output_text(
        self,
        output_text: str | None,
    ) -> None:
        text = output_text or ""

        self.output_character_count = len(text)
        self.estimated_output_tokens = (
            estimate_tokens_from_characters(
                len(text)
            )
        )

    def set_attribute(
        self,
        key: str,
        value: Any,
    ) -> None:
        if value is not None:
            self.attributes[key] = value

    def update_attributes(
        self,
        values: dict[str, Any],
    ) -> None:
        for key, value in values.items():
            self.set_attribute(
                key,
                value,
            )


@contextmanager
def observe_operation(
    *,
    operation_type: OperationType,
    operation_name: str,
    provider: str | None = None,
    model_name: str | None = None,
    input_text: str | None = None,
    retry_count: int | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[OperationObservation]:
    """
    Emit start, completion, and failure events for an operation.

    Large request and response bodies are never logged. Only counts,
    identifiers, and selected metadata are recorded.
    """

    input_character_count = (
        len(input_text)
        if input_text is not None
        else None
    )

    estimated_input_tokens = (
        estimate_tokens_from_characters(
            input_character_count
        )
        if input_character_count is not None
        else None
    )

    observation = OperationObservation()
    observation.retry_count = retry_count

    if attributes:
        observation.update_attributes(
            attributes
        )

    common = {
        "operation_type": operation_type,
        "operation_name": operation_name,
        "run_id": get_run_id(),
        "thread_id": get_thread_id(),
        "workflow_name": get_workflow_name(),
        "node_name": get_node_name(),
        "provider": provider,
        "model_name": model_name,
        "input_character_count": (
            input_character_count
        ),
        "estimated_input_tokens": (
            estimated_input_tokens
        ),
    }

    emit_operation_event(
        OperationEvent(
            event_type=(
                OperationEventType.OPERATION_STARTED
            ),
            status="running",
            attributes=(
                observation.attributes.copy()
            ),
            **common,
        )
    )

    started = time.perf_counter()

    try:
        yield observation

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        emit_operation_event(
            OperationEvent(
                event_type=(
                    OperationEventType
                    .OPERATION_COMPLETED
                ),
                status="completed",
                latency_ms=round(
                    latency_ms,
                    3,
                ),
                output_character_count=(
                    observation
                    .output_character_count
                ),
                estimated_output_tokens=(
                    observation
                    .estimated_output_tokens
                ),
                retry_count=(
                    observation.retry_count
                ),
                attributes=(
                    observation.attributes.copy()
                ),
                **common,
            )
        )

    except Exception as exc:
        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        emit_operation_event(
            OperationEvent(
                event_type=(
                    OperationEventType
                    .OPERATION_FAILED
                ),
                status="failed",
                latency_ms=round(
                    latency_ms,
                    3,
                ),
                retry_count=(
                    observation.retry_count
                ),
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
                attributes=(
                    observation.attributes.copy()
                ),
                **common,
            )
        )

        raise