# src/observability/operation_events.py

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OperationType(StrEnum):
    LLM = "llm"
    EMBEDDING = "embedding"
    VECTOR_SEARCH = "vector_search"
    RERANK = "rerank"
    COVERAGE = "coverage"
    VALIDATION = "validation"
    REPAIR = "repair"
    DATABASE = "database"
    ARTIFACT = "artifact"
    OTHER = "other"


class OperationEventType(StrEnum):
    OPERATION_STARTED = "operation_started"
    OPERATION_COMPLETED = "operation_completed"
    OPERATION_FAILED = "operation_failed"


class OperationEvent(BaseModel):
    """
    Vendor-neutral telemetry event for an internal operation.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    event_type: OperationEventType
    operation_type: OperationType

    operation_name: str = Field(min_length=1)

    run_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    workflow_name: str = Field(min_length=1)
    node_name: str = Field(min_length=1)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    status: str

    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
    )

    provider: str | None = None
    model_name: str | None = None

    input_character_count: int | None = Field(
        default=None,
        ge=0,
    )
    output_character_count: int | None = Field(
        default=None,
        ge=0,
    )

    estimated_input_tokens: int | None = Field(
        default=None,
        ge=0,
    )
    estimated_output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    retry_count: int | None = Field(
        default=None,
        ge=0,
    )

    error_type: str | None = None
    error_message: str | None = None

    attributes: dict[str, Any] = Field(
        default_factory=dict
    )