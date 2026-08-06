# src/observability/events.py

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ExecutionEventType(StrEnum):
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"


class ExecutionEvent(BaseModel):
    """
    Vendor-neutral event emitted for one graph-node transition.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    event_type: ExecutionEventType

    run_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)

    workflow_name: str = Field(min_length=1)
    node_name: str = Field(min_length=1)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
    )

    status: str

    workflow_status: str | None = None

    error_type: str | None = None
    error_message: str | None = None

    attributes: dict[str, Any] = Field(
        default_factory=dict
    )