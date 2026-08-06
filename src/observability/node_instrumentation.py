# src/observability/node_instrumentation.py

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.observability.context import (
    bind_execution_context,
)
from src.observability.events import (
    ExecutionEvent,
    ExecutionEventType,
)


LOGGER = logging.getLogger(
    "sec_copilot.execution"
)

WORKFLOW_NAME = "bounded_sec_research"


NodeFunction = Callable[
    [dict[str, Any]],
    dict[str, Any],
]


def _safe_length(value: Any) -> int:
    if isinstance(
        value,
        (list, tuple, set, dict),
    ):
        return len(value)

    return 0


def _build_output_attributes(
    *,
    node_name: str,
    output: dict[str, Any],
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}

    if node_name == "plan_query":
        plan = output.get(
            "query_plan"
        ) or {}

        if isinstance(plan, dict):
            attributes.update(
                {
                    "query_in_scope": (
                        plan.get("in_scope")
                    ),
                    "clarification_needed": (
                        plan.get(
                            "clarification_needed"
                        )
                    ),
                    "symbol_count": _safe_length(
                        plan.get("symbols")
                    ),
                    "event_category": (
                        plan.get(
                            "event_category"
                        )
                    ),
                    "task_type": plan.get(
                        "task_type"
                    ),
                }
            )

    elif node_name == "retrieve_evidence":
        evidence = (
            output.get("evidence")
            or []
        )

        coverage = (
            output.get("coverage")
            or {}
        )

        attributes.update(
            {
                "evidence_count": (
                    _safe_length(evidence)
                ),
                "retrieval_attempts": (
                    output.get(
                        "retrieval_attempts",
                        0,
                    )
                ),
                "retried_symbol_count": (
                    _safe_length(
                        output.get(
                            "retried_symbols"
                        )
                    )
                ),
            }
        )

        if isinstance(coverage, dict):
            attributes.update(
                {
                    "filing_count": (
                        coverage.get(
                            "filing_count"
                        )
                    ),
                    "coverage_complete": (
                        coverage.get(
                            "complete"
                        )
                    ),
                    "missing_symbol_count": (
                        _safe_length(
                            coverage.get(
                                "missing_symbols"
                            )
                        )
                    ),
                }
            )

    elif node_name == "analyze_evidence":
        analysis = (
            output.get("analysis")
            or {}
        )

        if isinstance(analysis, dict):
            attributes.update(
                {
                    "finding_count": (
                        _safe_length(
                            analysis.get(
                                "findings"
                            )
                        )
                    ),
                    "analysis_complete": (
                        analysis.get(
                            "analysis_complete"
                        )
                    ),
                    "analyzed_evidence_count": (
                        _safe_length(
                            analysis.get(
                                "analyzed_evidence_ids"
                            )
                        )
                    ),
                    "unused_evidence_count": (
                        _safe_length(
                            analysis.get(
                                "unused_evidence_ids"
                            )
                        )
                    ),
                }
            )

    elif node_name == "verify_analysis":
        verification = (
            output.get("verification")
            or {}
        )

        if isinstance(
            verification,
            dict,
        ):
            summary = (
                verification.get(
                    "summary"
                )
                or {}
            )

            attributes.update(
                {
                    "answerable": (
                        verification.get(
                            "answerable"
                        )
                    ),
                    "verification_complete": (
                        verification.get(
                            "verification_complete"
                        )
                    ),
                }
            )

            if isinstance(summary, dict):
                attributes.update(
                    {
                        "total_claims": (
                            summary.get(
                                "total_claims"
                            )
                        ),
                        "verified_claims": (
                            summary.get(
                                "verified_claims"
                            )
                        ),
                        "unsupported_claims": (
                            summary.get(
                                "unsupported_claims"
                            )
                        ),
                        "contradicted_claims": (
                            summary.get(
                                "contradicted_claims"
                            )
                        ),
                        "verified_ratio": (
                            summary.get(
                                "verified_ratio"
                            )
                        ),
                    }
                )

    elif node_name == "generate_answer":
        final_answer = (
            output.get("final_answer")
            or {}
        )

        if isinstance(
            final_answer,
            dict,
        ):
            answer_text = str(
                final_answer.get(
                    "answer"
                )
                or ""
            )

            attributes.update(
                {
                    "answerable": (
                        final_answer.get(
                            "answerable"
                        )
                    ),
                    "verification_complete": (
                        final_answer.get(
                            "verification_complete"
                        )
                    ),
                    "citation_count": (
                        _safe_length(
                            final_answer.get(
                                "citations"
                            )
                        )
                    ),
                    "warning_count": (
                        _safe_length(
                            final_answer.get(
                                "warnings"
                            )
                        )
                    ),
                    "answer_character_count": (
                        len(answer_text)
                    ),
                }
            )

    return {
        key: value
        for key, value in attributes.items()
        if value is not None
    }


def _emit_event(
    event: ExecutionEvent,
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


def instrument_node(
    node_name: str,
    node_function: NodeFunction,
) -> NodeFunction:
    @wraps(node_function)
    def wrapped(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        run_id = str(
            state.get("run_id")
            or "unknown-run"
        )

        thread_id = str(
            state.get("thread_id")
            or "unknown-thread"
        )

        started = time.perf_counter()

        _emit_event(
            ExecutionEvent(
                event_type=(
                    ExecutionEventType
                    .NODE_STARTED
                ),
                run_id=run_id,
                thread_id=thread_id,
                workflow_name=(
                    WORKFLOW_NAME
                ),
                node_name=node_name,
                status="running",
                workflow_status=(
                    state.get(
                        "workflow_status"
                    )
                ),
                attributes={},
            )
        )

        try:
            with bind_execution_context(
                run_id=run_id,
                thread_id=thread_id,
                workflow_name=(
                    WORKFLOW_NAME
                ),
                node_name=node_name,
            ):
                output = node_function(
                    state
                )

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000

            output_attributes = (
                _build_output_attributes(
                    node_name=node_name,
                    output=output,
                )
            )

            _emit_event(
                ExecutionEvent(
                    event_type=(
                        ExecutionEventType
                        .NODE_COMPLETED
                    ),
                    run_id=run_id,
                    thread_id=thread_id,
                    workflow_name=(
                        WORKFLOW_NAME
                    ),
                    node_name=node_name,
                    status="completed",
                    workflow_status=(
                        output.get(
                            "workflow_status"
                        )
                    ),
                    latency_ms=round(
                        latency_ms,
                        3,
                    ),
                    attributes=(
                        output_attributes
                    ),
                )
            )

            return output

        except Exception as exc:
            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000

            _emit_event(
                ExecutionEvent(
                    event_type=(
                        ExecutionEventType
                        .NODE_FAILED
                    ),
                    run_id=run_id,
                    thread_id=thread_id,
                    workflow_name=(
                        WORKFLOW_NAME
                    ),
                    node_name=node_name,
                    status="failed",
                    workflow_status=(
                        state.get(
                            "workflow_status"
                        )
                    ),
                    latency_ms=round(
                        latency_ms,
                        3,
                    ),
                    error_type=(
                        type(exc).__name__
                    ),
                    error_message=(
                        str(exc)[:1000]
                    ),
                    attributes={},
                )
            )

            LOGGER.exception(
                "Graph node execution failed",
                extra={
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "workflow_name": (
                        WORKFLOW_NAME
                    ),
                    "node_name": node_name,
                },
            )

            raise

    return wrapped