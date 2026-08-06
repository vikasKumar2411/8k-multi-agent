from typing import Any, TypedDict


class BoundedResearchState(
    TypedDict,
    total=False,
):
    run_id: str
    thread_id: str

    user_query: str

    query_plan: dict[str, Any]

    evidence: list[dict[str, Any]]
    coverage: dict[str, Any]
    retrieval_attempts: int
    retried_symbols: list[str]

    analysis: dict[str, Any]
    verification: dict[str, Any]
    final_answer: dict[str, Any]

    current_node: str
    workflow_status: str
    error: str | None