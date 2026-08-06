from typing import Literal

from src.agents.schemas import QueryPlan
from src.workflows.bounded_research.state import (
    BoundedResearchState,
)


PlannerRoute = Literal[
    "research",
    "reject",
    "clarify",
]


def route_after_planning(
    state: BoundedResearchState,
) -> PlannerRoute:
    raw_plan = state.get(
        "query_plan"
    )

    if raw_plan is None:
        raise ValueError(
            "query_plan is required "
            "before planner routing"
        )

    plan = QueryPlan.model_validate(
        raw_plan
    )

    if not plan.in_scope:
        return "reject"

    if plan.clarification_needed:
        return "clarify"

    return "research"