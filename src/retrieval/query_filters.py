from datetime import date

from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
)

from src.agents.schemas import QueryPlan


def build_query_filter(
    plan: QueryPlan,
) -> Filter | None:
    """
    Convert a validated QueryPlan into Qdrant metadata filters.

    Semantic meaning remains in plan.retrieval_query.
    This function only handles structured constraints:

    - ticker symbols
    - filing release dates

    Event category is intentionally not applied as a hard filter
    because event_category is inferred at query time and is not
    currently guaranteed to exist in every indexed chunk payload.
    It remains available to the deterministic reranker.
    """
    if not plan.in_scope:
        raise ValueError(
            "Cannot build a retrieval filter for an out-of-scope query"
        )

    conditions: list[FieldCondition] = []

    if plan.symbols:
        conditions.append(
            FieldCondition(
                key="symbol",
                match=MatchAny(any=plan.symbols),
            )
        )

    date_condition = _build_date_condition(
        start_date=plan.start_date,
        end_date=plan.end_date,
    )

    if date_condition is not None:
        conditions.append(date_condition)

    if not conditions:
        return None

    return Filter(must=conditions)


def _build_date_condition(
    *,
    start_date: date | None,
    end_date: date | None,
) -> FieldCondition | None:
    if start_date is None and end_date is None:
        return None

    return FieldCondition(
        key="release_datetime",
        range=DatetimeRange(
            gte=(
                _start_of_day(start_date)
                if start_date is not None
                else None
            ),
            lte=(
                _end_of_day(end_date)
                if end_date is not None
                else None
            ),
        ),
    )


def _start_of_day(value: date) -> str:
    return f"{value.isoformat()}T00:00:00Z"


def _end_of_day(value: date) -> str:
    return f"{value.isoformat()}T23:59:59.999999Z"