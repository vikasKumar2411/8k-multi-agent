from datetime import date
from pprint import pprint

from src.agents.schemas import (
    EventCategory,
    QueryPlan,
    TaskType,
)
from src.retrieval.query_filters import build_query_filter


def print_filter(
    label: str,
    plan: QueryPlan,
) -> None:
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    query_filter = build_query_filter(plan)

    if query_filter is None:
        print("No metadata filter required.")
        return

    pprint(
        query_filter.model_dump(
            mode="json",
            exclude_none=True,
        )
    )


def main() -> None:
    symbol_and_date_plan = QueryPlan(
        in_scope=True,
        rejection_reason=None,
        symbols=["TSLA", "F"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        event_category=EventCategory.OPERATIONAL,
        task_type=TaskType.COMPANY_COMPARISON,
        retrieval_query=(
            "production deliveries manufacturing shipments "
            "operational results"
        ),
        clarification_needed=False,
        clarification_question=None,
    )

    date_only_plan = QueryPlan(
        in_scope=True,
        rejection_reason=None,
        symbols=[],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        event_category=EventCategory.CYBERSECURITY,
        task_type=TaskType.EVENT_SEARCH,
        retrieval_query=(
            "cybersecurity incident data breach unauthorized access"
        ),
        clarification_needed=False,
        clarification_question=None,
    )

    symbol_only_plan = QueryPlan(
        in_scope=True,
        rejection_reason=None,
        symbols=["AAPL"],
        start_date=None,
        end_date=None,
        event_category=EventCategory.LEADERSHIP,
        task_type=TaskType.EVENT_SEARCH,
        retrieval_query=(
            "executive resignation appointment retirement "
            "leadership transition"
        ),
        clarification_needed=False,
        clarification_question=None,
    )

    unrestricted_plan = QueryPlan(
        in_scope=True,
        rejection_reason=None,
        symbols=[],
        start_date=None,
        end_date=None,
        event_category=EventCategory.LEADERSHIP,
        task_type=TaskType.EVENT_SEARCH,
        retrieval_query=(
            "executive resignation appointment retirement "
            "leadership transition"
        ),
        clarification_needed=False,
        clarification_question=None,
    )

    print_filter(
        "Symbol and date filter",
        symbol_and_date_plan,
    )
    print_filter(
        "Date-only filter",
        date_only_plan,
    )
    print_filter(
        "Symbol-only filter",
        symbol_only_plan,
    )
    print_filter(
        "No structured constraints",
        unrestricted_plan,
    )


if __name__ == "__main__":
    main()