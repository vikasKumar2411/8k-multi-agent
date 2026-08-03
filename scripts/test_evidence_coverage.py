from datetime import datetime, timezone
from pprint import pprint

from src.agents.schemas import (
    EventCategory,
    QueryPlan,
    TaskType,
)
from src.retrieval.coverage import (
    validate_evidence_coverage,
)
from src.retrieval.schemas import EvidenceItem


def make_evidence(
    *,
    evidence_id: str,
    symbol: str,
    accession_number: str,
    chunk_id: int,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        point_id=f"point-{evidence_id}",
        accession_number=accession_number,
        chunk_id=chunk_id,
        symbol=symbol,
        company_name=(
            "Tesla, Inc."
            if symbol == "TSLA"
            else "Ford Motor Company"
        ),
        title="Operational update",
        release_datetime=datetime(
            2024,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        filing_type="8-K",
        excerpt="Example filing excerpt.",
        chunk_text=(
            "The company reported production and delivery results."
        ),
        keywords=["production", "deliveries"],
        chunk_item_numbers=[],
        filing_item_numbers=["2.02", "9.01"],
        vector_score=0.58,
        rerank_score=0.84,
    )


def build_comparison_plan() -> QueryPlan:
    return QueryPlan(
        in_scope=True,
        rejection_reason=None,
        symbols=["F", "TSLA"],
        start_date=None,
        end_date=None,
        event_category=EventCategory.OPERATIONAL,
        task_type=TaskType.COMPANY_COMPARISON,
        retrieval_query=(
            "production deliveries manufacturing shipments "
            "operational results"
        ),
        clarification_needed=False,
        clarification_question=None,
    )


def print_case(
    title: str,
    evidence: list[EvidenceItem],
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    coverage = validate_evidence_coverage(
        plan=build_comparison_plan(),
        evidence=evidence,
    )

    pprint(
        coverage.model_dump(
            mode="json",
        ),
        sort_dicts=False,
    )


def main() -> None:
    tesla_evidence = make_evidence(
        evidence_id="evidence-1",
        symbol="TSLA",
        accession_number="tesla-accession",
        chunk_id=2,
    )

    ford_evidence = make_evidence(
        evidence_id="evidence-2",
        symbol="F",
        accession_number="ford-accession",
        chunk_id=1,
    )

    print_case(
        "Incomplete comparison coverage",
        [tesla_evidence],
    )

    print_case(
        "Complete comparison coverage",
        [tesla_evidence, ford_evidence],
    )

    print_case(
        "No evidence",
        [],
    )


if __name__ == "__main__":
    main()