import argparse
import textwrap
from pprint import pprint

from src.agents.query_planner import QueryPlanningAgent
from src.retrieval.coordinator import RetrievalCoordinator


DEFAULT_QUERY = (
    "Compare Tesla and Ford operational updates from 2024."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test initial retrieval, evidence coverage, and "
            "bounded per-symbol retries."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
    )

    return parser.parse_args()


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def preview(
    value: str,
    *,
    limit: int = 450,
) -> str:
    normalized = " ".join(value.split())

    if len(normalized) > limit:
        normalized = normalized[:limit].rstrip() + "..."

    return textwrap.fill(
        normalized,
        width=100,
    )


def main() -> None:
    args = parse_args()
    user_query = args.query.strip()

    planner = QueryPlanningAgent()
    plan = planner.plan(user_query)

    print_section("QUERY PLAN")
    pprint(
        plan.model_dump(mode="json"),
        sort_dicts=False,
    )

    if not plan.in_scope:
        print_section("REQUEST REJECTED")
        print(plan.rejection_reason)
        return

    if plan.clarification_needed:
        print_section("CLARIFICATION REQUIRED")
        print(plan.clarification_question)
        return

    coordinator = RetrievalCoordinator()

    outcome = coordinator.retrieve(
        plan=plan,
        candidate_limit=30,
        result_limit=8,
        retry_candidate_limit=60,
        retry_result_limit=8,
        max_chunks_per_filing=2,
    )

    print_section("COORDINATOR SUMMARY")
    print(f"Attempts: {outcome.attempts}")
    print(
        "Retried symbols: "
        f"{outcome.retried_symbols or []}"
    )

    pprint(
        outcome.coverage.model_dump(mode="json"),
        sort_dicts=False,
    )

    print_section("FINAL EVIDENCE")

    if not outcome.evidence:
        print("No evidence found.")
        return

    for index, item in enumerate(
        outcome.evidence,
        start=1,
    ):
        print(
            f"\n[{index}] {item.symbol} — {item.title}"
        )
        print("-" * 88)
        print(
            f"Released: {item.release_datetime.isoformat()}"
        )
        print(
            f"Accession: {item.accession_number}"
        )
        print(
            f"Chunk: {item.chunk_id}"
        )
        print(
            f"Vector score: {item.vector_score:.4f}"
        )
        print(
            f"Rerank score: {item.rerank_score:.4f}"
        )
        print("\nEvidence:")
        print(preview(item.chunk_text))


if __name__ == "__main__":
    main()