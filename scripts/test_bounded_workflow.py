import argparse
from pprint import pprint

from src.agents.answer_generator import AnswerGenerator
from src.agents.evidence_analyzer import (
    EvidenceAnalysisAgent,
)
from src.agents.query_planner import QueryPlanningAgent
from src.agents.verifier import VerificationAgent
from src.retrieval.coordinator import RetrievalCoordinator


DEFAULT_QUERY = (
    "What operational metrics did Tesla report in 2024?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete bounded SEC research workflow."
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


def main() -> None:
    args = parse_args()
    user_query = args.query.strip()

    planner = QueryPlanningAgent()
    plan = planner.plan(user_query)

    print_section("QUERY PLAN")
    pprint(
        plan.model_dump(
            mode="json",
            exclude_none=False,
        ),
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

    retrieval_outcome = coordinator.retrieve(
        plan=plan,
        candidate_limit=30,
        result_limit=8,
        retry_candidate_limit=60,
        retry_result_limit=8,
        max_chunks_per_filing=2,
    )

    print_section("COVERAGE")
    pprint(
        retrieval_outcome.coverage.model_dump(
            mode="json",
            exclude_none=False,
        ),
        sort_dicts=False,
    )

    analyzer = EvidenceAnalysisAgent()

    analysis = analyzer.analyze(
        user_query=user_query,
        plan=plan,
        evidence=retrieval_outcome.evidence,
        coverage=retrieval_outcome.coverage,
    )

    print_section("ANALYSIS")
    pprint(
        analysis.model_dump(
            mode="json",
            exclude_none=False,
        ),
        sort_dicts=False,
    )

    verifier = VerificationAgent()

    verification = verifier.verify(
        user_query=user_query,
        plan=plan,
        analysis=analysis,
        evidence=retrieval_outcome.evidence,
        coverage=retrieval_outcome.coverage,
    )

    print_section("VERIFICATION")
    pprint(
        verification.model_dump(
            mode="json",
            exclude_none=False,
        ),
        sort_dicts=False,
    )

    answer_generator = AnswerGenerator()

    final_answer = answer_generator.generate(
        user_query=user_query,
        plan=plan,
        analysis=analysis,
        verification=verification,
        coverage=retrieval_outcome.coverage,
        evidence=retrieval_outcome.evidence,
    )

    print_section("FINAL ANSWER")
    print(final_answer.answer)

    print_section("FINAL ANSWER METADATA")
    pprint(
        final_answer.model_dump(
            mode="json",
            exclude_none=False,
        ),
        sort_dicts=False,
    )


if __name__ == "__main__":
    main()