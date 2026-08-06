import argparse
from pprint import pprint
from uuid import uuid4

from src.agents.answer_schemas import FinalAnswer
from src.workflows.bounded_research.graph import (
    build_bounded_research_graph,
)
from src.observability import (
    configure_observability_logging,
)


DEFAULT_QUERY = (
    "What operational metrics did Tesla report in 2024?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded SEC research workflow through LangGraph."
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
    configure_observability_logging()
    args = parse_args()
    user_query = args.query.strip()

    if not user_query:
        raise ValueError("query cannot be empty")

    run_id = str(uuid4())
    thread_id = str(uuid4())

    graph = build_bounded_research_graph()

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = {
        "run_id": run_id,
        "thread_id": thread_id,
        "user_query": user_query,
        "workflow_status": "started",
        "current_node": "start",
        "error": None,
    }

    print_section("RUN")
    print(f"Run ID:    {run_id}")
    print(f"Thread ID: {thread_id}")
    print(f"Query:     {user_query}")

    print_section("LANGGRAPH NODE UPDATES")

    for update in graph.stream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():
            print(f"\nNODE: {node_name}")
            pprint(
                node_update,
                sort_dicts=False,
            )

    snapshot = graph.get_state(config)
    final_state = snapshot.values

    print_section("FINAL WORKFLOW STATUS")
    print(final_state.get("workflow_status"))

    raw_final_answer = final_state.get("final_answer")

    if raw_final_answer is None:
        print_section("ERROR")
        print(
            final_state.get("error")
            or "Workflow finished without a final answer."
        )
        return

    final_answer = FinalAnswer.model_validate(
        raw_final_answer
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

    print_section("CHECKPOINT STATE")
    print(
        "Current node:",
        final_state.get("current_node"),
    )
    print(
        "Evidence count:",
        len(final_state.get("evidence", [])),
    )
    print(
        "Retrieval attempts:",
        final_state.get("retrieval_attempts", 0),
    )
    print(
        "Retried symbols:",
        final_state.get("retried_symbols", []),
    )


if __name__ == "__main__":
    main()