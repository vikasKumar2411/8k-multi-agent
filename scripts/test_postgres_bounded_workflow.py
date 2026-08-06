# scripts/test_postgres_bounded_workflow.py

import argparse
import pprint
import uuid
from typing import Any

from src.observability import (
    configure_observability_logging,
)
from src.persistence import (
    create_postgres_checkpointer,
)
from src.workflows.bounded_research.graph import (
    build_bounded_research_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded SEC workflow using "
            "PostgreSQL checkpoint persistence."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "What operational metrics did "
            "Tesla report in 2024?"
        ),
    )

    parser.add_argument(
        "--thread-id",
        required=True,
        help=(
            "Stable LangGraph thread ID used to "
            "persist and retrieve checkpoints."
        ),
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional execution ID. A UUID is "
            "generated when omitted."
        ),
    )

    return parser.parse_args()


def build_initial_state(
    *,
    query: str,
    run_id: str,
    thread_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "user_query": query,
        "query_plan": None,
        "evidence": [],
        "coverage": None,
        "retrieval_attempts": 0,
        "retried_symbols": [],
        "analysis": None,
        "verification": None,
        "final_answer": None,
        "current_node": None,
        "workflow_status": "started",
        "error": None,
    }


def main() -> None:
    configure_observability_logging()

    args = parse_args()

    query = args.query.strip()

    if not query:
        raise ValueError(
            "query cannot be empty"
        )

    run_id = (
        args.run_id
        or str(uuid.uuid4())
    )

    thread_id = args.thread_id.strip()

    if not thread_id:
        raise ValueError(
            "thread-id cannot be empty"
        )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = build_initial_state(
        query=query,
        run_id=run_id,
        thread_id=thread_id,
    )

    print("=" * 88)
    print("PERSISTENT WORKFLOW RUN")
    print("=" * 88)
    print(f"Run ID:    {run_id}")
    print(f"Thread ID: {thread_id}")
    print(f"Query:     {query}")

    with create_postgres_checkpointer() as checkpointer:
        graph = build_bounded_research_graph(
            checkpointer=checkpointer
        )

        final_state = graph.invoke(
            initial_state,
            config=config,
        )

        snapshot = graph.get_state(
            config
        )

        print()
        print("=" * 88)
        print("FINAL WORKFLOW STATUS")
        print("=" * 88)
        print(
            final_state.get(
                "workflow_status"
            )
        )

        print()
        print("=" * 88)
        print("FINAL ANSWER")
        print("=" * 88)

        final_answer = (
            final_state.get("final_answer")
            or {}
        )

        print(
            final_answer.get(
                "answer",
                "No answer generated.",
            )
        )

        print()
        print("=" * 88)
        print("PERSISTED CHECKPOINT")
        print("=" * 88)
        print(
            "Checkpoint exists:",
            snapshot is not None,
        )
        print(
            "Next nodes:",
            snapshot.next,
        )
        print(
            "Created at:",
            snapshot.created_at,
        )
        print(
            "Metadata:"
        )
        pprint.pp(snapshot.metadata)


if __name__ == "__main__":
    main()