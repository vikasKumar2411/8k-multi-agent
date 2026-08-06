# scripts/inspect_postgres_thread.py

import argparse
import pprint

from src.persistence import (
    create_postgres_checkpointer,
)
from src.workflows.bounded_research.graph import (
    build_bounded_research_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a persisted LangGraph thread."
        )
    )

    parser.add_argument(
        "--thread-id",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = {
        "configurable": {
            "thread_id": args.thread_id,
        }
    }

    with create_postgres_checkpointer() as checkpointer:
        graph = build_bounded_research_graph(
            checkpointer=checkpointer
        )

        snapshot = graph.get_state(
            config
        )

        print("=" * 88)
        print("PERSISTED THREAD")
        print("=" * 88)
        print(
            "Thread ID:",
            args.thread_id,
        )
        print(
            "Created at:",
            snapshot.created_at,
        )
        print(
            "Next nodes:",
            snapshot.next,
        )
        print(
            "Workflow status:",
            snapshot.values.get(
                "workflow_status"
            ),
        )
        print(
            "Current node:",
            snapshot.values.get(
                "current_node"
            ),
        )

        final_answer = (
            snapshot.values.get(
                "final_answer"
            )
            or {}
        )

        print()
        print("Final answer:")
        print(
            final_answer.get(
                "answer",
                "No persisted answer.",
            )
        )

        print()
        print("Checkpoint metadata:")
        pprint.pp(snapshot.metadata)


if __name__ == "__main__":
    main()