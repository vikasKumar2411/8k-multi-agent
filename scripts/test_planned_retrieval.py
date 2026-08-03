import argparse
import textwrap
from pprint import pprint
from typing import Any

from src.agents.query_planner import QueryPlanningAgent
from src.embeddings.ollama_client import (
    OllamaEmbeddingClient,
)
from src.retrieval.qdrant_store import SECQdrantStore
from src.retrieval.query_filters import build_query_filter


DEFAULT_QUERY = (
    "Compare Tesla and Ford operational updates from 2024."
)

CANDIDATE_LIMIT = 30
RESULT_LIMIT = 8
EVIDENCE_PREVIEW_LENGTH = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run query planning, filtered Qdrant retrieval, "
            "and deterministic reranking."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Natural-language SEC 8-K research question.",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=CANDIDATE_LIMIT,
        help="Number of vector candidates requested from Qdrant.",
    )

    parser.add_argument(
        "--result-limit",
        type=int,
        default=RESULT_LIMIT,
        help="Number of reranked results to display.",
    )

    return parser.parse_args()


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def format_preview(
    text: str,
    *,
    limit: int = EVIDENCE_PREVIEW_LENGTH,
) -> str:
    normalized = " ".join(text.split())

    if len(normalized) > limit:
        normalized = normalized[:limit].rstrip() + "..."

    return textwrap.fill(
        normalized,
        width=100,
    )


def print_plan(plan: Any) -> None:
    print_section("1. QUERY PLAN")

    pprint(
        plan.model_dump(
            mode="json",
            exclude_none=False,
        ),
        sort_dicts=False,
    )


def print_filter(query_filter: Any) -> None:
    print_section("2. QDRANT FILTER")

    if query_filter is None:
        print("No metadata filter required.")
        return

    pprint(
        query_filter.model_dump(
            mode="json",
            exclude_none=True,
        ),
        sort_dicts=False,
    )


def print_results(
    results: list[dict[str, Any]],
) -> None:
    print_section("3. RERANKED EVIDENCE")

    if not results:
        print(
            "No results were returned. Check whether the indexed "
            "sample contains filings matching the requested "
            "symbols and date range."
        )
        return

    for rank, result in enumerate(results, start=1):
        payload = result.get("payload") or {}

        vector_score = float(
            result.get(
                "vector_score",
                result.get("score", 0.0),
            )
        )
        rerank_score = float(
            result.get(
                "rerank_score",
                vector_score,
            )
        )

        print(f"\n[{rank}] {payload.get('title', 'Untitled filing')}")
        print("-" * 88)

        print(
            "Company: "
            f"{payload.get('company_name', 'Unknown')} "
            f"({payload.get('symbol', 'N/A')})"
        )
        print(
            "Released: "
            f"{payload.get('release_datetime', 'Unknown')}"
        )
        print(
            "Accession: "
            f"{payload.get('accession_number', 'Unknown')}"
        )
        print(
            "Chunk: "
            f"{payload.get('chunk_id', 'Unknown')}"
        )
        print(
            "Chunk SEC items: "
            f"{payload.get('chunk_item_numbers') or []}"
        )
        print(
            "Filing SEC items: "
            f"{payload.get('filing_item_numbers') or []}"
        )
        print(f"Vector score: {vector_score:.4f}")
        print(f"Rerank score: {rerank_score:.4f}")

        excerpt = str(payload.get("excerpt") or "").strip()

        if excerpt:
            print("\nFiling excerpt:")
            print(format_preview(excerpt, limit=350))

        chunk_text = str(
            payload.get("chunk_text") or ""
        ).strip()

        print("\nEvidence chunk:")
        print(format_preview(chunk_text))


def main() -> None:
    args = parse_args()
    user_query = args.query.strip()

    if not user_query:
        raise ValueError("The user query cannot be empty.")

    if args.candidate_limit <= 0:
        raise ValueError(
            "candidate-limit must be positive"
        )

    if args.result_limit <= 0:
        raise ValueError(
            "result-limit must be positive"
        )

    if args.candidate_limit < args.result_limit:
        raise ValueError(
            "candidate-limit cannot be smaller than result-limit"
        )

    print_section("USER QUESTION")
    print(user_query)

    planner = QueryPlanningAgent()
    plan = planner.plan(user_query)

    print_plan(plan)

    if not plan.in_scope:
        print_section("REQUEST REJECTED")
        print(
            plan.rejection_reason
            or "The request is outside the supported scope."
        )
        return

    if plan.clarification_needed:
        print_section("CLARIFICATION REQUIRED")
        print(
            plan.clarification_question
            or (
                "Please provide a more specific event category "
                "or date range."
            )
        )
        return

    if plan.retrieval_query is None:
        raise RuntimeError(
            "The planner returned no retrieval query."
        )

    if plan.event_category is None:
        raise RuntimeError(
            "The planner returned no event category."
        )

    query_filter = build_query_filter(plan)
    print_filter(query_filter)

    print_section("EMBEDDING QUERY")
    print(plan.retrieval_query)

    embedding_client = OllamaEmbeddingClient()
    query_vector = embedding_client.embed_query(
        plan.retrieval_query
    )

    print(
        f"\nEmbedding dimensions: {len(query_vector)}"
    )

    store = SECQdrantStore()

    results = store.search(
        query_vector=query_vector,
        query_filter=query_filter,
        event_category=plan.event_category.value,
        candidate_limit=args.candidate_limit,
        limit=args.result_limit,
        max_chunks_per_filing=2,
    )

    print_results(results)


if __name__ == "__main__":
    main()