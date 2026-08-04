from dataclasses import dataclass
from typing import Any

from src.agents.schemas import QueryPlan
from src.embeddings.ollama_client import OllamaEmbeddingClient
from src.retrieval.coverage import validate_evidence_coverage
from src.retrieval.evidence import build_evidence_items
from src.retrieval.qdrant_store import SECQdrantStore
from src.retrieval.query_filters import build_query_filter
from src.retrieval.schemas import CoverageResult, EvidenceItem


@dataclass(frozen=True)
class RetrievalOutcome:
    """
    Final result produced by the deterministic retrieval coordinator.
    """

    evidence: list[EvidenceItem]
    coverage: CoverageResult
    attempts: int
    retried_symbols: list[str]


class RetrievalCoordinator:
    """
    Coordinates initial retrieval and bounded retries.

    Retry policy:
    1. Run the original combined retrieval.
    2. Check entity coverage.
    3. For each missing symbol, run one symbol-specific retry.
    4. Merge and deduplicate evidence.
    5. Validate final coverage.

    This class never performs open-ended retries.
    """

    def __init__(
        self,
        *,
        embedding_client: OllamaEmbeddingClient | None = None,
        store: SECQdrantStore | None = None,
    ) -> None:
        self.embedding_client = (
            embedding_client or OllamaEmbeddingClient()
        )
        self.store = store or SECQdrantStore()

    def retrieve(
        self,
        *,
        plan: QueryPlan,
        candidate_limit: int = 30,
        result_limit: int = 8,
        retry_candidate_limit: int = 50,
        retry_result_limit: int = 6,
        max_chunks_per_filing: int = 2,
    ) -> RetrievalOutcome:
        self._validate_plan(plan)
        self._validate_limits(
            candidate_limit=candidate_limit,
            result_limit=result_limit,
            retry_candidate_limit=retry_candidate_limit,
            retry_result_limit=retry_result_limit,
        )

        query_vector = self.embedding_client.embed_query(
            plan.retrieval_query or ""
        )

        initial_results = self._search(
            plan=plan,
            query_vector=query_vector,
            candidate_limit=candidate_limit,
            result_limit=result_limit,
            max_chunks_per_filing=max_chunks_per_filing,
        )

        evidence = build_evidence_items(initial_results)

        coverage = validate_evidence_coverage(
            plan=plan,
            evidence=evidence,
        )

        attempts = 1
        retried_symbols: list[str] = []

        if not coverage.retry_recommended:
            return RetrievalOutcome(
                evidence=evidence,
                coverage=coverage,
                attempts=attempts,
                retried_symbols=retried_symbols,
            )

        for symbol in coverage.missing_symbols:
            retry_plan = plan.model_copy(
                update={
                    "symbols": [symbol],
                }
            )

            retry_query = self._build_retry_query(
                plan=retry_plan,
                symbol=symbol,
            )

            retry_vector = self.embedding_client.embed_query(
                retry_query
            )

            retry_results = self._search(
                plan=retry_plan,
                query_vector=retry_vector,
                candidate_limit=retry_candidate_limit,
                result_limit=retry_result_limit,
                max_chunks_per_filing=max_chunks_per_filing,
            )

            retry_evidence = build_evidence_items(
                retry_results
            )

            evidence = merge_evidence(
                existing=evidence,
                incoming=retry_evidence,
            )

            attempts += 1
            retried_symbols.append(symbol)

        final_coverage = validate_evidence_coverage(
            plan=plan,
            evidence=evidence,
        )

        return RetrievalOutcome(
            evidence=evidence,
            coverage=final_coverage,
            attempts=attempts,
            retried_symbols=retried_symbols,
        )

    def _search(
        self,
        *,
        plan: QueryPlan,
        query_vector: list[float],
        candidate_limit: int,
        result_limit: int,
        max_chunks_per_filing: int,
    ) -> list[dict[str, Any]]:
        query_filter = build_query_filter(plan)

        if plan.event_category is None:
            raise ValueError(
                "Retrieval requires an event category"
            )

        return self.store.search(
            query_vector=query_vector,
            query_filter=query_filter,
            event_category=plan.event_category.value,
            candidate_limit=candidate_limit,
            limit=result_limit,
            max_chunks_per_filing=max_chunks_per_filing,
        )

    @staticmethod
    def _build_retry_query(
        *,
        plan: QueryPlan,
        symbol: str,
    ) -> str:
        """
        Build a broader semantic query for a missing company.

        Symbol restrictions remain metadata filters. The symbol is
        included in the semantic text only as an additional retrieval
        clue.
        """
        base_query = plan.retrieval_query or ""

        return (
            f"{symbol} {base_query} company announcement "
            "reported disclosed results update"
        ).strip()

    @staticmethod
    def _validate_plan(plan: QueryPlan) -> None:
        if not plan.in_scope:
            raise ValueError(
                "Cannot retrieve for an out-of-scope plan"
            )

        if plan.clarification_needed:
            raise ValueError(
                "Cannot retrieve until clarification is resolved"
            )

        if not plan.retrieval_query:
            raise ValueError(
                "Retrieval plan is missing retrieval_query"
            )

        if plan.event_category is None:
            raise ValueError(
                "Retrieval plan is missing event_category"
            )

    @staticmethod
    def _validate_limits(
        *,
        candidate_limit: int,
        result_limit: int,
        retry_candidate_limit: int,
        retry_result_limit: int,
    ) -> None:
        values = {
            "candidate_limit": candidate_limit,
            "result_limit": result_limit,
            "retry_candidate_limit": retry_candidate_limit,
            "retry_result_limit": retry_result_limit,
        }

        for name, value in values.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if candidate_limit < result_limit:
            raise ValueError(
                "candidate_limit cannot be smaller than result_limit"
            )

        if retry_candidate_limit < retry_result_limit:
            raise ValueError(
                "retry_candidate_limit cannot be smaller than "
                "retry_result_limit"
            )


def merge_evidence(
    *,
    existing: list[EvidenceItem],
    incoming: list[EvidenceItem],
) -> list[EvidenceItem]:
    """
    Merge evidence while removing duplicate Qdrant chunks.

    point_id is the canonical identifier because it is deterministically
    derived from accession_number and chunk_id.
    """
    merged_by_point_id: dict[str, EvidenceItem] = {
        item.point_id: item
        for item in existing
    }

    for item in incoming:
        current = merged_by_point_id.get(item.point_id)

        if current is None:
            merged_by_point_id[item.point_id] = item
            continue

        if item.rerank_score > current.rerank_score:
            merged_by_point_id[item.point_id] = item

    ordered = sorted(
        merged_by_point_id.values(),
        key=lambda item: (
            item.rerank_score,
            item.vector_score,
        ),
        reverse=True,
    )

    return [
        item.model_copy(
            update={
                "evidence_id": f"evidence-{index}",
            }
        )
        for index, item in enumerate(
            ordered,
            start=1,
        )
    ]